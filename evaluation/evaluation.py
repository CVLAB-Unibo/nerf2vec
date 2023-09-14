import json
import math
import os
import time
from nerfacc import OccupancyGrid
import torch
from torch.utils.data import DataLoader, Dataset

from classification import config_classifier as config
from classification.create_renderings import get_rays
from classification.utils import get_mlp_params_as_matrix
from models.baseline_classifier import Resnet50Classifier
from models.encoder import Encoder
from models.fc_classifier import FcClassifier

from typing import Any, Dict, List, Tuple

from nerf.intant_ngp import NGPradianceField
from nerf.utils import namedtuple_map

from nerfacc import OccupancyGrid, ray_marching, rendering

"""
    1) Prepare a dataset of 1k NeRFs (or just take val/test sets)
    2) nerf2vec:
        - load nerf2vec's encoder weights from classification/train
        - load fc_classifier's weights from classification/classifier_embeddings folder
        - create a module that calls the encoder and then classify each NeRF, one at a time
        - record the time required for the encoding operation, and for classify the embeddings
    3) ResNet:
        - get minimal code so as to get a single rendering for a single NeRF (see create_renderings.py module)
        - load baseline_classifier's weights from classification/classifier_baseline_images_no_interp (or the current training in progress)
        - create a module that creates a rendering and then classifies each NeRF, one at a time
        - record the time required for the creation of the rendering, and for classify the image
    4) do comparisons between results obtained in the previous samples
    
"""

from classification.train_nerf2vec import NeRFDataset, Nerf2vecTrainer

class NeRFWeightsDataset(Dataset):
    def __init__(self, split_json: str, device: str) -> None:
        super().__init__()

        with open(split_json) as file:
            self.nerf_paths = json.load(file)
        
        # self.nerf_paths = self._get_nerf_paths('data\\data_TRAINED')
        assert isinstance(self.nerf_paths, list), 'The json file provided is not a list.'

        self.device = device

    def __len__(self) -> int:
        return len(self.nerf_paths)

    def __getitem__(self, index) -> Any:

        data_dir = self.nerf_paths[index]
        weights_file_path = 'bb07_steps3000_encodingFrequency_mlpFullyFusedMLP_activationReLU_hiddenLayers3_units64_encodingSize24.pth'

        matrix_unflattened = torch.load(weights_file_path, map_location=torch.device(self.device))  # The NeRF weights obtained from NerfAcc
        matrix_flattened = get_mlp_params_as_matrix(matrix_unflattened['mlp_base.params'])  # The NeRF weights with proper padding

        grid_weights_path = os.path.join(data_dir, 'grid.pth')  
        grid_weights = torch.load(grid_weights_path, map_location=self.device)
        grid_weights['_binary'] = grid_weights['_binary'].to_dense()#.unsqueeze(dim=0)
        n_total_cells = 884736  # TODO: add this as config parameter
        grid_weights['occs'] = torch.empty([n_total_cells])   # 884736 if resolution == 96 else 2097152
        

        return matrix_unflattened, matrix_flattened, grid_weights, data_dir


@torch.no_grad()
def evaluate_nerf2vec_classification(device='cuda:0'):

    # DATASET
    val_dset_json = os.path.abspath(os.path.join('data', 'validation.json'))  
    val_dset = NeRFWeightsDataset(val_dset_json, device='cpu')   
    val_loader = DataLoader(
        val_dset,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )

    # NERF2VEC ENCODER
    encoder = Encoder(
        config.MLP_UNITS,
        config.ENCODER_HIDDEN_DIM,
        config.ENCODER_EMBEDDING_DIM
    )
    encoder = encoder.to(device)
    encoder.eval()

    encoder_ckpt_path = os.path.join('classification', 'train', 'ckpts', '830.pt')
    print(f'loading weights: {encoder_ckpt_path}')
    ckpt = torch.load(encoder_ckpt_path)
    encoder.load_state_dict(ckpt["encoder"])
    
    # CLASSIFIER
    layers_dim = config.LAYERS_DIM
    num_classes = config.NUM_CLASSES
    classifier = FcClassifier(layers_dim, num_classes)
    classifier = classifier.to(device)
    classifier.eval()

    classifier_ckpt_path = os.path.join('classification', 'classifier_embeddings', 'ckpts', '149.pt')
    ckpt = torch.load(classifier_ckpt_path)
    classifier.load_state_dict(ckpt["net"])

    # EVALUATION
    encoding_times = []
    classification_times = []
    for batch_idx, batch in enumerate(val_loader):
        matrix_unflattened, matrix_flattened, grid_weights, data_dir = batch
        
        start = time.time()
        embeddings = encoder(matrix_flattened)
        end = time.time()
        encoding_times.append(end-start)

        start = time.time()
        pred = classifier(embeddings)
        end = time.time()
        classification_times.append(end-start)
    
    average_encoding_time = sum(encoding_times) / len(encoding_times)
    average_classification_time = sum(classification_times) / len(classification_times)

    print(f'average_encoding_time: {average_encoding_time}s')
    print(f'average_classification_time: {average_classification_time}s')


def generate_image(radiance_field, radiance_field_weights, device, occupancy_grid, grid_weights, scene_aabb, render_step_size, bkgd):
    
    ngp_mlp_weights = torch.load(radiance_field_weights, map_location=torch.device(device))
    radiance_field.load_state_dict(ngp_mlp_weights)
    
    if not grid_weights:
        occupancy_grid = None
    else:
        grid_weights['_binary'] = grid_weights['_binary'].to_dense()
        n_total_cells = 884736  # TODO: add this as config parameter
        grid_weights['occs'] = torch.empty([n_total_cells])   # 884736 if resolution == 96 else 2097152
        occupancy_grid.load_state_dict(grid_weights)

    rays = get_rays()
    rays_shape = rays.origins.shape
    height, width, _ = rays_shape
    num_rays = height * width
    rays_reshaped = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        
        _, density = radiance_field._query_density_and_rgb(positions, None)
        return density

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        
        return radiance_field(positions, t_dirs)
    
    chunk = 4096
    results = []

    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays_reshaped)
        ray_indices, t_starts, t_ends = ray_marching(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            grid=occupancy_grid,
            sigma_fn=sigma_fn,
            near_plane=None,
            far_plane=None,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=0.0,
            alpha_thre=0.0,
        )
        rgb, opacity, depth = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)

    rgbs, _, _, _ = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]

    rgbs = rgbs.view((*rays_shape[:-1], -1))
    return rgbs
    
        

@torch.no_grad()
def evaluate_baseline_classification(device='device', use_occupancy_grid=False):
    # DATASET
    val_dset_json = os.path.abspath(os.path.join('data', 'validation.json'))  
    val_dset = NeRFWeightsDataset(val_dset_json, device='cpu')   
    val_loader = DataLoader(
        val_dset,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )

    # Radiance field 
    radiance_field = NGPradianceField(**config.INSTANT_NGP_MLP_CONF).to(device)
    radiance_field.eval()

    # Occupancy grid
    occupancy_grid = OccupancyGrid(
                roi_aabb=config.GRID_AABB,
                resolution=config.GRID_RESOLUTION,
                contraction_type=config.GRID_CONTRACTION_TYPE,
            ).to(device)
    occupancy_grid.eval()

    # Other parameters required 
    scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / config.GRID_CONFIG_N_SAMPLES
    ).item()
    
    bkgd = torch.zeros(3, device=device)

    # CLASSIFIER
    num_classes = config.NUM_CLASSES
    classifier =  Resnet50Classifier(num_classes, interpolate_features=False)
    classifier = classifier.to(device)

    classifier_ckpt_path = os.path.join('classification', 'classifier_baseline_images_no_interp', 'ckpts', '149.pt')
    ckpt = torch.load(classifier_ckpt_path)
    classifier.load_state_dict(ckpt["net"])

    # EVALUATION
    rendering_times = []
    classification_times = []
    for batch_idx, batch in enumerate(val_loader):
        matrix_unflattened, matrix_flattened, grid_weights, data_dir = batch

        if not use_occupancy_grid:
            grid_weights = None
        
        start = time.time()
        image = generate_image(radiance_field, matrix_unflattened, device, occupancy_grid, grid_weights, scene_aabb, render_step_size, bkgd)
        end = time.time()
        rendering_times.append(end-start)

        start = time.time()
        pred = classifier(image)
        end = time.time()
        classification_times.append(end-start)
    
    average_rendering_time = sum(rendering_times) / len(rendering_times)
    average_classification_time = sum(classification_times) / len(classification_times)

    print(f'average_rendering_time: {average_rendering_time}s')
    print(f'average_classification_time: {average_classification_time}s')

