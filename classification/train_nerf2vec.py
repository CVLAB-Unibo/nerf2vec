import math
import time

from nerfacc import ContractionType, OccupancyGrid
from nerf.utils import get_mlp_params_as_matrix, next_multiple

import os
import torch
from torch.utils.data import DataLoader, Dataset

from nerf.loader import NeRFLoader

from pathlib import Path
from random import randint
from typing import Any, Dict, Tuple

from classification.ngp_nerf2vec import NGPradianceField
from classification import config

class NeRFDataset(Dataset):
    def __init__(self, nerfs_root: str, sample_sd: Dict[str, Any], device: str) -> None:
        super().__init__()

        self.nerf_paths = self._get_nerf_paths(nerfs_root)

        self.device = device

    def __len__(self) -> int:
        return len(self.nerf_paths)

    def __getitem__(self, index) -> Any:

        dataset_kwargs = {}
        data_dir = self.nerf_paths[index]

        nerf_loader = NeRFLoader(
            data_dir=data_dir,
            num_rays=config.NUM_RAYS,
            device=self.device,
            **dataset_kwargs)

        # Get data for the batch
        data = nerf_loader[0]
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        matrix = torch.load(nerf_loader.weights_file_path, map_location=torch.device(self.device))
        matrix = get_mlp_params_as_matrix(matrix['mlp_base.params'])
        
        return rays, pixels, render_bkgd, matrix, nerf_loader.weights_file_path
    
    
    def _get_nerf_paths(self, nerfs_root: str):
        
        nerf_paths = []

        for class_name in os.listdir(nerfs_root):

            subject_dirs = os.path.join(nerfs_root, class_name)

            # Sometimes there are hidden files (e.g., when unzipping a file from a Mac)
            if not os.path.isdir(subject_dirs):
                continue
            
            for subject_name in os.listdir(subject_dirs):
                subject_dir = os.path.join(subject_dirs, subject_name)
                nerf_paths.append(subject_dir)
        
        return nerf_paths

class Nerf2vecTrainer:
    def __init__(self, device='cuda:0') -> None:
        train_dset = NeRFDataset(os.path.abspath('data'), None, device='cpu') 

        self.device = device
        
        self.train_loader = DataLoader(
        train_dset,
        batch_size=16,
        num_workers=4,
        shuffle=True
        )
    
    def train(self):
        for batch in self.train_loader:
            # pixels, color = batch
            rays, pixels, render_bkgd, matrix, nerf_weights_path = batch

            # Move tensors to CUDA 
            start = time.time()
            rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
            pixels = pixels.cuda()
            render_bkgd = render_bkgd.cuda()
            matrix = matrix.cuda()

            end = time.time()
            print(f'moving tensor to CUDA: {end-start}')

            grids = []
            start = time.time()
            for elem in nerf_weights_path:
                grid = self._generate_occupancy_grid(self.device, elem)
                grids.append(grid)
            end = time.time()
            print(f'elapsed: {end-start}')
    
    def _generate_occupancy_grid(self, device, nerf_weights_path):

        radiance_field = NGPradianceField(
                aabb=config.AABB,
                unbounded=False,
                encoding='Frequency',
                mlp='FullyFusedMLP',
                activation='ReLU',
                n_hidden_layers=config.MLP_HIDDEN_LAYERS,
                n_neurons=config.MLP_UNITS,
                encoding_size=config.MLP_ENCODING_SIZE
            ).to(device)
        
        matrix = torch.load(nerf_weights_path)
        radiance_field.load_state_dict(matrix)
        radiance_field.eval()

        # Create the OccupancyGrid
        render_n_samples = 1024
        grid_resolution = 128
        
        contraction_type = ContractionType.AABB
        scene_aabb = torch.tensor(config.AABB, dtype=torch.float32, device=device)
        near_plane = None
        far_plane = None
        render_step_size = (
            (scene_aabb[3:] - scene_aabb[:3]).max()
            * math.sqrt(3)
            / render_n_samples
        ).item()
        alpha_thre = 0.0

        occupancy_grid = OccupancyGrid(
            roi_aabb=config.AABB,
            resolution=grid_resolution,
            contraction_type=contraction_type,
        ).to(device)
        occupancy_grid.eval()

        with torch.no_grad():
            for i in range(config.OCCUPANCY_GRID_RECONSTRUCTION_ITERATIONS):
                def occ_eval_fn(x):
                    step_size = render_step_size
                    _ , density = radiance_field._query_density_and_rgb(x, None)
                    return density * step_size

                # update occupancy grid
                occupancy_grid._update(
                    step=i,
                    occ_eval_fn=occ_eval_fn,
                    occ_thre=1e-2,
                    ema_decay=0.95,
                    warmup_steps=config.OCCUPANCY_GRID_WARMUP_ITERATIONS
                )

        return occupancy_grid