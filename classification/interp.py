import math
import os
from pathlib import Path
from random import randint
import uuid

import tqdm
import torch
import wandb
from classification import config_classifier as config
from classification.train_nerf2vec import NeRFDataset
from classification.utils import generate_rays, get_class_label_from_nerf_root_path, pose_spherical
from models.encoder import Encoder
from models.idecoder import ImplicitDecoder
from nerf.utils import Rays, namedtuple_map, render_image
from nerf.intant_ngp import NGPradianceField
from nerfacc import OccupancyGrid, contract_inv, ray_marching, render_visibility, rendering

import numpy as np
import imageio.v2 as imageio

from torch.cuda.amp import autocast
import torch.nn.functional as F


def render_with_multiple_camera_poses(device, embeddings, decoder, renderings_root_path, scene_aabb, render_step_size, color_bkgds):

    width = 224
    height = 224
    camera_angle_x = 0.8575560450553894 # Parameter taken from traned NeRFs
    focal_length = 0.5 * width / np.tan(0.5 * camera_angle_x)

    max_images = 6
    array = np.linspace(-30.0, 30.0, max_images//2, endpoint=False)
    array = np.append(array, np.linspace(
        30.0, -30.0, max_images//2, endpoint=False))
    
    for emb_idx in range(len(embeddings)):
        # rgb_frames = []
        for n_img, theta in tqdm.tqdm(enumerate(np.linspace(0.0, 360.0, max_images, endpoint=False))):
            c2w = pose_spherical(torch.tensor(theta), torch.tensor(array[n_img]), torch.tensor(1.5))
            c2w = c2w.to(device)
            rays = generate_rays(device, width, height, focal_length, c2w)
            with autocast():
                rgb, _, _, _, _, _ = render_image(
                        radiance_field=decoder,
                        embeddings=embeddings[emb_idx].unsqueeze(dim=0),
                        occupancy_grid=None,
                        rays=Rays(origins=rays.origins.unsqueeze(dim=0), viewdirs=rays.viewdirs.unsqueeze(dim=0)),
                        scene_aabb=scene_aabb,
                        render_step_size=render_step_size,
                        render_bkgd=color_bkgds.unsqueeze(dim=0),
                        grid_weights=None
                    )
            
            full_path = os.path.join(renderings_root_path, f'int{emb_idx}_img{n_img}.png')
            imageio.imwrite(
                full_path,
                (rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8)
            )

@torch.no_grad()
def baseline_interpolation(
    device, 
    ngp_mlp_weights,
    rays,
    curr_bkgd,
    img_path):

    radiance_field = NGPradianceField(**config.INSTANT_NGP_MLP_CONF).to(device)
    radiance_field.eval()

    scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / config.GRID_CONFIG_N_SAMPLES
    ).item()

    radiance_field.load_state_dict(ngp_mlp_weights)

    rays_shape = rays.origins.shape
    height, width, _ = rays_shape
    num_rays = height * width
    rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)

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
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = ray_marching(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            grid=None,
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
            render_bkgd=curr_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)

    rgbs, _, _, _ = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]

    rgbs = rgbs.view((*rays_shape[:-1], -1))
    
    if img_path:
        imageio.imwrite(
            img_path,
            (rgbs.cpu().detach().numpy() * 255).astype(np.uint8)
        )
    

def nerf2vec_baseline_comparisons(
        rays, 
        color_bkgds, 
        embeddings, 
        matrices_unflattened_A, 
        matrices_unflattened_B,
        decoder,
        scene_aabb,
        render_step_size,
        curr_folder_path,
        device):

    # #########################
    # nerf2vec EMBEDDINGS
    # #########################
    for idx in range(len(embeddings)):
        with autocast():
            rgb, _, _, _, _, _ = render_image(
                    radiance_field=decoder,
                    embeddings=embeddings[idx].unsqueeze(dim=0),
                    occupancy_grid=None,
                    rays=Rays(origins=rays.origins.unsqueeze(dim=0), viewdirs=rays.viewdirs.unsqueeze(dim=0)),
                    scene_aabb=scene_aabb,
                    render_step_size=render_step_size,
                    render_bkgd=color_bkgds.unsqueeze(dim=0),
                    grid_weights=None
                )
        
        img_name = f'{idx}.png'
        full_path = os.path.join(curr_folder_path, img_name)
        
        imageio.imwrite(
            full_path,
            (rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8)
        )

    # ##################################################
    # BASELINE (direct interpolation of NeRF weights)
    # ##################################################
    """
    ngp_weights = [{'mlp_base.params':matrices_unflattened_A}]
    for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        emb_interp = (1 - gamma) * matrices_unflattened_A + gamma * matrices_unflattened_B
        ngp_weights.append({'mlp_base.params':emb_interp})
    ngp_weights.append({'mlp_base.params': matrices_unflattened_B})

    for idx in range(len(ngp_weights)):
        img_name = f'{idx}_baseline.png'
        full_path = os.path.join(curr_folder_path, img_name)

        baseline_interpolation(
            device, 
            ngp_weights[idx], 
            rays,
            color_bkgds,
            full_path
        )
    """


# TODO: a couple of new tasks has been added, without being very well structured in the current module.
#       Therefore, it is necessary to do some refactoring.
@torch.no_grad()
def interpolate():

    device = 'cuda'
    scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / config.GRID_CONFIG_N_SAMPLES
    ).item()

    # TODO: decide which weight to load (best vs last)
    # ckpt_path = ckpt_path / ""ckpts/best.pt""

    ckpts_path = Path(os.path.join('classification', 'train', 'ckpts'))
    ckpt_paths = [p for p in ckpts_path.glob("*.pt") if "best" not in p.name]
    ckpt_path = ckpt_paths[0]
    ckpt = torch.load(ckpt_path)
    
    print(f'loaded weights: {ckpt_path}')

    encoder = Encoder(
                config.MLP_UNITS,
                config.ENCODER_HIDDEN_DIM,
                config.ENCODER_EMBEDDING_DIM
                )
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.cuda()
    encoder.eval()

    decoder = ImplicitDecoder(
            embed_dim=config.ENCODER_EMBEDDING_DIM,
            in_dim=config.DECODER_INPUT_DIM,
            hidden_dim=config.DECODER_HIDDEN_DIM,
            num_hidden_layers_before_skip=config.DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP,
            num_hidden_layers_after_skip=config.DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP,
            out_dim=config.DECODER_OUT_DIM,
            encoding_conf=config.INSTANT_NGP_ENCODING_CONF,
            aabb=torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
        )
    decoder.load_state_dict(ckpt["decoder"])
    decoder = decoder.cuda()
    decoder.eval()

    split = 'validation'
    dset_json = os.path.abspath(os.path.join('data', f'{split}.json'))  
    dset = NeRFDataset(dset_json, device='cpu')  

    n_images = 0
    max_images = 100
    
    while n_images < max_images:
        idx_A = randint(0, len(dset) - 1)
        train_nerf_A, test_nerf_A, matrices_unflattened_A, matrices_flattened_A, _, data_dir_A, _, _ = dset[idx_A]
        class_id_A = get_class_label_from_nerf_root_path(data_dir_A)
        if "_A1" in data_dir_A or "_A2" in data_dir_A:
            continue
        matrices_unflattened_A = matrices_unflattened_A['mlp_base.params']

        class_id_B = -1
        while class_id_B != class_id_A:
            idx_B = randint(0, len(dset) - 1)
            train_nerf_B, test_nerf_B, matrices_unflattened_B, matrices_flattened_B, _, data_dir_B, _, _ = dset[idx_B]
            class_id_B = get_class_label_from_nerf_root_path(data_dir_B)
        
        if "_A1" in data_dir_B or "_A2" in data_dir_B:
            continue
        matrices_unflattened_B = matrices_unflattened_B['mlp_base.params']
        
        print(f'Progress: {n_images}/{max_images}')
        
        matrices_flattened_A = matrices_flattened_A.cuda().unsqueeze(0)
        matrices_flattened_B = matrices_flattened_B.cuda().unsqueeze(0)

        with autocast():
            embedding_A = encoder(matrices_flattened_A).squeeze(0)  
            embedding_B = encoder(matrices_flattened_B).squeeze(0)  
        

        embeddings = [embedding_A]
        for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            emb_interp = (1 - gamma) * embedding_A + gamma * embedding_B
            embeddings.append(emb_interp)
        embeddings.append(embedding_B)

        plots_path = f'plots_interp_{split}'
        curr_folder_path = os.path.join(plots_path, str(uuid.uuid4()))    
        os.makedirs(curr_folder_path, exist_ok=True)

        rays = test_nerf_A['rays']
        color_bkgds = test_nerf_A['color_bkgd']
        rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
        color_bkgds = color_bkgds.cuda()
        
        # Interpolation
        nerf2vec_baseline_comparisons(
            rays, 
            color_bkgds, 
            embeddings,
            matrices_unflattened_A,
            matrices_unflattened_B,
            decoder,
            scene_aabb,
            render_step_size,
            curr_folder_path,
            device
        )

        """
        # 3D shape consistency 
        render_with_multiple_camera_poses(
            device, 
            embeddings, 
            decoder, 
            curr_folder_path, 
            scene_aabb, 
            render_step_size, 
            color_bkgds)
        """
        
        n_images += 1
