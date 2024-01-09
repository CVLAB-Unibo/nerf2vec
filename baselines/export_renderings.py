"""
Module used to export renderings to train baseline classifiers based on ResNet50
"""

import json
import math
import os
import shutil
from classification import config_classifier as config
from classification.export_embeddings import find_duplicate_folders
from classification.utils import generate_rays, pose_spherical
from nerf.utils import namedtuple_map
from nerf.loader_gt import _load_renderings
from nerf.intant_ngp import NGPradianceField
from nerfacc import OccupancyGrid, ray_marching, rendering
import torch

import numpy as np
import imageio.v2 as imageio

def get_rays(
        device,
        camera_angle_x=0.8575560450553894,  # Parameter taken from traned NeRFs
        width=224,
        height=224):

    # Get camera pose
    theta = torch.tensor(90.0, device=device) # The horizontal camera position (change the value between and 360 to make a full cycle around the object)
    phi = torch.tensor(-30.0, device=device) # The vertical camera position
    t = torch.tensor(1.5, device=device) # camera distance from object
    c2w = pose_spherical(theta, phi, t)
    c2w = c2w.to(device)

    # Compute the focal_length 
    focal_length = 0.5 * width / np.tan(0.5 * camera_angle_x)

    rays = generate_rays(device, width, height, focal_length, c2w)
    
    return rays

def read_json(split, root_folder='data'):
    f_path = os.path.abspath(os.path.join(root_folder, f'{split}.json'))
    with open(f_path) as file:
        return json.load(file)

def process_split(
        split,
        nerf_paths, 
        radiance_field,
        occupancy_grid,
        device,
        rays,
        scene_aabb,
        render_step_size,
        bkgd,
        ngp_mlp_weights_file_name: str = "bb07_steps3000_encodingFrequency_mlpFullyFusedMLP_activationReLU_hiddenLayers3_units64_encodingSize24.pth",
        grid_file_name: str = "grid.pth",
        rendering_path: str = 'baseline_classifier_renderings'):

    # Create directories if required
    split_path = os.path.join(rendering_path, split)
    os.makedirs(split_path, exist_ok=True)
    
    n_generated_images = 0
    for data_dir in nerf_paths:
        ngp_mlp_weights_file_path = os.path.join(data_dir, ngp_mlp_weights_file_name)
        ngp_mlp_weights = torch.load(ngp_mlp_weights_file_path, map_location=torch.device(device))
        radiance_field.load_state_dict(ngp_mlp_weights)
        
        grid_weights_path = os.path.join(data_dir, grid_file_name) 
        grid_weights = torch.load(grid_weights_path, map_location=device)
        
        grid_weights['_binary'] = grid_weights['_binary'].to_dense()
        n_total_cells = 884736  # TODO: add this as config parameter
        grid_weights['occs'] = torch.empty([n_total_cells])   # 884736 if resolution == 96 else 2097152
        occupancy_grid.load_state_dict(grid_weights)

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
        
        data_dir_splitted = data_dir.split('/')
        img_name = f'{data_dir_splitted[-2]}_{data_dir_splitted[-1]}.png'
        img_path = os.path.join(split_path, img_name)
        imageio.imwrite(
            img_path,
            (rgbs.cpu().detach().numpy() * 255).astype(np.uint8)
        )

        n_generated_images+=1
        if n_generated_images % 500 == 0:
            print(f'{split} split: {n_generated_images}/{len(nerf_paths)}')

def process_split_multiple_poses(
        split,
        nerf_paths, 
        radiance_field,
        occupancy_grid,
        device,
        rays,
        scene_aabb,
        render_step_size,
        bkgd,
        ngp_mlp_weights_file_name: str = "bb07_steps3000_encodingFrequency_mlpFullyFusedMLP_activationReLU_hiddenLayers3_units64_encodingSize24.pth",
        grid_file_name: str = "grid.pth",
        rendering_path: str = 'baseline_classifier_renderings_multi_poses'):

    # Create directories if required
    split_path = os.path.join(rendering_path, split)
    os.makedirs(split_path, exist_ok=True)
    
    with open(os.path.join('data', 'duplicated_NeRFs.json'), 'r') as f:
        folders_to_skip = json.load(f)
    
    n_processed_nerfs = 0
    n_skipped_folders = 0

    for data_dir in nerf_paths:

        folder_to_check = f"{data_dir.split('/')[-2]}_{data_dir.split('/')[-1]}"
        if folder_to_check in folders_to_skip:
            n_skipped_folders += 1
            continue
        
        WIDTH = 224
        HEIGHT = 224
        camtoworlds, focal = _load_renderings(data_dir, 'train', HEIGHT, WIDTH)

        ngp_mlp_weights_file_path = os.path.join(data_dir, ngp_mlp_weights_file_name)
        ngp_mlp_weights = torch.load(ngp_mlp_weights_file_path, map_location=torch.device(device))
        radiance_field.load_state_dict(ngp_mlp_weights)
        
        grid_weights_path = os.path.join(data_dir, grid_file_name) 
        grid_weights = torch.load(grid_weights_path, map_location=device)
        
        grid_weights['_binary'] = grid_weights['_binary'].to_dense()
        n_total_cells = 884736  # TODO: add this as config parameter
        grid_weights['occs'] = torch.empty([n_total_cells])   # 884736 if resolution == 96 else 2097152
        occupancy_grid.load_state_dict(grid_weights)

        

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
        
        # chunk = 4096
        chunk = 200000

        for pose_idx, c2w in enumerate(camtoworlds):

            results = []

            # Generate rays
            c2w = torch.tensor(camtoworlds[pose_idx], device=device, dtype=torch.float32)
            rays = generate_rays(device, WIDTH, HEIGHT, focal, c2w)

            rays_shape = rays.origins.shape
            height, width, _ = rays_shape
            num_rays = height * width
            rays_reshaped = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)

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
            
            data_dir_splitted = data_dir.split('/')
            img_name = f'{data_dir_splitted[-2]}_{data_dir_splitted[-1]}_{pose_idx}.png'
            img_path = os.path.join(split_path, img_name)
            imageio.imwrite(
                img_path,
                (rgbs.cpu().detach().numpy() * 255).astype(np.uint8)
            )

        n_processed_nerfs+=1
        if n_processed_nerfs % 1000 == 0:
            print(f'{split} split: {n_processed_nerfs}/{len(nerf_paths)}')
        
        # if n_processed_nerfs == 2:
        #    break
    
    print(f'Skipped {n_skipped_folders} in {split} split')

def export_baseline_renderings():
    device = 'cuda:0'
    train_nerf_paths = read_json('train')
    validation_nerf_paths = read_json('validation')
    test_nerf_paths = read_json('test')

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


    rays = get_rays(device)
    
    bkgd = torch.zeros(3, device=device)

    splits = ['train', 'val', 'test']
    nerf_paths = [train_nerf_paths, validation_nerf_paths, test_nerf_paths]
    for i in range(len(splits)):
        split = splits[i]
        nerf_path = nerf_paths[i]
        
        process_split_multiple_poses(
            split, 
            nerf_path, 
            radiance_field, 
            occupancy_grid, 
            device, 
            rays, 
            scene_aabb, 
            render_step_size, 
            bkgd
        )

# ################################################################################
# TEMPORARY CODE
# Keep consistency with the elements used for training the embeddings classifier
# ################################################################################     
"""   
def clear_baseline_renderings():
    
    folders_to_skip_file = 'folders_to_skip.json'
    if not os.path.exists(folders_to_skip_file):
        folders_to_skip = {}
        root_folders = ['data/data_TRAINED', 'data/data_TRAINED_A1', 'data/data_TRAINED_A2']
        for root_folder in root_folders:
            print(f'processing folder: {root_folder}')
            duplicated_folders = find_duplicate_folders(root_folder)
            for folder1, folder2 in duplicated_folders:
                # print(f"Duplicated folders: {folder1} and {folder2}")
                # print(folder1)
                # print(folder2)
                folders_to_skip[f"{folder1.split('/')[-2]}_{folder1.split('/')[-1]}.png"] = True
                folders_to_skip[f"{folder2.split('/')[-2]}_{folder2.split('/')[-1]}.png"] = True
        
        # Open a file in write mode
        with open("folders_to_skip.json", "w") as f:
            # Save the dictionary to the file
            json.dump(folders_to_skip, f)
    else:
       with open(folders_to_skip_file, 'r') as json_file:
            folders_to_skip = json.load(json_file) 

    invalid_classes = ['02992529', '03948459']
    
    
    root_folder = 'data/baseline_classifier_renderings'
    discarded_folder = os.path.join(root_folder, 'discarded')

    splits = ['train', 'validation', 'test']
    total = 0
    for split in splits:
        full_path = os.path.join(root_folder, split)

        for file in os.listdir(full_path):
            
            
            discard_file = False

            class_name = file.split('_')[0]
            if class_name in invalid_classes:
                discard_file = True
            
            if file in folders_to_skip and not discard_file:
                discard_file = True
            
            if discard_file:
                curr_file_path = os.path.join(full_path, file)
                new_file_path = os.path.join(discarded_folder, f'{split}_{file}')
                print(f'from {curr_file_path} to {new_file_path}')
                shutil.move(curr_file_path, new_file_path)
                total += 1
    
    print(total)
"""