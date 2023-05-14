import gzip
import math
import os
from nerfacc import ContractionType, OccupancyGrid
import torch
from classification import config
from nerf.intant_ngp import NGPradianceField


def generate_occupancy_grid(
        device, 
        weights_path, 
        nerf_dict, 
        aabb, 
        n_iterations, 
        n_warmups):

        radiance_field = NGPradianceField(
                **nerf_dict
            ).to(device)
        
        matrix = torch.load(weights_path)
        radiance_field.load_state_dict(matrix)
        radiance_field.eval()

        # Create the OccupancyGrid
        render_n_samples = 1024
        grid_resolution = 128
        
        contraction_type = ContractionType.AABB
        scene_aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        near_plane = None
        far_plane = None
        render_step_size = (
            (scene_aabb[3:] - scene_aabb[:3]).max()
            * math.sqrt(3)
            / render_n_samples
        ).item()
        alpha_thre = 0.0

        occupancy_grid = OccupancyGrid(
            roi_aabb=aabb,
            resolution=grid_resolution,
            contraction_type=contraction_type,
        ).to(device)
        occupancy_grid.eval()

        with torch.no_grad():
            for i in range(n_iterations):
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
                    warmup_steps=n_warmups
                )

        return occupancy_grid


nerf_paths = []
nerfs_root = 'data'
nerf_weights_file_name = 'bb07_steps3000_encodingFrequency_mlpFullyFusedMLP_activationReLU_hiddenLayers3_units64_encodingSize24.pth'
compressed_grid_file_name = 'grid.pth.gz'
grid_file_name = 'grid.pth'

for class_name in os.listdir(nerfs_root):

    subject_dirs = os.path.join(nerfs_root, class_name)

    # Sometimes there are hidden files (e.g., when unzipping a file from a Mac)
    if not os.path.isdir(subject_dirs):
        continue
    
    for subject_name in os.listdir(subject_dirs):
        subject_dir = os.path.join(subject_dirs, subject_name)
        weights_dir = os.path.join(subject_dir, nerf_weights_file_name)

        grid = generate_occupancy_grid(
             device='cuda:0',
             weights_path=weights_dir,
             nerf_dict=config.INSTANT_NGP_MLP_CONF,
             aabb=config.AABB,
             n_iterations=config.OCCUPANCY_GRID_RECONSTRUCTION_ITERATIONS,
             n_warmups=config.OCCUPANCY_GRID_WARMUP_ITERATIONS

        )

        dict = {
            'occs': grid.state_dict()['occs'].half(),
            '_roi_aabb': grid.state_dict()['_roi_aabb'].half(),
            '_binary': grid.state_dict()['_binary'].to_sparse(),
            'resolution': grid.state_dict()['resolution'].half()
        }

        grid_dir = os.path.join(subject_dir, compressed_grid_file_name)
        with gzip.open(grid_dir, 'wb') as f:
            torch.save(dict, f)
        
        grid_dir = os.path.join(subject_dir, grid_file_name)
        torch.save(dict, grid_dir)

