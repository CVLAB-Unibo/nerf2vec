import gzip
import math
import os
import time
from nerfacc import ContractionType, OccupancyGrid
import torch
from classification import config
from nerf.intant_ngp import NGPradianceField


def generate_occupancy_grid(
        device, 
        weights_path, 
        aabb, 
        n_iterations, 
        n_warmups,
        radiance_field):

        matrix = torch.load(weights_path)
        radiance_field.load_state_dict(matrix)
        radiance_field.eval()

        # Create the OccupancyGrid
        render_n_samples = 1024
        grid_resolution = 128
        
        contraction_type = ContractionType.AABB
        scene_aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        render_step_size = (
            (scene_aabb[3:] - scene_aabb[:3]).max()
            * math.sqrt(3)
            / render_n_samples
        ).item()

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


def start_grids_generation(nerf_root):

    nerf_weights_file_name = 'bb07_steps3000_encodingFrequency_mlpFullyFusedMLP_activationReLU_hiddenLayers3_units64_encodingSize24.pth'
    compressed_grid_file_name = 'grid.pth.gz'
    grid_file_name = 'grid.pth'

    generated_grids = 0
    N_GRIDS_LOG = 1000

    device = 'cuda:0'
    radiance_field = NGPradianceField(**config.INSTANT_NGP_MLP_CONF).to(device)
    

    start_time = time.time()

    to_skip = ['/media/data4TB/sirocchi/nerfacc_nerf2vec/data_TRAINED_A1/02691156/604392af2cbb7d1fe30ec10233e7931a_A1']
    to_skip = []

    for class_name in os.listdir(nerf_root):

        subject_dirs = os.path.join(nerf_root, class_name)

        # Sometimes there are hidden files (e.g., when unzipping a file from a Mac)
        if not os.path.isdir(subject_dirs):
            continue
        
        for subject_name in os.listdir(subject_dirs):
            
            subject_dir = os.path.join(subject_dirs, subject_name)
            weights_dir = os.path.join(subject_dir, nerf_weights_file_name)

            print(f'Generating grid: {subject_dir}')

            grid_compressed_dir = os.path.join(subject_dir, compressed_grid_file_name)
            if os.path.exists(grid_compressed_dir):
                print('ALREADY EXISTS!')
            elif not os.path.exists(weights_dir):
                with open(os.path.join('dataset', 'error_log.txt'), 'a') as f:
                    f.write(f'weights not found: {subject_dir}\n')
            elif subject_dir in to_skip:
                with open(os.path.join('dataset', 'error_log.txt'), 'a') as f:
                    f.write(f'skipped: {subject_dir}\n')
            else:
                grid = generate_occupancy_grid(
                    device=device,
                    weights_path=weights_dir,
                    aabb=config.GRID_AABB,
                    n_iterations=config.GRID_RECONSTRUCTION_TOTAL_ITERATIONS,
                    n_warmups=config.GRID_RECONSTRUCTION_WARMUP_ITERATIONS,
                    radiance_field=radiance_field
                )

                dict = {
                    'occs': grid.state_dict()['occs'].half(),
                    '_roi_aabb': grid.state_dict()['_roi_aabb'].half(),
                    '_binary': grid.state_dict()['_binary'].to_sparse(),
                    'resolution': grid.state_dict()['resolution'].half()
                }

                
                with gzip.open(grid_compressed_dir, 'wb') as f:
                    torch.save(dict, f)
                
                """
                # Uncompressed dictionary
                dict = {
                    'occs': grid.state_dict()['occs'],
                    '_roi_aabb': grid.state_dict()['_roi_aabb'],
                    '_binary': grid.state_dict()['_binary'],
                    'resolution': grid.state_dict()['resolution']
                }
                """

                """
                # Uncompressed file
                grid_dir = os.path.join(subject_dir, grid_file_name)
                torch.save(dict, grid_dir)
                """
                
                generated_grids += 1

                if generated_grids > 0 and generated_grids % N_GRIDS_LOG == 0:
                    end_time = time.time()
                    print(f'Generated {N_GRIDS_LOG} grids in {end_time - start_time:.2f}s')
                    start_time = time.time()
                    

                

        

