import math
import os
from pathlib import Path
import time
import uuid
from nerfacc import ContractionType, OccupancyGrid
import torch
import numpy as np

from torchvision.models import resnet18, ResNet50_Weights
import tqdm
from classification import config, create_video
from classification.ngp_nerf2vec import NGPradianceField

from classification.train_nerf2vec import NeRFDataset
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from classification.utils import render_image
from nerf.loader import NeRFLoader


def generate_occupancy_grid(device, weights_file_path, focal, train_dataset):
    radiance_field = NGPradianceField(
            aabb=config.aabb,
            unbounded=False,
            encoding='Frequency',
            mlp='FullyFusedMLP',
            activation='ReLU',
            n_hidden_layers=3,
            n_neurons=64,
            encoding_size=24
        ).to(device)
    
    matrix = torch.load(weights_file_path)
    radiance_field.load_state_dict(matrix)
    radiance_field.eval()

    # Create the OccupancyGrid
    render_n_samples = 1024
    grid_resolution = 128
    
    contraction_type = ContractionType.AABB
    scene_aabb = torch.tensor(config.aabb, dtype=torch.float32, device=device)
    near_plane = None
    far_plane = None
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / render_n_samples
    ).item()
    alpha_thre = 0.0

    occupancy_grid = OccupancyGrid(
        roi_aabb=config.aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
    ).to(device)
    occupancy_grid.eval()

    occupancy_grid_iterations = [5]
    warmup_steps = [10]

    file_name = str(uuid.uuid4())

    with torch.no_grad():
    
        # Iterate a certain number of times so as to improve the reconstructed occupancy grid.
        # Note that this is useful only for speeding up the video creation. Otherwise, the rendering
        # of a single image would take too long. However, the occupancy_grid object could also be None.
        # The number of iterations has been set empirically, and it requires an irrelevant amount of time.
        # 
        for wm_step in warmup_steps:

            with open('log.txt', 'a') as f:
                f.write(f'wm_steps: {wm_step}\n\t')

            for its in occupancy_grid_iterations:
                psnrs = []

                for i in range(its):
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
                        warmup_steps=wm_step
                    )
                
                create_video.create_video(
                        720, 
                        480, 
                        device, 
                        focal, 
                        radiance_field, 
                        occupancy_grid, 
                        scene_aabb,
                        near_plane, 
                        far_plane, 
                        render_step_size,
                        render_bkgd= torch.zeros(3, device=device),
                        cone_angle=0.0,
                        alpha_thre=alpha_thre,
                        # test options
                        test_chunk_size=8192,
                        path=f'{file_name}_{its}.mp4'
                    )
                
                """
                for i in tqdm.tqdm(range(len(train_dataset))):  # was test_set !
                    data = train_dataset[i] # was test_set !
                    render_bkgd = data["color_bkgd"]
                    rays = data["rays"]
                    pixels = data["pixels"]
                    
                    # rendering
                    rgb, acc, depth, _ = render_image(
                        radiance_field,
                        occupancy_grid,
                        rays,
                        scene_aabb=scene_aabb,
                        # rendering options
                        near_plane=near_plane,
                        far_plane=far_plane,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        cone_angle=0.0,
                        alpha_thre=alpha_thre,
                        # test options
                        test_chunk_size=8192,
                    )
                    mse = F.mse_loss(rgb, pixels)
                    psnr = -10.0 * torch.log(mse) / np.log(10.0)
                    psnrs.append(psnr.item())
                
                psnr_avg = sum(psnrs) / len(psnrs)
                # print(f'PSNR with {its} iterations is {psnr_avg}')
                with open('log.txt', 'a') as f:
                    f.write(f'{its}: {psnr_avg:.2f}')
                    if not its == occupancy_grid_iterations[-1]:
                        f.write(' - ')
                    else:
                        f.write('\n')
                """
                
    """
    with open('log.txt', 'a') as f:
        f.write('\n')
        f.write('*'*40)
    """
        

    return occupancy_grid
        

def train_nerf2vec():


    train_dset = NeRFDataset(os.path.abspath('data'), None, device='cpu')
    #curr_nerf_loader = a[0]
    #data = curr_nerf_loader[0]
    #render_bkgd = data["color_bkgd"]
    #rays = data["rays"]
    #pixels = data["pixels"]

    train_loader = DataLoader(
        train_dset,
        batch_size=16,
        num_workers=4,
        shuffle=True
    )
    
    print('loading batch')
    
    for batch in train_loader:
        # pixels, color = batch
        rays, pixels, render_bkgd, weights_file_path, focal, data_dir = batch

       

        # Move tensors to CUDA 
        start = time.time()
        rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
        pixels = pixels.cuda()
        render_bkgd.cuda()
        end = time.time()
        print(f'moving tensor to CUDA: {end-start}')

        grids = []
        start = time.time()
        for idx, elem in enumerate(weights_file_path):
            
            test_dataset_kwargs = {}
            nerf_loader = NeRFLoader(
                data_dir=data_dir[idx],
                num_rays=None,
                device='cuda:0',
                **test_dataset_kwargs)

            g = generate_occupancy_grid('cuda:0', elem, focal[idx], nerf_loader)
            grids.append(g)
        end = time.time()
        print(f'elapsed: {end-start}')

    print()


if __name__ == '__main__':
    train_nerf2vec()



