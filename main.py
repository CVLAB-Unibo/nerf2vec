import math
import os
from pathlib import Path
import time
from nerfacc import ContractionType, OccupancyGrid
import torch

from torchvision.models import resnet18, ResNet50_Weights
from classification import config
from classification.ngp_nerf2vec import NGPradianceField

from classification.train_nerf2vec import NeRFDataset
from torch.utils.data import DataLoader, Dataset


def worker_init_fn(worker_id):
    torch.cuda.set_device('cuda:0')


def generate_occupancy_grid(device, weights_file_path):
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

    with torch.no_grad():
    
        # Iterate a certain number of times so as to improve the reconstructed occupancy grid.
        # Note that this is useful only for speeding up the video creation. Otherwise, the rendering
        # of a single image would take too long. However, the occupancy_grid object could also be None.
        # The number of iterations has been set empirically, and it requires an irrelevant amount of time.
        # 
        for i in range(20):
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
                warmup_steps=256
            )

    return occupancy_grid
        



def train_nerf2vec():


    train_dset = NeRFDataset(os.path.abspath('data'), None, device='cuda:0')
    #curr_nerf_loader = a[0]
    #data = curr_nerf_loader[0]
    #render_bkgd = data["color_bkgd"]
    #rays = data["rays"]
    #pixels = data["pixels"]

    train_loader = DataLoader(
        train_dset,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        worker_init_fn=worker_init_fn
    )
    
    print('loading batch')
    
    for batch in train_loader:
        # pixels, color = batch
        pixels, weights_file_path = batch


        grids = []
        start = time.time()
        for elem in weights_file_path:
            g = generate_occupancy_grid('cuda:0', elem)
            grids.append(g)
        end = time.time()
        print(f'elapsed: {end-start}')
        #exit()
        
    

    print()


if __name__ == '__main__':
    train_nerf2vec()



