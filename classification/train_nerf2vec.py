import math
import time

from nerfacc import ContractionType, OccupancyGrid
from classification.utils import get_mlp_params_as_matrix, next_multiple

import os
import torch
from torch.utils.data import DataLoader, Dataset

from nerf.loader import NeRFLoader

from pathlib import Path
from random import randint
from typing import Any, Dict, Tuple

from nerf.intant_ngp import NGPradianceField
from classification import config
from nerf.utils import generate_occupancy_grid

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

        self.epoch = 0
    
    def train(self):
        num_epochs = config.NUM_EPOCHS
        start_epoch = self.epoch

        for epoch in range(start_epoch, num_epochs):

            self.epoch = epoch
            # self.encoder.train()
            # self.decoder.train()

            desc = f"Epoch {epoch}/{num_epochs}"

            for batch in self.train_loader:

                rays, pixels, render_bkgds, matrices, nerf_weights_path = batch
                rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
                pixels = pixels.cuda()
                render_bkgds = render_bkgds.cuda()
                matrices = matrices.cuda()

                grids = []
                # start = time.time()
                for elem in nerf_weights_path:
                    grid = generate_occupancy_grid(self.device, 
                                                elem, 
                                                config.INSTANT_NGP_MLP_CONF, 
                                                config.AABB, 
                                                config.OCCUPANCY_GRID_RECONSTRUCTION_ITERATIONS, 
                                                config.OCCUPANCY_GRID_WARMUP_ITERATIONS)
                    grids.append(grid)
                # end = time.time()
                # print(f'elapsed: {end-start}')

                embeddings = self.encoder(matrices)
                pred = self.decoder(embeddings, selected_coords)




    
    