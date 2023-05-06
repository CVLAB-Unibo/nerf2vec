import sys

sys.path.append("..")

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
    def __init__(self, nerfs_root: str, sample_sd: Dict[str, Any], device: str = 'cuda:0') -> None:
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
            **dataset_kwargs)
        
        # Load radiance field
        # Load the occupancy grid
        # Load the positions that must be passed as input of the decoder
        # Load the matrices (i.e., parameters)
        #  
        # Free occupied memory from the RadianceField
        #
        radiance_field = NGPradianceField(
            aabb=config.AABB,
            unbounded=False,
            encoding='Frequency',
            mlp='FullyFusedMLP',
            activation='ReLU',
            n_hidden_layers=3,
            n_neurons=64,
            encoding_size=24
        ).to(self.device)
        matrix = torch.load(nerf_loader.weights_file_path)
        radiance_field.load_state_dict(matrix)
        radiance_field.eval()

        # Get data for the batch
        data = nerf_loader[0]
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        return nerf_loader
        
    
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