import math
import sys

from nerfacc import ContractionType, OccupancyGrid

from nerf.loader2 import NeRFLoader2
from nerf.loader3 import NeRFLoader3

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
        # self.device = 'cpu'

    def __len__(self) -> int:
        return len(self.nerf_paths)

    def __getitem__(self, index) -> Any:

        # print(f'get item index: {index}')

        dataset_kwargs = {}

        data_dir = self.nerf_paths[index]

        nerf_loader = NeRFLoader2(
            data_dir=data_dir,
            num_rays=config.NUM_RAYS,
            device=self.device,
            **dataset_kwargs)
        
        # Load radiance field
        # Load the occupancy grid
        # Load the positions that must be passed as input of the decoder
        # Load the matrices (i.e., parameters)
        #  
        # Free occupied memory from the RadianceField
        #

        
        


        
        

        # Get data for the batch
        # data = nerf_loader[0]
        data = nerf_loader.get_sample()
        render_bkgd = data["color_bkgd"]
        # rays = data["rays"]
        pixels = data["pixels"]


        # del radiance_field
        # torch.cuda.empty_cache()
        # del nerf_loader
        # del scene_aabb

        return pixels.to('cpu'), nerf_loader.weights_file_path
        
    
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