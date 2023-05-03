import os
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
from random import randint
from typing import Any, Dict, Tuple

from nerf_loader import NeRFLoader
from ngp_nerf2vec import NGPradianceField
import config

class NeRFDataset(Dataset):
    def __init__(self, nerfs_root: str, sample_sd: Dict[str, Any]) -> None:
        super().__init__()
        
        self.nerfs_root = nerfs_root

        self.nerf_paths = self._get_nerf_paths()
    
    def __len__(self) -> int:
        return len(self.nerf_paths)

    def __getitem__(self, index) -> Any:
        dataset_kwargs = {}

        nerf_loader = NeRFLoader(
            subject_id=self.subject_id,
            root_fp=self.nerfs_root,
            num_rays=None,
            **dataset_kwargs,)
        
        radiance_field = NGPradianceField(
            aabb=config.AABB,
            unbounded=args.unbounded,
            encoding=args.coordinate_encoding,
            mlp=args.mlp,
            activation=args.activation,
            n_hidden_layers=args.n_hidden_layers,
            n_neurons=args.n_neurons,
            encoding_size=args.encoding_size
        ).to(device)
        
        return nerf_loader
        
    
    def _get_nerf_paths(self):
        
        nerf_paths = []

        for class_name in os.listdir(self.nerfs_root):

            subject_dirs = os.path.join(self.nerfs_root, class_name)

            # Sometimes there are hidden files (e.g., when unzipping a file from a Mac)
            if not os.path.isdir(subject_dirs):
                continue
            
            for subject_name in os.listdir(subject_dirs):
                subject_dir = os.path.join(class_name, subject_name)
                nerf_paths.append(subject_dir)
        
        return nerf_paths

        
a = NeRFDataset('data', None)