import os
import torch
from nerf.utils import Rays, namedtuple_map
from typing import Callable, Tuple
from torch import Tensor, nn

def _get_nerf_sizes(nerfs_root: str):
        
        n_files = 0
        file_size = 0
        zipped_file_size = 0

        for class_name in os.listdir(nerfs_root):

            subject_dirs = os.path.join(nerfs_root, class_name)

            # Sometimes there are hidden files (e.g., when unzipping a file from a Mac)
            if not os.path.isdir(subject_dirs):
                continue
            
            for subject_name in os.listdir(subject_dirs):
                subject_dir = os.path.join(subject_dirs, subject_name)
                
                file_size += os.path.getsize(os.path.join(subject_dir, 'grid.pth')) / (1024 * 1024)
                zipped_file_size += os.path.getsize(os.path.join(subject_dir, 'grid.pth.gz')) / (1024 * 1024)
                n_files += 1
        
        average_final_size = file_size/n_files
        average_final_zipped_size = zipped_file_size/n_files

        print(average_final_size, average_final_zipped_size, n_files)
        



_get_nerf_sizes('data')