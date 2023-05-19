import gzip
import os
import shutil

import torch
from classification import config

from dataset.generate_grids import generate_occupancy_grid


nerf_paths = []

nerfs_root = os.path.join('/', 'media', 'data4TB', 'sirocchi', 'nerfacc_nerf2vec', 'data_TRAINED')
dest_data_dir = 'data'

elements_per_class = 100
curr_elements = 0

for class_name in os.listdir(nerfs_root):   
    curr_elements = 0
    

    subject_dirs = os.path.join(nerfs_root, class_name)

    # Sometimes there are hidden files (e.g., when unzipping a file from a Mac)
    if not os.path.isdir(subject_dirs):
        continue
    
    for subject_name in os.listdir(subject_dirs):
        curr_elements += 1
        if curr_elements >= elements_per_class:
            break

        subject_dir = os.path.join(subject_dirs, subject_name)
        dest_nerf_dir = os.path.join(dest_data_dir, class_name, subject_name)

        shutil.copytree(subject_dir, dest_nerf_dir)