import json
import math
import os
import random
import shutil

import torch
import numpy as np


def cycle_path(nerfs_root):
    

    dict_result = {}

    last_two_parts = os.path.join(*os.path.splitdrive(nerfs_root)[1].split(os.sep)[-2:])
    base_folder = os.path.join('.', last_two_parts)

    for class_name in os.listdir(nerfs_root):

        class_nerf_paths = []

        subject_dirs = os.path.join(nerfs_root, class_name)

        # Sometimes there are hidden files (e.g., when unzipping a file from a Mac)
        if not os.path.isdir(subject_dirs):
            continue
        
        for subject_name in os.listdir(subject_dirs):
            subject_dir = os.path.join(subject_dirs, subject_name)
            class_nerf_paths.append(subject_dir.replace(nerfs_root, base_folder))        
        dict_result[class_name] = class_nerf_paths

    return dict_result


def read_grids():
    
    root_paths = [
        '/media/data4TB/sirocchi/nerf2vec/data/data_TRAINED', 
        '/media/data4TB/sirocchi/nerf2vec/data/data_TRAINED_A1', 
        '/media/data4TB/sirocchi/nerf2vec/data/data_TRAINED_A2'
    ]
    

    train = []
    validation = []
    test = []

    TRAIN_SPLIT = 80
    VALIDATION_SPLIT = 10
    TEST_SPLIT = 10

    random.seed(1203)
    values = []

    with open('occupancy.txt', 'a') as f:
        for curr_path in root_paths:
            
            # Get 
            nerfs_dict = cycle_path(curr_path)

            for class_name in nerfs_dict:

                # Get elements related to the current class
                class_elements = nerfs_dict[class_name]

                for data_dir in class_elements:
                    grid_weights_path = os.path.join(data_dir, 'grid.pth')  
                    grid_weights = torch.load(grid_weights_path, map_location='cuda')
                    grid_weights['_binary'] = grid_weights['_binary'].to_dense()#.unsqueeze(dim=0)
                    grid_weights['occs'] = torch.empty([884736])   # 884736 if resolution == 96 else 2097152

                    binary = grid_weights['_binary']
                    one_indices = torch.nonzero(binary.flatten() == 1)[:, 0]

                    print(f'data:dir: {data_dir} - one_indices: {len(one_indices)}')

                    values.append(len(one_indices))
                    f.write(f'{data_dir} {len(one_indices)}\n')
                
        f.write(f'{np.min(values)} - {np.max(values)} \n')
        print(np.min(values), np.max(values))

# read_grids()
# exit()

def find_closest_key(dictionary, target):
    closest_key = min(dictionary.keys(), key=lambda x: abs(x - target))
    return closest_key

def contains_substring(array, substring):
    for element in array:
        if substring in element:
            return True
    return False

def copy_folder(source_path, destination_path):
    try:
        shutil.copytree(source_path, destination_path)
        print(f"Folder copied from '{source_path}' to '{destination_path}'")
    except FileExistsError:
        print(f"Destination folder '{destination_path}' already exists.")


required_sizes = {
    1000: [],
    5000: [],
    10000:[],
    15000:[],
    20000:[],
    25000:[],
    30000:[],
    35000:[],
    40000:[],
    45000:[],
    50000:[],
    55000:[],
    60000:[],
    65000:[],
    70000:[],
    75000:[],
    80000:[],
    85000:[],
    90000:[],
    95000:[],
    100000:[],
    105000:[],
    110000:[],
    115000:[],
    120000:[],
    125000:[],
    130000:[],
    135000:[],
    140000:[],
    145000:[],
    150000:[],
    155000:[],
    160000:[],
    165000:[],
    170000:[],
    175000:[],
    180000:[],
    185000:[],
    190000:[],
    195000:[],
    200000:[],
    205000:[],
    210000:[]
}


with open('occupancy.txt', 'r') as f:
    lines = f.readlines()

    # Shuffle the lines
    random.shuffle(lines)

    for l in lines:
        if l == '\n' or l == '':
            continue
        
        data_dir = l.split(' ')[0]
        rays = int(l.split(' ')[-1].replace('\n', ''))
        new_dir = f'{data_dir.split("/")[-2]}_{data_dir.split("/")[-1]}'
        
        key = find_closest_key(required_sizes, rays)

        if len(required_sizes[key]) == 16:
            continue
        
        if 'TRAINED/' in data_dir:
            if contains_substring(required_sizes[key], 'TRAINED/'):
                continue
            else:
                required_sizes[key].append('TRAINED_N/')
                copy_folder(data_dir, os.path.join('copied_NeRFs', new_dir))
        
        elif 'TRAINED_A1/' in data_dir:
            if contains_substring(required_sizes[key], 'TRAINED_A1/'):
                continue
            else:
                required_sizes[key].append('TRAINED_A1/')
                copy_folder(data_dir, os.path.join('copied_NeRFs', new_dir))
        
        elif 'TRAINED_A2/' in data_dir:
            if contains_substring(required_sizes[key], 'TRAINED_A2/'):
                continue
            else:
                required_sizes[key].append('TRAINED_A2/')
                copy_folder(data_dir, os.path.join('copied_NeRFs', new_dir))

print(required_sizes)
