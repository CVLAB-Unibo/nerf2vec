from collections import OrderedDict
import datetime
import gzip
import os
from pathlib import Path
import shutil
from typing import Any, Dict
from torch import Tensor
import torch
# import numpy as np


from classification import config


def next_multiple(val, divisor):
    """
    Implementation ported directly from TinyCuda implementation
    See https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/common.h#L300
    """
    return next_pot(div_round_up(val, divisor) * divisor)


def div_round_up(val, divisor):
	return next_pot((val + divisor - 1) / divisor)


def next_pot(v):
    v=int(v)
    v-=1
    v | v >> 1
    v | v >> 2
    v | v >> 4
    v | v >> 8
    v | v >> 16
    return v+1


def next_multiple_2(val, divisor):
    """
    Additional implementation added for testing purposes
    """
    return ((val - 1) | (divisor -1)) + 1


def get_mlp_params_as_matrix(flattened_params: Tensor, sd: Dict[str, Any] = None) -> Tensor:

    if sd is None:
         sd = get_mlp_sample_sd()
    
    params_shapes = [p.shape for p in sd.values()]
    feat_dim = params_shapes[0][0]
    """
    # Remove start/end layer weights
    start = params_shapes[0].numel() #+ params_shapes[1].numel()
    end = params_shapes[-1].numel() #+ params_shapes[-2].numel()
    params = flattened_params[start:-end]
    """
    padding_size = (feat_dim-params_shapes[-1][0]) * params_shapes[-1][1]
    padding_tensor = torch.zeros(padding_size)
    params = torch.cat((flattened_params, padding_tensor), dim=0)

    # TODO: for the future, it could be possible to try to transpose the returned matrix.
    # this would be more compliant to inr2vec
    return params.reshape((-1, feat_dim))


def get_mlp_sample_sd():
    sample_sd = OrderedDict()
    sample_sd['input'] = torch.zeros(config.MLP_UNITS, next_multiple(config.MLP_INPUT_SIZE_AFTER_ENCODING, config.TINY_CUDA_MIN_SIZE))
    for i in range(config.MLP_HIDDEN_LAYERS):
        sample_sd[f'hid_{i}'] = torch.zeros(config.MLP_UNITS, config.MLP_UNITS)
    sample_sd['output'] = torch.zeros(next_multiple(config.MLP_OUTPUT_SIZE, config.TINY_CUDA_MIN_SIZE), config.MLP_UNITS)

    return sample_sd


def get_grid_file_name(file_path):
    # Split the path into individual directories
    directories = os.path.normpath(file_path).split(os.sep)
    # Get the last two directories
    last_two_dirs = directories[-2:]
    # Join the last two directories with an underscore
    file_name = '_'.join(last_two_dirs) + '.pth'
    return file_name


def get_class_label(file_path):
    directories = os.path.normpath(file_path).split(os.sep)
    class_label = directories[-3]

    # TODO: REMOVE THIS AND BETTER COPE WITH THESE CLASSES
    if class_label == '02992529' or class_label == '03948459':
        return -1
    
    return class_label

def get_class_label_from_nerf_root_path(file_path):
    directories = os.path.normpath(file_path).split(os.sep)
    class_label = directories[-2]
    
    return class_label


def get_nerf_name_from_grid(file_path):
    grid_name = os.path.basename(file_path)
    nerf_name = os.path.splitext(grid_name)[0]
    return nerf_name


def unzip_file(file_path, extract_dir, file_name):
    with gzip.open(os.path.join(file_path, 'grid.pth.gz'), 'rb') as f_in:
        output_path = os.path.join(extract_dir, file_name) 
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


"""
X = [826, 1001, 1067, 1412, 1525, 1638, 1787, 1812, 1836, 1946, 2065, 2186, 2216, 2308, 2413, 2561, 2570, 3050, 3139, 3463, 3515, 3870, 4059, 4075, 4140, 4212, 4271, 4983, 5183, 5236, 5390, 5395, 6817, 7095, 7406, 7894, 8110, 8545, 8609, 8666, 8747, 8925, 9120, 9304, 9929, 10012, 10186, 10537, 10918, 11309, 11492, 12024, 12087, 12544, 12675, 12784, 12867, 12947, 12986, 13176, 13279, 13671, 14180, 14585, 14843, 15250, 15975, 16266, 16604, 17005, 17426, 17544, 17709, 18052, 18149, 18172, 18613, 18719, 18744, 18779, 18972, 19755, 20082, 20095, 20115, 20239, 21577, 21932, 22212, 22570, 22682, 22728, 23065, 23289, 23756, 23849, 24005, 24582, 24721, 25181, 25606, 25715, 25772, 26446, 26632, 27557, 27896, 27989, 28165, 28196, 28286, 28374, 28479, 28663, 29232, 29558, 29662, 30542, 30938, 30945, 31214, 31778, 32265, 32922, 33046, 33240, 33248, 33489, 33523, 34039, 34091, 34135, 34367, 34546, 34622, 35093, 35167, 35628, 36757, 37105, 37198, 37674, 37758, 37915, 38077, 39164, 39248, 39584, 39601, 39815, 40247, 40397, 40621, 40682, 40809, 41170, 41470, 41624, 42380, 42609, 42658, 42803, 43014, 43050, 43298, 43341, 43690, 43979, 44031, 44406, 45468, 45499, 45867, 46071, 46202, 46872, 47201, 47926, 47963, 48061, 48176, 48471, 48517, 48884, 48988, 49304, 50275, 50536, 51623, 51781, 52033, 52051, 52319, 52373, 52541, 52880, 53352, 53444, 53618, 54011, 54100, 54820, 55209, 55475, 55837, 56249, 56387, 56489, 56568, 56672, 57682, 57698, 57978, 58000, 58282, 58955, 59719, 59753, 60292, 60450, 60917, 61193, 61201, 61418, 61640, 62231, 62487, 62578, 62726, 62841, 63072, 63196, 63454, 63515, 63911, 64348, 64780, 65212, 65596, 65628, 66619, 68030, 68762, 74006, 74659, 77762, 77842, 78942, 80975, 86208, 87215, 87257, 87308, 88191, 88936, 93022, 97773, 99819, 101609, 103756, 104123, 106694, 106767, 108544, 110576, 113353, 114884, 116169, 119496, 121250, 121509, 121842, 123068, 126472, 129081, 130255, 133084, 133472, 134768, 137412, 137814, 139335, 141422, 143622, 145007, 150743, 151848, 152335, 153158, 155056, 157673, 159960, 161651, 162935, 163054, 165644, 165755, 170052, 170178, 170491, 173201, 174844, 182729, 186237, 186979, 202727]
Y=[0.1, 0.05, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.05, 0.1, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1, 0.2, 0.1, 0.005, 0.1, 0.1, 0.01, 0.01, 0.05, 0.1, 0.05, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.2, 0.01, 0.05, 0.05, 0.05, 0.1, 0.05, 0.1, 0.2, 0.1, 0.05, 0.1, 0.01, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.1, 0.05, 0.005, 0.05, 0.05, 0.1, 0.01, 0.05, 0.01, 0.05, 0.01, 0.01, 0.01, 0.05, 0.005, 0.05, 0.05, 0.05, 0.05, 0.01, 0.05, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.005, 0.05, 0.05, 0.05, 0.01, 0.05, 0.05, 0.1, 0.05, 0.1, 0.05, 0.05, 0.005, 0.01, 0.05, 0.01, 0.005, 0.05, 0.01, 0.05, 0.05, 0.1, 0.005, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.05, 0.01, 0.01, 0.01, 0.05, 0.01, 0.01, 0.01, 0.05, 0.05, 0.005, 0.01, 0.01, 0.01, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.01, 0.05, 0.1, 0.01, 0.01, 0.05, 0.005, 0.01, 0.005, 0.05, 0.01, 0.01, 0.01, 0.05, 0.01, 0.1, 0.05, 0.005, 0.05, 0.005, 0.05, 0.005, 0.05, 0.01, 0.01, 0.05, 0.05, 0.005, 0.05, 0.005, 0.005, 0.1, 0.01, 0.01, 0.05, 0.1, 0.05, 0.01, 0.01, 0.05, 0.01, 0.005, 0.01, 0.01, 0.01, 0.05, 0.05, 0.01, 0.01, 0.05, 0.005, 0.005, 0.005, 0.005, 0.05, 0.005, 0.01, 0.005, 0.01, 0.01, 0.05, 0.05, 0.01, 0.01, 0.01, 0.005, 0.05, 0.01, 0.05, 0.01, 0.05, 0.005, 0.05, 0.05, 0.005, 0.01, 0.01, 0.01, 0.05, 0.01, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.005, 0.01, 0.01, 0.01, 0.05, 0.05, 0.005, 0.05, 0.01, 0.01, 0.005, 0.01, 0.05, 0.05, 0.01, 0.005, 0.01, 0.01, 0.005, 0.01, 0.01, 0.05, 0.01, 0.01, 0.005, 0.01, 0.01, 0.005, 0.1, 0.005, 0.01, 0.01, 0.005, 0.01, 0.05, 0.01, 0.01, 0.05, 0.01, 0.01, 0.005, 0.005, 0.01, 0.01, 0.005, 0.01, 0.05, 0.005, 0.005, 0.01, 0.005, 0.005, 0.005, 0.01, 0.01, 0.001, 0.05, 0.01, 0.01, 0.01, 0.005, 0.01, 0.01, 0.05, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01, 0.005, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.05, 0.005, 0.005]
coefficients = np.polyfit(X, Y, 3)
polynomial = np.poly1d(coefficients)

def get_bg_weight(x):
    new_weights = torch.tensor(polynomial(x))

    return torch.clamp(new_weights, 0.0001, 0.5)
"""