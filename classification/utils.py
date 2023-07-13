from collections import OrderedDict
import datetime
import gzip
import os
import shutil
from typing import Any, Dict
from torch import Tensor
import torch

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
    start = params_shapes[0].numel() #+ params_shapes[1].numel()
    end = params_shapes[-1].numel() #+ params_shapes[-2].numel()
    params = flattened_params[start:-end]
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
    