from collections import OrderedDict
import datetime
import gzip
import os
from pathlib import Path
import shutil
from typing import Any, Dict
from torch import Tensor
import torch
import numpy as np


from classification import config
from nerf.utils import Rays

import torch.nn.functional as F


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


# ################################################################################
# CAMERA POSE MATRIX GENERATION METHODS
# ################################################################################
def get_translation_t(t):
    """Get the translation matrix for movement in t."""
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ]

    return torch.tensor(matrix, dtype=torch.float32)


def get_rotation_phi(phi):
    """Get the rotation matrix for movement in phi."""
    matrix = [
        [1, 0, 0, 0],
        [0, torch.cos(phi), -torch.sin(phi), 0],
        [0, torch.sin(phi), torch.cos(phi), 0],
        [0, 0, 0, 1],
    ]
    return torch.tensor(matrix, dtype=torch.float32)


def get_rotation_theta(theta):
    """Get the rotation matrix for movement in theta."""
    matrix = [
        [torch.cos(theta), 0, -torch.sin(theta), 0],
        [0, 1, 0, 0],
        [torch.sin(theta), 0, torch.cos(theta), 0],
        [0, 0, 0, 1],
    ]
    return torch.tensor(matrix, dtype=torch.float32)


def pose_spherical(theta, phi, t):
    """
    Get the camera to world matrix for the corresponding theta, phi
    and t.
    """
    c2w = get_translation_t(t)
    c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = torch.from_numpy(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [
                           0, 0, 0, 1]], dtype=np.float32)) @ c2w  
    return c2w

# ################################################################################
# RAYS GENERATION
# ################################################################################
def generate_rays(device, width, height, focal, c2w, OPENGL_CAMERA=True):
    x, y = torch.meshgrid(
            torch.arange(width, device=device),
            torch.arange(height, device=device),
            indexing="xy",
        )
    x = x.flatten()
    y = y.flatten()

    K = torch.tensor(
        [
            [focal, 0, width / 2.0],
            [0, focal, height / 2.0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )  # (3, 3)

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - K[0, 2] + 0.5) / K[0, 0],
                (y - K[1, 2] + 0.5)
                / K[1, 1]
                * (-1.0 if OPENGL_CAMERA else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if OPENGL_CAMERA else 1.0),
    )  # [num_rays, 3]
    camera_dirs.to(device)

    directions = (camera_dirs[:, None, :] * c2w[:3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )

    origins = torch.reshape(origins, (height, width, 3))#.unsqueeze(0)
    viewdirs = torch.reshape(viewdirs, (height, width, 3))#.unsqueeze(0)
    
    rays = Rays(origins=origins, viewdirs=viewdirs)
    
    return rays