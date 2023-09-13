import math
import os
from pathlib import Path
from random import randint
import uuid

import torch

from nerf.utils import Rays, namedtuple_map
from nerf.intant_ngp import NGPradianceField
from nerfacc import ray_marching, rendering

import numpy as np
import imageio.v2 as imageio

import torch.nn.functional as F

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

# ################################################################################
# IMAGE GENERATION
# ################################################################################
@torch.no_grad()
def generate_image(
    device, 
    ngp_mlp_weights,
    rays,
    curr_bkgd,
    img_path):

    GRID_AABB = [-0.7, -0.7, -0.7, 0.7, 0.7, 0.7]
    GRID_CONFIG_N_SAMPLES = 1024

    INSTANT_NGP_MLP_CONF = {
        'aabb': GRID_AABB,
        'unbounded':False,
        'encoding':'Frequency',
        'mlp':'FullyFusedMLP',
        'activation':'ReLU',
        'n_hidden_layers':3,
        'n_neurons':64,
        'encoding_size':24
    }

    radiance_field = NGPradianceField(**INSTANT_NGP_MLP_CONF).to(device)
    radiance_field.eval()

    scene_aabb = torch.tensor(GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / GRID_CONFIG_N_SAMPLES
    ).item()

    radiance_field.load_state_dict(ngp_mlp_weights)

    rays_shape = rays.origins.shape
    height, width, _ = rays_shape
    num_rays = height * width
    rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        
        _, density = radiance_field._query_density_and_rgb(positions, None)
        return density

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        
        return radiance_field(positions, t_dirs)
    
    chunk = 4096
    results = []

    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = ray_marching(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            grid=None,
            sigma_fn=sigma_fn,
            near_plane=None,
            far_plane=None,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=0.0,
            alpha_thre=0.0,
        )
        rgb, opacity, depth = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=curr_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)

    rgbs, _, _, _ = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]

    rgbs = rgbs.view((*rays_shape[:-1], -1))
    
    imageio.imwrite(
        img_path,
        (rgbs.cpu().detach().numpy() * 255).astype(np.uint8)
    )

device = 'cuda:0'
color_bkgd = torch.zeros(3, device=device)
ngp_weights = torch.load('/media/data4TB/sirocchi/nerf2vec/data/data_TRAINED_A2/02958343/1a1dcd236a1e6133860800e6696b8284_A2/bb07_steps3000_encodingFrequency_mlpFullyFusedMLP_activationReLU_hiddenLayers3_units64_encodingSize24.pth') # NGP WEIGHTS

# Image output size
width = 224
height = 224

# Get camera pose
theta = torch.tensor(120.0, device=device) # The horizontal camera position (change the value between and 360 to make a full cycle around the object)
phi = torch.tensor(-30.0, device=device) # The vertical camera position
t = torch.tensor(1.5, device=device) # camera distance from object
c2w = pose_spherical(theta, phi, t)
c2w = c2w.to(device)

# Compute the focal_length 
camera_angle_x = 0.8575560450553894 # Parameter taken from traned NeRFs
focal_length = 0.5 * width / np.tan(0.5 * camera_angle_x)

rays = generate_rays(device, width, height, focal_length, c2w)
path = "temp_plots/img.png"

generate_image(
    device, 
    ngp_weights, 
    rays,
    color_bkgd,
    path
)