from collections import defaultdict
import copy
import logging
import math
import os
from random import randint
import sys
import time
import uuid
from classification.export_renderings import get_rays
from classification.utils import get_mlp_params_as_matrix, pose_spherical, generate_rays
import h5py
import datetime
import tqdm

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from classification import config as config
from models.idecoder import ImplicitDecoder
from nerf.utils import Rays, render_image

import imageio.v2 as imageio
from torch.cuda.amp import autocast

def render_with_multiple_camera_poses(decoder, embeddings, device='cuda:0'):

    img_name_prefix = str(uuid.uuid4())

    scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / config.GRID_CONFIG_N_SAMPLES
    ).item()

    width = 224
    height = 224
    camera_angle_x = 0.8575560450553894 # Parameter taken from traned NeRFs
    focal_length = 0.5 * width / np.tan(0.5 * camera_angle_x)

    max_images = 6
    array = np.linspace(-30.0, 30.0, max_images//2, endpoint=False)
    array = np.append(array, np.linspace(
        30.0, -30.0, max_images//2, endpoint=False))
    
    color_bkgd = torch.ones((1,3), device=device)  # WHITE BACKGROUND!
    
    for emb_idx in range(len(embeddings)):
        # rgb_frames = []
        for n_img, theta in tqdm.tqdm(enumerate(np.linspace(0.0, 360.0, max_images, endpoint=False))):
            c2w = pose_spherical(torch.tensor(theta), torch.tensor(array[n_img]), torch.tensor(1.5))
            c2w = c2w.to(device)
            rays = generate_rays(device, width, height, focal_length, c2w)
            with autocast():
                rgb, _, _, _, _, _ = render_image(
                        radiance_field=decoder,
                        embeddings=embeddings[emb_idx].unsqueeze(dim=0),
                        occupancy_grid=None,
                        rays=Rays(origins=rays.origins.unsqueeze(dim=0), viewdirs=rays.viewdirs.unsqueeze(dim=0)),
                        scene_aabb=scene_aabb,
                        render_step_size=render_step_size,
                        render_bkgd=color_bkgd,
                        grid_weights=None
                    )
            
            plots_path = 'GAN_plots'
            full_path = os.path.join(plots_path, f'{img_name_prefix}_img{n_img}.png')
            imageio.imwrite(
                full_path,
                (rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8)
            )

@torch.no_grad()
def draw_images(decoder, embeddings, device='cuda:0'):

    scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / config.GRID_CONFIG_N_SAMPLES
    ).item()

    rays = get_rays(device)
    color_bkgd = torch.ones((1,3), device=device)  # WHITE BACKGROUND!

    img_name = str(uuid.uuid4())

    for idx, emb in enumerate(embeddings):
        emb = torch.tensor(emb, device=device, dtype=torch.float32)
        emb = emb.unsqueeze(dim=0)
        with autocast():
            rgb_A, alpha, b, c, _, _ = render_image(
                            radiance_field=decoder,
                            embeddings=emb,
                            occupancy_grid=None,
                            rays=Rays(origins=rays.origins.unsqueeze(dim=0), viewdirs=rays.viewdirs.unsqueeze(dim=0)),
                            scene_aabb=scene_aabb,
                            render_step_size=render_step_size,
                            render_bkgd=color_bkgd,
                            grid_weights=None
            )

        plots_path = 'GAN_plots'
        imageio.imwrite(
            os.path.join(plots_path, f'{img_name}_{idx}.png'),
            (rgb_A.squeeze(dim=0).cpu().detach().numpy() * 255).astype(np.uint8)
        )


@torch.no_grad()
def create_renderings_from_GAN_embeddings(device='cuda:0'):

    # Init nerf2vec 
    decoder = ImplicitDecoder(
            embed_dim=config.ENCODER_EMBEDDING_DIM,
            in_dim=config.DECODER_INPUT_DIM,
            hidden_dim=config.DECODER_HIDDEN_DIM,
            num_hidden_layers_before_skip=config.DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP,
            num_hidden_layers_after_skip=config.DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP,
            out_dim=config.DECODER_OUT_DIM,
            encoding_conf=config.INSTANT_NGP_ENCODING_CONF,
            aabb=torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
        )
    decoder.eval()
    decoder = decoder.to(device)

    ckpt_path = os.path.join('classification','train','ckpts','499.pt')
    print(f'loading weights: {ckpt_path}')
    ckpt = torch.load(ckpt_path)
    decoder.load_state_dict(ckpt["decoder"])

    latent_gan_embeddings_path = "/media/data4TB/sirocchi/nerf2vec/shape_generation/experiments/nerf2vec_3/generated_embeddings/epoch_2000.npz"
    embeddings = np.load(latent_gan_embeddings_path)["embeddings"]
    embeddings = torch.from_numpy(embeddings)


    for count in range(0, 100):
        idx = randint(0, embeddings.shape[0]-1)
        emb = embeddings[idx].unsqueeze(0).cuda()
        draw_images(decoder, emb, device)
        # render_with_multiple_camera_poses(decoder, emb, device)
