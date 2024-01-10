import math
import os
from random import randint
import uuid
from nerf2vec.utils import get_rays
from nerf2vec.utils import pose_spherical, generate_rays
import tqdm


import numpy as np

import torch

from nerf2vec import config as nerf2vec_config

from models.idecoder import ImplicitDecoder
from nerf.utils import Rays, render_image

import imageio.v2 as imageio
from torch.cuda.amp import autocast


@torch.no_grad()
def draw_images(decoder, embeddings, device='cuda:0'):

    scene_aabb = torch.tensor(nerf2vec_config.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / nerf2vec_config.GRID_CONFIG_N_SAMPLES
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
            embed_dim=nerf2vec_config.ENCODER_EMBEDDING_DIM,
            in_dim=nerf2vec_config.DECODER_INPUT_DIM,
            hidden_dim=nerf2vec_config.DECODER_HIDDEN_DIM,
            num_hidden_layers_before_skip=nerf2vec_config.DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP,
            num_hidden_layers_after_skip=nerf2vec_config.DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP,
            out_dim=nerf2vec_config.DECODER_OUT_DIM,
            encoding_conf=nerf2vec_config.INSTANT_NGP_ENCODING_CONF,
            aabb=torch.tensor(nerf2vec_config.GRID_AABB, dtype=torch.float32, device=device)
        )
    decoder.eval()
    decoder = decoder.to(device)

    ckpt_path = os.path.join('classification','train','ckpts','499.pt')  # TODO: update path
    print(f'loading weights: {ckpt_path}')
    ckpt = torch.load(ckpt_path)
    decoder.load_state_dict(ckpt["decoder"])

    latent_gan_embeddings_path = "/media/data4TB/sirocchi/nerf2vec/shape_generation/experiments/nerf2vec_3/generated_embeddings/epoch_2000.npz" # TODO: update path
    embeddings = np.load(latent_gan_embeddings_path)["embeddings"]
    embeddings = torch.from_numpy(embeddings)

    for count in range(0, 100):
        idx = randint(0, embeddings.shape[0]-1)
        emb = embeddings[idx].unsqueeze(0).cuda()
        draw_images(decoder, emb, device)
