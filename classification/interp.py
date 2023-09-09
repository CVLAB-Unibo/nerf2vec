import math
import os
from pathlib import Path
from random import randint
import uuid

import torch
import wandb
from classification import config_classifier as config
from classification.train_nerf2vec import NeRFDataset
from classification.utils import get_class_label_from_nerf_root_path
from models.encoder import Encoder
from models.idecoder import ImplicitDecoder
from nerf.utils import Rays, render_image

import numpy as np
import imageio.v2 as imageio

from torch.cuda.amp import autocast

@torch.no_grad()
def interpolate():

    device = 'cuda'

    scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / config.GRID_CONFIG_N_SAMPLES
    ).item()

    # TODO: decide which weight to load (best vs last)
    # ckpt_path = ckpt_path / ""ckpts/best.pt""

    ckpts_path = Path(os.path.join('classification', 'train', 'ckpts'))
    ckpt_paths = [p for p in ckpts_path.glob("*.pt") if "best" not in p.name]
    ckpt_path = ckpt_paths[0]
    ckpt = torch.load(ckpt_path)
    
    print(f'loaded weights: {ckpt_path}')

    encoder = Encoder(
                config.MLP_UNITS,
                config.ENCODER_HIDDEN_DIM,
                config.ENCODER_EMBEDDING_DIM
                )
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.cuda()
    encoder.eval()

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
    decoder.load_state_dict(ckpt["decoder"])
    decoder = decoder.cuda()
    decoder.eval()

    split = 'train'
    dset_json = os.path.abspath(os.path.join('data', f'{split}.json'))  
    dset = NeRFDataset(dset_json, device='cpu')  

    idx = 0
   
    while idx < 1000:
        idx_A = randint(0, len(dset) - 1)
        train_nerf_A, test_nerf_A, matrices_unflattened_A, matrices_flattened_A, grid_weights_A, data_dir_A, _, _ = dset[idx_A]
        class_id_A = get_class_label_from_nerf_root_path(data_dir_A)

        class_id_B = -1
        while class_id_B != class_id_A:
            idx_B = randint(0, len(dset) - 1)
            train_nerf_B, test_nerf_B, matrices_unflattened_B, matrices_flattened_B, _, data_dir_B, _, _ = dset[idx_B]
            class_id_B = get_class_label_from_nerf_root_path(data_dir_B)
        
        print(data_dir_A, data_dir_B)
        
        matrices_flattened_A = matrices_flattened_A.cuda().unsqueeze(0)
        matrices_flattened_B = matrices_flattened_B.cuda().unsqueeze(0)

        with autocast():
            embedding_A = encoder(matrices_flattened_A).squeeze(0)  
            embedding_B = encoder(matrices_flattened_B).squeeze(0)  
        

        embeddings = [embedding_A]
        for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            emb_interp = (1 - gamma) * embedding_A + gamma * embedding_B
            embeddings.append(emb_interp)
        embeddings.append(embedding_B)


        rays = test_nerf_A['rays']
        color_bkgds = test_nerf_A['color_bkgd']
        rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
        color_bkgds = color_bkgds.cuda()

        # renderings = []

        plots_path = f'plots_interp_{split}'
        curr_folder_path = os.path.join(plots_path, str(uuid.uuid4()))    
        os.makedirs(curr_folder_path, exist_ok=True)

        for idx in range(len(embeddings)):
            with autocast():
                rgb, _, _, _, _, _ = render_image(
                        radiance_field=decoder,
                        embeddings=embeddings[idx].unsqueeze(dim=0),
                        occupancy_grid=None,
                        rays=Rays(origins=rays.origins.unsqueeze(dim=0), viewdirs=rays.viewdirs.unsqueeze(dim=0)),
                        scene_aabb=scene_aabb,
                        render_step_size=render_step_size,
                        render_bkgd=color_bkgds.unsqueeze(dim=0),
                        grid_weights=None
                    )
            
            
           
            
            img_name = f'{idx}.png'
            full_path = os.path.join(curr_folder_path, img_name)
            
            imageio.imwrite(
                        full_path,
                        (rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8)
                    )
        
        idx += 1

            # renderings.append(rgb)


        # pred_image = wandb.Image((rgb.to('cpu').detach().numpy()[0] * 255).astype(np.uint8)) 
        # self.logfn({f"{split}/nerf_{idx}": [gt_image, pred_image]})
