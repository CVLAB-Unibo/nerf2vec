import math
import time
import imageio

from nerfacc import ContractionType, OccupancyGrid
import tqdm
from classification.utils import get_mlp_params_as_matrix, next_multiple

import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from models.encoder import Encoder
from models.idecoder import ImplicitDecoder

from nerf.loader import NeRFLoader

from pathlib import Path
from random import randint
from typing import Any, Dict, Tuple

from nerf.intant_ngp import NGPradianceField
from classification import config
from nerf.utils import generate_occupancy_grid, render_image

class NeRFDataset(Dataset):
    def __init__(self, nerfs_root: str, sample_sd: Dict[str, Any], device: str) -> None:
        super().__init__()

        self.nerf_paths = self._get_nerf_paths(nerfs_root)

        self.device = device

    def __len__(self) -> int:
        return len(self.nerf_paths)

    def __getitem__(self, index) -> Any:

        dataset_kwargs = {}
        data_dir = self.nerf_paths[index]

        nerf_loader = NeRFLoader(
            data_dir=data_dir,
            num_rays=config.NUM_RAYS,
            device=self.device,
            **dataset_kwargs)
        nerf_loader.training = True

        # Get data for the batch
        data = nerf_loader[0]
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        matrix = torch.load(nerf_loader.weights_file_path, map_location=torch.device(self.device))
        matrix = get_mlp_params_as_matrix(matrix['mlp_base.params'])
        

        nerf_loader.training = False
        # Get data for the batch
        test_data = nerf_loader[0]
        test_render_bkgd = test_data["color_bkgd"]
        test_rays = test_data["rays"]
        test_pixels = test_data["pixels"]


        return rays, pixels, render_bkgd, matrix, nerf_loader.weights_file_path, test_rays, test_pixels, test_render_bkgd
    
    
    def _get_nerf_paths(self, nerfs_root: str):
        
        nerf_paths = []

        for class_name in os.listdir(nerfs_root):

            subject_dirs = os.path.join(nerfs_root, class_name)

            # Sometimes there are hidden files (e.g., when unzipping a file from a Mac)
            if not os.path.isdir(subject_dirs):
                continue
            
            for subject_name in os.listdir(subject_dirs):
                subject_dir = os.path.join(subject_dirs, subject_name)
                nerf_paths.append(subject_dir)
        
        return nerf_paths

class Nerf2vecTrainer:
    def __init__(self, device='cuda:0') -> None:
        train_dset = NeRFDataset(os.path.abspath('data'), None, device='cpu') 

        self.device = device
        
        self.train_loader = DataLoader(
            train_dset,
            batch_size=config.BATCH_SIZE,#16,
            num_workers=0,#4,
            shuffle=True
        )
        
        encoder = Encoder(
            config.MLP_UNITS,
            config.ENCODER_HIDDEN_DIM,
            config.ENCODER_EMBEDDING_DIM
        )
        self.encoder = encoder.cuda()

        decoder = ImplicitDecoder(
            embed_dim=config.ENCODER_EMBEDDING_DIM,
            in_dim=config.DECODER_INPUT_DIM,
            hidden_dim=config.DECODER_HIDDEN_DIM,
            num_hidden_layers_before_skip=config.DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP,
            num_hidden_layers_after_skip=config.DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP,
            out_dim=config.DECODER_OUT_DIM,
            encoding_conf=config.INSTANT_NGP_ENCODING_CONF,
            aabb=torch.tensor(config.AABB, dtype=torch.float32, device=self.device)
        )
        self.decoder = decoder.cuda()

        lr = config.LR
        wd = config.WD
        params = list(self.encoder.parameters())
        params.extend(self.decoder.parameters())
        self.optimizer = AdamW(params, lr, weight_decay=wd)
        
        self.epoch = 0
        self.global_step = 0
        # TODO....
        """
        self.best_chamfer = float("inf")

        self.ckpts_path = get_out_dir() / "ckpts"
        self.all_ckpts_path = get_out_dir() / "all_ckpts"

        if self.ckpts_path.exists():
            self.restore_from_last_ckpt()

        self.ckpts_path.mkdir(exist_ok=True)
        self.all_ckpts_path.mkdir(exist_ok=True)
        """
        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)

    def logfn(self, values: Dict[str, Any]) -> None:
        #wandb.log(values, step=self.global_step, commit=False)
        print(values)
    
    def train(self):
        num_epochs = config.NUM_EPOCHS
        start_epoch = self.epoch

        start = time.time()
        for epoch in range(start_epoch, num_epochs):

            self.epoch = epoch
            self.encoder.train()
            self.decoder.train()

            desc = f"Epoch {epoch}/{num_epochs}"

            for batch in self.train_loader:

                # rays, pixels, render_bkgds, matrices, nerf_weights_path = batch
                rays, pixels, render_bkgds, matrices, nerf_weights_path, test_rays, test_pixels, test_render_bkgds = batch
                # TODO: check rays, it is not created properly
                rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
                pixels = pixels.cuda()
                render_bkgds = render_bkgds.cuda()
                matrices = matrices.cuda()

                test_rays = test_rays._replace(origins=test_rays.origins.cuda(), viewdirs=test_rays.viewdirs.cuda())
                test_pixels = test_pixels.cuda()
                test_render_bkgds = test_render_bkgds.cuda()
                

                grids = []
                
                for elem in nerf_weights_path:
                    grid = generate_occupancy_grid(self.device, 
                                                elem, 
                                                config.INSTANT_NGP_MLP_CONF, 
                                                config.AABB, 
                                                config.OCCUPANCY_GRID_RECONSTRUCTION_ITERATIONS, 
                                                config.OCCUPANCY_GRID_WARMUP_ITERATIONS)
                    grids.append(grid)
                # end = time.time()
                # print(f'elapsed: {end-start}')

                #embeddings = self.encoder(matrices)
                #pred = self.decoder(embeddings, selected_coords)
                # embeddings = torch.rand(config.BATCH_SIZE, config.ENCODER_EMBEDDING_DIM).cuda() # TODO: This is the output of the encoder!
                embeddings = self.encoder(matrices)
                
                render_n_samples = 1024
                scene_aabb = torch.tensor(config.AABB, dtype=torch.float32, device=self.device)
                render_step_size = (
                    (scene_aabb[3:] - scene_aabb[:3]).max()
                    * math.sqrt(3)
                    / render_n_samples
                ).item()
                alpha_thre = 0.0

                # start_time = time.time()
                rgb, acc, depth, n_rendering_samples = render_image(
                    self.decoder,
                    embeddings,
                    grids,
                    rays,
                    scene_aabb,
                    # rendering options
                    near_plane=None,
                    far_plane=None,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgds,
                    cone_angle=0.0,
                    alpha_thre=alpha_thre
                )
                # end_time = time.time()
                # print(f'elapsed: {end_time-start_time}')

                # TODO: evaluate whether to add this condition or not
                if 0 in n_rendering_samples:
                    continue

                alive_ray_mask = acc.squeeze(-1) > 0

                # compute loss
                loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                self.optimizer.zero_grad()
                # do not unscale it because we are using Adam.
                self.grad_scaler.scale(loss).backward()
                self.optimizer.step()
                
                """
                INTERVAL = 100
                if step % INTERVAL == 0:

                    # elapsed_time = time.time() - tic
                    loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                    print(f'loss={loss:.5f}')
                """

                if self.global_step % 100 == 0:
                    end = time.time()
                    # self.logfn(f'"train/loss": {loss.item()} - elapsed: {end-start}')
                    print(f'"train/loss": {loss.item()} - elapsed: {end-start}')


                    self.encoder.eval()
                    self.decoder.eval()
                    # for i in tqdm.tqdm(range(len(self.TEMP_nerf_loader))):
                    #data = self.TEMP_nerf_loader[i]
                    #render_bkgd = data["color_bkgd"]
                    #rays = data["rays"]
                    #pixels = data["pixels"]

                    rgb, acc, depth, n_rendering_samples = render_image(
                        self.decoder,
                        embeddings,
                        grids,
                        test_rays,
                        scene_aabb,
                        # rendering options
                        near_plane=None,
                        far_plane=None,
                        render_step_size=render_step_size,
                        render_bkgd=test_render_bkgds,
                        cone_angle=0.0,
                        alpha_thre=alpha_thre
                    )

                    imageio.imwrite(
                        os.path.join('temp_sanity_check', f'rgb_test_{self.global_step}.png'),
                        (rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8),
                    )
                    """
                    imageio.imwrite(
                                "acc_binary_test.png",
                                ((acc.cpu().detach().numpy()[0] > 0).float().cpu().numpy() * 255).astype(np.uint8),
                    )
                    """
                    

                    self.encoder.train()
                    self.decoder.train()
                    start = time.time()
                        


                self.global_step += 1



