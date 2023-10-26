import datetime
import json
import math
import random
import time

from nerfacc import OccupancyGrid, contract_inv
from classification.utils import get_mlp_params_as_matrix

import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from models.encoder import Encoder
from models.idecoder import ImplicitDecoder
from nerf.intant_ngp import NGPradianceField

from pathlib import Path
from typing import Any, Dict

from classification import config
from nerf.loader_gt import NeRFLoaderGT
from nerf.utils import Rays, render_image, render_image_GT

from torch.cuda.amp import autocast

import wandb
import uuid
import imageio.v2 as imageio

class NeRFDataset(Dataset):
    def __init__(self, split_json: str, device: str) -> None:
        super().__init__()

        with open(split_json) as file:
            self.nerf_paths = json.load(file)
        
        assert isinstance(self.nerf_paths, list), 'The json file provided is not a list.'

        self.device = device

    def __len__(self) -> int:
        return len(self.nerf_paths)

    def __getitem__(self, index) -> Any:

        data_dir = self.nerf_paths[index]

        nerf_loader = NeRFLoaderGT(
            data_dir=data_dir,
            num_rays=config.NUM_RAYS,
            device=self.device)

        nerf_loader.training = True
        data = nerf_loader[0]  # The index has not any relevance when training is True
        color_bkgd = data["color_bkgd"]
        rays = data["rays"]
        train_nerf = {
            'rays': rays,
            'color_bkgd': color_bkgd
        }

        nerf_loader.training = False
        
        test_data = nerf_loader[0]  # TODO: getting just the first image in the dataset. Evaluate whether to return more.
        test_color_bkgd = test_data["color_bkgd"]
        test_rays = test_data["rays"]
        test_nerf = {
            'rays': test_rays,
            'color_bkgd': test_color_bkgd
        }

        # ################################################################################
        # TEMP CODE: will be removed soon! This returns ALL the images contained in the
        #            test set. Used to compute the PSNR on all poses.
        # ################################################################################
        """
        test_nerf = {
            'rays': [],
            'color_bkgd': []
        }
        for test_data in nerf_loader:
            test_color_bkgd = test_data["color_bkgd"]
            test_rays = test_data["rays"]
            
            test_nerf['rays'].append(test_rays)
            test_nerf['color_bkgd'].append(color_bkgd)
        """
        # ################################################################################

        """
        # TODO: refactor these variable names. It is incorrect to say that the first matrix is not flattened, and the second one is it.
        # Actually it should be the opposite: 
        #  - matrix_unflattened = the torch.load returns the original MLP weights, that are a flattened verctors of roughly 18 elements.
        #  - matrix_flattened = it is the result of a reshape operation, that creates a bidimensional matrix, where the second dimension is the number of feature.
        """
        matrix_unflattened = torch.load(nerf_loader.weights_file_path, map_location=torch.device(self.device))  # The NeRF weights obtained from NerfAcc
        matrix_flattened = get_mlp_params_as_matrix(matrix_unflattened['mlp_base.params'])  # The NeRF weights with proper padding

        grid_weights_path = os.path.join(data_dir, 'grid.pth')  
        grid_weights = torch.load(grid_weights_path, map_location=self.device)
        grid_weights['_binary'] = grid_weights['_binary'].to_dense()#.unsqueeze(dim=0)
        n_total_cells = 884736  # TODO: add this as config parameter (884736 if resolution == 96 else 2097152)
        grid_weights['occs'] = torch.empty([n_total_cells]) 
        
        N = 32000  # TODO: add this as config parameter
        background_indices, n_true_coordinates = self._sample_unoccupied_cells(N, grid_weights['_binary'], data_dir, n_total_cells)

        return train_nerf, test_nerf, matrix_unflattened, matrix_flattened, grid_weights, data_dir, background_indices, n_true_coordinates
    
    def _sample_unoccupied_cells(self, n: int, binary: torch.Tensor, data_dir, n_total_cells: int) -> torch.Tensor:
        
        # 0 -> PERMUTAION
        # 1 -> BETAVARIATE
        # 2 -> UNIFORM SAMPLE WITHOUT REPLACEMENT
        APPROACH = 2

        zero_indices = torch.nonzero(binary.flatten() == 0)[:, 0]
        # one_indices = torch.nonzero(binary.flatten() == 1)[:, 0]
        n_one_indices = n_total_cells - len(zero_indices)

        if len(zero_indices) < n:
            print(f'ERROR: {len(zero_indices)} - {data_dir}')

        
        # PERMUTATION
        if APPROACH == 0:
            randomized_indices = zero_indices[torch.randperm(zero_indices.size(0))][:n]

        # BETAVARIATE
        elif APPROACH == 1:
            alpha = 4
            beta = 4

            probabilities = [random.betavariate(alpha, beta) for _ in range(len(zero_indices))]

            # Normalize the probabilities to sum up to 1
            total_prob = sum(probabilities)
            probabilities = [p / total_prob for p in probabilities]

            # Perform weighted random sampling to get n indices
            randomized_indices = random.choices(range(0, len(zero_indices)), probabilities, k=n)

        # UNIFORM SAMPLE WITHOUT REPLACEMENT
        elif APPROACH == 2:

            randomized_indices = random.sample(range(0, len(zero_indices)), n)
        
        randomized_indices = zero_indices[randomized_indices]
        return randomized_indices, n_one_indices
        
    
class Nerf2vecTrainer:
    def __init__(self, device='cuda:0') -> None:

        self.device = device

        train_dset_json = os.path.abspath(os.path.join('data', 'train.json'))
        train_dset = NeRFDataset(train_dset_json, device='cpu') 

        self.train_loader = DataLoader(
            train_dset,
            batch_size=config.BATCH_SIZE,
            shuffle=True, 
            num_workers=8, 
            persistent_workers=False, 
            prefetch_factor=2
        )

        val_dset_json = os.path.abspath(os.path.join('data', 'validation.json'))  
        val_dset = NeRFDataset(val_dset_json, device='cpu')   
        self.val_loader = DataLoader(
            val_dset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=8, 
            persistent_workers=False
        )
        self.val_loader_shuffled = DataLoader(
            val_dset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=8, 
            persistent_workers=False
        )
        
        encoder = Encoder(
            config.MLP_UNITS,
            config.ENCODER_HIDDEN_DIM,
            config.ENCODER_EMBEDDING_DIM
        )
        self.encoder = encoder.to(self.device)

        decoder = ImplicitDecoder(
            embed_dim=config.ENCODER_EMBEDDING_DIM,
            in_dim=config.DECODER_INPUT_DIM,
            hidden_dim=config.DECODER_HIDDEN_DIM,
            num_hidden_layers_before_skip=config.DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP,
            num_hidden_layers_after_skip=config.DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP,
            out_dim=config.DECODER_OUT_DIM,
            encoding_conf=config.INSTANT_NGP_ENCODING_CONF,
            aabb=torch.tensor(config.GRID_AABB, dtype=torch.float32, device=self.device)
        )
        self.decoder = decoder.to(self.device)
        occupancy_grid = OccupancyGrid(
            roi_aabb=config.GRID_AABB,
            resolution=config.GRID_RESOLUTION,
            contraction_type=config.GRID_CONTRACTION_TYPE,
        )
        self.occupancy_grid = occupancy_grid.to(self.device)
        self.occupancy_grid.eval()

        self.ngp_mlp = NGPradianceField(**config.INSTANT_NGP_MLP_CONF).to(device)
        self.ngp_mlp.eval()

        self.scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device=self.device)
        self.render_step_size = (
            (self.scene_aabb[3:] - self.scene_aabb[:3]).max()
            * math.sqrt(3)
            / config.GRID_CONFIG_N_SAMPLES
        ).item()

        lr = config.LR
        wd = config.WD
        params = list(self.encoder.parameters())
        params.extend(self.decoder.parameters())
        
        self.optimizer = AdamW(params, lr, weight_decay=wd)
        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)
        
        self.epoch = 0
        self.global_step = 0
        self.best_psnr = float("-inf")

        self.ckpts_path = Path(os.path.join('classification', 'train', 'ckpts'))
        self.all_ckpts_path = Path(os.path.join('classification', 'train', 'all_ckpts'))

        if self.ckpts_path.exists():
            self.restore_from_last_ckpt()

        self.ckpts_path.mkdir(parents=True, exist_ok=True)
        self.all_ckpts_path.mkdir(parents=True, exist_ok=True)

    def logfn(self, values: Dict[str, Any]) -> None:
        wandb.log(values, step=self.global_step, commit=False)

    def train(self):

        self.config_wandb()

        num_epochs = config.NUM_EPOCHS
        start_epoch = self.epoch

        # ################################################################################
        # TEMP CODE: used to compute PSNR on specific saved model's weights
        # ################################################################################
        """
        # ckpt_names = ['0.pt', '100.pt', '200.pt', '300.pt', '400.pt', '500.pt', '600.pt', '700.pt']
        # ckpt_names = ['0.pt', '100.pt', '200.pt', '300.pt']
        
        for ck in ckpt_names:
            self.restore_ckpt(ck)
            self.plot2('train')
            self.plot2('val')
            # self.val('train')
        exit()
        """
        
        for epoch in range(start_epoch, num_epochs):

            self.epoch = epoch
            self.encoder.train()
            self.decoder.train()
            
            print(f'Epoch {epoch} started...')          
            epoch_start = time.time()
            batch_start = time.time()

            for batch_idx, batch in enumerate(self.train_loader):
                train_nerf, _, matrices_unflattened, matrices_flattened, grid_weights, data_dir, background_indices, n_true_coordinates = batch

                rays = train_nerf['rays']
                color_bkgds = train_nerf['color_bkgd']
                color_bkgds = color_bkgds[0][None].expand(len(matrices_flattened), -1)

                rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
                color_bkgds = color_bkgds.cuda()
                matrices_flattened = matrices_flattened.cuda()

                # Enable autocast for mixed precision
                with autocast():
                    pixels, alpha, filtered_rays = render_image_GT(
                            radiance_field=self.ngp_mlp, 
                            occupancy_grid=self.occupancy_grid, 
                            rays=rays, 
                            scene_aabb=self.scene_aabb, 
                            render_step_size=self.render_step_size,
                            color_bkgds=color_bkgds,
                            grid_weights=grid_weights,
                            ngp_mlp_weights=matrices_unflattened,
                            # ngp_mlp=self.ngp_mlp,
                            device=self.device)
                    pixels = pixels * alpha + color_bkgds.unsqueeze(1) * (1.0 - alpha)

                    embeddings = self.encoder(matrices_flattened)
                    
                    rgb, acc, _, n_rendering_samples,  bg_rgb_pred, bg_rgb_label = render_image(
                        self.decoder,
                        embeddings,
                        self.occupancy_grid,
                        filtered_rays,
                        self.scene_aabb,
                        render_step_size=self.render_step_size,
                        render_bkgd=color_bkgds,
                        grid_weights=grid_weights,
                        background_indices=background_indices
                    )
                    
                    fg_loss = F.smooth_l1_loss(rgb, pixels) * config.FG_WEIGHT
                    bg_loss = F.smooth_l1_loss(bg_rgb_pred, bg_rgb_label) * config.BG_WEIGHT
                    loss = fg_loss + bg_loss
                    
                self.optimizer.zero_grad()
                self.grad_scaler.scale(loss).backward()

                self.optimizer.step()

                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss.item()})
                
                self.global_step += 1

                batch_end = time.time()
                if batch_idx % 1000 == 0:
                    print(f'Completed {batch_idx} batches in {batch_end-batch_start}s')

            if (epoch > 0 and epoch % 10 == 0) or epoch == num_epochs - 1:
                self.val(split='train')
                self.val(split='validation')

                self.plot(split='train')
                self.plot(split='validation')

                
            if epoch % 50 == 0:
                self.save_ckpt(all=True)
            
            self.save_ckpt()
            
            epoch_end = time.time()
            print(f'Epoch {epoch} completed in {epoch_end-epoch_start}s')        
    
    @torch.no_grad()
    def val(self, split: str) -> None:
        
        loader = self.train_loader if split == "train" else self.val_loader

        self.encoder.eval()
        self.decoder.eval()

        # psnrs_masked = []
        psnrs = []
        psnrs_bg = []
        idx = 0

        print(f'Validating on {split} set')

        for batch_idx, batch in enumerate(loader):

            train_nerf, _, matrices_unflattened, matrices_flattened, grid_weights, data_dir, background_indices, n_true_coordinates = batch
            rays = train_nerf['rays']
            color_bkgds = train_nerf['color_bkgd']
            color_bkgds = color_bkgds[0].unsqueeze(0).expand(len(matrices_flattened), -1) # TODO: refactor this
            
            rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
            color_bkgds = color_bkgds.cuda()
            matrices_flattened = matrices_flattened.cuda()
            with autocast():
                pixels, alpha, filtered_rays = render_image_GT(
                            radiance_field=self.ngp_mlp, 
                            occupancy_grid=self.occupancy_grid, 
                            rays=rays, 
                            scene_aabb=self.scene_aabb, 
                            render_step_size=self.render_step_size,
                            color_bkgds=color_bkgds,
                            grid_weights=grid_weights,
                            ngp_mlp_weights=matrices_unflattened,
                            # ngp_mlp=self.ngp_mlp,
                            device=self.device)
                pixels = pixels * alpha + color_bkgds.unsqueeze(1) * (1.0 - alpha)
                
                embeddings = self.encoder(matrices_flattened)
                
                rgb, acc, _, n_rendering_samples, bg_rgb_pred, bg_rgb_label = render_image(
                    self.decoder,
                    embeddings,
                    self.occupancy_grid,
                    filtered_rays,
                    self.scene_aabb,
                    render_step_size=self.render_step_size,
                    render_bkgd=color_bkgds,
                    grid_weights=grid_weights,
                    background_indices=background_indices
                )
                
                fg_mse = F.mse_loss(rgb, pixels) * config.FG_WEIGHT
                bg_mse = F.mse_loss(bg_rgb_pred, bg_rgb_label) * config.BG_WEIGHT

                mse_bg = fg_mse + bg_mse
                mse = F.mse_loss(rgb, pixels)
            
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs.append(psnr.item())

            psnr_bg = -10.0 * torch.log(mse_bg) / np.log(10.0)
            psnrs_bg.append(psnr_bg.item())

            if idx > 99:
                break
            idx+=1
        
        mean_psnr = sum(psnrs) / len(psnrs)
        mean_psnr_bg = sum(psnrs_bg) / len(psnrs_bg)

        self.logfn({f'{split}/PSNR': mean_psnr})
        self.logfn({f'{split}/PSNR_BG': mean_psnr_bg})
        
        if split == 'validation' and mean_psnr > self.best_psnr:
            self.best_psnr = mean_psnr
            self.save_ckpt(best=True)
    
    @torch.no_grad()
    def plot(self, split: str) -> None:
        
        loader = self.train_loader if split == "train" else self.val_loader_shuffled

        print('Plot started...')

        self.encoder.eval()
        self.decoder.eval()

        loader_iter = iter(loader)
        _, test_nerf, matrices_unflattened, matrices_flattened, grid_weights, data_dir, _, _ = next(loader_iter)
        
        rays = test_nerf['rays']
        color_bkgds = test_nerf['color_bkgd']
        
        rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())

        color_bkgds = color_bkgds.cuda()
        matrices_flattened = matrices_flattened.cuda()
        
        with autocast():
            pixels, alpha, _ = render_image_GT(
                            radiance_field=self.ngp_mlp, 
                            occupancy_grid=self.occupancy_grid, 
                            rays=rays, 
                            scene_aabb=self.scene_aabb, 
                            render_step_size=self.render_step_size,
                            color_bkgds=color_bkgds,
                            grid_weights=grid_weights,
                            ngp_mlp_weights=matrices_unflattened,
                            # ngp_mlp=self.ngp_mlp,
                            device=self.device,
                            training=False)
            pixels = pixels * alpha + color_bkgds.unsqueeze(1).unsqueeze(1) * (1.0 - alpha)
        
            embeddings = self.encoder(matrices_flattened)

            for idx in range(len(matrices_flattened)):
                
                curr_grid_weights = {
                    '_roi_aabb': [grid_weights['_roi_aabb'][idx]],
                    '_binary': [grid_weights['_binary'][idx]],
                    'resolution': [grid_weights['resolution'][idx]],
                    'occs': [grid_weights['occs'][idx]],
                }
        
                rgb, _, _, _, _, _ = render_image(
                    radiance_field=self.decoder,
                    embeddings=embeddings[idx].unsqueeze(dim=0),
                    occupancy_grid=self.occupancy_grid,
                    rays=Rays(origins=rays.origins[idx].unsqueeze(dim=0), viewdirs=rays.viewdirs[idx].unsqueeze(dim=0)),
                    scene_aabb=self.scene_aabb,
                    render_step_size=self.render_step_size,
                    render_bkgd=color_bkgds[idx].unsqueeze(dim=0),
                    grid_weights=curr_grid_weights
                )
                
                rgb_A, alpha, _, _, _, _ = render_image(
                                radiance_field=self.decoder,
                                embeddings=embeddings[idx].unsqueeze(dim=0),
                                occupancy_grid=None,
                                rays=Rays(origins=rays.origins[idx].unsqueeze(dim=0), viewdirs=rays.viewdirs[idx].unsqueeze(dim=0)),
                                scene_aabb=self.scene_aabb,
                                render_step_size=self.render_step_size,
                                render_bkgd=color_bkgds[idx].unsqueeze(dim=0),
                                grid_weights=None
                )
                
                gt_image = wandb.Image((pixels.to('cpu').detach().numpy()[idx] * 255).astype(np.uint8))
                pred_image_grid = wandb.Image((rgb.to('cpu').detach().numpy()[0] * 255).astype(np.uint8)) 
                pred_image_no_grid = wandb.Image((rgb_A.to('cpu').detach().numpy() * 255).astype(np.uint8))

                self.logfn({f"{split}/nerf_{idx}": [gt_image, pred_image_grid, pred_image_no_grid]})
                                
                """
                img_name = str(uuid.uuid4())
                plots_path = 'plots'

                imageio.imwrite(
                    os.path.join(plots_path, f'{img_name}_{split}_gt_{self.global_step}.png'),
                    (pixels.cpu().detach().numpy()[idx] * 255).astype(np.uint8)
                )
                
                imageio.imwrite(
                    os.path.join(plots_path, f'{img_name}_{split}_pred_{self.global_step}_GRID.png'),
                    (rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8)
                )
                
                img_name = f'{img_name}_{split}_pred_{self.global_step}'
                plots_path = 'plots'
                imageio.imwrite(
                    os.path.join(plots_path, f'{img_name}.png'),
                    (rgb_A.numpy() * 255).astype(np.uint8)
                )"""
    
    """
    @torch.no_grad()
    def plot2(self, split: str) -> None:
        
        loader = self.train_loader if split == "train" else self.val_loader_shuffled

        print('Plot started...')

        self.encoder.eval()
        self.decoder.eval()

        loader_iter = iter(loader)
        # _, test_nerf, matrices_unflattened, matrices_flattened, grid_weights, data_dir, _, _ = next(loader_iter)
        
        psnrs = []
        psnrs_no_bg = []

        batch = next(loader_iter)

        with open(f'{split}_{self.epoch}ep_PSNR.txt', 'a') as f:
            n_nerfs = 0

            while batch:
                _, test_nerf, matrices_unflattened, matrices_flattened, grid_weights, data_dir, _, _ = batch
            
                rays = test_nerf['rays']
                color_bkgds = test_nerf['color_bkgd']
                
                rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())

                color_bkgds = color_bkgds.cuda()
                matrices_flattened = matrices_flattened.cuda()
                
                with autocast():
                    pixels, alpha, _ = render_image_GT(
                                    radiance_field=self.ngp_mlp, 
                                    occupancy_grid=self.occupancy_grid, 
                                    rays=rays, 
                                    scene_aabb=self.scene_aabb, 
                                    render_step_size=self.render_step_size,
                                    color_bkgds=color_bkgds,
                                    grid_weights=grid_weights,
                                    ngp_mlp_weights=matrices_unflattened,
                                    # ngp_mlp=self.ngp_mlp,
                                    device=self.device,
                                    training=False)
                    pixels = pixels * alpha + color_bkgds.unsqueeze(1).unsqueeze(1) * (1.0 - alpha)
                
                    embeddings = self.encoder(matrices_flattened)

                    for idx in range(len(matrices_flattened)):
                        
                        curr_grid_weights = {
                            '_roi_aabb': [grid_weights['_roi_aabb'][idx]],
                            '_binary': [grid_weights['_binary'][idx]],
                            'resolution': [grid_weights['resolution'][idx]],
                            'occs': [grid_weights['occs'][idx]],
                        }
                
                        rgb, _, _, _, _, _ = render_image(
                            radiance_field=self.decoder,
                            embeddings=embeddings[idx].unsqueeze(dim=0),
                            occupancy_grid=self.occupancy_grid,
                            rays=Rays(origins=rays.origins[idx].unsqueeze(dim=0), viewdirs=rays.viewdirs[idx].unsqueeze(dim=0)),
                            scene_aabb=self.scene_aabb,
                            render_step_size=self.render_step_size,
                            render_bkgd=color_bkgds[idx].unsqueeze(dim=0),
                            grid_weights=curr_grid_weights
                        )
                        
                        rgb_A, alpha, b, c, _, _ = render_image(
                                        radiance_field=self.decoder,
                                        embeddings=embeddings[idx].unsqueeze(dim=0),
                                        occupancy_grid=None,
                                        rays=Rays(origins=rays.origins[idx].unsqueeze(dim=0), viewdirs=rays.viewdirs[idx].unsqueeze(dim=0)),
                                        scene_aabb=self.scene_aabb,
                                        render_step_size=self.render_step_size,
                                        render_bkgd=color_bkgds[idx].unsqueeze(dim=0),
                                        grid_weights=None
                        )
                        
                        pred_image_2=wandb.Image((rgb_A.to('cpu').detach().numpy() * 255).astype(np.uint8))
                        pred_image = wandb.Image((rgb.to('cpu').detach().numpy()[0] * 255).astype(np.uint8)) 
                        gt_image = wandb.Image((pixels.to('cpu').detach().numpy()[idx] * 255).astype(np.uint8))

                        self.logfn({f"{split}/nerf_{idx}": [gt_image, pred_image, pred_image_2]})

                        # ################################################################################
                        # TEMP CODE
                        # ################################################################################
                        mse = F.mse_loss(rgb_A[0], pixels[idx])
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)

                        mse_no_bg = F.mse_loss(rgb[0], pixels[idx])
                        psnr_no_bg = -10.0 * torch.log(mse_no_bg) / np.log(10.0)

                        print(f'{data_dir[idx]}: {psnr.item()} - {psnr_no_bg.item()}')
                        f.write(f'{data_dir[idx]}: {psnr.item()} - {psnr_no_bg.item()}\n')
                        

                        psnrs.append(psnr.item())
                        psnrs_no_bg.append(psnr_no_bg.item())

                        '''
                        plots_path = 'plots'
                        img_name = '_'.join(data_dir[idx].split('/')[-2:])
                        imageio.imwrite(
                            os.path.join(plots_path, f'{img_name}_{psnr.item()}.png'),
                            (rgb_A.cpu().detach().numpy()[0] * 255).astype(np.uint8)
                        )
                        '''
                        # ################################################################################
                        
                        '''
                        img_name = str(uuid.uuid4())
                        plots_path = 'plots'
                        imageio.imwrite(
                            os.path.join(plots_path, f'{img_name}_{split}_gt_{self.global_step}.png'),
                            (pixels.cpu().detach().numpy()[idx] * 255).astype(np.uint8)
                        )
                        
                        imageio.imwrite(
                            os.path.join(plots_path, f'{img_name}_{split}_pred_{self.global_step}_GRID.png'),
                            (rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8)
                        )

                        
                        img_name = f'{img_name}_{split}_pred_{self.global_step}'
                        plots_path = 'plots'
                        imageio.imwrite(
                            os.path.join(plots_path, f'{img_name}.png'),
                            (rgb_A.numpy() * 255).astype(np.uint8)
                        )'''
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    batch = None

                n_nerfs += 1
                print(f'processed: {n_nerfs * config.BATCH_SIZE} NeRFs')

                if n_nerfs >= 64:  # with BS=16 it means that 1024 NeRFs will be processed before interrupting this loop
                    break

        
            print(f'mean PSRN: {sum(psnrs) / len(psnrs)}')
            print(f'mean PSRN (no BG): {sum(psnrs_no_bg) / len(psnrs_no_bg)}')

            f.write('\n')
            f.write(f'mean PSRN: {sum(psnrs) / len(psnrs)}\n')
            f.write(f'mean PSRN (no BG): {sum(psnrs_no_bg) / len(psnrs_no_bg)}')
    
    @torch.no_grad()
    def plot3(self, split: str) -> None:
        
        loader = self.train_loader if split == "train" else self.val_loader_shuffled

        print('Plot started...')

        self.encoder.eval()
        self.decoder.eval()

        loader_iter = iter(loader)
        # _, test_nerfs, matrices_unflattened, matrices_flattened, grid_weights, data_dir, _, _ = next(loader_iter)
        
        N_RENDERINGS_PER_NERF = 36

        N_NERFS = 128
        N_BATCHES = N_NERFS / config.BATCH_SIZE
        curr_batch = 0
        
        psnrs = np.zeros(N_NERFS)
        psnrs_no_bg = np.zeros(N_NERFS)

        batch = next(loader_iter)

        with open(f'{split}_{self.epoch}ep_PSNR.txt', 'a') as f:

            while batch:
                _, test_nerfs, matrices_unflattened, matrices_flattened, grid_weights, data_dir, _, _ = batch

                assert len(test_nerfs['rays']) == N_RENDERINGS_PER_NERF and len(test_nerfs['color_bkgd']) == N_RENDERINGS_PER_NERF, "Wrong number of renderings found."

                for pose_idx in range(N_RENDERINGS_PER_NERF):
                    rays = test_nerfs['rays'][pose_idx]
                    color_bkgds = test_nerfs['color_bkgd'][pose_idx]
                
                    rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())

                    color_bkgds = color_bkgds.cuda()
                    matrices_flattened = matrices_flattened.cuda()
                
                    with autocast():
                        pixels, alpha, _ = render_image_GT(
                            radiance_field=self.ngp_mlp, 
                            occupancy_grid=self.occupancy_grid, 
                            rays=rays, 
                            scene_aabb=self.scene_aabb, 
                            render_step_size=self.render_step_size,
                            color_bkgds=color_bkgds,
                            grid_weights=grid_weights,
                            ngp_mlp_weights=matrices_unflattened,
                            # ngp_mlp=self.ngp_mlp,
                            device=self.device,
                            training=False)
                        pixels = pixels * alpha + color_bkgds.unsqueeze(1).unsqueeze(1) * (1.0 - alpha)
                
                        embeddings = self.encoder(matrices_flattened)

                        for idx in range(len(matrices_flattened)):
                            
                            curr_grid_weights = {
                                '_roi_aabb': [grid_weights['_roi_aabb'][idx]],
                                '_binary': [grid_weights['_binary'][idx]],
                                'resolution': [grid_weights['resolution'][idx]],
                                'occs': [grid_weights['occs'][idx]],
                            }
                    
                            rgb, _, _, _, _, _ = render_image(
                                radiance_field=self.decoder,
                                embeddings=embeddings[idx].unsqueeze(dim=0),
                                occupancy_grid=self.occupancy_grid,
                                rays=Rays(origins=rays.origins[idx].unsqueeze(dim=0), viewdirs=rays.viewdirs[idx].unsqueeze(dim=0)),
                                scene_aabb=self.scene_aabb,
                                render_step_size=self.render_step_size,
                                render_bkgd=color_bkgds[idx].unsqueeze(dim=0),
                                grid_weights=curr_grid_weights
                            )
                            
                            rgb_A, alpha, b, c, _, _ = render_image(
                                            radiance_field=self.decoder,
                                            embeddings=embeddings[idx].unsqueeze(dim=0),
                                            occupancy_grid=None,
                                            rays=Rays(origins=rays.origins[idx].unsqueeze(dim=0), viewdirs=rays.viewdirs[idx].unsqueeze(dim=0)),
                                            scene_aabb=self.scene_aabb,
                                            render_step_size=self.render_step_size,
                                            render_bkgd=color_bkgds[idx].unsqueeze(dim=0),
                                            grid_weights=None
                            )
                            
                            pred_image_2=wandb.Image((rgb_A.to('cpu').detach().numpy() * 255).astype(np.uint8))
                            pred_image = wandb.Image((rgb.to('cpu').detach().numpy()[0] * 255).astype(np.uint8)) 
                            gt_image = wandb.Image((pixels.to('cpu').detach().numpy()[idx] * 255).astype(np.uint8))

                            self.logfn({f"{split}/nerf_{idx}": [gt_image, pred_image, pred_image_2]})

                            # ################################################################################
                            # TEMP CODE
                            # ################################################################################
                            mse = F.mse_loss(rgb_A[0], pixels[idx])
                            psnr = -10.0 * torch.log(mse) / np.log(10.0)

                            mse_no_bg = F.mse_loss(rgb[0], pixels[idx])
                            psnr_no_bg = -10.0 * torch.log(mse_no_bg) / np.log(10.0)

                            print(f'{data_dir[idx]}: {psnr.item()} - {psnr_no_bg.item()}')
                            f.write(f'{data_dir[idx]}: {psnr.item()} - {psnr_no_bg.item()}\n')
                            
                            curr_idx = idx + curr_batch*config.BATCH_SIZE
                            psnrs[curr_idx] += psnr.item()
                            psnrs_no_bg[curr_idx] += psnr_no_bg.item()
                            print(f'curr_pose: {pose_idx} - curr_nerf: {curr_idx}')
                            print()

                            '''
                            plots_path = 'plots'
                            img_name = '_'.join(data_dir[idx].split('/')[-2:])
                            imageio.imwrite(
                                os.path.join(plots_path, f'{img_name}_{psnr.item()}.png'),
                                (rgb_A.cpu().detach().numpy()[0] * 255).astype(np.uint8)
                            )
                            '''
                            # ################################################################################
                            
                            '''
                            img_name = str(uuid.uuid4())
                            plots_path = 'plots'
                            imageio.imwrite(
                                os.path.join(plots_path, f'{img_name}_{split}_gt_{self.global_step}.png'),
                                (pixels.cpu().detach().numpy()[idx] * 255).astype(np.uint8)
                            )
                            
                            imageio.imwrite(
                                os.path.join(plots_path, f'{img_name}_{split}_pred_{self.global_step}_GRID.png'),
                                (rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8)
                            )

                            
                            img_name = f'{img_name}_{split}_pred_{self.global_step}'
                            plots_path = 'plots'
                            imageio.imwrite(
                                os.path.join(plots_path, f'{img_name}.png'),
                                (rgb_A.numpy() * 255).astype(np.uint8)
                            )'''
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    batch = None

                curr_batch += 1 
                print(f'processed: {curr_batch * config.BATCH_SIZE} NeRFs')

                if curr_batch >= N_BATCHES:
                    break
            
            # Divide for the number of images found for each NeRF
            psnrs /=  N_RENDERINGS_PER_NERF  
            psnrs_no_bg /= N_RENDERINGS_PER_NERF

            print(f'mean PSRN: {sum(psnrs) / len(psnrs)}')
            print(f'mean PSRN (no BG): {sum(psnrs_no_bg) / len(psnrs_no_bg)}')

            f.write('\n')
            f.write(f'mean PSRN: {sum(psnrs) / len(psnrs)}\n')
            f.write(f'mean PSRN (no BG): {sum(psnrs_no_bg) / len(psnrs_no_bg)}')
    """
                 
    def save_ckpt(self, best: bool = False, all: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_psnr": self.best_psnr,
        }

        if all:
            ckpt_path = self.all_ckpts_path / f"{self.epoch}.pt"
            torch.save(ckpt, ckpt_path)

        else:
            for previous_ckpt_path in self.ckpts_path.glob("*.pt"):
                if "best" not in previous_ckpt_path.name:
                    previous_ckpt_path.unlink()

            ckpt_path = self.ckpts_path / f"{self.epoch}.pt"
            torch.save(ckpt, ckpt_path)

        if best:
            ckpt_path = self.ckpts_path / "best.pt"
            torch.save(ckpt, ckpt_path)
    
    def restore_from_last_ckpt(self) -> None:
        if self.ckpts_path.exists():
            ckpt_paths = [p for p in self.ckpts_path.glob("*.pt") if "best" not in p.name]
            error_msg = "Expected only one ckpt apart from best, found none or too many."
            assert len(ckpt_paths) == 1, error_msg

            ckpt_path = ckpt_paths[0]
            print(f'loading weights: {ckpt_path}')
            ckpt = torch.load(ckpt_path)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = self.epoch * len(self.train_loader)
            self.best_psnr = ckpt["best_psnr"]

            self.encoder.load_state_dict(ckpt["encoder"])
            self.decoder.load_state_dict(ckpt["decoder"])
            self.optimizer.load_state_dict(ckpt["optimizer"])

            # self.optimizer.param_groups[0]['lr'] = 1e-5
    
    """
    def restore_ckpt(self, name) -> None:
        ckpt_path = os.path.join(self.all_ckpts_path, name)
        print(f'loading weights: {ckpt_path}')
        ckpt = torch.load(ckpt_path)

        self.epoch = ckpt["epoch"] + 1
        self.global_step = self.epoch * len(self.train_loader)
        self.best_psnr = ckpt["best_psnr"]

        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
    """
    
    
    def config_wandb(self):
        wandb.init(
            entity='dsr-lab',
            project='nerf2vec',
            name=f'run_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
            config=config.WANDB_CONFIG
        )
