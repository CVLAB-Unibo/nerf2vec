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
        
        # self.nerf_paths = self._get_nerf_paths('data\\data_TRAINED')
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

        # print(f'focal: {nerf_loader.focal}')

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

        matrix_unflattened = torch.load(nerf_loader.weights_file_path, map_location=torch.device(self.device))
        matrix_flattened = get_mlp_params_as_matrix(matrix_unflattened['mlp_base.params'])

        grid_weights_path = os.path.join(data_dir, 'grid.pth')  
        grid_weights = torch.load(grid_weights_path, map_location=self.device)
        grid_weights['_binary'] = grid_weights['_binary'].to_dense().unsqueeze(dim=0)
        grid_weights['occs'] = torch.empty([884736])   # 884736 if resolution == 96 else 2097152
        
        N = 10000
        background_indices = self._sample_unoccupied_cells(N, grid_weights['_binary'][0])

        return train_nerf, test_nerf, matrix_unflattened, matrix_flattened, grid_weights, data_dir, background_indices
    
    def _sample_unoccupied_cells(self, n: int, binary: torch.Tensor) -> torch.Tensor:
        
        # 0 -> PERMUTAION
        # 1 -> BETAVARIATE
        # 2 -> UNIFORM SAMPLE WITHOUT REPLACEMENT
        APPROACH = 2

        zero_indices = torch.nonzero(binary.flatten() == 0)[:, 0]

        
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
        return randomized_indices
        
    
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
            prefetch_factor=2,
            #pin_memory=True
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
        # self.optimizer = AdamW(params, lr, weight_decay=wd, eps=1e-15)
        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)

        """
        max_steps = config.NUM_EPOCHS * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.01, total_iters=20
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=[
                        max_steps // 2,
                        max_steps * 3 // 4,
                        max_steps * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )
        
        num_steps = config.NUM_EPOCHS * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, config.LR, total_steps=num_steps)
        

        total_training_steps = config.NUM_EPOCHS * len(self.train_loader)
        warmup_steps = 100
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                             max_lr=1e-2, 
                                                             div_factor=100,
                                                             total_steps=total_training_steps, 
                                                             final_div_factor=10,
                                                             pct_start=(warmup_steps)/total_training_steps
                                                            )
        """
        
        self.epoch = 0
        self.global_step = 0
        self.best_psnr = float(0)  # float("inf")

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

        # self.plot('train')
        # exit()

        for epoch in range(start_epoch, num_epochs):

            self.epoch = epoch
            self.encoder.train()
            self.decoder.train()
            
            print(f'Epoch {epoch} started...')          
            epoch_start = time.time()
            batch_start = time.time()

            for batch_idx, batch in enumerate(self.train_loader):
                # print(f'Batch {batch_idx} started...')
                # rays, pixels, render_bkgds, matrices, nerf_weights_path = batch
                train_nerf, _, matrices_unflattened, matrices_flattened, grid_weights, data_dir, background_indices = batch

                # print(data_dir)
                rays = train_nerf['rays']
                color_bkgds = train_nerf['color_bkgd']
                # color_bkgds2 = color_bkgds[0].unsqueeze(0).expand(len(matrices_flattened), -1)
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
                    
                    rgb, acc, _, n_rendering_samples, bg_positions, n_points = render_image(
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


                    print(n_points)

                    # TODO: evaluate whether to add this condition or not
                    if 0 in n_rendering_samples:
                        print(f'0 n_rendering_samples. Skip iteration.')
                        continue

                    #alive_ray_mask = acc.squeeze(-1) > 0
                    alive_ray_mask = acc > 0

                    # bg_pred = torch.nn.Sigmoid()(bg_pred)
                    
                    
                    # ################################################################################
                    # VERSION WITH DOUBLE LOSS - COMPUTING RGB FOR BACKGROUND
                    # ###############################################################################
                    
                    bg_rgb_labels = color_bkgds[0].view(1, 3).expand(len(matrices_flattened), -1, -1)
                    bg_rgb_labels = bg_rgb_labels.repeat(1, bg_positions.shape[1], 1)
                    
                    # Compute background predictions
                    bg_rgb_pred , bg_alpha_pred = self.decoder(embeddings, bg_positions)  # bg_positions are all positions in the background
                    # bg_rgb_pred = bg_rgb_pred + bg_rgb_labels.clone() * (1.0 - bg_alpha_pred)
                    bg_rgb_pred = bg_rgb_pred * bg_alpha_pred + bg_rgb_labels.clone() * (1.0 - bg_alpha_pred)
                    
                    # Append to the foreground predictions and ground truths
                    #rgb = torch.cat((rgb, bg_rgb_pred), dim=1)
                    #pixels = torch.cat((pixels, bg_rgb_labels), dim=1)
                    print(f'len_rgb: {rgb.shape[1]} - len_bg_rgb: {bg_rgb_pred.shape[1]}')
                    print(f'losses: {F.smooth_l1_loss(rgb, pixels), F.smooth_l1_loss(bg_rgb_pred, bg_rgb_labels)}')
                    # Compute the loss
                    total = rgb.shape[1] + bg_rgb_pred.shape[1]
                    fg_weight = bg_rgb_pred.shape[1] / total
                    bg_weight = rgb.shape[1] / total
                    
                    #fg_weight = 1
                    #n_points_average = np.sum(n_points) / len(n_points)
                    #min_range = np.min(n_points)
                    #max_range = np.max(n_points)
                    #bg_weight = (n_points_average - min_range) / (max_range - min_range)
                    #print(f'bg_weight: {bg_weight}')
                    
                    fg_weight = 1.0
                    bg_weight = 0.2533
        
                    loss = F.smooth_l1_loss(rgb, pixels) * fg_weight + F.smooth_l1_loss(bg_rgb_pred, bg_rgb_labels) * bg_weight
                    # loss = F.smooth_l1_loss(rgb, pixels) + (F.smooth_l1_loss(bg_rgb_pred, bg_rgb_labels) * (512/10000))
                    # loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                    print(loss)
                    
                    # ################################################################################
                    # VERSION WITH DOUBLE LOSS - COMPUTING ONLY ALPHA FOR BACKGROUND
                    # ################################################################################
                    """
                    bg_labels_shape = (bg_positions.shape[0], bg_positions.shape[1], 1)
                    bg_alpha_labels = torch.zeros(bg_labels_shape, device=bg_positions.device, dtype=torch.float32)
                    _ , bg_alpha_pred = self.decoder(embeddings, bg_positions)
                    
                    total = rgb.shape[1] + bg_alpha_pred.shape[1]
                    fg_weight = bg_alpha_pred.shape[1] / total
                    bg_weight = rgb.shape[1] / total
                    loss = F.smooth_l1_loss(rgb, pixels) * fg_weight + F.smooth_l1_loss(bg_alpha_pred, bg_alpha_labels) * bg_weight
                    """ 
                    
                    

                self.optimizer.zero_grad()
                self.grad_scaler.scale(loss).backward()
                self.optimizer.step()
                # self.scheduler.step()
                
                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss.item()})
                    # self.logfn({"train/LR":self.optimizer.param_groups[0]['lr']})
                    # print(self.optimizer.param_groups[0]['lr'])
                    # self.logfn({"train/LR":self.scheduler.get_last_lr()[0]})
                
                self.global_step += 1

                batch_end = time.time()
                if batch_idx % 1000 == 0:
                    print(f'Completed {batch_idx} batches in {batch_end-batch_start}s')

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print()
                self.val(split='train')
                #self.val(split='validation')

                self.plot(split='train')
                #self.plot(split='validation')

                
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
        idx = 0

        print(f'Validating on {split} set')

        for batch_idx, batch in enumerate(loader):

            train_nerf, _, matrices_unflattened, matrices_flattened, grid_weights, data_dir, background_indices = batch
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
                
                rgb, acc, _, n_rendering_samples, _, _ = render_image(
                    self.decoder,
                    embeddings,
                    self.occupancy_grid,
                    filtered_rays,
                    self.scene_aabb,
                    render_step_size=self.render_step_size,
                    render_bkgd=color_bkgds,
                    grid_weights=grid_weights,
                    background_indices=None
                )

                # TODO: evaluate whether to add this condition or not
                if 0 in n_rendering_samples:
                    self.logfn({'ERROR': '0 n_rendering_samples. Skip iteration.'})
                    continue
                # alive_ray_mask = acc.squeeze(-1) > 0
                # mse_masked = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                mse = F.mse_loss(rgb, pixels)
            
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs.append(psnr.item())

            # psnr_masked = -10.0 * torch.log(mse_masked) / np.log(10.0)
            #psnrs_masked.append(psnr_masked.item())

            if idx > 99:
                break
            idx+=1
        
        mean_psnr = sum(psnrs) / len(psnrs)
        # mean_psnr_masked = sum(psnrs_masked) / len(psnrs_masked)

        self.logfn({f'{split}/PSNR': mean_psnr})
        # self.logfn({f'{split}/PSNR_UNMASKED': mean_psnr_unmasked})
        
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
        _, test_nerf, matrices_unflattened, matrices_flattened, grid_weights, _, _ = next(loader_iter)
        
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

                
                rgb_A = torch.empty((224,224,3), device='cpu')
                for i in range(224):
                    rgb_A_chunck, alpha, b, c, _, _ = render_image(
                                radiance_field=self.decoder,
                                embeddings=embeddings[idx].unsqueeze(dim=0),
                                occupancy_grid=None,
                                rays=Rays(origins=rays.origins[idx, i, :, :].unsqueeze(dim=0), viewdirs=rays.viewdirs[idx, i, :, :].unsqueeze(dim=0)),
                                scene_aabb=self.scene_aabb,
                                render_step_size=self.render_step_size,
                                render_bkgd=color_bkgds[idx].unsqueeze(dim=0),
                                grid_weights=None
                            )
                    rgb_A[i, :, :] = rgb_A_chunck[0].cpu().detach()
                    del alpha
                    del b
                    del c
                    del rgb_A_chunck
                    torch.cuda.empty_cache()
                
                pred_image_2=wandb.Image((rgb_A.numpy() * 255).astype(np.uint8))
                pred_image = wandb.Image((rgb.to('cpu').detach().numpy()[0] * 255).astype(np.uint8)) 
                gt_image = wandb.Image((pixels.to('cpu').detach().numpy()[idx] * 255).astype(np.uint8))
                

                self.logfn({f"{split}/nerf_{idx}": [gt_image, pred_image, pred_image_2]})
                
                
                img_name = str(uuid.uuid4())
                plots_path = 'plots'
                """
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
    
    def config_wandb(self):
        wandb.init(
            entity='dsr-lab',
            project='nerf2vec',
            name=f'run_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
            config=config.WANDB_CONFIG
        )