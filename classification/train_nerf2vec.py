import gzip
import math
import multiprocessing
import random
import shutil
import time
import imageio

from nerfacc import ContractionType, OccupancyGrid
import tqdm
from classification.utils import get_mlp_params_as_matrix

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
from typing import Any, Dict, List

from classification import config
from nerf.utils import Rays, render_image


def get_grid_file_name(file_path):
    # Split the path into individual directories
    directories = os.path.normpath(file_path).split(os.sep)
    # Get the last two directories
    last_two_dirs = directories[-2:]
    # Join the last two directories with an underscore
    file_name = '_'.join(last_two_dirs) + '.pth'
    return file_name


def unzip_file(file_path, extract_dir, file_name):
    with gzip.open(os.path.join(file_path, 'grid.pth.gz'), 'rb') as f_in:
        output_path = os.path.join(extract_dir, file_name) 
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


class NeRFDataset(Dataset):
    def __init__(self, nerfs_root: str, device: str) -> None:
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

        # print(f'focal: {nerf_loader.focal}')

        # Get data for the batch
        data = nerf_loader[0]
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        matrix = torch.load(nerf_loader.weights_file_path, map_location=torch.device(self.device))
        matrix = get_mlp_params_as_matrix(matrix['mlp_base.params'])
        
        #nerf_loader.training = False
        # Get data for the batch
        #test_data = nerf_loader[0]
        #test_render_bkgd = test_data["color_bkgd"]
        #test_rays = test_data["rays"]
        #test_pixels = test_data["pixels"]
        
        grid_weights_path = os.path.join('grids', get_grid_file_name(data_dir))

        return rays, pixels, render_bkgd, matrix, grid_weights_path
    

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


class CyclicIndexIterator:
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
    
class Nerf2vecTrainer:
    def __init__(self, device='cuda:0') -> None:

        self.device = device

        self.train_dset = NeRFDataset(os.path.abspath('data'), device='cpu') 
        self.val_dset = NeRFDataset(os.path.abspath('data'), device='cpu')   # TODO: TO UPDATE WITH THE VALIDATION SET!

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
        self.best_psnr = float("inf")

        self.ckpts_path = Path(os.path.join('classification', 'train', 'ckpts'))
        self.all_ckpts_path = Path(os.path.join('classification', 'train', 'all_ckpts'))

        if self.ckpts_path.exists():
            self.restore_from_last_ckpt()

        self.ckpts_path.mkdir(exist_ok=True)
        self.all_ckpts_path.mkdir(exist_ok=True)
        

    def logfn(self, values: Dict[str, Any]) -> None:
        #wandb.log(values, step=self.global_step, commit=False)
        print(values)
    
    def unzip_occupancy_grids(self, batch_idx: int, all_indices: List[int], nerf_paths: List[str], unzip_folder='grids'):
        N_CACHED_GRIDS = 512
        if batch_idx == 0 or batch_idx % (N_CACHED_GRIDS/config.BATCH_SIZE) == 0: 
            self.logfn('Grid unzip started...')
            start_unzip = time.time()
            
            shutil.rmtree(unzip_folder)
            path = Path(unzip_folder)
            path.mkdir(parents=True, exist_ok=True)
            
            # Retrieve the list of gz files to unzip
            start_idx = int(batch_idx / (N_CACHED_GRIDS/config.BATCH_SIZE)) * N_CACHED_GRIDS
            end_idx = start_idx + N_CACHED_GRIDS
            current_split_indices = all_indices[start_idx:end_idx]

            gz_files = [nerf_paths[i] for i in current_split_indices]

            # Unzip files in parallel
            pool = multiprocessing.Pool()

            for file_path in gz_files:
                file_name = get_grid_file_name(file_path)
                pool.apply_async(unzip_file, args=(file_path, unzip_folder, file_name))

            # Close the pool and wait for the processes to complete
            pool.close()
            pool.join()

            end_unzip=time.time()
            self.logfn(f'Grids unzip completed in {end_unzip-start_unzip}s')
    
    def create_data_loader(self, dataset, shuffle=True) -> DataLoader:
        num_samples = len(dataset)

        # Create a list of all possible indices
        all_indices = list(range(num_samples))

        if shuffle:
            random.shuffle(all_indices)

        # Create the cyclic index iterator
        index_iterator = CyclicIndexIterator(all_indices)

        # Create the DataLoader with the cyclic index iterator
        loader = DataLoader(
            self.train_dset,
            batch_size=config.BATCH_SIZE,
            sampler=index_iterator,
            shuffle=False,
            num_workers=8, # Important for performances! (if batch size = 16, set to 8)
            persistent_workers=True
        )

        return loader, all_indices

    def train(self):
        num_epochs = config.NUM_EPOCHS
        start_epoch = self.epoch

        for epoch in range(start_epoch, num_epochs):

            train_loader, train_indices = self.create_data_loader(self.train_dset, shuffle=True)

            self.epoch = epoch
            self.encoder.train()
            self.decoder.train()
            
            self.logfn(f'Epoch {epoch} started...')         
            epoch_start = time.time()
            
            # TODO: remove this variable. For the moment used for debugging. 
            elem_per_batch = 0

            for batch_idx, batch in enumerate(train_loader):
                
                # rays, pixels, render_bkgds, matrices, nerf_weights_path = batch
                rays, pixels, render_bkgds, matrices, grid_weights_path = batch

                elem_per_batch += len(grid_weights_path)
                
                self.unzip_occupancy_grids(batch_idx=batch_idx, 
                                           all_indices=train_indices, 
                                           nerf_paths=self.train_dset.nerf_paths)
                
                rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
                pixels = pixels.cuda()
                render_bkgds = render_bkgds.cuda()
                matrices = matrices.cuda()
                
                embeddings = self.encoder(matrices)
                
                rgb, acc, depth, n_rendering_samples = render_image(
                    self.decoder,
                    embeddings,
                    self.occupancy_grid,
                    rays,
                    self.scene_aabb,
                    # rendering options
                    near_plane=None,
                    far_plane=None,
                    render_step_size=self.render_step_size,
                    render_bkgd=render_bkgds,
                    cone_angle=0.0,
                    alpha_thre=0.0,
                    grid_weights=grid_weights_path
                )

                # TODO: evaluate whether to add this condition or not
                if 0 in n_rendering_samples:
                    print(f'0 n_rendering_samples. Skip iteration.')
                    continue

                alive_ray_mask = acc.squeeze(-1) > 0

                # compute loss
                loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                self.optimizer.zero_grad()
                # do not unscale it because we are using Adam.
                self.grad_scaler.scale(loss).backward()
                self.optimizer.step()
                
                """
                if self.global_step % 300 == 0:
                    end = time.time()
                    print(f'{self.global_step} - "train/loss": {loss.item()} - elapsed: {end-start}')

                    self.encoder.eval()
                    self.decoder.eval()
                    with torch.no_grad():
                        
                        for i in range(config.BATCH_SIZE):
                            idx_to_draw = i
                            rgb, acc, depth, n_rendering_samples = render_image(
                                self.decoder,
                                embeddings[idx_to_draw].unsqueeze(dim=0),
                                self.occupancy_grid,#[grids[idx_to_draw]],
                                Rays(origins=test_rays.origins[idx_to_draw].unsqueeze(dim=0), viewdirs=test_rays.viewdirs[idx_to_draw].unsqueeze(dim=0)),
                                self.scene_aabb,
                                # rendering options
                                near_plane=None,
                                far_plane=None,
                                render_step_size=self.render_step_size,
                                render_bkgd=test_render_bkgds,
                                cone_angle=0.0,
                                alpha_thre=0.0,
                                grid_weights=[grid_weights_path[i]]
                            )

                            imageio.imwrite(
                                os.path.join('temp_sanity_check', 'images', f'{i}_rgb_test_{self.global_step}.png'),
                                (rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8),
                            )
                        
                        # ####################
                        # EVAL
                        # ####################
                        psnrs = []
                        psnrs_avg = []
                        idx_to_draw = random.randrange(0, config.BATCH_SIZE)
                        test_dataset_kwargs = {}
                        test_nerf_loader = NeRFLoader(
                            data_dir=nerf_weights_path[idx_to_draw],
                            num_rays=config.NUM_RAYS,
                            device=self.device,
                            **test_dataset_kwargs)
                        test_nerf_loader.training = False
                        
                        
                        for i in tqdm.tqdm(range(len(test_nerf_loader))):
                            data = test_nerf_loader[i]
                            render_bkgd = data["color_bkgd"]
                            test_rays_2 = data["rays"]
                            
                            pixels = data["pixels"].unsqueeze(dim=0)
                            
                            
                            rgb, acc, depth, n_rendering_samples = render_image(
                                self.decoder,
                                embeddings[idx_to_draw].unsqueeze(dim=0),
                                self.occupancy_grid,
                                Rays(origins=test_rays_2.origins.unsqueeze(dim=0), viewdirs=test_rays_2.viewdirs.unsqueeze(dim=0)),
                                self.scene_aabb,
                                # rendering options
                                near_plane=None,
                                far_plane=None,
                                render_step_size=self.render_step_size,
                                render_bkgd=test_render_bkgds,
                                cone_angle=0.0,
                                alpha_thre=0.0,
                                grid_weights=[grid_weights_path[idx_to_draw]]
                            )

                            if i == 0:
                                imageio.imwrite(
                                    os.path.join('temp_sanity_check', 'images', f'{i}_rgb_test_{self.global_step}.png'),
                                    (rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8),
                                )

                            mse = F.mse_loss(rgb, pixels)
                            psnr = -10.0 * torch.log(mse) / np.log(10.0)
                            psnrs.append(psnr.item())
                        
                        psnr_avg = sum(psnrs) / len(psnrs)
                        print(f'PSNR: {psnr_avg}')
                        psnrs_avg.append(psnr_avg)

                        
                        
                        '''
                        if self.global_step == 9900:
                            create_video(
                                    448, 
                                    448, 
                                    self.device, 
                                    245.0, 
                                    self.decoder, 
                                    occupancy_grid, 
                                    scene_aabb,
                                    None, 
                                    None, 
                                    render_step_size,
                                    render_bkgd=test_render_bkgds[0],
                                    cone_angle=0.0,
                                    alpha_thre=alpha_thre,
                                    # test options
                                    path=os.path.join('temp_sanity_check', f'video_{self.global_step}.mp4'),
                                    embeddings=embeddings[idx_to_draw].unsqueeze(dim=0),
                                    grid_weights=[grid_weights_path[idx_to_draw]]
                                )
                        '''
                        
                        
                    print(psnrs_avg)
                    
                    self.encoder.train()
                    self.decoder.train()
                """
                    
                self.global_step += 1
                # print(self.global_step)

            
            if epoch % 10 == 0 or epoch == num_epochs - 1:

                # TODO: The non-shuffled version could be created only once
                # validation_loader, validation_indices = self.create_data_loader(self.val_dset, shuffle=False)
                # validation_loader_shuffled, validation_indices_shuffled = self.create_data_loader(self.val_dset, shuffle=False)

                self.val(loader=train_loader, 
                         all_indices=train_indices, 
                         nerf_paths=self.train_dset.nerf_paths, 
                         split='train')
                
                # self.val("val")
                # self.plot("train")
                # self.plot("val")
            
            if epoch % 50 == 0:
                self.save_ckpt(all=True)
            
            self.save_ckpt()
            
            
            epoch_end = time.time()
            print(f'Epoch {epoch} completed in {epoch_end-epoch_start}s. Processed {elem_per_batch} elements')        
    
    @torch.no_grad()
    def val(self, loader: DataLoader, all_indices: List[int], nerf_paths: List[str], split: str) -> None:
        
        self.encoder.eval()
        self.decoder.eval()

        psnrs = []
    
        for batch_idx, batch in enumerate(loader):
            rays, pixels, render_bkgds, matrices, grid_weights_path = batch
                
            self.unzip_occupancy_grids(batch_idx=batch_idx, all_indices=all_indices, nerf_paths=nerf_paths)
            
            rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
            pixels = pixels.cuda()
            render_bkgds = render_bkgds.cuda()
            matrices = matrices.cuda()
            
            embeddings = self.encoder(matrices)
            
            rgb, acc, depth, n_rendering_samples = render_image(
                self.decoder,
                embeddings,
                self.occupancy_grid,
                rays,
                self.scene_aabb,
                # rendering options
                near_plane=None,
                far_plane=None,
                render_step_size=self.render_step_size,
                render_bkgd=render_bkgds,
                cone_angle=0.0,
                alpha_thre=0.0,
                grid_weights=grid_weights_path
            )

            # TODO: evaluate whether to add this condition or not
            if 0 in n_rendering_samples:
                self.logfn(f'ERROR: 0 n_rendering_samples. Skip iteration.')
                continue

            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs.append(psnr.item())


        mean_psnr = sum(psnrs) / len(psnrs)

        self.logfn({f"{split}/PSNR": mean_psnr})
        
        if split == "val" and mean_psnr > self.best_psnr:
            self.best_psnr = mean_psnr
            self.save_ckpt(best=True)

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
        return
        # TODO: FIX...
        if self.ckpts_path.exists():
            ckpt_paths = [p for p in self.ckpts_path.glob("*.pt") if "best" not in p.name]
            error_msg = "Expected only one ckpt apart from best, found none or too many."
            assert len(ckpt_paths) == 1, error_msg

            ckpt_path = ckpt_paths[0]
            ckpt = torch.load(ckpt_path)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = self.epoch * len(self.train_loader)
            self.best_psnr = ckpt["best_psnr"]

            self.encoder.load_state_dict(ckpt["encoder"])
            self.decoder.load_state_dict(ckpt["decoder"])
            self.optimizer.load_state_dict(ckpt["optimizer"])