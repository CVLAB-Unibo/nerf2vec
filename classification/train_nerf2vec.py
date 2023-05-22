import gzip
import json
import math
import multiprocessing
import random
import shutil
import time
import imageio

from nerfacc import OccupancyGrid
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
from typing import Any, List

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


def get_nerf_name_from_grid(file_path):
    grid_name = os.path.basename(file_path)
    nerf_name = os.path.splitext(grid_name)[0]
    return nerf_name


def unzip_file(file_path, extract_dir, file_name):
    with gzip.open(os.path.join(file_path, 'grid.pth.gz'), 'rb') as f_in:
        output_path = os.path.join(extract_dir, file_name) 
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


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

        dataset_kwargs = {}
        data_dir = self.nerf_paths[index]

        nerf_loader = NeRFLoader(
            data_dir=data_dir,
            num_rays=config.NUM_RAYS,
            device=self.device,
            **dataset_kwargs)

        # print(f'focal: {nerf_loader.focal}')

        nerf_loader.training = True
        data = nerf_loader[0]  # The index has not any relevance when training is True
        color_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]
        train_nerf = {
            'rays': rays,
            'pixels': pixels,
            'color_bkgd': color_bkgd
        }

        nerf_loader.training = False
        test_data = nerf_loader[0]  # TODO: getting just the first image in the dataset. Evaluate whether to return more.
        test_color_bkgd = test_data["color_bkgd"]
        test_rays = test_data["rays"]
        test_pixels = test_data["pixels"]
        test_nerf = {
            'rays': test_rays,
            'pixels': test_pixels,
            'color_bkgd': test_color_bkgd
        }

        matrix = torch.load(nerf_loader.weights_file_path, map_location=torch.device(self.device))
        matrix = get_mlp_params_as_matrix(matrix['mlp_base.params'])
        
        grid_weights_path = os.path.join('grids', get_grid_file_name(data_dir))

        return train_nerf, test_nerf, matrix, grid_weights_path
    

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

        train_dset_json = os.path.abspath(os.path.join('data', 'train.json'))
        self.train_dset = NeRFDataset(train_dset_json, device='cpu') 

        val_dset_json = os.path.abspath(os.path.join('data', 'train.json'))  # TODO: TO UPDATE WITH THE VALIDATION SET!
        self.val_dset = NeRFDataset(val_dset_json, device='cpu')   

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
        self.plots_path = Path(os.path.join('classification', 'plots'))

        if self.ckpts_path.exists():
            self.restore_from_last_ckpt()

        self.ckpts_path.mkdir(parents=True, exist_ok=True)
        self.all_ckpts_path.mkdir(parents=True, exist_ok=True)
        self.plots_path.mkdir(parents=True, exist_ok=True)
        

    def logfn(self, values: str) -> None:
        #wandb.log(values, step=self.global_step, commit=False)
        print(values)
        with open('train_logs.txt', 'a') as f:
            f.write(f'{values}\n')
    
    def unzip_occupancy_grids(self, batch_idx: int, nerf_indices: List[int], nerf_paths: List[str], unzip_folder='grids'):
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
            current_split_indices = nerf_indices[start_idx:end_idx]

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
        
        # Create a list of all the possible indices
        num_samples = len(dataset)
        indices = list(range(num_samples))
        if shuffle:
            random.shuffle(indices)

        # Create the cyclic index iterator
        index_iterator = CyclicIndexIterator(indices=indices)

        # Create the DataLoader with the cyclic index iterator
        loader = DataLoader(
            self.train_dset,
            batch_size=config.BATCH_SIZE,
            sampler=index_iterator,
            shuffle=False,
            num_workers=8, 
            persistent_workers=True
        )

        return loader, indices, dataset.nerf_paths

    def train(self):
        num_epochs = config.NUM_EPOCHS
        start_epoch = self.epoch

        for epoch in range(start_epoch, num_epochs):

            train_loader, train_indices, train_paths = self.create_data_loader(self.train_dset, shuffle=True)

            self.epoch = epoch
            self.encoder.train()
            self.decoder.train()
            
            self.logfn(f'Epoch {epoch} started...')         
            epoch_start = time.time()

            for batch_idx, batch in enumerate(train_loader):
                
                # rays, pixels, render_bkgds, matrices, nerf_weights_path = batch
                train_nerf, _, matrices, grid_weights_path = batch
                rays = train_nerf['rays']
                pixels = train_nerf['pixels']
                color_bkgds = train_nerf['color_bkgd']

                self.unzip_occupancy_grids(batch_idx=batch_idx, 
                                           nerf_indices=train_indices, 
                                           nerf_paths=train_paths)
                
                rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
                pixels = pixels.cuda()
                color_bkgds = color_bkgds.cuda()
                matrices = matrices.cuda()
                
                embeddings = self.encoder(matrices)
                
                rgb, acc, _, n_rendering_samples = render_image(
                    self.decoder,
                    embeddings,
                    self.occupancy_grid,
                    rays,
                    self.scene_aabb,
                    # rendering options
                    near_plane=None,
                    far_plane=None,
                    render_step_size=self.render_step_size,
                    render_bkgd=color_bkgds,
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
                
                self.global_step += 1

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                
                # Create the validation data loaders
                val_loader_shuffled, val_indices_shuffled, val_paths_shuffled = self.create_data_loader(self.val_dset, shuffle=True)
                val_loader, val_indices, val_paths = self.create_data_loader(self.val_dset, shuffle=False)

                self.val(loader=train_loader, nerf_indices=train_indices, nerf_paths=train_paths, split='train')

                self.val(loader=val_loader, nerf_indices=val_indices, nerf_paths=val_paths, split='validation')

                self.plot(loader=train_loader, nerf_indices=train_indices, nerf_paths=train_paths, split='train')

                self.plot(loader=val_loader_shuffled, nerf_indices=val_indices_shuffled, nerf_paths=val_paths_shuffled, split='validation')
                
            if epoch % 50 == 0:
                self.save_ckpt(all=True)
            
            self.save_ckpt()
            
            epoch_end = time.time()
            print(f'Epoch {epoch} completed in {epoch_end-epoch_start}s')        
    
    @torch.no_grad()
    def val(self, loader: DataLoader, nerf_indices: List[int], nerf_paths: List[str], split: str) -> None:
        
        self.encoder.eval()
        self.decoder.eval()

        psnrs = []
        idx = 0

        self.logfn(f'Validating on {split} set')

        for batch_idx, batch in enumerate(loader):

            train_nerf, _, matrices, grid_weights_path = batch
            rays = train_nerf['rays']
            pixels = train_nerf['pixels']
            color_bkgds = train_nerf['color_bkgd']

            self.unzip_occupancy_grids(batch_idx=batch_idx, nerf_indices=nerf_indices, nerf_paths=nerf_paths)
            
            rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
            pixels = pixels.cuda()
            color_bkgds = color_bkgds.cuda()
            matrices = matrices.cuda()
            
            embeddings = self.encoder(matrices)
            
            rgb, _, _, n_rendering_samples = render_image(
                self.decoder,
                embeddings,
                self.occupancy_grid,
                rays,
                self.scene_aabb,
                render_step_size=self.render_step_size,
                render_bkgd=color_bkgds,
                grid_weights=grid_weights_path
            )

            # TODO: evaluate whether to add this condition or not
            if 0 in n_rendering_samples:
                self.logfn(f'ERROR: 0 n_rendering_samples. Skip iteration.')
                continue

            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs.append(psnr.item())

            if idx > 99:
                break
            idx+=1


        mean_psnr = sum(psnrs) / len(psnrs)

        self.logfn({f"{split}/PSNR": mean_psnr})
        
        if split == "val" and mean_psnr > self.best_psnr:
            self.best_psnr = mean_psnr
            self.save_ckpt(best=True)
    
    @torch.no_grad()
    def plot(self, loader: DataLoader, nerf_indices: List[int], nerf_paths: List[str], split: str) -> None:
        self.encoder.eval()
        self.decoder.eval()

        for batch_idx, batch in enumerate(loader):

            _, test_nerf, matrices, grid_weights_path = batch
            rays = test_nerf['rays']
            color_bkgds = test_nerf['color_bkgd']
                
            self.unzip_occupancy_grids(batch_idx=batch_idx, nerf_indices=nerf_indices, nerf_paths=nerf_paths)
            
            rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
            color_bkgds = color_bkgds.cuda()
            matrices = matrices.cuda()
            
            embeddings = self.encoder(matrices)

            for idx in range(len(matrices)):

                rgb, _, _, _ = render_image(
                    self.decoder,
                    embeddings[idx].unsqueeze(dim=0),
                    self.occupancy_grid,
                    Rays(origins=rays.origins[idx].unsqueeze(dim=0), viewdirs=rays.viewdirs[idx].unsqueeze(dim=0)),
                    self.scene_aabb,
                    render_step_size=self.render_step_size,
                    render_bkgd=color_bkgds[idx].unsqueeze(dim=0),
                    grid_weights=[grid_weights_path[idx]]
                )

                img_name = get_nerf_name_from_grid(grid_weights_path[idx])
                
                imageio.imwrite(
                    os.path.join(self.plots_path, f'{split}_{img_name}_{self.global_step}.png'),
                    (rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8)
                )

                break
            break

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
            ckpt = torch.load(ckpt_path)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = self.epoch * math.ceil(len(self.train_dset)/config.BATCH_SIZE)  # len(self.train_loader)
            self.best_psnr = ckpt["best_psnr"]

            self.encoder.load_state_dict(ckpt["encoder"])
            self.decoder.load_state_dict(ckpt["decoder"])
            self.optimizer.load_state_dict(ckpt["optimizer"])