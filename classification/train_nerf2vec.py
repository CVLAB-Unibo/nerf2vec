import gzip
import math
import multiprocessing
import pickle
import shutil
import subprocess
import time
import zipfile
import imageio

from nerfacc import ContractionType, OccupancyGrid
import psutil
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
from nerf.utils import Rays, generate_occupancy_grid, render_image
from temp_sanity_check.create_video import create_video

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

        # print(f'focal: {nerf_loader.focal}')

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

        
        """
        grid = generate_occupancy_grid('cuda:0', 
                    nerf_loader.weights_file_path, 
                    config.INSTANT_NGP_MLP_CONF, 
                    config.AABB, 
                    config.OCCUPANCY_GRID_RECONSTRUCTION_ITERATIONS, 
                    config.OCCUPANCY_GRID_WARMUP_ITERATIONS)

        
        dict = {
            'occs': grid.state_dict()['occs'].half(),
            '_roi_aabb': grid.state_dict()['_roi_aabb'].half(),
            '_binary': grid.state_dict()['_binary'].to_sparse(),
            'resolution': grid.state_dict()['resolution'].half()
        }
        
      

        #dict = {key: tensor.to(torch.float16) for key, tensor in grid.state_dict().items()}

        torch.save(dict, 'grid.pth')

        #grid.load_state_dict(dict)

        # Save the model in a compressed format
        #with gzip.open('grid.pth.gz', 'wb') as f:
        #    torch.save(dict, f)
        with open(file_path, 'w') as file:
            json.dump(my_dict, file)
        exit()
        """
        
        # with gzip.open('grid.pth.gz', 'rb') as f:
        #    grid_weights = torch.load(f)
        


        # grid_weights = torch.load('grid.pth', map_location=torch.device(self.device))
        # grid_weights = []
        #grid_weights = torch.load('grid.pth', map_location=torch.device(self.device))
        #grid_weights['_binary'] = grid_weights['_binary'].to_dense()
        
        grid_weights = os.path.join(data_dir, 'grid.pth')
        #grid_weights = torch.load('grid.pth', map_location=torch.device(self.device))
        #grid_weights['_binary'] = grid_weights['_binary'].to_dense()

        return rays, pixels, render_bkgd, matrix, nerf_loader.weights_file_path, test_rays, test_pixels, test_render_bkgd, grid_weights
    
    
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

def unzip_file(file_path, extract_dir):
    with gzip.open(file_path, 'rb') as f_in:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(extract_dir, file_name[:-3])  # Remove the .gz extension
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

class Nerf2vecTrainer:
    def __init__(self, device='cuda:0') -> None:
        train_dset = NeRFDataset(os.path.abspath('data'), None, device='cuda:0') 

        self.device = device
        
        self.train_loader = DataLoader(
            train_dset,
            batch_size=config.BATCH_SIZE,
            num_workers=0,#4,
            shuffle=False#True
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

        grid_resolution = 128
        contraction_type = ContractionType.AABB
        occupancy_grid = OccupancyGrid(
            roi_aabb=config.AABB,
            resolution=grid_resolution,
            contraction_type=contraction_type,
        ).to(self.device)
        occupancy_grid.eval()

        #weights = torch.load('grid.pth')
        #weights['_binary'] = weights['_binary'].to_dense()
        # occupancy_grid.load_state_dict(weights)


        for epoch in range(start_epoch, num_epochs):
            
            """
            shutil.rmtree('zipped_grids')
            path = Path('zipped_grids')
            path.mkdir(parents=True, exist_ok=True)
            for i in range(5000):
                shutil.copy2('grid.pth.gz', os.path.join('zipped_grids', f'grid.pth_{i}.gz'))
            exit()
            """


            
            # OPTIMIZED CODE WHEN BATCH_SIZE 16
            N_CACHED_GRIDS = 512
            if self.global_step % (N_CACHED_GRIDS/config.BATCH_SIZE)  == 0:
                print('Prepare the grids!')
                start_unzip = time.time()
                shutil.rmtree('grids')
                path = Path('grids')
                path.mkdir(parents=True, exist_ok=True)
                
                folder_path = 'zipped_grids'  # Replace with the path to the folder containing the zip files
                extract_dir = 'grids'  # Replace with the desired extraction directory

                # Get a list of GZ files in the folder
                gz_files = [file for file in os.listdir(folder_path) if file.endswith('.gz')][N_CACHED_GRIDS]

                # Create a pool of worker processes
                pool = multiprocessing.Pool()

                # Unzip files in parallel
                for gz_file in gz_files:
                    file_path = os.path.join(folder_path, gz_file)
                    pool.apply_async(unzip_file, args=(file_path, extract_dir))

                # Close the pool and wait for the processes to complete
                pool.close()
                pool.join()



                end_unzip=time.time()
                print(end_unzip-start_unzip)
            
            

            self.epoch = epoch
            self.encoder.train()
            self.decoder.train()

            desc = f"Epoch {epoch}/{num_epochs}"

            for batch in self.train_loader:
                
                batch_start = time.time()
                # rays, pixels, render_bkgds, matrices, nerf_weights_path = batch
                rays, pixels, render_bkgds, matrices, nerf_weights_path, test_rays, test_pixels, test_render_bkgds, grid_weights = batch
                
                # TODO: check rays, it is not created properly
                #rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())
                #pixels = pixels.cuda()
                #render_bkgds = render_bkgds.cuda()
                #matrices = matrices.cuda()

                #test_rays = test_rays._replace(origins=test_rays.origins.cuda(), viewdirs=test_rays.viewdirs.cuda())
                #test_pixels = test_pixels.cuda()
                #test_render_bkgds = test_render_bkgds.cuda()
                
                
                """
                grid_resolution = 128
                contraction_type = ContractionType.AABB
                grids = []
                grid_start = time.time()

                for idx in range(len(test_render_bkgds)):
                    occupancy_grid = OccupancyGrid(
                        roi_aabb=config.AABB,
                        resolution=grid_resolution,
                        contraction_type=contraction_type,
                    ).to(self.device)
                    occupancy_grid.eval()
                    
                    #c_dict = {}
                    #for key in grid_weights:
                    #    c_dict[key] = grid_weights[key][idx]
                    #occupancy_grid.load_state_dict(c_dict)
                        # occupancy_grid.load_state_dict(torch.load(elem))
                    
                    #with gzip.open('grid.pth.gz', 'rb') as f:
                    #    loaded_dict = torch.load(f)
                    #    occupancy_grid.load_state_dict(loaded_dict)
                    # occupancy_grid.load_state_dict(torch.load('grid.pth'))

                    # occupancy_grid.eval()
                    grids.append(occupancy_grid)
                """
                
                
                #grid_end = time.time()
                #print(f'[{self.global_step}] grid creation elapsed: {grid_end-grid_start}')
                
                
                
                """
                grids = []
                #grid_start = time.time()
                for elem in nerf_weights_path:
                    grid = generate_occupancy_grid(self.device, 
                                                elem, 
                                                config.INSTANT_NGP_MLP_CONF, 
                                                config.AABB, 
                                                config.OCCUPANCY_GRID_RECONSTRUCTION_ITERATIONS, 
                                                config.OCCUPANCY_GRID_WARMUP_ITERATIONS)
                    grids.append(grid)
                #grid_end = time.time()
                # print(f'[{self.global_step}] grid creation elapsed: {grid_end-grid_start}')
                
                # grids = [None] * config.BATCH_SIZE
                """
                
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
                    occupancy_grid,
                    rays,
                    scene_aabb,
                    # rendering options
                    near_plane=None,
                    far_plane=None,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgds,
                    cone_angle=0.0,
                    alpha_thre=alpha_thre,
                    grid_weights=grid_weights
                )
                # end_time = time.time()
                # print(f'elapsed: {end_time-start_time}')

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
                INTERVAL = 100
                if step % INTERVAL == 0:

                    # elapsed_time = time.time() - tic
                    loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                    print(f'loss={loss:.5f}')
                """
                
                # self.logfn(f'"train/loss": {loss.item()} - elapsed: {end-start}')
                batch_end = time.time()
                #print(f'{self.global_step} - Single batch: {batch_end-batch_start}')
                batch_start = time.time()

                if self.global_step % 100 == 0:
                    end = time.time()
                    print(f'{self.global_step} - "train/loss": {loss.item()} - elapsed: {end-start}')

                    self.encoder.eval()
                    self.decoder.eval()
                    with torch.no_grad():
                        # for i in tqdm.tqdm(range(len(self.TEMP_nerf_loader))):
                        #data = self.TEMP_nerf_loader[i]
                        #render_bkgd = data["color_bkgd"]
                        #rays = data["rays"]
                        #pixels = data["pixels"]
                        
                        for i in range(config.BATCH_SIZE):
                            idx_to_draw = i
                            rgb, acc, depth, n_rendering_samples = render_image(
                                self.decoder,
                                embeddings[idx_to_draw].unsqueeze(dim=0),
                                occupancy_grid,#[grids[idx_to_draw]],
                                Rays(origins=test_rays.origins[idx_to_draw].unsqueeze(dim=0), viewdirs=test_rays.viewdirs[idx_to_draw].unsqueeze(dim=0)),
                                scene_aabb,
                                # rendering options
                                near_plane=None,
                                far_plane=None,
                                render_step_size=render_step_size,
                                render_bkgd=test_render_bkgds,
                                cone_angle=0.0,
                                alpha_thre=alpha_thre,
                                grid_weights=[grid_weights[i]]
                            )

                            imageio.imwrite(
                                os.path.join('temp_sanity_check', f'{i}_rgb_test_{self.global_step}.png'),
                                (rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8),
                            )
                        
                        
                        """
                        if self.global_step == 2900:
                            create_video(
                                    448, 
                                    448, 
                                    self.device, 
                                    245.0, 
                                    self.decoder, 
                                    grids, 
                                    scene_aabb,
                                    None, 
                                    None, 
                                    render_step_size,
                                    render_bkgd=test_render_bkgds,
                                    cone_angle=0.0,
                                    alpha_thre=alpha_thre,
                                    # test options
                                    path=os.path.join('temp_sanity_check', f'video_{self.global_step}.mp4'),
                                    embeddings=embeddings
                                )
                        """
                        
                        
                    start = time.time()
                    self.encoder.train()
                    self.decoder.train()
                    
                    
                self.global_step += 1



