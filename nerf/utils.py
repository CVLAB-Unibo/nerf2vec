import math
import random
import time
from typing import Optional

import numpy as np
import torch
# from utils import Rays, namedtuple_map
import collections

from nerfacc import ContractionType, OccupancyGrid, ray_marching, render_visibility, rendering

from nerf.intant_ngp import NGPradianceField

import nerfacc.cuda as _C

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))

def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_image(
    # scene
    radiance_field: torch.nn.Module,             # This should be the DECODER
    embeddings: torch.Tensor,
    occupancy_grid: OccupancyGrid,               # This should be [batch_size, grids]
    rays: Rays,                                  # This should be [batch_size, rays]
    scene_aabb: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,  # This should be [batch_size, render_bkgds]
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0
):
    """Render the pixels of an image."""

    rays_shape = rays.origins.shape
    if len(rays_shape) == 4:
        batch_size, height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([batch_size] + [num_rays] + list(r.shape[3:])), rays
        )
    else:
        batch_size, num_rays, _ = rays_shape
 
    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        _ = t_starts
        _ = t_ends
        _ = ray_indices

        rgb, sigmas = curr_rgb[curr_batch_idx], curr_sigmas[curr_batch_idx]
        return rgb, sigmas

    results = []
    
    # chunk = torch.iinfo(torch.int32).max
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else 4096
    )
    
    # start_time = time.time()
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[:, i : i + chunk], rays)

        b_positions = []
        b_t_starts = []
        b_t_ends = []
        b_ray_indices = []

        # Compute the positions for all the elements in the batch.
        # These position will be used for calling the decoder.
        for batch_idx in range(batch_size):

            """
            The ray_marching internally calls sigma_fn that, for the moment, has not be used.
            This because:
             - it is an optimization that, hopefully, can be skipped. By testing some models, it seems that they can be trained also without it.
             - it requires the value 'packed_info', which is returned from the ray marching algorithm, and it is not exposed to external callers.
            See the ray_marching.py file, at the end it uses this variable.

            Moreover, this avoids an additional call to the model (i.e., the nerf2vec decoder)
            """
            ray_indices, t_starts, t_ends = ray_marching(
                chunk_rays.origins[batch_idx],  # [batch_size, chunk, 3d coord]
                chunk_rays.viewdirs[batch_idx], # [batch_size, chunk, 3d coord]
                scene_aabb=scene_aabb, 
                grid=occupancy_grid[batch_idx], # [batch_size, occupancy_grids]
                sigma_fn=None,#sigma_fn,
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
                stratified=radiance_field.training,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
            )

            b_t_starts.append(t_starts)
            b_t_ends.append(t_ends)
            b_ray_indices.append(ray_indices)
            
            t_origins = chunk_rays.origins[batch_idx][ray_indices]
            t_dirs = chunk_rays.viewdirs[batch_idx][ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
            b_positions.append(positions)

        # Get the minimum size among all tensors, so as to have a tensor that can be passed to the decoder 
        # (i.e., all tensors will have the same dimensions)
        MIN_SIZE = min([tensor.size(0) for tensor in b_positions])
        # print(MIN_SIZE)
        
        if radiance_field.training: # Avoid OOM in case of too many rays
            if MIN_SIZE > 100000:
                MIN_SIZE = 100000
            #if MIN_SIZE > 4096:
            #    MIN_SIZE = 4096

       


        # #####
        # RAMDOM SELECTION
        # #####
        """
        for b in range(batch_size):
            n_elements = b_positions[b].shape[0]
            indices = torch.randperm(n_elements)[:MIN_SIZE]
            b_positions[b] = b_positions[b][indices]
            b_t_starts[b] = b_t_starts[b][indices]
            b_t_ends[b] = b_t_ends[b][indices]
            b_ray_indices[b] = b_ray_indices[b][indices]

            
        b_positions = torch.stack(b_positions, dim=0)
        b_t_starts = torch.stack(b_t_starts, dim=0)
        b_t_ends = torch.stack(b_t_ends, dim=0)
        b_ray_indices = torch.stack(b_ray_indices, dim=0)
        """



        # ################################################################################
        # RENDER VISIBILITY
        # ################################################################################
        b_positions = torch.stack([tensor[:MIN_SIZE] for tensor in b_positions], dim=0)
        #b_t_starts = torch.stack([tensor[:MIN_SIZE] for tensor in b_t_starts], dim=0)
        #b_t_ends = torch.stack([tensor[:MIN_SIZE] for tensor in b_t_ends], dim=0)
        #b_ray_indices = torch.stack([tensor[:MIN_SIZE] for tensor in b_ray_indices], dim=0)
        
        curr_rgb, curr_sigmas = radiance_field(embeddings, b_positions)
        # Compute visibility of the samples, and filter out invisible samples
        for batch_idx in range(batch_size): 
            sigmas = curr_sigmas[batch_idx]

            alphas = 1.0 - torch.exp(-sigmas * (b_t_ends[batch_idx] - b_t_starts[batch_idx]))
            masks = render_visibility(
                alphas,
                ray_indices=b_ray_indices[batch_idx],
                packed_info=None,
                early_stop_eps=1e-4,
                alpha_thre=alpha_thre,
                n_rays=chunk_rays.origins.shape[1]
            )

            b_ray_indices[batch_idx] = b_ray_indices[batch_idx][masks]
            b_t_starts[batch_idx] = b_t_starts[batch_idx][masks]
            b_t_ends[batch_idx] = b_t_ends[batch_idx][masks]

        
        b_positions = []
        for batch_idx in range(batch_size):
            
            batch_idx_indices = b_ray_indices[batch_idx]

            t_origins = chunk_rays.origins[batch_idx][batch_idx_indices]
            t_dirs = chunk_rays.viewdirs[batch_idx][batch_idx_indices]
            positions = t_origins + t_dirs * (b_t_starts[batch_idx] + b_t_ends[batch_idx]) / 2.0
            b_positions.append(positions)
        
        b_positions = torch.stack([tensor[:MIN_SIZE] for tensor in b_positions], dim=0)
        b_t_starts = torch.stack([tensor[:MIN_SIZE] for tensor in b_t_starts], dim=0)
        b_t_ends = torch.stack([tensor[:MIN_SIZE] for tensor in b_t_ends], dim=0)
        b_ray_indices = torch.stack([tensor[:MIN_SIZE] for tensor in b_ray_indices], dim=0)


        # ################################################################################
        # VOLUME RENDERING
        # ################################################################################
        curr_rgb, curr_sigmas = radiance_field(embeddings, b_positions)
        
        
        for curr_batch_idx in range(batch_size):
            rgb, opacity, depth = rendering(
                b_t_starts[curr_batch_idx],
                b_t_ends[curr_batch_idx],
                b_ray_indices[curr_batch_idx],
                n_rays=chunk_rays.origins.shape[1], #num_rays,
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=render_bkgd[curr_batch_idx],
            )
            chunk_results = [rgb, opacity, depth, len(b_t_starts[curr_batch_idx])]
            
            # results.append(chunk_results)
            if curr_batch_idx < len(results):
                # results[curr_batch_idx][0].append(chunk_results)
                #results[curr_batch_idx][0][0] = torch.cat([results[curr_batch_idx][0][0], chunk_results[0]], dim=0)
                #results[curr_batch_idx][0][1] = torch.cat([results[curr_batch_idx][0][1], chunk_results[1]], dim=0)
                #results[curr_batch_idx][0][2] = torch.cat([results[curr_batch_idx][0][2], chunk_results[2]], dim=0)
                #results[curr_batch_idx][0][3] = results[curr_batch_idx][0][3] + chunk_results[3]


                results[curr_batch_idx].append(chunk_results)
                #results[curr_batch_idx][0][1].append(chunk_results[1])
                #results[curr_batch_idx][0][2].append(chunk_results[2])
                #results[curr_batch_idx][0][3].append(chunk_results[3])
            else:
                results.append([chunk_results])
            
    
    colors, opacities, depths, n_rendering_samples = zip(*[
        (
            torch.cat([r[0] for r in batch], dim=0),
            torch.cat([r[1] for r in batch], dim=0),
            torch.cat([r[2] for r in batch], dim=0),
            [r[3] for r in batch]
        ) for batch in results
    ])
    
    
    colors = torch.stack(colors, dim=0).view((*rays_shape[:-1], -1))
    opacities = torch.stack(opacities, dim=0).view((*rays_shape[:-1], -1))
    depths = torch.stack(depths, dim=0).view((*rays_shape[:-1], -1))
    n_rendering_samples = [elem[0] for elem in n_rendering_samples]
    
    
    # end_time = time.time()
    # print(f'\t Passing data through encoder/decoder required: {end_time - start_time}')
    return (
        colors, opacities, depths, n_rendering_samples
    )

def generate_occupancy_grid(
        device, 
        weights_path, 
        nerf_dict, 
        aabb, 
        n_iterations, 
        n_warmups):

        radiance_field = NGPradianceField(
                **nerf_dict
            ).to(device)
        
        matrix = torch.load(weights_path)
        radiance_field.load_state_dict(matrix)
        radiance_field.eval()

        # Create the OccupancyGrid
        render_n_samples = 1024
        grid_resolution = 64
        
        contraction_type = ContractionType.AABB
        scene_aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        near_plane = None
        far_plane = None
        render_step_size = (
            (scene_aabb[3:] - scene_aabb[:3]).max()
            * math.sqrt(3)
            / render_n_samples
        ).item()
        alpha_thre = 0.0

        occupancy_grid = OccupancyGrid(
            roi_aabb=aabb,
            resolution=grid_resolution,
            contraction_type=contraction_type,
        ).to(device)
        occupancy_grid.eval()

        with torch.no_grad():
            for i in range(n_iterations):
                def occ_eval_fn(x):
                    step_size = render_step_size
                    _ , density = radiance_field._query_density_and_rgb(x, None)
                    return density * step_size

                # update occupancy grid
                occupancy_grid._update(
                    step=i,
                    occ_eval_fn=occ_eval_fn,
                    occ_thre=1e-2,
                    ema_decay=0.95,
                    warmup_steps=n_warmups
                )

        return occupancy_grid
