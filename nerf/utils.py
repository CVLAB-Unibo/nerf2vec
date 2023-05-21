import random
from typing import Optional

import numpy as np
import torch
# from utils import Rays, namedtuple_map
import collections
import torch.nn.functional as F

from nerfacc import OccupancyGrid, ray_marching, rendering

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
    radiance_field: torch.nn.Module,             
    embeddings: torch.Tensor,
    occupancy_grid: OccupancyGrid,               
    rays: Rays,                                  
    scene_aabb: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,  
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    grid_weights:dict = None
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

        rgb =  curr_rgb[curr_batch_idx][curr_mask] if curr_mask is not None else curr_rgb[curr_batch_idx]
        sigmas = curr_sigmas[curr_batch_idx][curr_mask] if curr_mask is not None else curr_sigmas[curr_batch_idx]

        return rgb, sigmas

    results = []
    
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training or batch_size > 1
        else 8192
    )
    
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[:, i : i + chunk], rays)

        b_positions = []
        b_t_starts = []
        b_t_ends = []
        b_ray_indices = []

        # ####################
        # RAY MARCHING
        # ####################
        with torch.no_grad():
            
            for batch_idx in range(batch_size):
                
                weights = torch.load(grid_weights[batch_idx])
                weights['_binary'] = weights['_binary'].to_dense()
                occupancy_grid.load_state_dict(weights)
                
                ray_indices, t_starts, t_ends = ray_marching(
                    chunk_rays.origins[batch_idx],  
                    chunk_rays.viewdirs[batch_idx], 
                    scene_aabb=scene_aabb, 
                    grid=occupancy_grid,
                    sigma_fn=None,  # Different from NerfAcc. Not beneficial/useful for nerf2vec.
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
                
                # Compute positions
                t_origins = chunk_rays.origins[batch_idx][ray_indices]
                t_dirs = chunk_rays.viewdirs[batch_idx][ray_indices]
                positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
                b_positions.append(positions)
                        

            MAX_SIZE = 35000  # Desired maximum size  # TODO: add a configuration variable
            padding_masks = [None]*batch_size

            if radiance_field.training or batch_size > 1:
                for batch_idx in range(batch_size):

                    # PADDING
                    if b_positions[batch_idx].size(0) < MAX_SIZE:

                        # Define padding dimensions                        
                        initial_size = b_positions[batch_idx].size(0)
                        padding_size = MAX_SIZE - initial_size

                        # Add padding
                        '''
                        For the positions pad with a value that is outside the bbox. This will force the decoder to consider
                        these points with density equal to 0, and they will not have any relevance in the computation of the loss.
                        '''
                        b_positions[batch_idx] = F.pad(b_positions[batch_idx], pad=(0, 0, 0, padding_size), value=0.8)  
                        b_t_starts[batch_idx] = F.pad(b_t_starts[batch_idx], pad=(0, 0, 0, padding_size))
                        b_t_ends[batch_idx] = F.pad(b_t_ends[batch_idx], pad=(0, 0, 0, padding_size))
                        b_ray_indices[batch_idx] = F.pad(b_ray_indices[batch_idx], pad=(0, padding_size))
                        
                        # Create masks used for ignoring the padding
                        padding_masks[batch_idx] = torch.zeros(MAX_SIZE, dtype=torch.bool)
                        padding_masks[batch_idx][:initial_size] = True

                    # TRUNCATION
                    else:

                        b_positions[batch_idx] = b_positions[batch_idx][:MAX_SIZE]
                        b_t_starts[batch_idx] = b_t_starts[batch_idx][:MAX_SIZE]
                        b_t_ends[batch_idx] = b_t_ends[batch_idx][:MAX_SIZE]
                        b_ray_indices[batch_idx] = b_ray_indices[batch_idx][:MAX_SIZE]
            
            # Convert arrays in tensors        
            b_t_starts = torch.stack(b_t_starts, dim=0)
            b_t_ends = torch.stack(b_t_ends, dim=0)
            b_ray_indices = torch.stack(b_ray_indices, dim=0)
            b_positions = torch.stack(b_positions, dim=0)
            
        # ####################
        # VOLUME RENDERING
        # ####################
        curr_rgb, curr_sigmas = radiance_field(embeddings, b_positions)
        
        for curr_batch_idx in range(batch_size):
            
            curr_mask = padding_masks[curr_batch_idx]

            rgb, opacity, depth = rendering(
                b_t_starts[curr_batch_idx][curr_mask] if curr_mask is not None else b_t_starts[curr_batch_idx],
                b_t_ends[curr_batch_idx][curr_mask] if curr_mask is not None else b_t_ends[curr_batch_idx],
                b_ray_indices[curr_batch_idx][curr_mask] if curr_mask is not None else b_ray_indices[curr_batch_idx],
                n_rays=chunk_rays.origins.shape[1], #original num_rays (important for the final output on which the loss will be computed)
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=render_bkgd[curr_batch_idx],
            )
            chunk_results = [rgb, opacity, depth, len(b_t_starts[curr_batch_idx][curr_mask]) if curr_mask is not None else len(b_t_starts[curr_batch_idx])]
            
            # Append to results array
            if curr_batch_idx < len(results):
                results[curr_batch_idx].append(chunk_results)
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
    n_rendering_samples = [sum(tensor) for tensor in n_rendering_samples] 
    
    return (
        colors, opacities, depths, n_rendering_samples
    )