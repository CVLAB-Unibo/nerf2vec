import math
import random
from typing import Optional

import numpy as np
import torch
# from utils import Rays, namedtuple_map
import collections

from nerfacc import ContractionType, OccupancyGrid, ray_marching, rendering

from nerf.intant_ngp import NGPradianceField

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
    occupancy_grid: OccupancyGrid,               # This should be [batch_size, grids]
    rays: Rays,                                  # This should be [batch_size, rays]
    scene_aabb: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,  # This should be [batch_size, render_bkgds]
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    
    """
    # ################################################################################
    # __D
    # __D this is the shape that arrives when testing. Will decide later what to do
    # __D
    # ################################################################################
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape
    """
    rays_shape = rays.origins.shape
    batch_size, num_rays, coordinates = rays_shape  # __D Remove unused variables once the debug is complete

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field.query_density(positions, t)
        
        _, density = radiance_field._query_density_and_rgb(positions, None)
        return density
        

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field(positions, t, t_dirs)
        return radiance_field(positions, t_dirs)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    
    
    # Divide in chunks [batch_size, ]
    for i in range(0, num_rays, chunk):
        # chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        chunk_rays = namedtuple_map(lambda r: r[:, i : i + chunk], rays)  # Add the batch size

        b_ray_indices = torch.zeros(batch_size, 1)
        b_t_starts = torch.zeros(batch_size, chunk, 1)
        b_t_ends = torch.zeros(batch_size, chunk, 1)


        for batch_idx in range(batch_size):
            ray_indices, t_starts, t_ends = ray_marching(
                chunk_rays[batch_idx].origins,  # [batch_size, chunk, 3d coord]
                chunk_rays[batch_idx].viewdirs, # [batch_size, chunk, 3d coord]
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

            # Add the variables to three different arrays, that keep in consideration the batch size
            b_ray_indices[batch_idx] = ray_indices  
            b_t_starts[batch_idx] = t_starts
            b_t_ends[batch_idx] = t_ends
        
        b_t_origins = rays.origins[:, b_ray_indices]
        b_t_dirs = rays.viewdirs[:, b_ray_indices]
        b_positions = b_t_origins + b_t_dirs * (b_t_starts + b_t_ends) / 2.0

        # __D Positions must have shape [batch_size, chunk, 3d_coord]
        _, sigmas = radiance_field._query_density_and_rgb(b_positions, None)
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)
        alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))

        
        # Compute visibility of the samples, and filter out invisible samples
        masks = render_visibility(
            alphas,
            ray_indices=ray_indices,
            packed_info=packed_info,
            early_stop_eps=early_stop_eps,
            alpha_thre=alpha_thre,
            n_rays=rays_o.shape[0],
        )
        ray_indices, t_starts, t_ends = (
            ray_indices[masks],
            t_starts[masks],
            t_ends[masks],
        )




        rgb, opacity, depth = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
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
        grid_resolution = 128
        
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


