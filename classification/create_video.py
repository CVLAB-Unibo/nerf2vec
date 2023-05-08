import os
import imageio
import torch
import tqdm
import numpy as np

import torch.nn.functional as F
from classification.utils import render_image
from nerf.utils import Rays



def get_translation_t(t):
    """Get the translation matrix for movement in t."""
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ]

    return torch.tensor(matrix, dtype=torch.float32)


def get_rotation_phi(phi):
    """Get the rotation matrix for movement in phi."""
    matrix = [
        [1, 0, 0, 0],
        [0, torch.cos(phi), -torch.sin(phi), 0],
        [0, torch.sin(phi), torch.cos(phi), 0],
        [0, 0, 0, 1],
    ]
    return torch.tensor(matrix, dtype=torch.float32)


def get_rotation_theta(theta):
    """Get the rotation matrix for movement in theta."""
    matrix = [
        [torch.cos(theta), 0, -torch.sin(theta), 0],
        [0, 1, 0, 0],
        [torch.sin(theta), 0, torch.cos(theta), 0],
        [0, 0, 0, 1],
    ]
    return torch.tensor(matrix, dtype=torch.float32)


def pose_spherical(theta, phi, t):
    """
    Get the camera to world matrix for the corresponding theta, phi
    and t.
    """
    c2w = get_translation_t(t)
    c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = torch.from_numpy(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [
                           0, 0, 0, 1]], dtype=np.float32)) @ c2w  # TODO: check if PyTorch can handle this
    return c2w


def create_video(width,
                 height,
                 device,
                 focal,
                 radiance_field,
                 occupancy_grid,
                 scene_aabb,
                 near_plane,
                 far_plane,
                 render_step_size,
                 render_bkgd,
                 cone_angle,
                 alpha_thre,
                 test_chunk_size,
                 path,
                 OPENGL_CAMERA=True):

    rgb_frames = []

    # Iterate over different theta value and generate scenes.
    max_images = 120
    array = np.linspace(-30.0, 30.0, max_images//2, endpoint=False)
    array = np.append(array, np.linspace(
        30.0, -30.0, max_images//2, endpoint=False))
    
    for index, theta in tqdm.tqdm(enumerate(np.linspace(0.0, 360.0, max_images, endpoint=False))):

        # Get the camera to world matrix.
        c2w = pose_spherical(torch.tensor(theta), torch.tensor(
            array[index]), torch.tensor(0.8))
        c2w = c2w.to(device)

        x, y = torch.meshgrid(
            torch.arange(width, device=device),
            torch.arange(height, device=device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()

        K = torch.tensor(
            [
                [focal, 0, width / 2.0],
                [0, focal, height / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        )  # (3, 3)

        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - K[0, 2] + 0.5) / K[0, 0],
                    (y - K[1, 2] + 0.5)
                    / K[1, 1]
                    * (-1.0 if OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]
        camera_dirs.to(device)

        directions = (camera_dirs[:, None, :] * c2w[:3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        origins = torch.reshape(origins, (height, width, 3))
        viewdirs = torch.reshape(viewdirs, (height, width, 3))
        
        rays = Rays(origins=origins, viewdirs=viewdirs)
        # render
        rgb, acc, depth, n_rendering_samples = render_image(
            radiance_field=radiance_field,
            occupancy_grid=occupancy_grid,
            rays=rays,
            scene_aabb=scene_aabb,
            # rendering options
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )

        numpy_image = (rgb.cpu().numpy() * 255).astype(np.uint8)
        rgb_frames.append(numpy_image)

    # rgb_video = os.path.join(path,"rgb_video.mp4")
    imageio.mimwrite(path, rgb_frames, fps=30,
                     quality=8, macro_block_size=None)
