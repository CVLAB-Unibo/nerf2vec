import collections
import json
import operator
import os
from PIL import Image

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from .utils import Rays


def _load_renderings(root_fp: str, subject_id: str, split: str):
    if not root_fp.startswith("/"):
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = os.path.join(root_fp, subject_id)
    print(f'Loading renderings from: {data_dir}')
    
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        rgba = imageio.imread(fname)
        
        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)
        

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal

class NeRFLoader(torch.utils.data.Dataset):

    WIDTH, HEIGHT = 224, 224  
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = True

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        color_bkgd_aug: str = "random",  # NERF2VEC (Originally, it was white)
        num_rays: int = None,
        near: float = None,
        far: float = None,
        device: str = "cuda:0",
    ):
        super().__init__()
        assert color_bkgd_aug in ["white", "black", "random"]
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far

        self.color_bkgd_aug = color_bkgd_aug

        weights_file_name = 'bb07_steps3000_encodingFrequency_mlpFullyFusedMLP_activationReLU_hiddenLayers3_units64_encodingSize24.pth'
        self.weights_file_path = os.path.join(root_fp, subject_id, weights_file_name)
        
        split = 'train'
        self.images, self.camtoworlds, self.focal = _load_renderings(
            root_fp, subject_id, split
        )
        self.images = torch.from_numpy(self.images).to(device).to(torch.uint8)
        self.camtoworlds = (
            torch.from_numpy(self.camtoworlds).to(device).to(torch.float32)
        )
        self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        )  # (3, 3)
        
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        # TODO: Evaluate the best background color
        color_bkgd = torch.ones(3, device=self.images.device)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        
        image_id = [index]
        x, y = torch.meshgrid(
            torch.arange(self.WIDTH, device=self.images.device),
            torch.arange(self.HEIGHT, device=self.images.device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)

        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )
        
        origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
        viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
        rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, 4))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgba": rgba,  # [h, w, 4] 
            "rays": rays,  # [h, w, 3] 
        }
