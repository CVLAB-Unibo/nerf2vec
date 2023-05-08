import collections
import json
import operator
import os
from PIL import Image

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from nerf.utils import Rays

class NeRFLoader2:
    WIDTH, HEIGHT = 224, 224  
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = True

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        color_bkgd_aug: str = "random",  # NERF2VEC (Originally, it was white)
        num_rays: int = None,
        near: float = None,
        far: float = None,
        device: str = "cuda:0",
        weights_file_name: str = "bb07_steps3000_encodingFrequency_mlpFullyFusedMLP_activationReLU_hiddenLayers3_units64_encodingSize24.pth",
    ):
        super().__init__()
        assert color_bkgd_aug in ["white", "black", "random"]
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.device = device

        self.color_bkgd_aug = color_bkgd_aug

        self.weights_file_path = os.path.join(data_dir, weights_file_name)
        # self.weights = torch.load(weights_file_path)
        
        self.images, self.camtoworlds, self.focal = self._load_renderings(
            data_dir, split
        )
        self.images = torch.from_numpy(self.images).to(self.device).to(torch.uint8)
        self.camtoworlds = (
            torch.from_numpy(self.camtoworlds).to(self.device).to(torch.float32)
        )
        self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=self.device,
        )  # (3, 3)
        
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)
    
    def get_sample(self):
        data = self._fetch_data()
        data = self._preprocess(data)

        return data

    def _preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        
        if self.color_bkgd_aug == "random":
            color_bkgd = torch.rand(3, device=self.device)
        elif self.color_bkgd_aug == "white":
            color_bkgd = torch.ones(3, device=self.device)
        elif self.color_bkgd_aug == "black":
            color_bkgd = torch.zeros(3, device=self.device)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def _fetch_data(self):
        """Fetch the data (it maybe cached for multiple batches)."""

        num_rays = self.num_rays

        image_id = torch.randint(
            0,
            len(self.images),
            size=(num_rays,),
            device=self.device,
        )

        x = torch.randint(
            0, self.WIDTH, size=(num_rays,), device=self.device
        )
        y = torch.randint(
            0, self.HEIGHT, size=(num_rays,), device=self.device
        )

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
        
        origins = torch.reshape(origins, (num_rays, 3))
        viewdirs = torch.reshape(viewdirs, (num_rays, 3))
        rgba = torch.reshape(rgba, (num_rays, 4))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgba": rgba,  # [h, w, 4] 
            "rays": rays,  # [h, w, 3] 
        }

    def _load_renderings(self, data_dir: str, split: str):
        """
        if not root_fp.startswith("/"):
            root_fp = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                root_fp,
            )
        """

        # data_dir = os.path.join(root_fp, subject_id)
        # print(f'Loading renderings from: {data_dir}')
        
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