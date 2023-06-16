import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import concurrent

from nerf.utils import Rays


def read_image(file_path):
    return imageio.imread(file_path)


def _load_renderings(data_dir: str, split: str):
    
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    file_paths = []
    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        
        file_paths.append(fname)

        camtoworlds.append(frame["transform_matrix"])
        #rgba = imageio.imread(fname)
        #images.append(rgba)
        
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(read_image, file_paths)
        images = list(results)
    
    
    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal

"""
def decode_images(dictionary, key):

    image_bytes_list = dictionary.get(key, [])
    decoded_image_list = []

    for image_bytes in image_bytes_list:
        try:
            decoded_image = imageio.imread(image_bytes)
            decoded_image_list.append(decoded_image)
        except Exception as e:
            print(f"Error decoding image for key '{key}': {str(e)}")

    return decoded_image_list

def _load_renderings_from_RAM2(data_dir: str, split: str, images_RAM: dict):
    
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    
    camtoworlds = []

    file_paths = []
    
    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        file_paths.append(fname)

        #rgba = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        #rgba = cv2.cvtColor(rgba, cv2.COLOR_BGR2RGBA)
        
        camtoworlds.append(frame["transform_matrix"])


        
    images = decode_images(images_RAM, data_dir) 
    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal

def _load_renderings_from_RAM( data_dir: str, split: str, images_RAM: dict):
    
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    file_paths = []
    
    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        file_paths.append(fname)

        #rgba = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        #rgba = cv2.cvtColor(rgba, cv2.COLOR_BGR2RGBA)
        
        camtoworlds.append(frame["transform_matrix"])

    if '_A1' in data_dir:
        images = decode_images(images_RAM[0], data_dir)
    #elif '_A2' in data_dir:
    #    images = decode_images(images_RAM[1], data_dir)
    else:
        # images = decode_images(images_RAM[2], data_dir)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(read_image, file_paths)
            images = list(results)
        
        
    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal
"""

class NeRFLoader(torch.utils.data.Dataset):

    WIDTH, HEIGHT = 224, 224  
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = True

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        color_bkgd_aug: str = "random",  
        num_rays: int = None,
        near: float = None,
        far: float = None,
        device: str = "cuda:0",
        weights_file_name: str = "bb07_steps3000_encodingFrequency_mlpFullyFusedMLP_activationReLU_hiddenLayers3_units64_encodingSize24.pth",
        training: bool = True,
        images_RAM = None
    ):
        super().__init__()
        assert color_bkgd_aug in ["white", "black", "random"]
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far

        self.training = training

        self.images_RAM = images_RAM

        self.color_bkgd_aug = color_bkgd_aug

        self.weights_file_path = os.path.join(data_dir, weights_file_name)
        
        self.images, self.camtoworlds, self.focal = _load_renderings(#_from_RAM(
            data_dir, split#, self.images_RAM
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

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            color_bkgd = torch.zeros(3, device=self.images.device)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""

        num_rays = self.num_rays

        if self.training:
            image_id = torch.randint(
                0,
                len(self.images),
                size=(num_rays,),
                device=self.images.device,
            )

            x = torch.randint(
                0, self.WIDTH, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device=self.images.device
            )
        else:
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

        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )
        
        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, 4))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, 4))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgba": rgba,  # [h, w, 4] 
            "rays": rays,  # [h, w, 3] 
        }
