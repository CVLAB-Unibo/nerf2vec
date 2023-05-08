import torch

class NeRFLoader3:
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
        self.device = device

    def get_sample(self):
        data = torch.rand((2,3), device='cuda:0')
        return data