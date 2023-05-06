import os
from pathlib import Path
import torch

from torchvision.models import resnet18, ResNet50_Weights

from classification.train_nerf2vec import NeRFDataset

def train_nerf2vec():
    
    a = NeRFDataset(os.path.abspath('data'), None)
    curr_nerf_loader = a[0]
    data = curr_nerf_loader[0]
    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]

    print()


if __name__ == '__main__':
    train_nerf2vec()



