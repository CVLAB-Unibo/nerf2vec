import os
import shutil

import torch
from classification.train_nerf2vec import Nerf2vecTrainer
from dataset.generate_grids import start_grids_generation

def train_nerf2vec():
    nerf2vec = Nerf2vecTrainer()
    nerf2vec.train()

def generate_grids():

    nerf_roots = [os.path.join('/', 'media', 'data4TB', 'sirocchi', 'nerfacc_nerf2vec', 'data_TRAINED'),
                  os.path.join('/', 'media', 'data4TB', 'sirocchi', 'nerfacc_nerf2vec', 'data_TRAINED_A1'),
                  os.path.join('/', 'media', 'data4TB', 'sirocchi', 'nerfacc_nerf2vec', 'data')]

    for nerf_root in nerf_roots:
        start_grids_generation(nerf_root)


if __name__ == '__main__':

    # torch.cuda.empty_cache()
    train_nerf2vec()

    # generate_grids()
