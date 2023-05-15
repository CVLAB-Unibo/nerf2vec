import shutil

import torch
from classification.train_nerf2vec import Nerf2vecTrainer

def train_nerf2vec():
    nerf2vec = Nerf2vecTrainer()
    nerf2vec.train()

if __name__ == '__main__':

    """
    source_file = 'grid.pth.gz'
    destination_directory = 'zipped_grids'
    num_duplicates = 1024

    for i in range(num_duplicates):
        destination_file = f'{destination_directory}/grid_{i+1}.pth.gz'
        shutil.copyfile(source_file, destination_file)
    """
    # torch.cuda.empty_cache()
    train_nerf2vec()
