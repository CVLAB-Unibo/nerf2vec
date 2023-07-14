import os
import sys

from classification.train_nerf2vec import Nerf2vecTrainer
from dataset.generate_grids import start_grids_generation
from classification.export_embeddings import export_embeddings
from classification.train_classifier import InrEmbeddingClassifier

def train_nerf2vec():
    nerf2vec = Nerf2vecTrainer()
    nerf2vec.train()

def train_classifier():
    classifier = InrEmbeddingClassifier()
    classifier.train()

def generate_grids():

    nerf_roots = [
        os.path.join('/', 'media', 'data4TB', 'sirocchi', 'nerf2vec', 'data', 'data_TRAINED'),
        os.path.join('/', 'media', 'data4TB', 'sirocchi', 'nerf2vec', 'data', 'data_TRAINED_A1'),
        os.path.join('/', 'media', 'data4TB', 'sirocchi', 'nerf2vec', 'data', 'data_TRAINED_A2')
    ]

    for nerf_root in nerf_roots:
        start_grids_generation(nerf_root)

        
if __name__ == '__main__':
    
    # TODO: uncomment this as soon as the training is complete (must be tested)
    """
    if len(sys.argv) < 2:
        print("Please provide a method name as an argument.")
    else:
        method_name = sys.argv[1]
        if hasattr(sys.modules[__name__], method_name):
            method = getattr(sys.modules[__name__], method_name)
            method()
        else:
            print(f"Method '{method_name}' does not exist.")
    """

    train_nerf2vec()
    # generate_grids()
    # export_embeddings()
    # train_classifier()
