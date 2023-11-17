import os
import sys
# from classification.export_renderings import clear_baseline_renderings, export_baseline_renderings
from classification.interp import interpolate
from classification.retrieval import do_retrieval
from classification.retrieval_baseline import do_retrieval_baseline
from classification.train_baseline_classifier import BaselineClassifier

from classification.train_nerf2vec import Nerf2vecTrainer
from dataset.generate_grids import start_grids_generation
from classification.export_embeddings import export_embeddings
from classification.export_baseline_embeddings import export_baseline_embeddings
from classification.train_classifier import InrEmbeddingClassifier
from evaluation.evaluation import evaluate_baseline_classification, evaluate_nerf2vec_classification

def train_nerf2vec():
    nerf2vec = Nerf2vecTrainer()
    nerf2vec.train()

def train_classifier():
    classifier = InrEmbeddingClassifier()
    classifier.train()

def interpolate_embeddings():
    interpolate()

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

    # train_nerf2vec()
    # generate_grids()
    # export_embeddings()
    # train_classifier()
    # interpolate()

    # export_baseline_renderings()
    # clear_baseline_renderings()
    
    
    # baseline_classifier = BaselineClassifier()
    # baseline_classifier.train()
    # baseline_classifier.val("val")

    # evaluate_nerf2vec_classification()
    # evaluate_baseline_classification()

    # do_retrieval()

    # export_baseline_embeddings(multi_view=False)
    do_retrieval_baseline(multi_view=True)