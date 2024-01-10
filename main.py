import os

import nerf2vec.export_embeddings as nerf2vec_generic_embeddings
import task_generation.export_embeddings as generation_embeddings
from task_classification.train_classifier import InrEmbeddingClassifier
from task_interp_and_retrieval.interp import do_interpolation
from task_interp_and_retrieval.retrieval import do_retrieval

from nerf2vec.train_nerf2vec import Nerf2vecTrainer
import torch

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_MODE"] = "disabled"

cuda_idx = 0
device_name = f'cuda:{cuda_idx}'
torch.cuda.set_device(cuda_idx)

def train_nerf2vec():
    nerf2vec = Nerf2vecTrainer(device=device_name)
    nerf2vec.train()

def export_generic_embeddings():
    nerf2vec_generic_embeddings.export_embeddings()

def export_retrieval_for_generation():
    generation_embeddings.export_embeddings()

def train_classifier():
    classifier = InrEmbeddingClassifier()
    classifier.train()

def execute_interpolation_task():
    do_interpolation(device=device_name)

def execute_retrieval_task():
    do_retrieval(device=device_name)

def generate_grids():

    nerf_roots = [
        os.path.join('/', 'media', 'data4TB', 'sirocchi', 'nerf2vec', 'data', 'data_TRAINED'),
        os.path.join('/', 'media', 'data4TB', 'sirocchi', 'nerf2vec', 'data', 'data_TRAINED_A1'),
        os.path.join('/', 'media', 'data4TB', 'sirocchi', 'nerf2vec', 'data', 'data_TRAINED_A2')
    ]

    for nerf_root in nerf_roots:
        start_grids_generation(nerf_root)



def main():
    method_to_call = None
    
    choices = {
        0: ['Train nerf2vec', train_nerf2vec],
        1: ['Export nerf2vec embeddings (necessary for classification and retrieval tasks)', export_generic_embeddings],
        2: ['\nInterpolation', execute_interpolation_task],
        3: ['Retrieval', execute_retrieval_task],
        4: ['Train classifier', train_classifier]
    }

    while True:
        # Display the menu options
        print(f'Select an option (0-{len(choices)-1}):')

        for i in choices:
            description = choices[i][0]
            print(f'{i}. {description}')

        # Get user input
        try:
            choice = int(input("Enter your choice: "))
            
            # Check if the choice is valid
            if 0 <= choice <= 10:
                print(f"You selected option {choice}.")
                method_to_call = choices[choice][1]
                # Here, you can add code to handle each choice
                # For example, call different functions based on the choice
                break  # Exit the loop if the choice is valid
            else:
                print("Invalid choice. Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    method_to_call()
    print()

if __name__ == '__main__':
    main()

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
    # do_retrieval_baseline(multi_view=True)
    # create_renderings_from_GAN_embeddings()
    # export_embeddings_for_mapping()
    # train_completion()

    # train_nerf2vec()

    # train_completion()
    # mapping_network_plot()
