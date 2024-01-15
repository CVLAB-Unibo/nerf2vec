import os

import nerf2vec.export_embeddings as nerf2vec_generic_embeddings
import task_generation.export_embeddings as generation_embeddings
from task_classification.train_classifier import InrEmbeddingClassifier
from task_generation.train_latent_gan import train
from task_generation.viz_nerf import create_renderings_from_GAN_embeddings
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

def train_classifier():
    classifier = InrEmbeddingClassifier()
    classifier.train()

def execute_interpolation_task():
    do_interpolation(device=device_name)

def execute_retrieval_task():
    do_retrieval(device=device_name)

def export_embeddings_for_gan():
    generation_embeddings.export_embeddings()

def ask_class_index() -> int:
    selected_class_idx = 0
    while True:
        print(f'Select a class (0-12):')

        # Get user input
        try:
            choice = int(input("Enter your choice: "))
            
            # Check if the choice is valid
            if 0 <= choice <= 12:
                print(f"You selected option {choice}.")
                # Here, you can add code to handle each choice
                # For example, call different functions based on the choice
                selected_class_idx = choice
                break  # Exit the loop if the choice is valid
            else:
                print("Invalid choice")
        except ValueError:
            print("Invalid input. Please enter a number.")
    return selected_class_idx


def train_latent_gan():
    selected_class_idx = ask_class_index()
    train(selected_class_idx=selected_class_idx)

def visualize_gan_results():   
    selected_class_idx = ask_class_index()
    create_renderings_from_GAN_embeddings(device=device_name, class_idx=selected_class_idx)

def main():
    method_to_call = None
    
    choices = {
        0: ['Train nerf2vec', train_nerf2vec],
        1: ['Export nerf2vec embeddings (necessary for the classification and retrieval tasks)', export_generic_embeddings],
        2: ['Interpolation', execute_interpolation_task],
        3: ['Retrieval', execute_retrieval_task],
        4: ['Train classifier', train_classifier],
        5: ['Export embeddings (necessary for the generation task)', export_embeddings_for_gan],
        6: ['Train latent GAN', train_latent_gan],
        7: ['Visualize GAN results', visualize_gan_results]
    }

    while True:
        print(f'Select an option (0-{len(choices)-1}):')

        for i in choices:
            if i == 0:
                print()
                print('*'*16)
                print('*** NERF2VEC ***')
                print('*'*16)
            if i == 5:
                print()
                print('*'*23)
                print('*** GENERATION TASK ***')
                print('*'*23)

            description = choices[i][0]
            print(f'  {i}. {description}')

        print()
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

