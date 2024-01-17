# nerf2vec

This repository contains the code related to the **nerf2vec** framework, which is detailed in the paper [nerf2vec: Neural Radiance Fields to Vector](https://arxiv.org/abs/2312.13277).

## Machine Configuration

Before running the code, ensure that your machine is properly configured. Here are the necessary steps:

```bash
conda install python=3.8.18
conda install -c anaconda pip
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install --upgrade nerfacc==0.3.5

conda install -c conda-forge einops
conda install -c conda-forge imageio
pip install --upgrade wandb==0.16.0



conda install -c conda-forge opencv

conda install -c conda-forge wandb
conda install -c conda-forge numba



conda install -c conda-forge pkg-config
conda install -c anaconda h5py

conda install -c conda-forge libjpeg-turbo
conda install -c conda-forge cupy




```
//conda install -c conda-forge ninja
//conda install -c conda-forge cudatoolkit-dev
//pip install ffcv
// conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
// conda install -c anaconda cudatoolkit=11.7
// conda install cudatoolkit=11.7 -c nvidia 

### Generation
The generation task is based on a Latent GAN model detailed at [THIS](https://github.com/optas/latent_3d_points) link. Please, follow the instructions provided at that link to properly configure your environment.

### Mapping Network
The mapping network task requires the training of the inr2vec framework. Please, refer to [THIS](https://github.com/CVLAB-Unibo/inr2vec?tab=readme-ov-file#setup) page to properly configure your environment.

## Training and experiments
This sections contains the details required to run the code.

**IMPORTANT NOTES**: 
1. each module cited below *must* be executed from the root of the project, and not within the corresponding packages.

2. the file *settings.py* contains all the paths (e.g., dataset location, model weights, etc...) and generic configurations that are used from each module explained below. 

3. Some training and experiments, such as the training of the *nerf2vec* framework and the classification task, are structured to use the *wandb* library. If you want to use it, then you need to change the following two variables: ``` os.environ["WANDB_SILENT"]``` and  ```os.environ["WANDB_MODE"]```, which are located at the beginning of the *settings.py* module. 

## Train *nerf2vec*

To train *nerf2vec* you need to have a dataset of trained NeRFs. The implemented code expects that there exist the following files:
* data/train.json
* data/validation.json
* data/test.json

These JSONs hold a list of file paths, with each path corresponding to a NeRF model that has been trained, and then used in a specific data split. In particular, each path corresponds to a folder, and each folder contains the following relevant files:
* the trained NeRF's weights
* the NeRF's occupancy grid
* JSON files with transform matrices and other paramters necessary to train NeRFs.

Note that the name of the files contained in these folders should not be changed. 


**TODO: ADD LINK!!!!!!** 
The original dataset used to train nerf2vec can be found here: ..... After you have downloaded it, you have to put the *data* folder in the root of the project.

Execute the following command to train the *nerf2vec* framework:
```bash
python nerf2vec/train_nerf2vec.py
```
If you have enabled *wandb*, then you should update its settings located in the *config_wandb* method.

## Export *nerf2vec* embeddings
Execute the following command to export the *nerf2vec*'s embeddings:
```bash
python nerf2vec/export_embeddings.py
```
Note that these embeddings are **necessary** for other tasks, such as classification, retrieval and generation.

## Retrieval task
Execute the following command to perform the retrieval task:
```bash
python task_interp_and_retrieval/retrieval.py
```
The results will be shown in the *task_interp_and_retrieval/retrieval_plots_X* folder, where X depends on the chosen split (i.e., train, validation or test). The split che be set in the main method of the retrieval.py module.

Each file generated during a specific retrieval iteration will be named using the same prefix represented by a randomly generated UUID.


## Interpolation task
Execute the following command to perform the interpolation task:
```bash
python task_interp_and_retrieval/interp.py
```
The results will be shown in the *task_interp_and_retrieval/interp_plots_X* folder, where X depends on the chosen split (i.e., train, validation or test). The split che be set in the main method of the retrieval.py module.

## Classification task
Execute the following command to perform the classification task:
```bash
python task_classification/train_classifier.py
```
If you have enabled *wandb*, then you should update its settings located in the *config_wandb* method.

## Generation task
In order to generate and visualize the new embeddings, it is necessary to execute some operations following a specific order.

### 1) Export embeddings
The following command creates the folder *task_generation/latent_embeddings*, which will contain the *nerf2vec*'s embedding properly organized for this task.
```bash 
python task_generation/export_embeddings.py
```

### 2) Train GANs
The following command creates the folder *task_generation/experiments*, which will contain both the weights of the trained models and the generated embeddings:
```bash
python task_generation/train_latent_gan.py
```
All the hyperparameters used to train the Latent GANs can be found inside the *train_latent_gan.py* module.

Note that this step requires to enable a specific environment, as explained before.

### 3) Create renderings
The following command create renderings from the embeddings generated during the previous step:
```bash
python task_generation/viz_nerf.py 
```
The renderings will be created in the *GAN_plots_X* folder, where X is the ID of a specific class.


## Mapping network map
Please, for this task refer to [THIS](task_mapping_network/README.md) README.


 