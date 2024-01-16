# nerf2vec

This repository contains the code related to the framework nerf2vec, as described in the paper [nerf2vec: Neural Radiance Fields to Vector](https://arxiv.org/abs/2312.13277).

## Machine Configuration

Before running the code, ensure that your machine is properly configured. Here are the necessary steps:

```bash

pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch



conda install python=3.8.18
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c anaconda pip
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
conda install -c conda-forge cudatoolkit-dev
nvcc --version
conda install -c conda-forge opencv
conda install -c conda-forge libjpeg-turbo
conda install -c conda-forge cupy
conda install -c conda-forge pkg-config
conda install -c conda-forge numba
/home/sirocchi/miniconda3/envs/cuda_test/bin/pip install ffcv
pip install --upgrade nerfacc==0.3.5
conda install -c conda-forge imageio
conda install -c conda-forge ninja
conda install -c anaconda h5py
conda install -c anaconda cudatoolkit=11.7
conda install cudatoolkit=11.7 -c nvidia 

# TODO: CHECK THESE DEPENDENCIES
### conda install -c anaconda cudatoolkit=11.7
### conda install cudatoolkit=11.7 -c nvidia 
```
### Generation
Describe the environment for this task.

### Mapping Network
The mapping network task requires the training of the inr2vec framework. Please, refer to [THIS](https://github.com/CVLAB-Unibo/inr2vec?tab=readme-ov-file#setup) page to properly configure your environment.

## Training and tasks
This sections contains the details required to run the code.

**IMPORTANT NOTE**: each module cited below *should* be executed from the root of the project, and not within the corresponding packages.

**settings.py**: explain...

### Train nerf2vec

```bash
python nerf2vec/train_nerf2vec.py
```

### Retrieval task

```bash
python task_interp_and_retrieval/retrieval.py
```

The results will be shown in the task_interp_and_retrieval/retrieval_plots_X folder, where X depends on the chosen split. The split che be set in the main method of the retrieval.py module.
Note that each file generated for the same retrieval iteration will have the same UUID, with a suffix that is the indices. 


### Interpolation task
The results will be shown in the task_interp_and_retrieval/called interp_plots_X folder, where X depends on the chosen split. The split che be set in the main method of the interp.py module.

```bash
python task_interp_and_retrieval/interp.py
```

### Generation task
In order to properly genereate e see new embeddings, it is necessary to execute the following operations respecting the order:

1. export embeddings
This task will create the folder task_generation/latent_embeddings...
```bash
python task_generation/export_embeddings.py
```

2. train GANs
This task will create the folder task_generation/experiments...
```bash
python task_generation/train_latent_gan.py
```
Specific environment required
This task will create one GAN for each class.
The train method contained in the train_latent_gan.py has all the hyper-parameters used to train the GANs.

3. create renderings
Change environment again.
```bash
python task_generation/viz_nerf.py 
```
GAN_plots_X where X depends on the chosen class
GENERATION_LATENT_GAN_FULL_CKPT_PATH this path depends on the number of epochs used to train the models.


 