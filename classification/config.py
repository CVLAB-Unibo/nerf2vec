# TODO: rename this module in config_nerf2vec.py, and update all the references to it.
"""
# ####################
# NERF2VEC
# ####################
"""
#
# DIMENSIONS
#
ENCODER_EMBEDDING_DIM = 1024
ENCODER_HIDDEN_DIM = [256, 256, 512, 512]


DECODER_INPUT_DIM = 3
DECODER_HIDDEN_DIM = 512
DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP = 2
DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP = 2
DECODER_OUT_DIM = 4

# 
# TRAIN 
#
NUM_EPOCHS = 600
BATCH_SIZE = 64
LR = 1e-3
WD = 1e-2

"""
# ####################
# NERFACC
# ####################
"""
#
# GRIDd
#
import os
from nerfacc import ContractionType
GRID_AABB = [-0.7, -0.7, -0.7, 0.7, 0.7, 0.7]
GRID_RESOLUTION = 96
GRID_CONTRACTION_TYPE = ContractionType.AABB
GRID_CONFIG_N_SAMPLES = 1024

GRID_RECONSTRUCTION_TOTAL_ITERATIONS = 20
GRID_RECONSTRUCTION_WARMUP_ITERATIONS = 5

#
# RAYS
#
NUM_RAYS = 25000

#
# INSTANT-NGP 
#
MLP_INPUT_SIZE = 3
MLP_ENCODING_SIZE = 24
MLP_INPUT_SIZE_AFTER_ENCODING = MLP_INPUT_SIZE * MLP_ENCODING_SIZE * 2
MLP_OUTPUT_SIZE = 4
MLP_HIDDEN_LAYERS = 3
MLP_UNITS = 64

INSTANT_NGP_MLP_CONF = {
    'aabb': GRID_AABB,
    'unbounded':False,
    'encoding':'Frequency',
    'mlp':'FullyFusedMLP',
    'activation':'ReLU',
    'n_hidden_layers':MLP_HIDDEN_LAYERS,
    'n_neurons':MLP_UNITS,
    'encoding_size':MLP_ENCODING_SIZE
}

INSTANT_NGP_ENCODING_CONF = {
    "otype": "Frequency",
    "n_frequencies": 24
}

NERF_WEIGHTS_FILE_NAME = 'bb07_steps3000_encodingFrequency_mlpFullyFusedMLP_activationReLU_hiddenLayers3_units64_encodingSize24.pth'

#
# TINY-CUDA
#
TINY_CUDA_MIN_SIZE = 16

"""
# ####################
# LOGGING
# ####################
"""
WANDB_CONFIG = {
    'ENCODER_EMBEDDING_DIM': ENCODER_EMBEDDING_DIM,
    'ENCODER_HIDDEN_DIM': ENCODER_HIDDEN_DIM,
    'DECODER_INPUT_DIM': DECODER_INPUT_DIM,
    'DECODER_HIDDEN_DIM': DECODER_HIDDEN_DIM,
    'DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP': DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP,
    'DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP': DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP,
    'DECODER_OUT_DIM': DECODER_OUT_DIM,
    'NUM_EPOCHS': NUM_EPOCHS,
    'BATCH_SIZE': BATCH_SIZE,
    'LR': LR,
    'WD': WD,
    "NUM_RAYS": NUM_RAYS,
    "GRID_RESOLUTION": GRID_RESOLUTION
}





