#
# NERFACC
#
AABB = [-0.7, -0.7, -0.7, 0.7, 0.7, 0.7]

NUM_RAYS = 20000

OCCUPANCY_GRID_RECONSTRUCTION_ITERATIONS = 20
OCCUPANCY_GRID_WARMUP_ITERATIONS = 5

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
    'aabb': AABB,
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

# 
# TINY-CUDA
#
TINY_CUDA_MIN_SIZE = 16

# 
# TRAIN
#
NUM_EPOCHS = 150
BATCH_SIZE = 2