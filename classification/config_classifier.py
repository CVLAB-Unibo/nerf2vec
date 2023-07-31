import os
from classification import config as base_config

LABELS_TO_IDS = {
    "02691156": 0,   # airplane
    "02828884": 1,   # bench
    "02933112": 2,   # cabinet
    "02958343": 3,   # car
    #'02992529': 4,   # tablet (delete?)
    "03001627": 4,   # chair
    "03211117": 5,   # display
    "03636649": 6,   # lamp
    "03691459": 7,   # speaker
    #"03948459": 9,   # gun (delete?)
    "04090263": 8,  # rifle
    "04256520": 9,  # sofa
    "04379243": 10,  # table
    "04401088": 11,  # phone
    "04530566": 12   # watercraft
}

EMBEDDINGS_DIR = os.path.join('classification', 'embeddings')
TRAIN_SPLIT = 'train'
VAL_SPLIT = 'val'
TEST_SPLIT = 'test'

TRAIN_BS = 256
VAL_BS = 256
# LAYERS_DIM = [1024, 512, 256]
LAYERS_DIM = [18432, 512, 256]
NUM_CLASSES = len(LABELS_TO_IDS)

LR = 1e-4
WD = 1e-2
NUM_EPOCHS = 150

OUTPUT_DIR = os.path.join('classification', 'classifier')


ENCODER_EMBEDDING_DIM = base_config.ENCODER_EMBEDDING_DIM  #1024
ENCODER_HIDDEN_DIM = base_config.ENCODER_HIDDEN_DIM  #[256, 256, 512, 512]
NERF_WEIGHTS_FILE_NAME = base_config.NERF_WEIGHTS_FILE_NAME  # 'bb07_steps3000_encodingFrequency_mlpFullyFusedMLP_activationReLU_hiddenLayers3_units64_encodingSize24.pth'
MLP_UNITS = base_config.MLP_UNITS  # 64

DECODER_INPUT_DIM = base_config.DECODER_INPUT_DIM
DECODER_HIDDEN_DIM = base_config.DECODER_HIDDEN_DIM
DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP = base_config.DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP
DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP = base_config.DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP
DECODER_OUT_DIM = base_config.DECODER_OUT_DIM
INSTANT_NGP_ENCODING_CONF = base_config.INSTANT_NGP_ENCODING_CONF
GRID_AABB = base_config.GRID_AABB

# TODO: complete this dictionary
WANDB_CONFIG = {

}
