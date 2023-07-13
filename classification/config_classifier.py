import os

LABELS_TO_IDS = {
    "02691156": 0,   # airplane
    "02828884": 1,   # bench
    "02933112": 2,   # cabinet
    "02958343": 3,   # car
    '02992529': 4,   # tablet
    "03001627": 5,   # chair
    "03211117": 6,   # display
    "03636649": 7,   # lamp
    "03691459": 8,   # speaker
    "03948459": 9,   # gun
    "04090263": 10,  # rifle
    "04256520": 11,  # sofa
    "04379243": 12,  # table
    "04401088": 13,  # phone
    "04530566": 14   # watercraft
}

EMBEDDINGS_DIR = os.path.join('classification', 'embeddings')
TRAIN_SPLIT = 'train'
VAL_SPLIT = 'val'
TEST_SPLIT = 'test'

TRAIN_BS = 256
VAL_BS = 256
LAYERS_DIM = [1024, 512, 256]
NUM_CLASSES = len(LABELS_TO_IDS)

LR = 1e-4
WD = 1e-2
NUM_EPOCHS = 150

OUTPUT_DIR = os.path.join('classification', 'classifier')


# TODO: take these values from the inr2vec configuration file
ENCODER_EMBEDDING_DIM = 1024
ENCODER_HIDDEN_DIM = [256, 256, 512, 512]
NERF_WEIGHTS_FILE_NAME = 'bb07_steps3000_encodingFrequency_mlpFullyFusedMLP_activationReLU_hiddenLayers3_units64_encodingSize24.pth'
MLP_UNITS = 64
