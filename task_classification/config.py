import os

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


TRAIN_BS = 256
VAL_BS = 256
LAYERS_DIM = [1024, 512, 256]
NUM_CLASSES = len(LABELS_TO_IDS)

LR = 1e-4
WD = 1e-2
NUM_EPOCHS = 150


# TODO: complete this dictionary
WANDB_CONFIG = {

}
