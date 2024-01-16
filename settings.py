import os

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_MODE"] = "disabled"

cuda_idx = 0
DEVICE_NAME = 'cuda:%s' % cuda_idx  # Keep compatibility with older code

"""
# ##################################################
# PATHS USED BY DIFFERENT MODULES
# ##################################################
"""

# DATASET
TRAIN_DSET_JSON = os.path.abspath(os.path.join('data', 'train.json'))
VAL_DSET_JSON = os.path.abspath(os.path.join('data', 'validation.json'))  
TEST_DSET_JSON = os.path.abspath(os.path.join('data', 'test.json'))  

# NERF2VEC
NERF2VEC_CKPTS_PATH = os.path.join('nerf2vec', 'train', 'ckpts')
NERF2VEC_ALL_CKPTS_PATH = os.path.join('nerf2vec', 'train', 'all_ckpts')
NERF2VEC_EMBEDDINGS_DIR = os.path.join('task_classification', 'embeddings') 

# CLASSIFICATION
CLASSIFICATION_OUTPUT_DIR = os.path.join('classification', 'classifier')

# GENERATION
GENERATION_EMBEDDING_DIR = os.path.join('task_generation', 'latent_embeddings')
GENERATION_OUT_DIR = os.path.join('task_generation', 'experiments', '{}')  # The placeholder will contain the class index
GENERATION_NERF2VEC_FULL_CKPT_PATH = os.path.join('task_classification', 'train', 'ckpts', '499.pt')
GENERATION_LATENT_GAN_FULL_CKPT_PATH = os.path.join('task_generation', 'experiments', 'nerf2vec_{}', 'generated_embeddings', 'epoch_2000.npz')  # The placeholder will contain the class index
    

