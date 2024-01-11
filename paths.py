import os

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

# CLASSIFICATION, INTERPOLATION AND RETRIEVAL
CLASSIFICATION_OUTPUT_DIR = os.path.join('classification', 'classifier')

# GENERATION
GENERATION_EMBEDDING_DIR = os.path.join('/', 'media', 'data4TB', 'sirocchi', 'nerf2vec', 'shape_generation', 'latent_embeddings')
GENERATION_OUT_DIR = os.path.join('task_generation', 'experiments', '{}')  # The placeholder will contain the class index
GENERATION_NERF2VEC_FULL_CKPT_PATH = os.path.join('task_classification', 'train', 'ckpts', '499.pt')
GENERATION_LATENT_GAN_FULL_CKPT_PATH = os.path.join('task_generation', 'experiments', 'nerf2vec_{}', 'latent_gan_ckpts', 'epoch_2000.ckpt')  # The placeholder will contain the class index
