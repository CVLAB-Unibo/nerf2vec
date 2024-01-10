import os

"""
# ############################################################
# PATHS AND REFERENCES USED BY DIFFERENT MODULES
# ############################################################
"""
TRAIN_DSET_JSON = os.path.abspath(os.path.join('data', 'train.json'))
VAL_DSET_JSON = os.path.abspath(os.path.join('data', 'validation.json'))  
TEST_DSET_JSON = os.path.abspath(os.path.join('data', 'test.json'))  

NERF2VEC_CKPTS_PATH = os.path.join('nerf2vec', 'train', 'ckpts')
NERF2VEC_ALL_CKPTS_PATH = os.path.join('nerf2vec', 'train', 'all_ckpts')

NERF2VEC_EMBEDDINGS_DIR = os.path.join('task_classification', 'embeddings') 

CLASSIFICATION_OUTPUT_DIR = os.path.join('classification', 'classifier')

GENERATION_EMBEDDING_DIR = '/media/data4TB/sirocchi/nerf2vec/shape_generation/latent_embeddings'