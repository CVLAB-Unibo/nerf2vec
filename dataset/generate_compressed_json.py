'''
This simple module is responsible to create a compressed version of the transform jsons
output by the rendering scripts.
This has been done for optimizing the I/O operation, rougly halving the size of these
jsons.

In the future, rather than executing this script, would be better to update directly
the rendering scripts.
'''
import json
import os
import re
import time
from classification import config
from nerf.intant_ngp import NGPradianceField
from nerf.loader import _load_renderings
import os
import random
import numpy as np
from PIL import Image
import imageio.v2 as imageio


def save_json(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, separators=(',', ':'))

nerf_roots = [
    os.path.join('/', 'media', 'data4TB', 'sirocchi', 'nerf2vec', 'data', 'data_TRAINED'),
    os.path.join('/', 'media', 'data4TB', 'sirocchi', 'nerf2vec', 'data', 'data_TRAINED_A1'),
    os.path.join('/', 'media', 'data4TB', 'sirocchi', 'nerf2vec', 'data', 'data_TRAINED_A2')
]

completed = 0
for nerf_root in nerf_roots:
    for class_name in os.listdir(nerf_root):

            subject_dirs = os.path.join(nerf_root, class_name)

            # Sometimes there are hidden files (e.g., when unzipping a file from a Mac)
            if not os.path.isdir(subject_dirs):
                continue
            
            for subject_name in os.listdir(subject_dirs):
                
                subject_dir = os.path.join(subject_dirs, subject_name)
                transforms_json = os.path.join(subject_dir, 'transforms_train.json')
                compressed_transforms_json = os.path.join(subject_dir, 'transforms_train_compressed.json')

                if os.path.exists(compressed_transforms_json):
                    print('ALREADY EXISTS!')
                else:
                    with open(transforms_json, "r") as fp:
                        meta = json.load(fp)
                    save_json(compressed_transforms_json, meta)
                completed+=1
                print(completed)

