import os
import time
from nerf.loader import _load_renderings
import os
import random
import numpy as np
from PIL import Image
import imageio.v2 as imageio

def _get_nerf_paths(nerfs_root: str):
    
    nerf_paths = []

    for class_name in os.listdir(nerfs_root):

        subject_dirs = os.path.join(nerfs_root, class_name)

        # Sometimes there are hidden files (e.g., when unzipping a file from a Mac)
        if not os.path.isdir(subject_dirs):
            continue
        
        for subject_name in os.listdir(subject_dirs):
            subject_dir = os.path.join(subject_dirs, subject_name)
            nerf_paths.append(subject_dir)
            
    
    return nerf_paths


def get_nerf_name_with_class(file_path):
    # Split the path into individual directories
    directories = os.path.normpath(file_path).split(os.sep)
    # Get the last two directories
    last_two_dirs = directories[-2:]
    # Join the last two directories with an underscore
    file_name = '_'.join(last_two_dirs)
    return file_name

dict_images = {}

nerf_paths = _get_nerf_paths(os.path.join('/', 'media','data4TB','sirocchi','nerf2vec','data','data_TRAINED_A1'))






def get_image_paths(folder_path):
    file_paths = []
    folder_path = os.path.join(folder_path, 'train')
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path[-3:] == 'png':
                file_paths.append(file_path)
    return file_paths


def sample_random_pixels(nerf_paths, num_samples):
    pixel_values = []

    for idx, nerf in enumerate(nerf_paths):

        nerf_images = get_image_paths(nerf)

        for path in nerf_images:
            
            with open(path, 'rb') as f:
                img = Image.open(f)
                img_width, img_height = img.size

                for _ in range(num_samples):
                    x = random.randint(0, img_width - 1)
                    y = random.randint(0, img_height - 1)
                    pixel_pos = (y * img_width + x) * 4  # Assuming RGBA format, 4 bytes per pixel

                    f.seek(pixel_pos)
                    pixel_bytes = f.read(4)
                    pixel_value = np.frombuffer(pixel_bytes, dtype=np.uint8)
                    pixel_values.append(pixel_value)
    return pixel_values

        

# Usage example
start = time.time()
num_samples = 228  # Number of random pixels to sample
pixels = sample_random_pixels(nerf_paths[0:160], num_samples)
end = time.time()
print(end-start)

print()
