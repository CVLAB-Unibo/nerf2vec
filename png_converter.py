import glob
import logging
from multiprocessing import Pool
import os
import imageio.v2 as imageio
from tqdm import tqdm


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


def process_png(image_path):
    rgba = imageio.imread(image_path)
    # Save a new copy of the image
    imageio.imwrite(image_path, rgba)


nerf_paths = _get_nerf_paths(os.path.join('data', 'data_TRAINED_A2'))
# nerf_paths = _get_nerf_paths(os.path.join('temp_data7', 'data_1'))


log_file_name = 'processing_log.txt'
dict = {}
if os.path.exists(log_file_name):
    with open(log_file_name, 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            dict[line] = True

    filtered_nerf_paths = []
    for path in nerf_paths:
        if path not in dict:
            filtered_nerf_paths.append(path)
        else:
            print('ALREADY EXISTS!')
else:
    filtered_nerf_paths = nerf_paths


# Configure logging to write to a log file
logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(message)s')


with tqdm(total=len(filtered_nerf_paths)) as pbar:
    for idx, path in enumerate(filtered_nerf_paths):

        train_images_path = os.path.join(path, 'train')
        # Find all PNG files in the current directory
        png_files = glob.glob(os.path.join(train_images_path, '*.png'))

        with Pool() as pool:
            pool.map(process_png, png_files)

        logging.info(path)
        
        pbar.update()

    

"""
for idx, path in enumerate(nerf_paths):
    train_images_path = os.path.join(path, 'train')
    # Find all PNG files in the current directory
    png_files = glob.glob(os.path.join(train_images_path, '*.png'))

    for image in png_files:
        # print(image)
        rgba = imageio.imread(image)
        # Save a new copy of the image
        imageio.imwrite(image, rgba)
    
    print(f'{idx}/{len(nerf_paths)}')
"""    


