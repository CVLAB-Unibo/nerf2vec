
import json
import math
import os
import random


def cycle_path(nerfs_root):
    

    dict_result = {}

    base_folder = os.path.join('.', os.path.basename(os.path.normpath(nerfs_root)))

    for class_name in os.listdir(nerfs_root):

        class_nerf_paths = []

        subject_dirs = os.path.join(nerfs_root, class_name)

        # Sometimes there are hidden files (e.g., when unzipping a file from a Mac)
        if not os.path.isdir(subject_dirs):
            continue
        
        for subject_name in os.listdir(subject_dirs):
            subject_dir = os.path.join(subject_dirs, subject_name)
            class_nerf_paths.append(subject_dir.replace(nerfs_root, base_folder))        
        dict_result[class_name] = class_nerf_paths

    return dict_result


def create():
    # root_paths = ['data', 'augmented1', 'augmented2']
    root_paths = ['/media/data4TB/sirocchi/nerf2vec/data', '/media/data4TB/sirocchi/nerf2vec/data_1']

    train = []
    validation = []
    test = []

    TRAIN_SPLIT = 80
    VALIDATION_SPLIT = 10
    TEST_SPLIT = 10

    for curr_path in root_paths:
        
        # Get 
        nerfs_dict = cycle_path(curr_path)

        for class_name in nerfs_dict:

            # Get elements related to the current class
            class_elements = nerfs_dict[class_name]
            random.shuffle(class_elements)
            
            n_elements = len(class_elements)

            # Define the dimensions of the splits
            n_test = math.floor(n_elements * TEST_SPLIT / 100)
            n_validation = math.floor(n_elements * VALIDATION_SPLIT / 100)
            n_train = n_elements - n_validation - n_test

            # Make the splits according to their sizes
            train_elements = class_elements[0:n_train]
            validation_elements = class_elements[n_train:n_train+n_validation]
            test_elements = class_elements[n_train+n_validation:]
            
            # Length validation
            total_elements = len(train_elements) + len(validation_elements) + len(test_elements)
            assert total_elements == n_elements and n_test > 0 and n_validation > 0 and n_train > 0, 'Not all elements were properly used.'

            # Elements uniqueness validation
            set1 = set(train_elements)
            set2 = set(validation_elements)
            set3 = set(test_elements)

            no_common_elements = set1.isdisjoint(set2) and set1.isdisjoint(set3) and set2.isdisjoint(set3)
            assert not no_common_elements == n_elements, 'Some elements are shared between splits'

            train = train + train_elements
            validation = validation + validation_elements
            test = test + test_elements

    base_path = 'dataset'
    with open(os.path.join(base_path, 'train.json'), 'w') as file:
        json.dump(train, file)
    with open(os.path.join(base_path, 'validation.json'), 'w') as file:
        json.dump(validation, file)
    with open(os.path.join(base_path, 'test.json'), 'w') as file:
        json.dump(test, file)
        

create()