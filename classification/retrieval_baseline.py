
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import imageio.v2 as imageio
import numpy as np
import h5py
import os

from sklearn.neighbors import KDTree

from classification import config_classifier as config

class BaselineEmbeddingDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        with h5py.File(self.item_paths[index], "r") as f:
            embedding = np.array(f.get("embedding"))
            embedding = torch.from_numpy(embedding)
            class_id = np.array(f.get("class_id"))
            class_id = torch.from_numpy(class_id).long()
            data_dir = f.get('data_dir')
            
        return embedding, class_id, data_dir

@torch.no_grad()
def draw_images(data_dirs, multi_view, renderings_base_path):

    for idx, data_dir in enumerate(data_dirs):
        plots_path = 'retrieval_baseline_plots_single_view' if not multi_view else 'retrieval_baseline_plots_multi_view'
        
        image = imageio.imread(os.path.join(renderings_base_path, data_dir))
        img_name = data_dir.replace('.png', '')
        imageio.imwrite(os.path.join(plots_path, img_name), image)

@torch.no_grad()
def get_recalls(multi_view: bool, 
                gallery: Tensor, 
                labels_gallery: Tensor, data_dirs: [str], 
                renderings_base_path: str, 
                kk: List[int]) -> Dict[int, float]:
    
    max_nn = max(kk)
    recalls = {idx: 0.0 for idx in kk}
    targets = labels_gallery.cpu().numpy()
    gallery = gallery.cpu().numpy()
    tree = KDTree(gallery)

    dic_renderings = defaultdict(int)

    for query, label_query, data_dir in zip(gallery, targets, data_dirs):
        with torch.no_grad():

            if not multi_view:
                query = np.expand_dims(query, 0)
                _, indices_matched = tree.query(query, k=max_nn + 1)
                indices_matched = indices_matched[0]

            else:
                
                indices_matched = []
                while len(indices_matched) != max_nn+1:

                    query = np.expand_dims(query, 0)
                    _, indices_matched = tree.query(query, k=max_nn + 1)
                    indices_matched = indices_matched[0]
                    
                    indices_matched_processed = []
                    indices_matched_processed.append(indices_matched[0])  # TODO: check equality between query and first element

                    query_nerf_id = f"{data_dir.split('_')[0]}_{data_dir.split('_')[1]}"                    
                    object_matches = {}
                    object_matches[query_nerf_id] = True
                    
                    for i in range(1, len(indices_matched)):
                        curr_idx = indices_matched[i]
                        curr_data_dir = data_dirs[curr_idx]

                        curr_nerf_id = f"{curr_data_dir.split('_')[0]}_{curr_data_dir.split('_')[1]}"
                        
                        # With this condition, it is impossible that an already matched image is considered more than once.
                        # This is something that could happen, because there are multiple images of the same object, although
                        # they are taken from different perspectives.
                        if curr_nerf_id not in object_matches:
                            indices_matched_processed.append(curr_idx)
                            object_matches[curr_nerf_id] = True
                
                    indices_matched = indices_matched_processed

            # Draw the query and the first 3 neighbours
            if dic_renderings[label_query] < 10:
                draw_images(gallery[indices_matched], multi_view, renderings_base_path)
                dic_renderings[label_query] += 1
                print(dic_renderings)
            
            for k in kk:
                indices_matched_temp = indices_matched[1 : k + 1]
                classes_matched = targets[indices_matched_temp]
                recalls[k] += np.count_nonzero(classes_matched == label_query) > 0

    for key, value in recalls.items():
        recalls[key] = value / (1.0 * len(gallery))

    return recalls

@torch.no_grad()
def do_retrieval(device='cuda:0', multi_view=False):

    if multi_view:
        dset_root = Path(config.EMBEDDINGS_BASELINE_DIR_MULTI_VIEW)
    else:
        dset_root = Path(config.EMBEDDINGS_BASELINE_DIR_SINGLE_VIEW)

    split = config.TEST_SPLIT
    dset = BaselineEmbeddingDataset(dset_root, split)

    renderings_base_path = os.path.join(dset_root, split)

    embeddings = []
    labels = []
    data_dirs = []

    for i in range(len(dset)):
        embedding, label, dir = dset[i]
        embeddings.append(embedding)
        labels.append(label)    
        data_dirs.append(dir)
    
    embeddings = torch.stack(embeddings)
    labels = torch.stack(labels)

    recalls = get_recalls(multi_view, embeddings, labels, data_dirs, renderings_base_path, [1, 5, 10])
    for key, value in recalls.items():
        print(f"Recall@{key} : {100. * value:.2f}%")