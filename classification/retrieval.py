from collections import defaultdict
import copy
import logging
import math
import os
import sys
import time
import uuid
from classification.export_renderings import get_rays
from classification.utils import get_mlp_params_as_matrix
import h5py
import datetime

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from classification import config_classifier as config
from models.fc_classifier import FcClassifier
from torchmetrics.classification.accuracy import Accuracy

from sklearn.neighbors import KDTree

from models.idecoder import ImplicitDecoder
from nerf.utils import Rays, render_image

import imageio.v2 as imageio
from torch.cuda.amp import autocast




class InrEmbeddingDataset(Dataset):
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
            
        return embedding, class_id

@torch.no_grad()
def draw_images(decoder, embeddings, device='cuda:0'):

    scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / config.GRID_CONFIG_N_SAMPLES
    ).item()

    rays = get_rays(device)
    color_bkgd = torch.ones((1,3), device=device)  # WHITE BACKGROUND!

    img_name = str(uuid.uuid4())

    for idx, emb in enumerate(embeddings):
        emb = torch.tensor(emb, device=device, dtype=torch.float32)
        emb = emb.unsqueeze(dim=0)
        with autocast():
            rgb_A, alpha, b, c, _, _ = render_image(
                            radiance_field=decoder,
                            embeddings=emb,
                            occupancy_grid=None,
                            rays=Rays(origins=rays.origins.unsqueeze(dim=0), viewdirs=rays.viewdirs.unsqueeze(dim=0)),
                            scene_aabb=scene_aabb,
                            render_step_size=render_step_size,
                            render_bkgd=color_bkgd,
                            grid_weights=None
            )

        plots_path = 'retrieval_plots'
        imageio.imwrite(
            os.path.join(plots_path, f'{img_name}_{idx}.png'),
            (rgb_A.squeeze(dim=0).cpu().detach().numpy() * 255).astype(np.uint8)
        )


@torch.no_grad()
def get_recalls(gallery: Tensor, labels_gallery: Tensor, kk: List[int], decoder) -> Dict[int, float]:
    max_nn = max(kk)
    recalls = {idx: 0.0 for idx in kk}
    targets = labels_gallery.cpu().numpy()
    gallery = gallery.cpu().numpy()
    tree = KDTree(gallery)

    # shuffling_indices = torch.randperm(gallery.shape[0])
    # gallery = gallery[shuffling_indices]
    # labels_gallery = labels_gallery[shuffling_indices]

    dic_renderings = defaultdict(int)

    for query, label_query in zip(gallery, targets):
        with torch.no_grad():
            query = np.expand_dims(query, 0)
            _, indices_matched = tree.query(query, k=max_nn + 1)
            indices_matched = indices_matched[0]

            # Draw the query and the first 3 neighbours
            if dic_renderings[label_query] < 10:
                # draw_images(decoder, gallery[indices_matched[:4]])
                draw_images(decoder, gallery[indices_matched])
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
def do_retrieval(device='cuda:0'):

    # Init nerf2vec 
    decoder = ImplicitDecoder(
            embed_dim=config.ENCODER_EMBEDDING_DIM,
            in_dim=config.DECODER_INPUT_DIM,
            hidden_dim=config.DECODER_HIDDEN_DIM,
            num_hidden_layers_before_skip=config.DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP,
            num_hidden_layers_after_skip=config.DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP,
            out_dim=config.DECODER_OUT_DIM,
            encoding_conf=config.INSTANT_NGP_ENCODING_CONF,
            aabb=torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
        )
    decoder.eval()
    decoder = decoder.to(device)

    ckpt_path = os.path.join('classification','train','ckpts','499.pt')
    print(f'loading weights: {ckpt_path}')
    ckpt = torch.load(ckpt_path)
    decoder.load_state_dict(ckpt["decoder"])
    


    dset_root = Path(config.EMBEDDINGS_DIR)
    dset = InrEmbeddingDataset(dset_root, config.TEST_SPLIT)

    embeddings = []
    labels = []

    for i in range(len(dset)):
        embedding, label = dset[i]
        embeddings.append(embedding)
        labels.append(label)    
    
    embeddings = torch.stack(embeddings)
    labels = torch.stack(labels)

    recalls = get_recalls(embeddings, labels, [1, 5, 10], decoder)
    for key, value in recalls.items():
        print(f"Recall@{key} : {100. * value:.2f}%")