import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple
import torch
from torch.utils.data import DataLoader, Dataset

from torch import Tensor

from classification.utils import get_class_label, get_mlp_params_as_matrix
from models.beseline_feature_extractor import Resnet50FeatureExtractor

from classification import config_classifier as config

import imageio.v2 as imageio
import numpy as np

import h5py


class BaselineDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        all_files = os.listdir(self.root)
        self.item_paths = sorted([file for file in all_files if file.lower().endswith(".png")])
        # self.item_paths = sorted(self.root.glob("*.png"), key=lambda x: str(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image_name = self.item_paths[index]
        image_path = os.path.join(self.root, image_name)
        rgba_np = imageio.imread(image_path)
        rgba = torch.from_numpy(rgba_np).permute(2, 0, 1).float() / 255.0  # (channels, width, height)
        
        class_label = image_name.split('_')[0]
        class_id = np.array(config.LABELS_TO_IDS[class_label])
        class_id = torch.from_numpy(class_id).long()
        
        return rgba, class_id, self.item_paths[index]


def export_baseline_embeddings(multi_view=True):

    device = 'cuda:0'
    # device = 'cpu'

    if multi_view:
        dset_root = Path(config.RENDERINGS_MULTI_VIEW)
    else:
        dset_root = Path(config.RENDERINGS_SINGLE_VIEW)

    train_dset = BaselineDataset(dset_root, config.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=1, num_workers=0, shuffle=False)

    val_dset = BaselineDataset(dset_root, config.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=0, shuffle=False)

    test_dset = BaselineDataset(dset_root, config.TEST_SPLIT)
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False)

    net = Resnet50FeatureExtractor(interpolate_features=False)
    net = net.to(device)
    net.eval()

    loaders = [train_loader, val_loader, test_loader]
    splits = [config.TRAIN_SPLIT, config.VAL_SPLIT, config.TEST_SPLIT]

    for loader, split in zip(loaders, splits):
        idx = 0

        for batch in loader:
            images, class_ids, data_dirs = batch
            images = images.cuda()

            with torch.no_grad():
                embeddings, _ = net(images)
            
            if multi_view:
                out_root = Path(config.EMBEDDINGS_BASELINE_DIR_MULTI_VIEW)
            else:
                out_root = Path(config.EMBEDDINGS_BASELINE_DIR_SINGLE_VIEW)

            h5_path = out_root / Path(f"{split}") / f"{idx}.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(h5_path, "w") as f:
                f.create_dataset("data_dir", data=data_dirs[0])
                f.create_dataset("embedding", data=embeddings[0].detach().cpu().numpy())
                f.create_dataset("class_id", data=class_ids[0].detach().cpu().numpy())

            idx += 1

            if idx % 5000 == 0:
                print(f'Created {idx} embeddings for {split} split')