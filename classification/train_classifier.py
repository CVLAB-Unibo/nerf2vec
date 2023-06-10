import copy
import logging
import os
import sys

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

# import h5py


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
    
    class InrEmbeddingClassifier:
        def __init__(self) -> None:
            dset_root = Path(config.EMBEDDINGS_PATH)
            train_dset = InrEmbeddingDataset(dset_root, config.TRAIN_SPLIT)



