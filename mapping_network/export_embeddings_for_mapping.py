import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple
import torch
from torch.utils.data import DataLoader, Dataset

from torch import Tensor

from classification.utils import get_class_label, get_mlp_params_as_matrix
from models.encoder import Encoder

from classification import config_classifier as config

import h5py

LABELS_TO_IDS_FOR_MAPPING = {
    "02691156": 0,   # airplane
    "02828884": 1,   # bench
    "02933112": 2,   # cabinet
    "02958343": 3,   # car
    '02992529': 4,   # tablet (delete?)
    "03001627": 5,   # chair
    "03211117": 6,   # display
    "03636649": 7,   # lamp
    "03691459": 8,   # speaker
    "03948459": 9,   # gun (delete?)
    "04090263": 10,  # rifle
    "04256520": 11,  # sofa
    "04379243": 12,  # table
    "04401088": 13,  # phone
    "04530566": 14   # watercraft
}

class InrDataset(Dataset):
    def __init__(self, split_json: str, device: str, nerf_weights_file_name: str) -> None:
        super().__init__()

        with open(split_json) as file:
            self.nerf_paths = json.load(file)
            self.nerf_paths = sorted(self.nerf_paths)
        
        # self.nerf_paths = self._get_nerf_paths('data\\data_TRAINED')
        assert isinstance(self.nerf_paths, list), 'The json file provided is not a list.'

        self.device = device
        self.nerf_weights_file_name = nerf_weights_file_name

    def __len__(self) -> int:
        return len(self.nerf_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:

        data_dir = self.nerf_paths[index]
        weights_file_path = os.path.join(data_dir, self.nerf_weights_file_name)

        # TODO: TEMP
        class_label = get_class_label(weights_file_path)
        class_id = LABELS_TO_IDS_FOR_MAPPING[get_class_label(weights_file_path)] if class_label != -1 else class_label

        matrix = torch.load(weights_file_path, map_location=torch.device(self.device))
        matrix = get_mlp_params_as_matrix(matrix['mlp_base.params'])

        return matrix, class_id, data_dir

def load_nerf2vec_checkpoint():
    ckpts_path = Path(os.path.join('classification', 'train', 'ckpts'))
    ckpt_paths = [p for p in ckpts_path.glob("*.pt") if "best" not in p.name]
    error_msg = "Expected only one ckpt apart from best, found none or too many."
    assert len(ckpt_paths) == 1, error_msg
    ckpt_path = ckpt_paths[0]
    print(f'loading path: {ckpt_path}')
    ckpt = torch.load(ckpt_path)
    
    return ckpt

def export_embeddings_for_mapping():

    device = 'cuda:3'

    train_dset_json = os.path.abspath(os.path.join('data', 'train.json'))
    train_dset = InrDataset(train_dset_json, device='cpu', nerf_weights_file_name=config.NERF_WEIGHTS_FILE_NAME)
    train_loader = DataLoader(train_dset, batch_size=1, num_workers=0, shuffle=False)

    """
    val_dset_json = os.path.abspath(os.path.join('data', 'validation.json'))
    val_dset = InrDataset(val_dset_json, device='cpu', nerf_weights_file_name=config.NERF_WEIGHTS_FILE_NAME)
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=0, shuffle=False)

    test_dset_json = os.path.abspath(os.path.join('data', 'test.json'))
    test_dset = InrDataset(test_dset_json, device='cpu', nerf_weights_file_name=config.NERF_WEIGHTS_FILE_NAME)
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False)
    """

    encoder = Encoder(
            config.MLP_UNITS,
            config.ENCODER_HIDDEN_DIM,
            config.ENCODER_EMBEDDING_DIM
            )
    encoder = encoder.to(device)
    ckpt = load_nerf2vec_checkpoint()
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    
    loaders = [train_loader]  # , val_loader, test_loader]
    splits = [config.TRAIN_SPLIT]  #, config.VAL_SPLIT, config.TEST_SPLIT]


    for loader, split in zip(loaders, splits):
        idx = 0

        for batch in loader:
            matrices, class_ids, data_dirs = batch
            matrices = matrices.cuda()

            with torch.no_grad():
                embeddings = encoder(matrices)

            out_root = Path('/media/data7/dsirocchi/nerf2vec/mapping_network/nerf_embeddings')
            h5_path = out_root / Path(f"{split}") / f"{idx}.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(h5_path, "w") as f:
                # print(f'dir: {data_dirs[0]}, class: {class_ids[0]}')

                p = Path(data_dirs[0])
                uuid = p.parts[-1].replace('.ply','')

                f.create_dataset("data_dir", data=data_dirs[0])
                f.create_dataset("embedding", data=embeddings[0].detach().cpu().numpy())
                f.create_dataset("class_id", data=class_ids[0].detach().cpu().numpy())
                f.create_dataset("uuid", data=uuid)

            idx += 1

            if idx % 5000 == 0:
                print(f'Created {idx} embeddings for {split} split')