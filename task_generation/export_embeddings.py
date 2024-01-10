import sys
from pathlib import Path

import h5py
import numpy as np

from task_generation import config


def export_embeddings() -> None:
    
    nerf_embeddings_root = Path(config.NERF_EMBEDDING_PATH)
    out_root = Path(config.GENERATION_EMBEDDING_PATH)
    out_root.mkdir(parents=True, exist_ok=True)
   
    num_classes = 13

    embeddings_paths = list(nerf_embeddings_root.glob("*.h5"))

    embeddings = {}
    for cls in range(num_classes):
        embeddings[cls] = []

    print('Extracting embeddings...')
    for idx, path in enumerate(embeddings_paths):
        with h5py.File(path, "r") as f:
            embedding = np.array(f.get("embedding"))
            class_id = np.array(f.get("class_id")).item()
            embeddings[class_id].append(embedding)
        
        if idx % 5000 == 0:
            print(f'\t {idx}/{len(embeddings_paths)}')

    for class_id in range(num_classes):
        print(f'Processing class: {class_id}')
        if class_id == 2:
            print()
        path_out = out_root / f"embeddings_{class_id}.npz"
        stacked_embeddings = np.stack(embeddings[class_id])
        np.savez_compressed(path_out, embeddings=stacked_embeddings)

