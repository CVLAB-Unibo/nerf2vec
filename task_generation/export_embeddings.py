import os
import sys
import settings
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from pathlib import Path

import h5py
import numpy as np


def export_embeddings() -> None:
    
    split = 'train'  # TODO: document this variable
    nerf_embeddings_root = Path(settings.NERF2VEC_EMBEDDINGS_DIR) / split  # TODO: TEST THIS!
    out_root = Path(settings.GENERATION_EMBEDDING_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
   
    num_classes = 13  # TODO: get this class from configuration file!

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

def main() -> None:
    export_embeddings()

if __name__ == "__main__":
    main()