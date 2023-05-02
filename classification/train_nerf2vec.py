from torch.utils.data import DataLoader, Dataset

from pathlib import Path
from random import randint
from typing import Any, Dict, Tuple

from classification.nerf_loader import NeRFLoader

class NeRFDataset(Dataset):
    def __init__(self, nerfs_root: Path, sample_sd: Dict[str, Any]) -> None:
        super().__init__()
        
        self.nerfs_root = nerfs_root

        # TODO: cycle all the files to correctly set this number
        self.nerf_paths = 10000
    
    def __len__(self) -> int:
        return len(self.nerf_paths)

    def __getitem__(self, index) -> Any:
        dataset_kwargs = {}

        nerf_loader = NeRFLoader(
            subject_id=self.subject_id,
            root_fp=self.nerfs_root,
            num_rays=None,
            **dataset_kwargs,)
        
