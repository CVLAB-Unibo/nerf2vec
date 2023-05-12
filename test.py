import torch
from nerf.utils import Rays, namedtuple_map
from typing import Callable, Tuple
from torch import Tensor, nn

N_BATCHES = 2

test = torch.tensor([[[1,2,3]]])
test2 = torch.tensor([[[6,7,8]]])



print()