import os
import torch


# Assuming you have a tensor of size [3]
original_tensor = torch.tensor([1, 2, 3])

N = 5  # Number of times to repeat the tensor

repeated_tensor = original_tensor.unsqueeze(0).expand(N, -1)

print()