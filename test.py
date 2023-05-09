import torch
from nerf.utils import Rays, namedtuple_map


origins = torch.rand(5, 20, 3) # batch_size, n_rays, coordinate
viewdirs = torch.rand(5, 20, 3) # batch_size, n_rays, coordinate

rays = Rays(origins=origins, viewdirs=viewdirs)
chunk = torch.iinfo(torch.int32).max
chunk = 5

rays_shape = rays.origins.shape
batch_size, num_rays, coordinates = rays_shape


b_ray_indices = torch.zeros(batch_size, 1, dtype=torch.int32)
b_t_starts = torch.zeros(batch_size, chunk, 1)
b_t_ends = torch.zeros(batch_size, chunk, 1)

b_ray_indices[0] = 1
b_ray_indices[1] = 2
b_ray_indices[2] = 3
b_ray_indices[3] = 4
b_ray_indices[4] = 5

for i in range(0, num_rays, chunk):
    chunk_rays = namedtuple_map(lambda r: r[:, i : i + chunk], rays)
    
    print()

b_t_origins = chunk_rays.origins[:, b_ray_indices]

print()