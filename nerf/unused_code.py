# ########################################
# RAMDOM SELECTION
# ########################################
# The following code has been tested for randomly select elements from the batch, 
# so as to avoid to exceed a predefined MIN_SIZE.
"""
# MIN_SIZE = min([tensor.size(0) for tensor in b_t_starts])
MIN_SIZE = 10000
if radiance_field.training:
    for b in range(batch_size):
        n_elements = b_positions[b].shape[0]
        indices = torch.randperm(n_elements)[:MIN_SIZE]
        indices, _ = torch.sort(indices)  # This is important to avoid problem with volume rendering
        b_positions[b] = b_positions[b][indices]
        b_t_starts[b] = b_t_starts[b][indices]
        b_t_ends[b] = b_t_ends[b][indices]
        b_ray_indices[b] = b_ray_indices[b][indices]

    
b_positions = torch.stack(b_positions, dim=0)
b_t_starts = torch.stack(b_t_starts, dim=0)
b_t_ends = torch.stack(b_t_ends, dim=0)
b_ray_indices = torch.stack(b_ray_indices, dim=0)
"""
# ########################################


# ########################################
# SEQUENTIAL SELECTION
# ########################################
# Truncate
# Similar as before, but with truncation
"""
if radiance_field.training:
    b_positions = torch.stack([tensor[:MIN_SIZE] for tensor in b_positions], dim=0)
    b_t_starts = torch.stack([tensor[:MIN_SIZE] for tensor in b_t_starts], dim=0)
    b_t_ends = torch.stack([tensor[:MIN_SIZE] for tensor in b_t_ends], dim=0)
    b_ray_indices = torch.stack([tensor[:MIN_SIZE] for tensor in b_ray_indices], dim=0)
else:
    b_t_starts = torch.stack(b_t_starts, dim=0)
    b_t_ends = torch.stack(b_t_ends, dim=0)
    b_ray_indices = torch.stack(b_ray_indices, dim=0)
    b_positions = torch.stack(b_positions, dim=0)
"""

# Others approaches tested to cope with padding
'''
# Repeat-last-element padding
b_position_last_row = b_positions[batch_idx][-1:].repeat(padding_size, 1)
b_positions[batch_idx] = torch.cat((b_positions[batch_idx], b_position_last_row), dim=0)

b_t_starts_last_row = b_t_starts[0][-1:].repeat(padding_size, 1)
b_t_starts[batch_idx] = torch.cat((b_t_starts[batch_idx], b_t_starts_last_row), dim=0)

b_t_ends_last_row = b_t_ends[0][-1:].repeat(padding_size, 1)
b_t_ends[batch_idx] = torch.cat((b_t_ends[batch_idx], b_t_ends_last_row), dim=0)

b_ray_indices_last_row = b_ray_indices[0][-1:].repeat(padding_size)
b_ray_indices[batch_idx] = torch.cat((b_ray_indices[batch_idx], b_ray_indices_last_row), dim=0)
'''

'''
# Compute indices
tensor_size = b_positions[batch_idx].size(0)
padding_indices = torch.randint(tensor_size, (padding_size,))

# Extend tensors
padded_positions = b_positions[batch_idx][padding_indices][:padding_size]
b_positions[batch_idx] = torch.cat((b_positions[batch_idx], padded_positions), dim=0)

padded_t_starts = b_t_starts[batch_idx][padding_indices][:padding_size]
b_t_starts[batch_idx] = torch.cat((b_t_starts[batch_idx], padded_t_starts), dim=0)

padded_t_ends = b_t_ends[batch_idx][padding_indices][:padding_size]
b_t_ends[batch_idx] = torch.cat((b_t_ends[batch_idx], padded_t_ends), dim=0)

padded_ray_indices = b_ray_indices[batch_idx][padding_indices][:padding_size]
b_ray_indices[batch_idx] = torch.cat((b_ray_indices[batch_idx], padded_ray_indices), dim=0)

'''


# ################################################################################
# RENDER VISIBILITY
# ################################################################################
# Method originally used by NerfAcc so as to remove elements that are classified
# as non-visible. This is not useful for nerf2vec, because we fix the number of 
# elements in the batches. Moreover, this add complexity to the model that seems
# to be not relevant.
"""
#with torch.no_grad():
'''
    The ray_marching internally calls sigma_fn that, for the moment, has not be used.
    This because:
    - it is an optimization that, hopefully, can be skipped. By testing some models, it seems that they can be trained also without it.
    - it requires the value 'packed_info', which is returned from the ray marching algorithm, and it is not exposed to external callers.
    See the ray_marching.py file, at the end it uses this variable.

    Moreover, this avoids an additional call to the model (i.e., the nerf2vec decoder)
'''


_, curr_sigmas = radiance_field(embeddings, b_positions)


b_t_starts_visible = []
b_t_ends_visible = []
b_ray_indices_visible = []

# Compute visibility of the samples, and filter out invisible samples
for batch_idx in range(batch_size): 
    sigmas = curr_sigmas[batch_idx]

    alphas = 1.0 - torch.exp(-sigmas * (b_t_ends[batch_idx] - b_t_starts[batch_idx]))
    masks = render_visibility(
        alphas,
        ray_indices=b_ray_indices[batch_idx],
        packed_info=None,
        early_stop_eps=1e-4,
        alpha_thre=alpha_thre,
        n_rays=chunk_rays.origins.shape[1]
    )
    
    b_ray_indices_visible.append(b_ray_indices[batch_idx][masks])
    b_t_starts_visible.append(b_t_starts[batch_idx][masks])
    b_t_ends_visible.append(b_t_ends[batch_idx][masks])

MAX_SIZE = 10000  # Desired maximum size
#MIN_SIZE = ([tensor.size(0) for tensor in b_t_starts])

if radiance_field.training:
    b_t_starts = [None]*batch_size
    b_t_ends = [None]*batch_size
    b_ray_indices = [None]*batch_size

    for batch_idx in range(batch_size):
        if b_t_starts_visible[batch_idx].size(0) < MAX_SIZE:
            
            padding_size = MAX_SIZE - b_t_starts_visible[batch_idx].size(0)

            b_t_starts[batch_idx] = F.pad(b_t_starts_visible[batch_idx], pad=(0, 0, 0, padding_size))
            b_t_ends[batch_idx] = F.pad(b_t_ends_visible[batch_idx], pad=(0, 0, 0, padding_size))
            b_ray_indices[batch_idx] = F.pad(b_ray_indices_visible[batch_idx], pad=(0, padding_size))
            
            
            '''
            padding_indices = torch.randperm(b_positions[batch_idx].size(0))

            b_positions[batch_idx] = F.pad(b_positions[batch_idx], pad=(0, 0, 0, padding_size))
            b_positions[batch_idx][-padding_size:] = b_positions[batch_idx][padding_indices][:padding_size]

            
            b_t_starts[batch_idx] = F.pad(b_t_starts[batch_idx], pad=(0, 0, 0, padding_size))
            b_t_starts[batch_idx][-padding_size:] = b_t_starts[batch_idx][padding_indices][:padding_size]


            
            b_t_ends[batch_idx] = F.pad(b_t_ends[batch_idx], pad=(0, 0, 0, padding_size))
            b_t_ends[batch_idx][-padding_size:] = b_t_ends[batch_idx][padding_indices][:padding_size]

            
            b_ray_indices[batch_idx] = F.pad(b_ray_indices[batch_idx], pad=(0, padding_size))
            b_ray_indices[batch_idx][-padding_size:] = b_ray_indices[batch_idx][padding_indices][:padding_size]
            '''

        else:
            b_t_starts[batch_idx] = b_t_starts_visible[batch_idx][:MAX_SIZE]
            b_t_ends[batch_idx] = b_t_ends_visible[batch_idx][:MAX_SIZE]
            b_ray_indices[batch_idx] = b_ray_indices_visible[batch_idx][:MAX_SIZE]
else:
    b_t_starts = b_t_starts_visible
    b_t_ends = b_t_ends_visible
    b_ray_indices = b_ray_indices_visible
        
b_t_starts = torch.stack(b_t_starts, dim=0)
b_t_ends = torch.stack(b_t_ends, dim=0)
b_ray_indices = torch.stack(b_ray_indices, dim=0)
b_positions = []


# Compute positions
for batch_idx in range(batch_size):
    
    batch_idx_indices = b_ray_indices[batch_idx]

    t_origins = chunk_rays.origins[batch_idx][batch_idx_indices]
    t_dirs = chunk_rays.viewdirs[batch_idx][batch_idx_indices]
    positions = t_origins + t_dirs * (b_t_starts[batch_idx] + b_t_ends[batch_idx]) / 2.0
    
    if radiance_field.training:
        if positions.size(0) < MAX_SIZE:
            positions = F.pad(positions, pad=(0, 0, 0, padding_size))
        else:
            positions = positions[:MAX_SIZE]

    b_positions.append(positions)

b_positions = torch.stack(b_positions, dim=0)
"""   