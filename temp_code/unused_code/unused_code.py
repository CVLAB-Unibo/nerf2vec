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


# ################################################################################
# CREATE VIDEO DURING TRAINING
# ################################################################################
"""
if self.global_step % 300 == 0:
    end = time.time()
    print(f'{self.global_step} - "train/loss": {loss.item()} - elapsed: {end-start}')

    self.encoder.eval()
    self.decoder.eval()
    with torch.no_grad():
        
        for i in range(config.BATCH_SIZE):
            idx_to_draw = i
            rgb, acc, depth, n_rendering_samples = render_image(
                self.decoder,
                embeddings[idx_to_draw].unsqueeze(dim=0),
                self.occupancy_grid,#[grids[idx_to_draw]],
                Rays(origins=test_rays.origins[idx_to_draw].unsqueeze(dim=0), viewdirs=test_rays.viewdirs[idx_to_draw].unsqueeze(dim=0)),
                self.scene_aabb,
                # rendering options
                near_plane=None,
                far_plane=None,
                render_step_size=self.render_step_size,
                render_bkgd=test_render_bkgds,
                cone_angle=0.0,
                alpha_thre=0.0,
                grid_weights=[grid_weights_path[i]]
            )

            imageio.imwrite(
                os.path.join('temp_sanity_check', 'images', f'{i}_rgb_test_{self.global_step}.png'),
                (rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8),
            )
        
        # ####################
        # EVAL
        # ####################
        psnrs = []
        psnrs_avg = []
        idx_to_draw = random.randrange(0, config.BATCH_SIZE)
        test_dataset_kwargs = {}
        test_nerf_loader = NeRFLoader(
            data_dir=nerf_weights_path[idx_to_draw],
            num_rays=config.NUM_RAYS,
            device=self.device,
            **test_dataset_kwargs)
        test_nerf_loader.training = False
        
        
        for i in tqdm.tqdm(range(len(test_nerf_loader))):
            data = test_nerf_loader[i]
            render_bkgd = data["color_bkgd"]
            test_rays_2 = data["rays"]
            
            pixels = data["pixels"].unsqueeze(dim=0)
            
            
            rgb, acc, depth, n_rendering_samples = render_image(
                self.decoder,
                embeddings[idx_to_draw].unsqueeze(dim=0),
                self.occupancy_grid,
                Rays(origins=test_rays_2.origins.unsqueeze(dim=0), viewdirs=test_rays_2.viewdirs.unsqueeze(dim=0)),
                self.scene_aabb,
                # rendering options
                near_plane=None,
                far_plane=None,
                render_step_size=self.render_step_size,
                render_bkgd=test_render_bkgds,
                cone_angle=0.0,
                alpha_thre=0.0,
                grid_weights=[grid_weights_path[idx_to_draw]]
            )

            if i == 0:
                imageio.imwrite(
                    os.path.join('temp_sanity_check', 'images', f'{i}_rgb_test_{self.global_step}.png'),
                    (rgb.cpu().detach().numpy()[0] * 255).astype(np.uint8),
                )

            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs.append(psnr.item())
        
        psnr_avg = sum(psnrs) / len(psnrs)
        print(f'PSNR: {psnr_avg}')
        psnrs_avg.append(psnr_avg)

        
        
        '''
        if self.global_step == 9900:
            create_video(
                    448, 
                    448, 
                    self.device, 
                    245.0, 
                    self.decoder, 
                    occupancy_grid, 
                    scene_aabb,
                    None, 
                    None, 
                    render_step_size,
                    render_bkgd=test_render_bkgds[0],
                    cone_angle=0.0,
                    alpha_thre=alpha_thre,
                    # test options
                    path=os.path.join('temp_sanity_check', f'video_{self.global_step}.mp4'),
                    embeddings=embeddings[idx_to_draw].unsqueeze(dim=0),
                    grid_weights=[grid_weights_path[idx_to_draw]]
                )
        '''
        
        
    print(psnrs_avg)
    
    self.encoder.train()
    self.decoder.train()
"""

# ################################################################################
# RETRIEVE NERF LIST FROM FILE SYSTEM
# ################################################################################
"""
def _get_nerf_paths(self, nerfs_root: str):
    
    nerf_paths = []

    for class_name in os.listdir(nerfs_root):

        subject_dirs = os.path.join(nerfs_root, class_name)

        # Sometimes there are hidden files (e.g., when unzipping a file from a Mac)
        if not os.path.isdir(subject_dirs):
            continue
        
        for subject_name in os.listdir(subject_dirs):
            subject_dir = os.path.join(subject_dirs, subject_name)
            nerf_paths.append(subject_dir)
    
    return nerf_paths
"""