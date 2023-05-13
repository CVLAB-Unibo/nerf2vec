import math
import os
from nerfacc import ContractionType, OccupancyGrid
import torch
from classification import config
from nerf.intant_ngp import NGPradianceField


class MyModule(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModule, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        
    def forward(self, x):
        out = self.linear(x)
        return torch.rand(4)

def generate_occupancy_grid2(
        device, 
        weights_path, 
        nerf_dict, 
        aabb, 
        n_iterations, 
        n_warmups,
        radiance_field,
        occupancy_grid,
        render_step_size):

        with torch.no_grad():
            for i in range(n_iterations):
                def occ_eval_fn(x):
                    step_size = render_step_size
                    _ , density = radiance_field._query_density_and_rgb(x, None)
                    return density * step_size

                # update occupancy grid
                occupancy_grid._update(
                    step=i,
                    occ_eval_fn=occ_eval_fn,
                    occ_thre=1e-2,
                    ema_decay=0.95,
                    warmup_steps=n_warmups
                )

        return occupancy_grid


def generate():
    with open('logs.txt', 'a') as f:
        f.write('generating!\n')
    
    device = 'cpu'
    
    """
    radiance_field = NGPradianceField(
                **config.INSTANT_NGP_MLP_CONF
            ).to(device)
    """
    radiance_field = MyModule(3,4)
    weights_path = os.path.join('data', '02691156', '1a04e3eab45ca15dd86060f189eb133', 'bb07_steps3000_encodingFrequency_mlpFullyFusedMLP_activationReLU_hiddenLayers3_units64_encodingSize24.pth')
    matrix = torch.load(weights_path)
    #radiance_field.load_state_dict(matrix)
    radiance_field.eval()

    # Create the OccupancyGrid
    render_n_samples = 1024
    grid_resolution = 128
    
    contraction_type = ContractionType.AABB
    scene_aabb = torch.tensor(config.AABB, dtype=torch.float32, device=device)
    near_plane = None
    far_plane = None
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / render_n_samples
    ).item()
    alpha_thre = 0.0

    occupancy_grid = OccupancyGrid(
            roi_aabb=config.AABB,
            resolution=grid_resolution,
            contraction_type=contraction_type,
        ).to(device)
    occupancy_grid.eval()

    for i in range(100):
        grid = generate_occupancy_grid2(device, 
                    os.path.join('data', '02691156', '1a04e3eab45ca15dd86060f189eb133', 'bb07_steps3000_encodingFrequency_mlpFullyFusedMLP_activationReLU_hiddenLayers3_units64_encodingSize24.pth'), 
                    config.INSTANT_NGP_MLP_CONF, 
                    config.AABB, 
                    config.OCCUPANCY_GRID_RECONSTRUCTION_ITERATIONS, 
                    config.OCCUPANCY_GRID_WARMUP_ITERATIONS,
                    radiance_field,
                    occupancy_grid,
                    render_step_size)

        torch.save(grid.state_dict(), os.path.join('grids', f'grid_{i}.pth'))

generate()        