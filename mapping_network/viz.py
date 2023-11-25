
import datetime
import open3d as o3d
import wandb
import numpy as np
from typing import Any, Dict, Tuple

def logfn(values: Dict[str, Any]) -> None:
        wandb.log(values, step=0, commit=False)

def config_wandb():
    wandb.init(
        entity='dsr-lab',
        project='point_clouds',
        name=f'run_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
        config={}
        )

# config_wandb()
gt_pcd = o3d.io.read_point_cloud('/media/data7/dsirocchi/nerf2vec/mapping_network/point_clouds/02691156/test/1a9b552befd6306cc8f2d5fe7449af61.ply')
# gt_pcd = np.asarray(gt_pcd.points)

vis = o3d.visualization.Visualizer()
vis.create_window(visible=True) #works for me with False, on some systems needs to be true
vis.add_geometry(gt_pcd)
vis.update_geometry(gt_pcd)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image('out.png')
vis.destroy_window()


# o3d.visualization.draw_geometries([gt_pcd], 'output')
# mesh_logs = {f"mesh": [gt_pcd]}
# logfn(mesh_logs)

# pcd = wandb.Object3D(gt_pcd)
