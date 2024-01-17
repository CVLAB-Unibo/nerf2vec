# Task Mapping Network
The mapping network task requires the training of the *inr2vec* framework. Please, refer to THIS page to properly configure your environment.

In order complete this task, it is necessary to execute some operations following a specific order.

Additionally, specifically for this task, a librar

## 1) Create point clouds
This step is necessary for creating the dataset on which *inr2vec* will be trained. It is important to update the variable *shapenet_root* found in *task_mapping_network/cfg/pcd_dataset.yaml*. This variable must point to the root of the *ShapeNet* folder.

Then, execute the following command:
```bash
python task_mapping_network/inr2vec/create_point_clouds_dataset.py 
```

## 2) Train *inr2vec*
## 3) Export *inr2vec* embeddings
## 4) Export *nerf2vec* embeddings
## 5) Train the mapping network
## 6) Export results


