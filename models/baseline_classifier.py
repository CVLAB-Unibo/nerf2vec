from collections import OrderedDict
from typing import List

from torch import Tensor, nn
import torch
from torchvision import models

from typing import Any, Dict, List, Tuple
from torch import Tensor

class Resnet50Classifier(nn.Module):
    def __init__(self, num_classes: int, interpolate_features: bool = False) -> None:
        super().__init__()

        self.interpolate_features = interpolate_features
        
        # resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet50 = models.resnet50(weights=None)
        self.feature_extractor = torch.nn.Sequential(OrderedDict([*(list(resnet50.named_children())[:-1])]))
        self.classification_head = nn.Linear(resnet50.fc.in_features, num_classes) 

    def interpolate(self, params: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        new_params = []
        new_labels = []

        for i, p1 in enumerate(params):
            same_class = labels == labels[i]
            num_same_class = torch.sum(same_class)

            if num_same_class > 2:
                indices = torch.where(same_class)[0]
                random_order = torch.randperm(len(indices))
                random_idx = indices[random_order][0]
                p2 = params[random_idx]

                random_uniform = torch.rand(len(p2))
                tsh = torch.rand(())
                from2 = random_uniform >= tsh
                p1_copy = p1.clone()
                p1_copy[from2] = p2[from2]
                new_params.append(p1_copy)
                new_labels.append(labels[i])

        if len(new_params) > 0:
            new_params = torch.stack(new_params)
            final_params = torch.cat([params, new_params], dim=0)
            new_labels = torch.stack(new_labels)
            final_labels = torch.cat([labels, new_labels], dim=0)
        else:
            final_params = params
            final_labels = labels

        return final_params, final_labels

    def forward(self, x: Tensor, y: Tensor = None) -> Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)

        if self.interpolate_features and y is not None:
            x, y = self.interpolate(x, y)

        return self.classification_head(x), y