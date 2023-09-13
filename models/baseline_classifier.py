from collections import OrderedDict
from typing import List

from torch import Tensor, nn
import torch
from torchvision import models


class Resnet50Classifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = torch.nn.Sequential(OrderedDict([*(list(resnet50.named_children())[:-1])]))
        self.classification_head = nn.Linear(resnet50.fc.in_features, num_classes)  #resnet50.fc.in_features


    def forward(self, x: Tensor, y: Tensor = None) -> Tensor:
        x = self.feature_extractor(x)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)

        # TODO: if y is not None, then apply the interpolation here!

        return self.classification_head(x)