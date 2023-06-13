"""
model.py is a file that you use when you load a pre-trained model and use it as it is, or change it.
"""

import torch.nn as nn
import timm
from torchvision import models


class CustomModel(nn.Module):
    def __init__(self, num_classes : int, backbone : str, pretrained : bool):
        super(CustomModel, self).__init__()

        self.backbone = timm.create_model(backbone, num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        x = self.backbone(x)
        return x
    
class SegModel(nn.Module):
    def __init__(self, num_classes : int, backbone : str, pretrained : bool):
        super(SegModel, self).__init__()
        

        self.backbone = models.segmentation.fcn_resnet50(weight=True)
        
        #Input Channel 수 조정
        # self.backbone.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        ###################
        
        self.backbone.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        # print(self.backbone)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    