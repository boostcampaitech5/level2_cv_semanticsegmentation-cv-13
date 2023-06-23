import torch 
from torch import nn 
from torchvision import models 
import torch.nn.functional as F 
import os 
import numpy as np 


class DeeplabV3_Resnet50(nn.Module): 
    def __init__(self, num_classes=29): 
        super().__init__() 
        self.deeplab_v3 = models.segmentation.deeplabv3_resnet50(pretrained=True) 
    
        self.deeplab_v3.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1)) 
    
    def forward(self, x): 
        return self.deeplab_v3(x)   


class DeeplabV3_Resnet101(nn.Module): 
    def __init__(self, num_classes=29): 
        super().__init__() 
        self.deeplab_v3 = models.segmentation.deeplabv3_resnet101(pretrained=True) 
    
        self.deeplab_v3.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1)) 
    
    def forward(self, x): 
        return self.deeplab_v3(x)  