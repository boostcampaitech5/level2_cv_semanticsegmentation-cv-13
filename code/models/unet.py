import torch 
from torch import nn 
from torchvision import models 
import os 
import numpy as np 
import segmentation_models_pytorch as smp


class EfficientUNet(nn.Module):
    def __init__(self, num_class=29): 
        super().__init__()
        self.model = smp.Unet(
            encoder_name="efficientnet-b7", 
            encoder_weights="imagenet",    
            in_channels=3,                
            classes=num_class,                     
        )
        
    def forward(self, x): 
        return self.model(x) 