"""
model.py is a file that you use when you load a pre-trained model and use it as it is, or change it.
"""
import torch
import torch.nn as nn
import timm
from torchvision import models
import segmentation_models_pytorch as smp


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

        self.backbone = models.segmentation.fcn_resnet50(pretrained=True)
        self.backbone.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.backbone(x)
        return x

class Unet(nn.Module):
    def __init__(self, num_classes : int, backbone : str, pretrained : bool):
        super(Unet, self).__init__()
        self.backbone =  smp.Unet(
                                encoder_name=backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                classes=num_classes,                      # model output channels (number of classes in your dataset)
                                )
    def forward(self, x):
        x = self.backbone(x)
        return x
    
class DeepLab(nn.Module):
    def __init__(self, num_classes : int, backbone : str, pretrained : bool):
        super(DeepLab, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
        self.backbone.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    def forward(self, x):
        x = self.backbone(x)
        return x