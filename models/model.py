"""
model.py is a file that you use when you load a pre-trained model and use it as it is, or change it.
"""

import torch.nn as nn
import timm
from torchvision import models
import numpy as np

from .intern_image import InternImage as InternModel
# from mmcv.utils import Config
# from mmseg.models import build_segmentor
# from mmcv.cnn.utils import revert_sync_batchnorm
from .uper_head import UPerHead as head

class CustomModel(nn.Module):
    def __init__(self, num_classes : int, backbone : str, pretrained : bool):
        super(CustomModel, self).__init__()

        self.backbone = timm.create_model(backbone, num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        x = self.backbone(x)
        return x
    
class SegModel(nn.Module):
    def __init__(self, num_classes : int, pretrained : bool):
        super(SegModel, self).__init__()

        self.backbone = models.segmentation.fcn_resnet50(pretrained=True)
        self.backbone.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    
class InternImage(nn.Module):
    def __init__(self, num_classes : int, pretrained : bool):
        super(InternImage, self).__init__()

        if pretrained == True:
            checkpoint = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth'
            self.backbone = InternModel(core_op='DCNv3',channels=64,depths=[4, 4, 18, 4],groups=[4, 8, 16, 32], mlp_ratio=4.,
                        drop_path_rate=0.4,norm_layer='LN',layer_scale=1.0, offset_scale=1.0,post_norm=False,with_cp=False,
                        out_indices=(0, 1, 2, 3),init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
                        )
        else:
            self.backbone = InternModel(core_op='DCNv3',channels=64,depths=[4, 4, 18, 4],groups=[4, 8, 16, 32], mlp_ratio=4.,
                        drop_path_rate=0.4,norm_layer='LN',layer_scale=1.0, offset_scale=1.0,post_norm=False,with_cp=False,
                        out_indices=(0, 1, 2, 3)
                        )
        
        self.head = head(in_channels=[64, 128, 256, 512], in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6),
                channels=256, dropout_ratio=0.1, num_classes=29, norm_cfg=dict(type='SyncBN', requires_grad=True), 
                align_corners=False, loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                )
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return x