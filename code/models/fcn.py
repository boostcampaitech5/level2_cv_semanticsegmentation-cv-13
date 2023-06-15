from torch import nn 
from torchvision import models 

class FCN_Resnet50(nn.Module): 
    def __init__(self, num_classes=29): 
        super().__init__() 
        self.fcn_resnet50 = models.segmentation.fcn_resnet50(pretrained=True) 
        
        self.fcn_resnet50.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1) 
        
    def forward(self, x): 
        return self.fcn_resnet50(x) 
    
    
class FCN_Resnet101(nn.Module): 
    def __init__(self, num_classes=29): 
        super().__init__() 
        self.fcn_resnet101 = models.segmentation.fcn_resnet101(pretrained=True) 
        
        self.fcn_resnet101.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1) 
        
    def forward(self, x): 
        return self.fcn_resnet101(x)  
    