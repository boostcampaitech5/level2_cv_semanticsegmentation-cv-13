import torch 
import torch.nn as nn 
import torch.nn.functional as F  


class StemBlock(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU() 
        ) 
    
    def forward(self, inputs): 
        return self.block(inputs) 
    

class Stage01Block(nn.Module): 
    def __init__(self, in_channels): 
        super().__init__() 
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            
            nn.Conv2d(64, 256, kernel_size=1, bias=False), 
            nn.BatchNorm2d(256),              
        ) 
        if in_channels == 64: 
            self.identity_block = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=1, bias=False), 
                nn.BatchNorm2d(256) 
            )
        self.relu = nn.ReLU() 
        self.in_channels = in_channels 
        
    def forward(self, inputs): 
        identity = inputs 
        out = self.block(inputs) 
        
        if self.in_channels == 64: 
            identity = self.identity_block(identity) 
        out += identity 
        
        return self.relu(out)  
    
    
class Stage01StreamGenerateBlock(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.high_res_block = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(48), 
            nn.ReLU() 
        )
        self.medium_res_block = nn.Sequential(
            nn.Conv2d(256, 96, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(96), 
            nn.ReLU() 
        )
        
    def forward(self, inputs): 
        out_high = self.high_res_block(inputs) 
        out_medium = self.medium_res_block(inputs) 
        return out_high, out_medium   
    

class StageBlock(nn.Module): 
    def __init__(self, in_channels):  # in_channels = 46 or 92 
        super().__init__() 
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(), 
            
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(in_channels), 
        ) 
        self.relu = nn.ReLU() 
        
    def forward(self, inputs): 
        identity = inputs 
        out = self.block(inputs) 
        out += identity 
        out = self.relu(out) 
        
        return out  
    
    
class Stage02(nn.Module): 
    def __init__(self): 
        super().__init__() 
        high_res_blocks = [StageBlock(48) for _ in range(4)] 
        medium_res_blocks = [StageBlock(96) for _ in range(4)] 
        
        self.high_res_blocks = nn.Sequential(*high_res_blocks) 
        self.medium_res_blocks = nn.Sequential(*medium_res_blocks) 
        
    def forward(self, inputs):
        inputs_high, inputs_medium = inputs[0], inputs[1] 
        
        out_high = self.high_res_blocks(inputs_high) 
        out_medium = self.medium_res_blocks(inputs_medium) 
        return (out_high, out_medium)
    

class Stage02Fuse(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.high_to_medium = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(96) 
        )
        self.medium_to_high = nn.Sequential(
            nn.Conv2d(96,  48, kernel_size=1, bias=False), 
            nn.BatchNorm2d(48) 
        )
        self.relu = nn.ReLU() 
        
    def forward(self, inputs): 
        inputs_high, inputs_medium = inputs[0], inputs[1] 
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2]) 
        
        med2high = self.medium_to_high(inputs_medium)  
        med2high = F.interpolate(
            med2high, size=high_size, mode='bilinear', align_corners=True 
        )
        
        high2med = self.high_to_medium(inputs_high)  
        
        out_high = inputs_high + med2high 
        out_medium = inputs_medium + high2med  
        
        out_high = self.relu(out_high)
        out_medium = self.relu(out_medium) 
        
        return out_high, out_medium  
    
    
class StreamGenerateBlock(nn.Module): 
    def __init__(self, in_channels): 
        super().__init__() 
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(in_channels*2), 
            nn.ReLU() 
        )
        
    def forward(self, inputs): 
        return self.block(inputs)  

 
class Stage03(nn.Module): 
    def __init__(self): 
        super().__init__() 
        high_res_blocks = [StageBlock(48) for _ in range(4)] 
        medium_res_blocks = [StageBlock(96) for _ in range(4)] 
        low_res_blocks = [StageBlock(192) for _ in range(4)] 
        
        self.high_res_blocks = nn.Sequential(*high_res_blocks) 
        self.medium_res_blocks = nn.Sequential(*medium_res_blocks) 
        self.low_res_blocks = nn.Sequential(*low_res_blocks)
        
    def forward(self, inputs): 
        inputs_high, inputs_medium, inputs_low = inputs[0], inputs[1], inputs[2] 
        out_high = self.high_res_blocks(inputs_high) 
        out_medium = self.medium_res_blocks(inputs_medium)
        out_low = self.low_res_blocks(inputs_low) 
        return (out_high, out_medium, out_low) 
    

class Stage03Fuse(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.high_to_medium = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(96)  
        )
        self.high_to_low = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(48), 
            nn.ReLU(), 
            nn.Conv2d(48, 192, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(192) 
        )
        self.medium_to_low = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(192) 
        )
        self.medium_to_high = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1, bias=False), 
            nn.BatchNorm2d(48) 
        )
        self.low_to_high = nn.Sequential(
            nn.Conv2d(192, 48, kernel_size=1, bias=False), 
            nn.BatchNorm2d(48)
        )
        self.low_to_medium = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1, bias=False), 
            nn.BatchNorm2d(96) 
        )
        self.relu = nn.ReLU() 
        
    def forward(self, inputs): 
        inputs_high, inputs_medium, inputs_low = inputs[0], inputs[1], inputs[2] 
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2]) 
        medium_size = (inputs_medium.shape[-1], inputs_medium.shape[-2]) 
        
        low2high = F.interpolate(
            inputs_low, size=high_size, mode='bilinear', align_corners=True 
        )
        low2high = self.low_to_high(low2high) 
        
        low2med = F.interpolate(
            inputs_low, size=medium_size, mode='bilinear', align_corners=True 
        )
        low2med = self.low_to_medium(low2med) 
        
        med2high = F.interpolate(
            inputs_medium, size=high_size, mode='bilinear', align_corners=True 
        )
        med2high = self.medium_to_high(med2high) 
        
        med2low = self.medium_to_low(inputs_medium) 
        
        high2med = self.high_to_medium(inputs_high) 
        high2low = self.high_to_low(inputs_high) 
        
        out_high = inputs_high + med2high + low2high 
        out_medium = inputs_medium + high2med + low2med 
        out_low = inputs_low + high2low + med2low 
        
        out_high = self.relu(out_high) 
        out_medium = self.relu(out_medium) 
        out_low = self.relu(out_low) 
        
        return (out_high, out_medium, out_low)  
    

class Stage04(nn.Module): 
    def __init__(self): 
        super().__init__() 
        high_res_blocks = [StageBlock(48) for _ in range(3)] 
        medium_res_blocks = [StageBlock(96) for _ in range(3)] 
        low_res_blocks = [StageBlock(192) for _ in range(3)]
        vlow_res_blocks = [StageBlock(384) for _ in range(3)] 
        
        self.high_res_blocks = nn.Sequential(*high_res_blocks) 
        self.medium_res_blocks = nn.Sequential(*medium_res_blocks) 
        self.low_res_blocks = nn.Sequential(*low_res_blocks)
        self.vlow_res_blocks = nn.Sequential(*vlow_res_blocks)
        
    def forward(self, inputs):
        inputs_high, inputs_medium, inputs_low, inputs_vlow = inputs[0], inputs[1], inputs[2], inputs[3]
         
        out_high = self.high_res_blocks(inputs_high) 
        out_medium = self.medium_res_blocks(inputs_medium)
        out_low = self.low_res_blocks(inputs_low) 
        out_vlow = self.vlow_res_blocks(inputs_vlow)
        return (out_high, out_medium, out_low, out_vlow) 


class Stage04Fuse(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.high_to_medium = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(96)  
        )
        self.high_to_low = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(48), 
            nn.ReLU(), 
            nn.Conv2d(48, 192, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(192) 
        )
        self.high_to_vlow = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(48), 
            nn.ReLU(), 
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(48), 
            nn.ReLU(), 
            nn.Conv2d(48, 384, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(384)
        )
        self.medium_to_low = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(192) 
        )
        self.medium_to_high = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1, bias=False), 
            nn.BatchNorm2d(48) 
        )
        self.medium_to_vlow = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(96), 
            nn.ReLU(),
            nn.Conv2d(96, 384, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(384) 
        )
        self.low_to_high = nn.Sequential(
            nn.Conv2d(192, 48, kernel_size=1, bias=False), 
            nn.BatchNorm2d(48)
        )
        self.low_to_medium = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1, bias=False), 
            nn.BatchNorm2d(96) 
        )
        self.low_to_vlow = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(384)
        )
        self.vlow_to_high = nn.Sequential(
            nn.Conv2d(384, 48, kernel_size=1, bias=False), 
            nn.BatchNorm2d(48) 
        )
        self.vlow_to_medium = nn.Sequential(
            nn.Conv2d(384, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96) 
        )
        self.vlow_to_low = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192)
        )
        self.relu = nn.ReLU() 
        
    def forward(self, inputs):
        inputs_high, inputs_medium, inputs_low, inputs_vlow = inputs[0], inputs[1], inputs[2], inputs[3] 
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2]) 
        medium_size = (inputs_medium.shape[-1], inputs_medium.shape[-2]) 
        low_size = (inputs_low.shape[-1], inputs_low.shape[-2]) 
        
        vlow2high = F.interpolate(
            inputs_vlow, size=high_size, mode='bilinear', align_corners=True
        )
        vlow2high = self.vlow_to_high(vlow2high)
        
        vlow2med = F.interpolate(
            inputs_vlow, size=medium_size, mode='bilinear', align_corners=True
        )
        vlow2med = self.vlow_to_medium(vlow2med)
        
        vlow2low = F.interpolate(
            inputs_vlow, size=low_size, mode='bilinear', align_corners=True 
        )
        vlow2low = self.vlow_to_low(vlow2low)
        
        low2high = F.interpolate(
            inputs_low, size=high_size, mode='bilinear', align_corners=True 
        )
        low2high = self.low_to_high(low2high) 
        
        low2med = F.interpolate(
            inputs_low, size=medium_size, mode='bilinear', align_corners=True 
        )
        low2med = self.low_to_medium(low2med) 
        
        med2high = F.interpolate(
            inputs_medium, size=high_size, mode='bilinear', align_corners=True 
        )
        med2high = self.medium_to_high(med2high) 
        
        low2vlow = self.low_to_vlow(inputs_low) 
        
        med2vlow = self.medium_to_vlow(inputs_medium)  
        med2low = self.medium_to_low(inputs_medium) 
        
        high2vlow = self.high_to_vlow(inputs_high) 
        high2med = self.high_to_medium(inputs_high) 
        high2low = self.high_to_low(inputs_high) 
        
        out_high = inputs_high + med2high + low2high + vlow2high 
        out_medium = inputs_medium + high2med + low2med + vlow2med
        out_low = inputs_low + high2low + med2low + vlow2low
        out_vlow = inputs_vlow + high2vlow + med2vlow + low2vlow 
        
        out_high = self.relu(out_high) 
        out_medium = self.relu(out_medium) 
        out_low = self.relu(out_low) 
        out_vlow = self.relu(out_vlow)
        
        return (out_high, out_medium, out_low, out_vlow) 


class LastBlock(nn.Module): 
    def __init__(self, num_classes=29): 
        super().__init__() 
        total_channels = 48 + 96 + 192 + 384 
        self.block = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(total_channels),
            nn.ReLU(), 
            
            nn.Conv2d(total_channels, num_classes, kernel_size=1, bias=False),         
        ) 
        
    def forward(self, inputs_high, inputs_medium, inputs_low, inputs_vlow): 
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2]) 
        original_size = (high_size[0]*4, high_size[1]*4) 
        
        med2high = F.interpolate(inputs_medium, size=high_size, mode='bilinear', align_corners=True)
        low2high = F.interpolate(inputs_low, size=high_size, mode='bilinear', align_corners=True) 
        vlow2high = F.interpolate(inputs_vlow, size=high_size, mode='bilinear', align_corners=True) 
        
        out = torch.cat([inputs_high, med2high, low2high, vlow2high], dim=1)  # 채널 방향 concat 
        out = self.block(out) 
        
        out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=True) 
        return out
    

class HRNet(nn.Module): 
    def __init__(self): 
        super().__init__() 
        
        self.stem_block = StemBlock()  
        
        self.stage01 = nn.Sequential(
            Stage01Block(in_channels=64), 
            Stage01Block(in_channels=256),
            Stage01Block(in_channels=256), 
            Stage01Block(in_channels=256), 
        )
        self.stage01_stream_block = Stage01StreamGenerateBlock()  # out_high: 48x128x128, out_medium: 96x64x64  
        
        self.stage02 = nn.Sequential(
            Stage02(),
            Stage02Fuse(), 
        ) 
        self.stage02_stream_block = StreamGenerateBlock(96) # out_low: 32x32x192  
        
        self.stage03 = nn.Sequential(
            Stage03(),
            Stage03Fuse(), 
            Stage03(),
            Stage03Fuse(), 
            Stage03(),
            Stage03Fuse(), 
            Stage03(),
            Stage03Fuse(), 
            Stage03(),
            Stage03Fuse(), 
        ) 
        self.stage03_stream_block = StreamGenerateBlock(192) # out_vlow: 16x16x384 
        
        self.stage04 = nn.Sequential(
            Stage04(), 
            Stage04Fuse(),
            Stage04(), 
            Stage04Fuse(),
            Stage04(), 
            Stage04Fuse(), 
        ) 
        
        self.last_block = LastBlock() 
        
    def forward(self, inputs): 
        out = self.stem_block(inputs)  # 64x128x128

        # stage01 
        out = self.stage01(out) 
        out_high, out_medium = self.stage01_stream_block(out) 
        
        # stage02 
        out_high, out_medium = self.stage02((out_high, out_medium)) 
        out_low = self.stage02_stream_block(out_medium) 
        
        # stage03 
        out_high, out_medium, out_low = self.stage03((out_high, out_medium, out_low)) 
        out_vlow = self.stage03_stream_block(out_low) 
        
        # stage04 
        out_high, out_medium, out_low, out_vlow = self.stage04((out_high, out_medium, out_low, out_vlow))
        
        out = self.last_block(out_high, out_medium, out_low, out_vlow)
        
        return out  
    
    