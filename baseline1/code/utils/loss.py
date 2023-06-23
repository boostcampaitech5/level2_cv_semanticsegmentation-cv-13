import torch 
import torch.nn as nn 
import torch.nn.functional as F 


def create_criterion(criterion_name, **kwargs): 
    if criterion_name in _criterion_entrypoints: 
        create_fn = _criterion_entrypoints[criterion_name] 
        criterion = create_fn(**kwargs)
    else: 
        return RuntimeError('Unknown loss (%s)' % criterion_name) 
    return criterion  

def IOU_loss(inputs, targets, smooth=1) : 
    inputs = F.sigmoid(inputs)      
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    IoU = (intersection + smooth)/(union + smooth)
    return 1 - IoU 


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean() 


class CombinedLoss1(): 
    '''
        Binart Cross Entropy Loss + Dice Loss 
    '''
    def __init__(self, bce_weight = 0.5): 
        self.bce_weight = bce_weight 
    
    def __call__(self, pred, target): 
        
        # Binary Cross Entropy Loss 
        bce = F.binary_cross_entropy_with_logits(pred, target) 
        # Dice Loss
        pred = F.sigmoid(pred)
        dice = dice_loss(pred, target) 
        
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        
        return loss 
    

class CombinedLoss2(): 
    
    def __call__(self, pred, target): 
        
        # Binary Cross Entropy Loss 
        bce = F.binary_cross_entropy_with_logits(pred, target) 
        
        pred = F.sigmoid(pred)
        # Dice Loss
        dice = dice_loss(pred, target) 
        
        # IOU Loss 
        iou = IOU_loss(pred, target)
        
        loss = bce * 0.4 + dice * 0.4 + iou * 0.2
        
        return loss


_criterion_entrypoints = {
    'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
    'CustomCombinedLoss': CombinedLoss1, 
    'CustomCombinedLoss1': CombinedLoss2,
}