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

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean() 


class CombinedLoss(): 
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
    

_criterion_entrypoints = {
    'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
    'CustomCombinedLoss': CombinedLoss,
}