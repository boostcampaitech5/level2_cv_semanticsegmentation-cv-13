import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True, device='cpu'):
        super(FocalLoss, self).__init__()
        """
        gamma(int) : focusing parameter.
        alpha(list) : alpha-balanced term.
        size_average(bool) : whether to apply reduction to the output.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.device = device

    def forward(self, input, target):
        # input : N * C (btach_size, num_class)
        # target : N (batch_size)

        CE = F.cross_entropy(input, target, reduction='none')  # -log(pt)
        pt = torch.exp(-CE)  # pt
        loss = (1 - pt) ** self.gamma * CE * self.alpha  # -(1-pt)^rlog(pt)

        if self.size_average:
            loss = torch.mean(loss)

        return 
    
    
class F1Loss(nn.Module):
    def __init__(self, classes=3, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()
    
class CombineLoss(nn.Module):
    def __init__(self, bce_weight = 0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = F.sigmoid(pred)
        dice = dice_loss(pred, target)
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        
        return loss

def crossentropy():
    criterion = nn.CrossEntropyLoss()
    return criterion


def bceloss():
    criterion = nn.BCELoss()
    return criterion


def mseloss():
    criterion = nn.MSELoss()
    return criterion


def f1loss(classes=3, epsilon=1e-7):
    criterion = F1Loss(classes=classes, epsilon=epsilon)
    return criterion


def focalloss(gamma=2,alpha=0.25,device='cpu'):
    criterion = FocalLoss(gamma=gamma, alpha=alpha, device=device)
    return criterion

def bce_with_logit_loss():
    criterion = nn.BCEWithLogitsLoss()
    return criterion

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()
    

def combineloss(bce_weight = 0.5):
    criterion = CombineLoss(bce_weight = 0.5)
    return criterion