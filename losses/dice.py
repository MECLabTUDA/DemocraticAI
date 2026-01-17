
import torch
from torch import nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_dice(pred: torch.Tensor, target: torch.Tensor, epsilon=1e-6, dim=None):
        intersection = (pred * target).sum(dim=dim)
        union = pred.sum(dim=dim) + target.sum(dim=dim)
        return 2 * intersection / (union + epsilon)
    
    def input(self, pred, target):
        return self.compute_dice(pred, target)
    
    
class DiceLossMulti(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_dice(pred: torch.Tensor, target: torch.Tensor, epsilon=1e-6):
        intersection = (pred * target).sum(dim=(0,2,3))
        union = pred.sum(dim=(0,2,3)) + target.sum(dim=(0,2,3))
        return 2 * intersection / (union + epsilon)
    
    def input(self, pred, target):
        # pred: BCHW
        # target: BCHW
        return self.compute_dice_multi(pred, target)