import torch.nn as nn
from metrics import dice_score

class DiceBCELoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.bce = nn.BCELoss()
        
    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        dice_loss = 1 - dice_score(preds, targets)
        return bce_loss + self.weight * dice_loss