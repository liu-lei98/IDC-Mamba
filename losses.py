import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        print("Charbonnier Loss (L1)")
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss



class L2(nn.Module):

    def __init__(self):
        super(L2, self).__init__()
        self.mse = torch.nn.MSELoss()
        print("l2 loss")
    def forward(self, prediction,gt):
        cost_mean = torch.sqrt(self.mse(prediction, gt))
        return cost_mean