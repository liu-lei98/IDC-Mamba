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

class CharbonnierLossGT(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3, gamma= 1e-2):
        super(CharbonnierLossGT, self).__init__()
        print("CharbonnierLossGT Loss (L1)")
        self.eps = eps
        self.gamma = gamma

    def forward(self, x, y, GT_rec):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss1 = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        diff2 = GT_rec - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss2 = torch.mean(torch.sqrt((diff2 * diff2) + (self.eps*self.eps)))
        loss = loss1 + self.gamma*loss2
        return loss

class CharbonnierLossY(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3, gamma= 1e-2):
        super(CharbonnierLossY, self).__init__()
        print("Charbonnier Loss Y")
        self.eps = eps
        self.gamma = gamma

    def shift(self, inputs,step=2):
        [bs, nC, row, col] = inputs.shape
        output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
        for i in range(nC):
            output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
        return output

    def gen_meas_torch(self, data_batch, mask3d_batch):
        [batch_size, nC, H, W] = data_batch.shape
        mask3d_batch = (mask3d_batch[0, :, :, :]).expand([batch_size, nC, H, W]).cuda().float()  # [10,28,256,256]
        temp = self.shift(mask3d_batch * data_batch, 2)
        meas = torch.sum(temp, 1)
        return meas

    def forward(self, x, y, phi, meas):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss1 = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))

        diff2 = self.gen_meas_torch(x,phi) - meas

        loss2 = torch.mean(torch.sqrt((diff2 * diff2) + (self.eps*self.eps)))

        loss = loss1 + self.gamma*loss2

        return loss


class lossFuc(nn.Module):

    def __init__(self):
        super(lossFuc, self).__init__()
        self.mse = torch.nn.MSELoss()
        print("ist loss")
    def forward(self, prediction, prediction_symmetric, gt):

        cost_mean = torch.sqrt(self.mse(prediction, gt))
        cost_symmetric = 0
        for k in range(len(prediction_symmetric)):
            cost_symmetric += torch.mean(torch.pow(prediction_symmetric[k], 2))

        cost_all = cost_mean + 0.01 * cost_symmetric
        return cost_all

class L2(nn.Module):

    def __init__(self):
        super(L2, self).__init__()
        self.mse = torch.nn.MSELoss()
        print("l2 loss")
    def forward(self, prediction,gt):
        cost_mean = torch.sqrt(self.mse(prediction, gt))
        return cost_mean