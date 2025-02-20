import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
from .Denoiser import UM


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class HyPaNet(nn.Module):
    def __init__(self, in_nc=29, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.fution = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.down_sample = nn.Conv2d(channel, channel, 3, 2, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = out_nc

    def forward(self, x,y):
        x = self.down_sample(self.relu(self.fution(torch.cat([x.unsqueeze(1),y],dim=1))))
        x = self.avg_pool(x)
        x = self.mlp(x) + 1e-6
        return x[:,:self.out_nc//2,:,:], x[:,self.out_nc//2:,:,:]

class ProximalMapping(nn.Module):
    def __init__(self,dim) -> None:
        super(ProximalMapping,self).__init__()
        self.Denoiser =UM(dim)
        self.embedding = nn.Conv2d(dim+1,dim, 3, 1, 1, bias=False)

    def forward(self,r, enc_fea=None, bot_fea=None, dec_fea=None): 
        b, c, h_inp, w_inp = r.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        r = F.pad(r, [0, pad_w, 0, pad_h], mode='reflect')
        r = self.embedding(r)
        xk, enc_fea, bot_fea, dec_fea  = self.Denoiser(r,enc_fea, bot_fea, dec_fea)
        return xk[:, :, :h_inp, :w_inp] ,enc_fea, bot_fea, dec_fea
    

class Gradient(nn.Module):
    def __init__(self,dim) -> None:
        super(Gradient,self).__init__()
        self.Degradation =  nn.Sequential(
            nn.Conv2d(dim*2, 64, kernel_size=1, stride=1), 
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,groups=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, padding=0, bias=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,groups=64),
            nn.Conv2d(64, dim, kernel_size=1, stride=1)
        )

    def forward(self,degradation,z):

        degradation_r = self.Degradation(torch.cat([degradation,z],dim=1))
        new_degradation = degradation + degradation_r
        return new_degradation


class IDCMamba(nn.Module):

    def __init__(self, num_iterations=1):
        super(IDCMamba, self).__init__()
        self.para_estimator = HyPaNet(in_nc=29, out_nc=num_iterations*2)
        self.stage_estimator = HyPaNet(in_nc=29, out_nc=2)
        self.fution = nn.Conv2d(56, 28, 1, padding=0, bias=True)
        self.num_iterations = num_iterations
        self.Gradient = Gradient(28)
        self.PM = ProximalMapping(28)

    def initial(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :return: temp: [b,28,256,310]; alpha: [b, num_iterations]; beta: [b, num_iterations]
        """
        nC, step = 28, 2
        y = y / nC * 2
        bs,row,col = y.shape
        y_shift = torch.zeros(bs, nC, row, col).cuda().float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fution(torch.cat([y_shift, Phi], dim=1))
        return z

    def A(self, x,Phi):
        temp = x*Phi
        y = torch.sum(temp,1)
        return y

    def At(self, y,Phi):
        temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
        x = temp*Phi
        return x

    def shift_3d(self, inputs,step=2):
        [bs, nC, row, col] = inputs.shape
        for i in range(nC):
            inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
        return inputs

    def shift_back_3d(self, inputs,step=2):
        [bs, nC, row, col] = inputs.shape
        for i in range(nC):
            inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
        return inputs

    def forward(self, y, Phi=None):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :param Phi_PhiT: [b,256,310]
        :return: z_crop: [b,28,256,256]
        """
        z  = self.initial(y, Phi)
        enc_fea = None
        bot_fea = None
        dec_fea = None
        alphas, betas = self.para_estimator(y,Phi)
        for i in range(self.num_iterations):
            alpha, beta = alphas[:,i,:,:], betas[:,i:i+1,:,:]
            Phi = self.Gradient(Phi,z)
            Phi_s= torch.sum(Phi**2,1)
            Phi_s[Phi_s==0] = 1
            Phi_z = self.A(z,Phi)
            alpha_r, beta_r = self.stage_estimator(Phi_z,Phi)
            alpha = alpha + alpha_r[:,0,:,:]
            beta = beta + beta_r[:,0:1,:,:]
            x = z + self.At(torch.div(y-Phi_z,alpha+Phi_s), Phi)
            x = self.shift_back_3d(x)
            beta_repeat = beta.repeat(1,1,x.shape[2], x.shape[3])
            z, enc_fea, bot_fea, dec_fea = self.PM(torch.cat([x, beta_repeat],dim=1),enc_fea, bot_fea, dec_fea)
            if i<self.num_iterations-1:
                z = self.shift_3d(z)
        return z[:, :, :, 0:256]

if __name__=="__main__":
    model = IDCMamba(2).cuda()
    M= torch.randn((1,28,256,310)).cuda().float()
    x= torch.randn((1,256,310)).cuda().float()
    y = model(x,M)
    print(y.shape)