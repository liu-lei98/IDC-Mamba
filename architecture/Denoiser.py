import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
import random
from .SSM import Block

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


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

class InformationFusion(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(InformationFusion, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.attentions = nn.Sequential(
                        nn.Conv2d(in_channels, d, 1, padding=0,bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(d, d, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(d, d, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(d, d, 1, padding=0, bias=True), 
                        nn.ReLU(inplace=True),
                        nn.Conv2d(d, in_channels * self.height, 1, padding=0,bias=True),
                        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        
        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_sum = torch.sum(inp_feats, dim=1)
        feats_pool = self.avg_pool(feats_sum)

        attention_vectors = self.attentions(feats_pool)
        attention_vectors = self.softmax(attention_vectors.view(batch_size, self.height, n_feats, 1, 1))
        
        feats_out = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_out  


class UM(nn.Module):
    def __init__(self, input_dim = 28, dim=28, stage=2, num_states=[4, 8, 16] ):
        super(UM, self).__init__()
        self.dim = dim
        self.stage = stage
        self.embedding = nn.Conv2d(input_dim, self.dim, 3, 1, 1, bias=False)
        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                InformationFusion(dim_stage),
                Block(hidden_dim=dim_stage,d_state=num_states[i]),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck_proj = InformationFusion(dim_stage)
        self.bottleneck = Block(hidden_dim=dim_stage,d_state=num_states[-1])
        

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, 0, bias=False),
                InformationFusion(dim_stage // 2),
                Block(hidden_dim=dim_stage // 2,d_state=num_states[stage - 1 - i]),
                # FFTInteraction_N(dim_stage // 2)
            ]))
            dim_stage //= 2

        # Output projection
        self.out_proj = nn.Conv2d(self.dim, input_dim, 3, 1, 1, bias=False)

    def forward(self, x_ori, enc_fea=None, bot_fea=None, dec_fea=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h, w = x_ori.shape
        x = self.embedding(x_ori)
        # Encoder
        fea_encoder = []
        fea_next = []
        fea = x
        for j, (Mixer, Blcok, FeaDownSample) in enumerate(self.encoder_layers):
            if enc_fea is None:
                fea_mixed = fea
            else:
                fea_mixed = Mixer([fea,enc_fea[j]])
            fea = Blcok(fea_mixed)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        if bot_fea is None:
            fea = fea
        else:
            fea = self.bottleneck_proj([fea,bot_fea])
        fea = self.bottleneck(fea)
        fea_bottle = fea
        # Decoder
        fea_decoder = []
        # Decoder
        for i, (FeaUpSample,Fusion,Mixer,Blcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)            
            fea = Fusion(torch.cat([fea,fea_encoder[self.stage - 1 - i]],dim=1))
            if dec_fea is None:
                fea = fea
            else:
                fea = Mixer([fea,dec_fea[i]])
            fea = Blcok(fea)
            fea_decoder.append(fea)
            # fea_next.append(Deliver(fea_encoder[self.stage - 1 - i],fea))

        # Output projection
        out = self.out_proj(fea) + x_ori
        return out, fea_encoder, fea_bottle, fea_decoder












