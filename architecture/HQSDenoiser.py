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


# class InformationFusion(nn.Module):
#     def __init__(self, in_channels, height=2, reduction=8, bias=False):
#         super(InformationFusion, self).__init__()
        
#         self.height = height
#         d = max(int(in_channels/reduction),4)
        
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#         self.attentions = nn.ModuleList([])
#         for i in range(self.height):
#             self.attentions.append(
#                 nn.Sequential(
#                         nn.Conv2d(in_channels, d, 1, padding=0,bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(d, d, 1, padding=0, bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(d, d, 1, padding=0, bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(d, d, 1, padding=0, bias=True), 
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(d, in_channels, 1, padding=0,bias=True),
#                         )
#             )
        
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, inp_feats):
#         batch_size = inp_feats[0].shape[0]
#         n_feats =  inp_feats[0].shape[1]
        
#         inp_feats = torch.cat(inp_feats, dim=1)
#         inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
#         feats_sum = torch.sum(inp_feats, dim=1)
#         feats_pool = self.avg_pool(feats_sum)

#         attention_vectors = [ca(feats_pool) for ca in self.attentions]
#         attention_vectors = torch.cat(attention_vectors, dim=1)
#         attention_vectors = self.softmax(attention_vectors.view(batch_size, self.height, n_feats, 1, 1))
        
#         feats_out = torch.sum(inp_feats*attention_vectors, dim=1)
        
#         return feats_out  


class GatedMixed(nn.Module):
    def __init__(self, in_c=28, expansion_factor=1):
        super(GatedMixed, self).__init__()
        self.channel = in_c
        dim = int(in_c*expansion_factor)
        self.conv11 = nn.Conv2d(2 * self.channel, dim, 1, 1)
        self.dwconv = nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=1, padding=1,
                                groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.project_out = nn.Conv2d(dim, in_c, kernel_size=1)

    def forward(self, feature_hi, feature_low):
        concat = torch.cat([feature_hi, feature_low], dim=1)
        conv11 = self.conv11(concat)
        dwconv1, dwconv2 = self.dwconv(conv11).chunk(2, dim=1)
        perception = torch.sigmoid(dwconv1) * dwconv2  + self.dwconv2(dwconv1)
        perception = self.project_out(perception)

        return perception

class FFTInteraction_N(nn.Module):
    def __init__(self, in_nc ):
        super(FFTInteraction_N,self).__init__()

        self.mid = nn.Conv2d(in_nc,in_nc,3,1,1,groups=in_nc)


    def forward(self,x_enc,x_dec):
        x_enc = torch.fft.rfft2(x_enc, norm='backward')
        x_dec = torch.fft.rfft2(x_dec, norm='backward')
        x_freq_amp = torch.abs(x_enc)
        x_freq_pha = torch.angle(x_dec)
        x_freq_pha = self.mid(x_freq_pha)
        real = x_freq_amp * torch.cos(x_freq_pha)
        imag = x_freq_amp * torch.sin(x_freq_pha)
        x_recom = torch.complex(real, imag)
        x_recom = torch.fft.irfft2(x_recom)

        return x_recom

class UM_head(nn.Module):
    def __init__(self, input_dim = 28, dim=28, stage=2, num_states=[4, 8, 16] ):
        super(UM_head, self).__init__()
        self.dim = dim
        self.stage = stage
        self.embedding = nn.Conv2d(input_dim, self.dim, 3, 1, 1, bias=False)
        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                Block(hidden_dim=dim_stage,d_state=num_states[i]),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        
        self.bottleneck = Block(hidden_dim=dim_stage,d_state=num_states[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, 0, bias=False),
                Block(hidden_dim=dim_stage // 2,d_state=num_states[stage - 1 - i]),
                FFTInteraction_N(dim_stage // 2)
            ]))
            dim_stage //= 2

        # Output projection
        self.out_proj = nn.Conv2d(self.dim, input_dim, 3, 1, 1, bias=False)

    def forward(self, x_ori, enc_fea=None):
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
        for j, (Blcok, FeaDownSample) in enumerate(self.encoder_layers):
            fea = Blcok(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        fea = self.bottleneck(fea)
        fea_next.append(fea)
        # Decoder
        for i, (FeaUpSample,Fusion, Blcok, Deliver) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fusion(torch.cat([fea,fea_encoder[self.stage - 1 - i]],dim=1))
            fea = Blcok(fea)
            fea_next.append(Deliver(fea_encoder[self.stage - 1 - i],fea))

        # Output projection
        out = self.out_proj(fea) + x_ori
        return out,fea_next

# class UM(nn.Module):
#     def __init__(self, input_dim = 28, dim=28, stage=2, num_states=[4, 8, 16] ):
#         super(UM, self).__init__()
#         self.dim = dim
#         self.stage = stage
#         self.embedding = nn.Conv2d(input_dim, self.dim, 3, 1, 1, bias=False)
#         # Encoder
#         self.encoder_layers = nn.ModuleList([])
#         dim_stage = dim
#         for i in range(stage):
#             self.encoder_layers.append(nn.ModuleList([
#                 Block(hidden_dim=dim_stage,d_state=num_states[i]),
#                 nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
#             ]))
#             dim_stage *= 2

#         # Bottleneck
#         self.bottleneck_proj = InformationFusion(dim_stage)
#         self.bottleneck = Block(hidden_dim=dim_stage,d_state=num_states[-1])
        

#         # Decoder
#         self.decoder_layers = nn.ModuleList([])
#         for i in range(stage):
#             self.decoder_layers.append(nn.ModuleList([
#                 nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
#                 InformationFusion(dim_stage // 2),
#                 nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, 0, bias=False),
#                 Block(hidden_dim=dim_stage // 2,d_state=num_states[stage - 1 - i]),
#                 FFTInteraction_N(dim_stage // 2)
#             ]))
#             dim_stage //= 2

#         # Output projection
#         self.out_proj = nn.Conv2d(self.dim, input_dim, 3, 1, 1, bias=False)

#     def forward(self, x_ori, enc_fea):
#         """
#         x: [b,c,h,w]
#         return out:[b,c,h,w]
#         """
#         b, c, h, w = x_ori.shape
#         x = self.embedding(x_ori)
#         # Encoder
#         fea_encoder = []
#         fea_next = []
#         fea = x
#         for j, (Blcok, FeaDownSample) in enumerate(self.encoder_layers):
#             fea = Blcok(fea)
#             fea_encoder.append(fea)
#             fea = FeaDownSample(fea)

#         fea = self.bottleneck_proj([fea,enc_fea[0]])
#         fea = self.bottleneck(fea)
#         fea_next.append(fea)
#         # Decoder
#         for i, (FeaUpSample, Mixer, Fusion, Blcok, Deliver) in enumerate(self.decoder_layers):
#             fea = FeaUpSample(fea)
#             fea_mixed = Mixer([fea_encoder[self.stage - 1 - i],enc_fea[i+1]])
#             fea = Fusion(torch.cat([fea,fea_mixed],dim=1))
#             fea = Blcok(fea)
#             fea_next.append(Deliver(fea_encoder[self.stage - 1 - i],fea))

#         # Output projection
#         out = self.out_proj(fea) + x_ori
#         return out,fea_next

# class UM(nn.Module):
#     def __init__(self, input_dim = 28, dim=28, stage=2, num_states=[4, 8, 16] ):
#         super(UM, self).__init__()
#         self.dim = dim
#         self.stage = stage
#         self.embedding = nn.Conv2d(input_dim, self.dim, 3, 1, 1, bias=False)
#         # Encoder
#         self.encoder_layers = nn.ModuleList([])
#         dim_stage = dim
#         for i in range(stage):
#             self.encoder_layers.append(nn.ModuleList([
#                 InformationFusion(dim_stage),
#                 Block(hidden_dim=dim_stage,d_state=num_states[i]),
#                 nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
#             ]))
#             dim_stage *= 2

#         # Bottleneck
#         self.bottleneck_proj = InformationFusion(dim_stage)
#         self.bottleneck = Block(hidden_dim=dim_stage,d_state=num_states[-1])
        

#         # Decoder
#         self.decoder_layers = nn.ModuleList([])
#         for i in range(stage):
#             self.decoder_layers.append(nn.ModuleList([
#                 nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
#                 nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, 0, bias=False),
#                 InformationFusion(dim_stage // 2),
#                 Block(hidden_dim=dim_stage // 2,d_state=num_states[stage - 1 - i]),
#                 # FFTInteraction_N(dim_stage // 2)
#             ]))
#             dim_stage //= 2

#         # Output projection
#         self.out_proj = nn.Conv2d(self.dim, input_dim, 3, 1, 1, bias=False)

#     def forward(self, x_ori, enc_fea, bot_fea, dec_fea):
#         """
#         x: [b,c,h,w]
#         return out:[b,c,h,w]
#         """
#         b, c, h, w = x_ori.shape
#         x = self.embedding(x_ori)
#         # Encoder
#         fea_encoder = []
#         fea = x
#         for j, (Mixer, Blcok, FeaDownSample) in enumerate(self.encoder_layers):
#             fea_mixed = Mixer([fea,enc_fea[j]])
#             fea = Blcok(fea_mixed)
#             fea_encoder.append(fea)
#             fea = FeaDownSample(fea)

#         fea = self.bottleneck_proj([fea,bot_fea])
#         fea = self.bottleneck(fea)

#         # Decoder
#         for i, (FeaUpSample,Fusion,Mixer,Blcok) in enumerate(self.decoder_layers):
#             fea = FeaUpSample(fea)            
#             fea = Fusion(torch.cat([fea,fea_encoder[self.stage - 1 - i]],dim=1))
#             fea = Mixer([fea,dec_fea[i]])
#             fea = Blcok(fea)
 
#             # fea_next.append(Deliver(fea_encoder[self.stage - 1 - i],fea))

#         # Output projection
#         out = self.out_proj(fea) + x_ori
#         return out

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
                Block(hidden_dim=dim_stage,d_state=num_states[i]),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = Block(hidden_dim=dim_stage,d_state=num_states[-1])
        

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, 0, bias=False),
                Block(hidden_dim=dim_stage // 2,d_state=num_states[stage - 1 - i]),
                # FFTInteraction_N(dim_stage // 2)
            ]))
            dim_stage //= 2

        # Output projection
        self.out_proj = nn.Conv2d(self.dim, input_dim, 3, 1, 1, bias=False)

    def forward(self, x_ori):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h, w = x_ori.shape
        x = self.embedding(x_ori)
        # Encoder
        fea_encoder = []
        fea = x
        for j, (Blcok, FeaDownSample) in enumerate(self.encoder_layers):
            
            fea = Blcok(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

 
        fea = self.bottleneck(fea)

        # Decoder

        # Decoder
        for i, (FeaUpSample,Fusion,Blcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)            
            fea = Fusion(torch.cat([fea,fea_encoder[self.stage - 1 - i]],dim=1))
            fea = Blcok(fea)
            # fea_next.append(Deliver(fea_encoder[self.stage - 1 - i],fea))

        # Output projection
        out = self.out_proj(fea) + x_ori
        return out


# class UM(nn.Module):
#     def __init__(self, input_dim = 28, dim=28, stage=2, num_states=[4, 8, 16] ):
#         super(UM, self).__init__()
#         self.dim = dim
#         self.stage = stage
#         self.embedding = nn.Conv2d(input_dim, self.dim, 3, 1, 1, bias=False)
#         # Encoder
#         self.encoder_layers = nn.ModuleList([])
#         dim_stage = dim
#         for i in range(stage):
#             self.encoder_layers.append(nn.ModuleList([
#                 InformationFusion(dim_stage),
#                 Block(hidden_dim=dim_stage,d_state=num_states[i]),
#                 nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
#             ]))
#             dim_stage *= 2

#         # Bottleneck
#         self.bottleneck_proj = InformationFusion(dim_stage)
#         self.bottleneck = Block(hidden_dim=dim_stage,d_state=num_states[-1])
        

#         # Decoder
#         self.decoder_layers = nn.ModuleList([])
#         for i in range(stage):
#             self.decoder_layers.append(nn.ModuleList([
#                 nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
#                 nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, 0, bias=False),
#                 InformationFusion(dim_stage // 2),
#                 Block(hidden_dim=dim_stage // 2,d_state=num_states[stage - 1 - i]),
#                 # FFTInteraction_N(dim_stage // 2)
#             ]))
#             dim_stage //= 2

#         # Output projection
#         self.out_proj = nn.Conv2d(self.dim, input_dim, 3, 1, 1, bias=False)

#     def forward(self, x_ori, enc_fea=None, bot_fea=None, dec_fea=None):
#         """
#         x: [b,c,h,w]
#         return out:[b,c,h,w]
#         """
#         b, c, h, w = x_ori.shape
#         x = self.embedding(x_ori)
#         # Encoder
#         fea_encoder = []
#         fea_next = []
#         fea = x
#         for j, (Mixer, Blcok, FeaDownSample) in enumerate(self.encoder_layers):
#             if enc_fea is None:
#                 fea_mixed = fea
#             else:
#                 fea_mixed = Mixer([fea,enc_fea[j]])
#             fea = Blcok(fea_mixed)
#             fea_encoder.append(fea)
#             fea = FeaDownSample(fea)

#         if bot_fea is None:
#             fea = fea
#         else:
#             fea = self.bottleneck_proj([fea,bot_fea])
#         fea = self.bottleneck(fea)
#         fea_bottle = fea
#         # Decoder
#         fea_decoder = []
#         # Decoder
#         for i, (FeaUpSample,Fusion,Mixer,Blcok) in enumerate(self.decoder_layers):
#             fea = FeaUpSample(fea)            
#             fea = Fusion(torch.cat([fea,fea_encoder[self.stage - 1 - i]],dim=1))
#             if dec_fea is None:
#                 fea = fea
#             else:
#                 fea = Mixer([fea,dec_fea[i]])
#             fea = Blcok(fea)
#             fea_decoder.append(fea)
#             # fea_next.append(Deliver(fea_encoder[self.stage - 1 - i],fea))

#         # Output projection
#         out = self.out_proj(fea) + x_ori
#         return out, fea_encoder, fea_bottle, fea_decoder

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class FeatureExtractor(nn.Module):
    def __init__(self, dim=28, expand=2, sparse=False):
        super(FeatureExtractor, self).__init__()
        self.dim = dim
        self.stage = 2
        self.sparse = sparse
        
        # Input projection
        self.in_proj = nn.Conv2d(28, dim, 1, 1, 0, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(2):
            self.encoder_layers.append(nn.ModuleList([
                nn.Conv2d(dim_stage, dim_stage * expand, 1, 1, 0, bias=False),
                nn.Conv2d(dim_stage * expand, dim_stage * expand, 3, 2, 1, bias=False, groups=dim_stage * expand),
                nn.Conv2d(dim_stage * expand, dim_stage*expand, 1, 1, 0, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = ASPP(dim_stage, [3,6], dim_stage)

        # Decoder:
        self.decoder_layers = nn.ModuleList([])
        for i in range(2):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage // 2, dim_stage, 1, 1, 0, bias=False),
                nn.Conv2d(dim_stage, dim_stage, 3, 1, 1, bias=False, groups=dim_stage),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, 0, bias=False),
            ]))
            dim_stage //= 2

        # Output projection

        self.out_conv2 = nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False)
            
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def shift_back_3d(self,inputs,step=2):
        [bs, nC, row, col] = inputs.shape
        for i in range(nC):
            inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
        return inputs
    
    def forward(self,x): 
        x = self.shift_back_3d(x)
        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        # Input projection
        fea = self.lrelu(self.in_proj(x))
        # Encoder
        fea_encoder = []  # [c 2c 4c 8c]
        for (Conv1, Conv2, Conv3) in self.encoder_layers:
            fea_encoder.append(fea)
            fea = Conv3(self.lrelu(Conv2(self.lrelu(Conv1(fea)))))
        # Bottleneck
        fea = self.bottleneck(fea)+fea
        fea_bottle = fea
        # Decoder
        fea_decoder = []
        for i, (FeaUpSample, Conv1, Conv2, Conv3) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Conv3(self.lrelu(Conv2(self.lrelu(Conv1(fea)))))
            fea = fea + fea_encoder[self.stage-1-i]
            fea_decoder.append(fea)
        # Output projection
        out = self.out_conv2(fea) + x

        return out[:, :, :h_inp, :w_inp], fea_encoder, fea_bottle, fea_decoder

# class FeatureExtractor(nn.Module):
#     def __init__(self,in_dim,dim=28,num_states=[4, 8, 16]) -> None:
#         super(FeatureExtractor,self).__init__()

#         self.embedding = nn.Conv2d(in_dim, dim, 3, 1, 1, bias=False)
#         self.layer1 = nn.Sequential(
#             Block(hidden_dim=dim,d_state=num_states[0]),
#             nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=False)
#         )
#         self.layer2 = nn.Sequential(
#             Block(hidden_dim=dim * 2,d_state=num_states[1]),
#             nn.Conv2d(dim * 2, dim * 4, 4, 2, 1, bias=False)
#         )
#         self.layer3 = nn.Sequential(
#             Block(hidden_dim=dim * 4,d_state=num_states[2])
#         )

#     def shift_back_3d(self,inputs,step=2):
#         [bs, nC, row, col] = inputs.shape
#         for i in range(nC):
#             inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
#         return inputs
    
#     def forward(self,r): 
#         r = self.shift_back_3d(r)
#         b, c, h_inp, w_inp = r.shape
#         hb, wb = 16, 16
#         pad_h = (hb - h_inp % hb) % hb
#         pad_w = (wb - w_inp % wb) % wb
#         r = F.pad(r, [0, pad_w, 0, pad_h], mode='reflect')

#         f1 = self.layer1(self.embedding(r))

#         f2 = self.layer2(f1)

#         f3 = self.layer3(f2)
       
#         return  [f3,f2,f1]
    
if __name__ == '__main__':

    x = torch.randn((1,28,256,256))
    net = UM()
    net.eval()
    print(net(x,x).shape)











