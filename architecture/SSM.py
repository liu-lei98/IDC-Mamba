import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type = 'WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        elif LayerNorm_type =='WithBias':
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class GatedFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(GatedFeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class WindowMamba(nn.Module):
    """
    The Window MSA partitions the input into non-overlapping windows of size M × M, treating each pixel within the window as a token, and computes self-attention within the window.
    """
    def __init__(self, 
                 dim, 
                 window_size = [16,16],
                 d_state = 16, d_conv = 4, expand = 2 
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        # self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        # self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))


    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape

        # b l d :token length is b0*b1
        x_partial = rearrange(
            x, 'b c (h b0) (w b1) -> (b h w) (b0 b1) c', 
            b0=self.window_size[0], b1=self.window_size[1]
            )
        # x_f = self.mamba(self.norm(x_partial)) + self.skip_scale*x_partial
        x_f = self.mamba(x_partial)
        out = rearrange(x_f, '(b h w) (b0 b1) c -> b c (h b0) (w b1)', h=h // self.window_size[0], w=w // self.window_size[1],
                            b0=self.window_size[0])
        # x_f = self.mamba(x_partial) + self.skip_scale*x_partial
        return out

    
class CrossMamba(nn.Module):
    """
    The Cross MSA divides the input into N × N non-overlapping windows, treating each window as a token, and computes self-attention across the windows.
    """
    def __init__(self, 
                dim,
                d_state = 16, d_conv = 4, expand = 2 
    ):
        super().__init__()
        self.dim = dim
        # self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        # self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))


    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape

        x_partial = rearrange(x, 'b c h w-> b (h w) c',
                                              h=h, w=w)
        # x_f = self.mamba(self.norm(x_partial)) + self.skip_scale*x_partial
        x_f = self.mamba(x_partial) 
        out = rearrange(x_f, 'b (h w) c -> b c h w', h=h, w=w)
        # x_f = self.mamba(x_partial) + self.skip_scale*x_partial
        return out
        

class Block(nn.Module):
    def __init__(
            self,
            hidden_dim,
            d_state = 16, d_conv = 4, expand = 2 ,
            window_size = [16,16] 
    ):
        super().__init__()
        # self.ln_1 = LayerNorm(hidden_dim)
        # self.self_attention_window = CrossMamba(hidden_dim,d_state, d_conv, expand)
        self.ln_2 = LayerNorm(hidden_dim)
        self.self_attention_corss = WindowMamba(hidden_dim, window_size,d_state, d_conv, expand)
        self.ln_3 = LayerNorm(hidden_dim)
        self.ffn = GatedFeedForward(hidden_dim)



    def forward(self, x):
        # input [B C H W]
        # x [B C H W]
        # x = x + self.self_attention_window(self.ln_1(x))
        x = x + self.self_attention_corss(self.ln_2(x))
        x = x + self.ffn(self.ln_3(x))
        return x
    
if __name__=="__main__":
    model = Block(28).cuda()
    x = torch.randn((1,28,256,256)).cuda().float()
    B, H, W, C = x.shape
    y = model(x)
    print(y.shape)