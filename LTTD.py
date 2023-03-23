import math
from collections import OrderedDict
from functools import partial
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, \
    resample_abs_pos_embed, RmsNorm

from einops import rearrange, einsum

__all__ = ['LTTD']  

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class Attention(nn.Module):
    fast_attn: Final[bool]
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm,):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # FIXME

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False, drop=0., attn_drop=0.,
            init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class LETBlock(nn.Module):

    def __init__(self, let_index):
        super().__init__()
        self.transformer = nn.Sequential(*[Block(
                                                dim=384,
                                                num_heads=6,
                                                mlp_ratio=4.,
                                                qkv_bias=True,
                                                qk_norm=False,
                                                init_values=None,
                                                drop=0.,
                                                attn_drop=0.,
                                                drop_path=0.,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                                act_layer=nn.GELU
                                            )
                                            for i in range(3)])

        self.conv3d = nn.Conv3d(64,
                               64,
                               kernel_size=(3, 3, 3),
                               stride=(1, 1, 1),
                               padding=(1, 1, 1),
                               bias=False)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.mlp_attn = nn.Linear(int(4096/ ((2**let_index)**2)), 384)


    def forward(self, x, y):
        x = self.transformer(x)

        y = self.conv3d(y)
        y = self.pool(y)
        
        x_a = y.flatten(3)
        x_a = x_a.permute(0, 2, 1, 3)
        x_a = x_a.flatten(2)
        x_a = self.mlp_attn(x_a)
        return x, y

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class LSE(nn.Module):
    def __init__(self):
        super().__init__()
        # as stated in Implementation Details
        self.in_planes = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(7, 7, 7),
                               stride=(1, 2, 2),
                               padding=(7 // 2, 3, 3),
                               bias=False)
        self.conv2_x = self._make_layer(BasicBlock, 64, 2)
        self.embedding = nn.Linear(4096, 384)

        # traditional ViT embedding
        # self.embedding = nn.Linear(768, 384)
        
        self.conv3d = nn.Conv3d(3,
                               64,
                               kernel_size=(3, 3, 3),
                               stride=(1, 1, 1),
                               padding=(1, 1, 1),
                               bias=False)

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.mlp_attn = nn.Linear(4096, 384)
        
        self.norm = nn.LayerNorm(384, eps=1e-6)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.temp_embedding = nn.Parameter(torch.zeros(1, 17, 384))
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)
    def forward(self, x):
        bHW, c, t, p0, p1 = x.shape

        y = self.conv3d(x)
        y = self.pool(y)

        x_a = y.flatten(3)
        x_a = x_a.permute(0, 2, 1, 3)
        x_a = x_a.flatten(2)
        x_a = self.mlp_attn(x_a)

        # x = x.permute(0, 2, 1, 3, 4).flatten(2)
        # x = self.embedding(x)
        x = self.conv1(x)
        x = self.conv2_x(x).permute(0, 2, 1, 3, 4)
        x = x.flatten(2)
        x = self.embedding(x)

        x = self.norm(x)
        x = torch.cat([self.cls_token.expand(x.shape[0],-1,-1), x], dim=1)
        x = x+self.temp_embedding

        x[:,1:] = x[:,1:] * x_a.sigmoid()
        return x, y

class LET(nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = LETBlock(let_index=1)
        self.module2 = LETBlock(let_index=2)
        self.module3 = LETBlock(let_index=3)

    def forward(self, x, y):
        x,y = self.module1(x,y)
        x,y = self.module2(x,y)
        x,y = self.module3(x,y)
        return x, y

class LocBranch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, H, W, T_1, c = x.shape
        x = x[:, :, :, 1:].mean(3)
        x = x.view(b, H*W, c)
        x = F.normalize(x,dim=2) 
        x = einsum(x, x ,'b n0 c, b n1 c-> b n0 n1')
        return x

class ClsBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Sequential(*[Block(
                                                dim=384,
                                                num_heads=6,
                                                mlp_ratio=4.,
                                                qkv_bias=True,
                                                qk_norm=False,
                                                init_values=None,
                                                drop=0.,
                                                attn_drop=0.,
                                                drop_path=0.,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                                act_layer=nn.GELU
                                            ) for i in range(3)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.fc_head = nn.Linear(384, 2)

    def forward(self, x):
        b, H, W, T_1, c = x.shape
        x = x[:, :, :, 0]
        x = x.view(b, H*W, c)
        x = torch.cat([self.cls_token.expand(x.shape[0],-1,-1), x], dim=1)
        x = self.transformer(x)
        x = x[:,0]
        x = self.fc_head(x)
        return x

class LTTD(nn.Module):
    def __init__(self):
        super().__init__()
        self.lst = LSE()
        self.let = LET()
        self.clsbranch = ClsBranch()
        self.locbranch = LocBranch()

    def forward(self, x):
        H = W = 14
        b, c, t, h, w = x.shape
        x = x.view(b, c, t, 14, 16, 14, 16).permute(0,3,5,1,2,4,6)
        x = rearrange(x, 'b H W c t p0 p1 -> (b H W) c t p0 p1')

        x,y = self.lst(x) 
        x,y = self.let(x,y)
        # torch.Size([b*196, 17, 384])
        x = rearrange(x, '(b H W) t c -> b H W t c', H=14, W=14)
        sim = self.locbranch(x)
        x = self.clsbranch(x)
        return x, sim


if __name__ == "__main__":
    a = torch.randn(2, 3, 16, 224, 224).cuda(0)
    module = LTTD().cuda(0)
    x, sim = module(a)
    print(x.shape)
    print(sim.shape)
