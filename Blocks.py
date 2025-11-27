import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
  def __init__(self,img_size=224,patch_size=16,in_chans=3,embed_dim=768):
    super().__init__()
    self.proj = nn.Conv2d(in_channels=in_chans,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size)
    self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(1,int(img_size/patch_size)**2+1,embed_dim))
  def forward(self, x):
    x = self.proj(x)
    x = x.flatten(2).transpose(1,2)
    cls_token = self.cls_token.expand(x.shape[0],-1,-1)
    x = torch.cat((cls_token,x),dim=1)
    x = x + self.pos_embed
    return x


class Encoder_Block(nn.Module):
    def __init__(self, dim, heads, drop=0.1, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class Encoder_stack(nn.Module):
  def __init__(self,dim,heads):
    super().__init__()
    self.layers = nn.Sequential(*[Encoder_Block(dim,heads)])
  def forward(self,x):
    x = self.layers(x)
    return x