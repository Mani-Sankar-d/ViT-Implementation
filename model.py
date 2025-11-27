import torch
import torch.nn as nn
import numpy as np
from Blocks import Encoder_stack

class Classfier(nn.Module):
  def __init__(self,dim,heads,num_classes):
    super().__init__()
    self.enc_stk = Encoder_stack(dim,heads)
    self.mlp_head = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim,num_classes)
    )
  def forward(self,x):
    x=self.Encoder_stack(x)
    cls_token = x[:,0]
    out = self.mlp_head(cls_token)
    return out