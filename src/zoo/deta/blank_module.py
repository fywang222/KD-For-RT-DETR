# Just for test.
import torch 
import torch.nn as nn 

from ...core import register

__all__ = ['BlankModule', ]

@register()
class BlankModule(nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, x):
        return x