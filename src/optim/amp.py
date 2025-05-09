"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch

from ..core import register


__all__ = ['GradScaler']

GradScaler = register()(torch.GradScaler)

