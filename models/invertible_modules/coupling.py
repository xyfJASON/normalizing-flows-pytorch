import torch
import torch.nn as nn
from torch import Tensor

from .invertible_module import InvertibleModule


class AdditiveCouplingLayer(InvertibleModule):
    def __init__(self, coupling_fn: nn.Module):
        super().__init__()
        self.coupling_fn = coupling_fn

    def forward(self, x1: Tensor, x2: Tensor):
        return x1, x2 + self.coupling_fn(x1)

    def backward(self, y1: Tensor, y2: Tensor):
        return y1, y2 - self.coupling_fn(y1)

    def log_abs_jac(self, x: Tensor):
        return torch.zeros((x.shape[0], ), device=x.device)
