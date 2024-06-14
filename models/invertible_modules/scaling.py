import torch
import torch.nn as nn
from torch import Tensor

from .invertible_module import InvertibleModule


def expand_dims(x: Tensor, ndim: int):
    return x[(...,) + (None, ) * (ndim - x.ndim)]


class ScalingLayer(InvertibleModule):
    def __init__(self, dim: int):
        super().__init__()
        self.logscales = nn.Parameter(torch.ones((dim, )))

    def forward(self, x: Tensor):
        logscales = expand_dims(self.logscales[None, :], x.ndim)
        return x * torch.exp(logscales)

    def backward(self, y: Tensor):
        logscales = expand_dims(self.logscales[None, :], y.ndim)
        return y * torch.exp(-logscales)

    def log_abs_jac(self, x: Tensor):
        return (torch.sum(self.logscales)).repeat((x.shape[0], ))
