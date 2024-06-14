import torch
import torch.nn as nn
from torch import Tensor

from .invertible_modules import InvertibleModule, AdditiveCouplingLayer, ScalingLayer


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dim: int, n_layers: int):
        super().__init__()
        layers = [nn.Linear(in_dim, dim), nn.ReLU()]
        for i in range(n_layers):
            layers.extend([nn.Linear(dim, dim), nn.ReLU()])
        layers.append(nn.Linear(dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.layers(x)


class NICE(InvertibleModule):
    def __init__(self, data_dim: int, n_coupling_layers: int, mlp_dim: int, mlp_n_layers: int):
        super().__init__()
        assert data_dim % 2 == 0, 'Data dimension must be even.'

        self.couplings = nn.ModuleList([])
        for i in range(n_coupling_layers):
            mlp = MLP(in_dim=data_dim // 2, out_dim=data_dim // 2, dim=mlp_dim, n_layers=mlp_n_layers)
            self.couplings.append(AdditiveCouplingLayer(mlp))
        self.scaling = ScalingLayer(dim=data_dim)

    def forward(self, x: Tensor):
        b, c, h, w = x.shape
        x = x.flatten(start_dim=1)
        x1, x2 = x[:, ::2], x[:, 1::2]
        for coupling in self.couplings:
            x1, x2 = coupling(x1, x2)
            x1, x2 = x2, x1
        y = torch.empty_like(x)
        y[:, ::2], y[:, 1::2] = x1, x2
        y = self.scaling(y)
        y = y.view(b, c, h, w)
        return y, self.log_abs_jac(x)

    def backward(self, y: Tensor):
        b, c, h, w = y.shape
        y = y.flatten(start_dim=1)
        y = self.scaling.backward(y)
        y1, y2 = y[:, ::2], y[:, 1::2]
        for coupling in reversed(self.couplings):
            y1, y2 = y2, y1
            y1, y2 = coupling.backward(y1, y2)
        x = torch.empty_like(y)
        x[:, ::2], x[:, 1::2] = y1, y2
        x = x.view(b, c, h, w)
        return x

    def log_abs_jac(self, x: Tensor):
        return self.scaling.log_abs_jac(x)
