from abc import abstractmethod

import torch.nn as nn


class InvertibleModule(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    @abstractmethod
    def log_abs_jac(self, *args, **kwargs):
        pass
