import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, model_shape):
        super().__init__()
        self.fcs = []
        self.activations = []
        for l_idx in range(len(model_shape) - 1):
            self.fcs.append(nn.Linear(model_shape[l_idx], model_shape[l_idx + 1]))
            self.activations.append(nn.ReLU())

        self.fcs = nn.ParameterList(self.fcs)
        self.activations = nn.ParameterList(self.activations)

    def forward(self, x):
        for l_idx in range(len(self.fcs)):
            x = self.fcs[l_idx](x)
            x = self.activations[l_idx](x)
        return x
    