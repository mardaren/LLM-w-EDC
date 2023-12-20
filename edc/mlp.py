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


# class CNN(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu3(x)
#         x = self.fc2(x)
#         return x
