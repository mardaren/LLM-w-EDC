import torch
from torch import nn


class ClassificationUncertainty(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_proba):
        max_proba, _ = torch.max(y_proba, dim=1)
        return 1 - max_proba


class ClassificationMargin(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_proba):
        max_2_proba, _ = torch.topk(y_proba, 2, dim=1)
        return max_2_proba[:, 0] - max_2_proba[:, 1]


class ClassificationEntropy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_proba):
        log_val = torch.log(y_proba)
        return -torch.sum(y_proba * log_val, dim=1)
