import numpy as np
import torch
from torch import nn


class ClassificationUncertainty(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_proba):
        max_proba, _ = np.max(y_proba, axis=1)
        return 1 - max_proba


class ClassificationMargin(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_proba):
        max_2_proba, _ = np.argpartition(y_proba, -2, axis=1)[:, -2:]
        return max_2_proba[:, 0] - max_2_proba[:, 1]


class ClassificationEntropy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_proba):
        log_val = np.log(y_proba + 1e-9)
        return -np.sum(y_proba * log_val, axis=1)
