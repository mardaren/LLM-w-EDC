from models.llm import LLMBasic
from models.edc.layer import EDCLayer, EDCLoss

import torch
from torch import nn


class EvidentialClassifier(nn.Module):

    def __init__(self, in_features, hidden_dim, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus(),         
        )
        self.edc = EDCLayer((hidden_dim, num_classes))

    def forward(self, x):
        z = self.backbone(x)
        e = self.head(z)            
        alpha, p, u = self.edc(e)
        return alpha, p, u
    

class LLMwEDC(nn.Module):
    def __init__(self, backbone: LLMBasic, d_model: int, hidden_dim: int,
                 num_classes: int, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = backbone
        self.classifier = EvidentialClassifier(
            in_features=d_model,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        hidden = self.backbone.forward_features(x)        # (batch, seq_len, d_model)
        batch_idx = torch.arange(x.shape[0], device=x.device)
        pooled = hidden[batch_idx, lengths - 1]            # (batch, d_model)
        alpha, p, u = self.classifier(pooled)
        return alpha, p, u