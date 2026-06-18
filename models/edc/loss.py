import torch
from torch import nn


class EDCLoss(nn.Module):

    def __init__(self, k, annealing_epochs=10):
        super().__init__()
        self.k = torch.tensor(k, dtype=torch.float32)      # düzeltme 1: float32
        self.annealing_epochs = annealing_epochs            # düzeltme 3: sabit yerine parametre

    def forward(self, e_values, y_true, t):

        alpha = e_values + 1
        S = torch.sum(alpha, dim=1, keepdim=True)

        y_pred = alpha / S

        samp_error = torch.sum((y_true - y_pred) ** 2 + y_pred * (1 - y_pred) / (S + 1), dim=1)

        alpha_hat = y_true + (1 - y_true) * alpha
        sum_alpha_hat = torch.sum(alpha_hat, dim=1, keepdim=True)

        kl_div_p1 = (
            torch.lgamma(sum_alpha_hat)
            - torch.lgamma(self.k)                            # düzeltme 2: log -> lgamma
            - torch.sum(torch.lgamma(alpha_hat), dim=1, keepdim=True)
        )
        kl_div_p2 = torch.sum(
            (alpha_hat - 1) * (torch.digamma(alpha_hat) - torch.digamma(sum_alpha_hat)),
            dim=1, keepdim=True
        )

        lambda_t = min(1.0, float(t) / self.annealing_epochs)

        return torch.mean(samp_error + lambda_t * (kl_div_p1 + kl_div_p2).squeeze(1))
