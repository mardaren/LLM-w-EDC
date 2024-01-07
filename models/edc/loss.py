import torch
from torch import nn


def gamma(x):
    return torch.exp(torch.lgamma(x))


class EDCLoss(nn.Module):

    def __init__(self, k):
        super().__init__()
        self.k = torch.tensor(k, dtype=torch.int32)

    def forward(self, e_values, y_true, t):

        alpha = e_values + 1
        S = torch.sum(alpha, dim=1).reshape(-1, 1)

        y_pred = alpha / S

        samp_error = torch.sum((y_true-y_pred) ** 2 + y_pred * (1 - y_pred) / (S + 1), dim=1)

        alpha_hat = y_true + (1 - y_true) * alpha
        sum_alpha_hat = torch.sum(alpha_hat, dim=1).reshape(-1, 1)

        kl_div_p1 = torch.lgamma(sum_alpha_hat) - torch.log(self.k) - torch.sum(torch.lgamma(alpha_hat), dim=1).reshape(-1, 1)
        kl_div_p2 = torch.sum((alpha_hat - 1) * (torch.digamma(alpha_hat) - torch.digamma(sum_alpha_hat)), dim=1).reshape(-1, 1)

        ranges = torch.tensor([1.0, t/100])
        lambda_t = torch.min(ranges)

        return torch.sum(samp_error) + lambda_t * torch.sum((kl_div_p1 + kl_div_p2))

