import numpy as np
import torch
from torch import optim, nn
# from torch import nn

from models.mlp import MLP
from models.edc import EDCLoss
from models.standard_nn_classifiers import BaseModel


# class EDCModel(BaseModel):

#     def __init__(self, model_shape, lr, batch_size):
#         super().__init__(batch_size=batch_size)
#         self.k = model_shape[-1]
#         self.mlp = MLP(model_shape)
#         self.criterion = EDCLoss(k=self.k)
#         self.optimizer = optim.Adam(self.mlp.parameters(), lr=lr)

#     def get_predictions(self, x_data):
#         e_values = self.mlp(x_data)
#         alpha = e_values + 1
#         S = torch.sum(alpha, dim=1).reshape(-1, 1)
#         y_proba = alpha / S
#         u = self.k / S
#         y_pred = torch.argmax(y_proba, dim=1)
#         return y_pred.detach().numpy(), y_proba.detach().numpy(), u.detach().numpy()

#     def get_criterion(self, outputs, y_train, epoch=None):
#         return self.criterion(outputs, y_train, epoch)


class EDCLayer(nn.Module):

    def __init__(self, model_shape):
        super().__init__()
        self.k = model_shape[-1]

    def forward(self, e):
        alpha = e + 1

        S = torch.sum(alpha, dim=1, keepdim=True)  # (batch, 1)
        
        p = alpha / S                     # (batch, k)
        u = self.k / S.squeeze(1)         # (batch,)
        
        return alpha, p, u