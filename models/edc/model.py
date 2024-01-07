import numpy as np
import torch
from torch import optim

from models.mlp import MLP
from models.edc import EDCLoss
from models.standard_nn_classifiers import BaseModel


class EDCModel(BaseModel):

    def __init__(self, model_shape, lr, batch_size):
        super().__init__(batch_size=batch_size)
        self.k = model_shape[-1]
        self.mlp = MLP(model_shape)
        self.criterion = EDCLoss(k=self.k)
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=lr)

    def get_predictions(self, x_data):
        e_values = self.mlp(x_data)
        alpha = e_values + 1
        S = torch.sum(alpha, dim=1).reshape(-1, 1)
        y_proba = alpha / S
        u = self.k / S
        y_pred = torch.argmax(y_proba, dim=1)
        return y_pred.detach().numpy(), y_proba.detach().numpy(), u.detach().numpy()

    def get_criterion(self, outputs, y_train, epoch=None):
        return self.criterion(outputs, y_train, epoch)

    # def train(self, x_train, y_train, num_epochs=1000):
    #     # Set the model in training mode
    #     self.mlp.train()
    #
    #     for epoch in range(num_epochs):
    #         # Forward pass
    #         outputs = self.mlp(x_train)
    #
    #         # Compute the loss
    #         loss = self.criterion(outputs, y_train, epoch)
    #
    #         # Backward pass and optimization
    #         self.optimizer.zero_grad()  # Clear gradients
    #         loss.backward()  # Backpropagation
    #         self.optimizer.step()  # Update weights
    #
    #         # Print training statistics
    #         if (epoch + 1) % 10 == 0:
    #             print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # def test(self, x_test, y_test):
    #     self.mlp.eval()
    #     y_pred, y_proba, u = self.get_predictions(x_test)
    #     y_test_cat = np.argmax(y_test.detach().numpy(), axis=1)
    #     accuracy = sum(y_pred == y_test_cat) / x_test.shape[0]
    #     print(f"Accuracy: {accuracy}")
