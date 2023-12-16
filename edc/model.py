import numpy as np
import torch
from torch import optim

from edc.mlp import MLP
from edc.loss import EDCLoss


class Model:

    def __init__(self, model_shape, lr):
        self.mlp = MLP(model_shape)
        self.criterion = EDCLoss(k=model_shape[-1])
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=lr)

    def train(self, x_train, y_train, num_epochs=1000):
        # Set the model in training mode
        self.mlp.train()

        for epoch in range(num_epochs):
            # Forward pass
            outputs = self.mlp(x_train)

            # Compute the loss
            loss = self.criterion(outputs, y_train, epoch)

            # Backward pass and optimization
            self.optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Update weights

            # Print training statistics
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def test(self, x_test, y_test):
        self.mlp.eval()
        outputs = self.mlp(x_test)
        y_pred = torch.argmax(outputs, dim=1)
        y_test_cat = np.argmax(y_test, axis=1)
        accuracy = sum(y_pred == y_test_cat) / x_test.shape[0]
        print(f"Accuracy: {accuracy}")
