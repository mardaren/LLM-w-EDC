import numpy as np
import torch
from torch import nn, optim

from models.mlp import MLP
from models.uncertainty_quantificaiton import ClassificationUncertainty, ClassificationEntropy, ClassificationMargin


class BaseModel:

    def get_predictions(self, x_data):
        pass

    def train(self, x_train, y_train, num_epochs=1000):
        pass

    def test(self, x_test, y_test):
        pass


class BinaryNNClassifier(BaseModel):

    def __init__(self, model_shape, lr):
        self.mlp = MLP(model_shape, output_activation='Sigmoid')
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=lr)
        self.quantifier = ClassificationEntropy()

    def get_predictions(self, x_data):
        y_proba = self.mlp(x_data)
        y_proba = torch.cat([1 - y_proba, y_proba], dim=1)
        y_pred = torch.argmax(y_proba, dim=1)
        u = self.quantifier(y_proba)
        return y_pred.detach().numpy(), y_proba.detach().numpy(), u.detach().numpy()

    def train(self, x_train, y_train, num_epochs=1000):
        # Set the model in training mode
        self.mlp.train()

        for epoch in range(num_epochs):
            # Forward pass
            outputs = self.mlp(x_train)

            # Compute the loss
            loss = self.criterion(outputs, y_train)

            # Backward pass and optimization
            self.optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Update weights

            # Print training statistics
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def test(self, x_test, y_test):
        self.mlp.eval()
        y_pred, _, u = self.get_predictions(x_test)
        y_test_cat = np.argmax(y_test.detach().numpy(), axis=1)
        accuracy = sum(y_pred == y_test_cat) / x_test.shape[0]
        print(f"Accuracy: {accuracy}")


class CategoricalNNClassifier(BaseModel):

    def __init__(self, model_shape, lr):
        self.mlp = MLP(model_shape, output_activation='Softmax')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=lr)
        self.quantifier = ClassificationEntropy()

    def get_predictions(self, x_data):
        y_proba = self.mlp(x_data)
        # y_proba = torch.cat([1 - y_proba, y_proba], dim=1)
        y_pred = torch.argmax(y_proba, dim=1)
        u = self.quantifier(y_proba)
        return y_pred.detach().numpy(), y_proba.detach().numpy(), u.detach().numpy()

    def train(self, x_train, y_train, num_epochs=1000):
        # Set the model in training mode
        self.mlp.train()

        for epoch in range(num_epochs):
            # Forward pass
            outputs = self.mlp(x_train)

            # Compute the loss
            loss = self.criterion(outputs, y_train)

            # Backward pass and optimization
            self.optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Update weights

            # Print training statistics
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def test(self, x_test, y_test):
        self.mlp.eval()
        y_pred, _, u = self.get_predictions(x_test)
        y_test_cat = np.argmax(y_test.detach().numpy(), axis=1)
        accuracy = sum(y_pred == y_test_cat) / x_test.shape[0]
        print(f"Accuracy: {accuracy}")
