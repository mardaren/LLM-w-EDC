import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

from models.mlp import MLP
from models.uncertainty_quantification import ClassificationUncertainty, ClassificationEntropy, ClassificationMargin
from sklearn.metrics import f1_score


class BaseModel:

    def __init__(self, batch_size):
        self.mlp = None
        self.criterion = None
        self.optimizer = None

        self.batch_size = batch_size

    def get_predictions(self, x_data):
        return None, None, None

    def train(self, tr_dataset, num_epochs=1000):

        dl = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True)

        # Set the model in training mode
        self.mlp.train()

        for epoch in range(num_epochs):
            losses = []
            for x_train, y_train in dl:
                # Forward pass
                outputs = self.mlp(x_train)

                # Compute the loss
                loss = self.get_criterion(outputs, y_train, epoch)
                losses.append(loss.item())

                # Backward pass and optimization
                self.optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights

            # Print training statistics
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {sum(losses)/len(losses):.4f}')

    def get_criterion(self, outputs, y_train, epoch):
        return self.criterion(outputs, y_train)

    def test(self, x_test, y_test):
        self.mlp.eval()
        y_pred, _, u = self.get_predictions(x_test)
        y_test_cat = np.argmax(y_test.detach().numpy(), axis=1)
        accuracy = sum(y_pred == y_test_cat) / x_test.shape[0]
        f1 = f1_score(y_test_cat, y_pred, average='weighted')
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")


class BinaryNNClassifier(BaseModel):

    def __init__(self, model_shape, lr, batch_size):
        super().__init__(batch_size=batch_size)
        self.mlp = MLP(model_shape, output_activation='Sigmoid')
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=lr)
        self.quantifier = ClassificationEntropy()

    def get_predictions(self, x_data):
        self.mlp.eval()
        y_proba = self.mlp(x_data).detach().numpy()
        y_proba = np.concatenate([1 - y_proba, y_proba], axis=1)
        y_pred = np.argmax(y_proba, axis=1)
        u = self.quantifier(y_proba)
        return y_pred, y_proba, u


class CategoricalNNClassifier(BaseModel):

    def __init__(self, model_shape, lr, batch_size):
        super().__init__(batch_size=batch_size)
        self.mlp = MLP(model_shape, output_activation='Softmax')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=lr)
        self.quantifier = ClassificationEntropy()

    def get_predictions(self, x_data):
        self.mlp.eval()
        y_proba = self.mlp(x_data).detach().numpy()
        y_pred = np.argmax(y_proba, axis=1)
        u = self.quantifier(y_proba)
        return y_pred, y_proba, u
