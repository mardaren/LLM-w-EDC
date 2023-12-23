import pandas as pd
import numpy as np
import torch

from models import EDCModel, BinaryNNClassifier, CategoricalNNClassifier
from test.plot import plot_colormap


df = pd.read_csv("data/basic/train.csv")

y_train = pd.get_dummies(df.pop('y')).values
# y_train = df.pop('y').values.reshape(-1, 1)
x_train = df.values

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

df = pd.read_csv("data/basic/test.csv")

y_test = pd.get_dummies(df.pop('y')).values
# y_test = df.pop('y').values.reshape(-1, 1)
x_test = df.values

x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

model = CategoricalNNClassifier(model_shape=(2, 16, 2), lr=1e-3)
model.train(x_train, y_train, num_epochs=2000)
model.test(x_train, y_train)
model.test(x_test, y_test)

plot_colormap(model, x_test, y_test, hidden_size=16)
