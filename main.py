import numpy as np

from data_processing import DataHolder
from models.standard_nn_classifiers import CategoricalNNClassifier
from models.edc.model import EDCModel
from utils import *


# Read the data
dh = DataHolder(target="clickbait_notclickbait_original")

# Standard Model
# model = CategoricalNNClassifier(model_shape=(768, 256, 64, 4), batch_size=256, lr=1e-3)
# model.train(dh.ds_train)

# Edc Model
model = EDCModel(model_shape=(768, 256, 128, 64, 2), batch_size=256, lr=5e-4)
model.train(dh.ds_train, num_epochs=500)

# Testing Phase ---------------------------------------------------------------------------
# Train Data
print("Training Data:")
model.test(dh.ds_train.x_data, dh.ds_train.y_data)
# Test Data
print("Test Data:")
model.test(dh.ds_test.x_data, dh.ds_test.y_data)

# Likelihood using uncertainty as a distribution
y_pred, _, u = model.get_predictions(dh.ds_test.x_data)
y_true = np.argmax(dh.ds_test.y_data.detach().numpy(), axis=1)
likelihood = calculate_likelihood(y_true, y_pred, u)
print(f"Likelihood: {likelihood}")
