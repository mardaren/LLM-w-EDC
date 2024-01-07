from data_processing import DataHolder
from models.standard_nn_classifiers import CategoricalNNClassifier
from models.edc.model import EDCModel


dh = DataHolder(target="ag_news_original")
# model = CategoricalNNClassifier(model_shape=(768, 256, 64, 4), batch_size=128, lr=1e-3)
#
# model.train(dh.ds_train)
# model.test(dh.ds_test.x_data, dh.ds_test.y_data)

model = EDCModel(model_shape=(768, 256, 128, 64, 4), batch_size=128, lr=5e-4)

model.train(dh.ds_train, num_epochs=500)

# Train Data
model.test(dh.ds_train.x_data, dh.ds_train.y_data)
# Test Data
model.test(dh.ds_test.x_data, dh.ds_test.y_data)

