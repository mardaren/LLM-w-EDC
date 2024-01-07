from data_processing import DataHolder
from models.standard_nn_classifiers import CategoricalNNClassifier


dh = DataHolder(target="ag_news_original")
model = CategoricalNNClassifier(model_shape=(768, 256, 64, 4), batch_size=128, lr=1e-3)

model.train(dh.ds_train)
model.test(dh.ds_test.x_data, dh.ds_test.y_data)

print()
