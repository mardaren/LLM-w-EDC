from data_processing import DataHolder
import pickle



data_holder = DataHolder("data/ag_news.csv")
print(data_holder.x_train)
print(data_holder.x_test)

with open('ag_news_x_train.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(data_holder.x_train, file)

with open('ag_news_x_test.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(data_holder.x_test, file)