import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel


class DataHolder:

    def __init__(self, dir_data: str):

        self.test_percentage = 0.3
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained("bert-base-uncased")

        text_train, self.y_train, text_test, self.y_test = self.get_dataset(dir_data=dir_data)

        # We can add preprocessing **********************************************************************

        # Embedding selection
        self.x_train = self.get_embeddings(text=text_train)
        self.x_test = self.get_embeddings(text=text_test)

        # Maybe we can change it to dataset

    def get_dataset(self, dir_data: str):
        df = pd.read_csv(dir_data)

        # Stratification may be needed here ***********************************************************
        df_test = df.sample(frac=self.test_percentage)
        df_train = df.drop(df_test.index)

        text_train = df_train["text"].values
        y_train = df_train["label"].values
        text_test = df_test["text"].values
        y_test = df_test["label"].values

        return text_train, y_train, text_test, y_test

    def get_embeddings(self, text: np.array):
        encoded_input = self.tokenizer(text.tolist(), return_tensors='pt', padding=True, truncation=True)
        output = self.model(**encoded_input)
        return output
