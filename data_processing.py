import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel


class DataConverter:

    def __init__(self, dir_data: str):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained("bert-base-uncased").to(self.device)

        df = pd.read_csv(dir_data)
        self.text = df["text"].values
        self.label = df["label"].values.reshape(-1, 1)
        del df

        self.embeddings = self.get_embeddings(text=self.text)

    def get_embeddings(self, text: np.array, batch_size=10):
        results = []
        for i in range(0, len(text), batch_size):
            print(f"Encoding texts {i}-{i + batch_size}")
            batch_text = text[i:i + batch_size].tolist()
            # Tokenize and pad dynamically
            encoded_input = self.tokenizer(batch_text, return_tensors='pt', padding='max_length', truncation=True,
                                           max_length=128).to(self.device)
            output = self.model(**encoded_input)
            # output = output.last_hidden_state.real.detach().cpu().numpy()
            # result = np.mean(output, axis=1)
            result = torch.mean(output.last_hidden_state.real, dim=1).detach().cpu().numpy()
            results.append(result)
            del encoded_input, output
            torch.cuda.empty_cache()  # Clear GPU memory
        return np.concatenate(results, axis=0)


class DataHolder:  # Will be fixed ****************************************************************************

    def __init__(self, dir_data):
        self.test_percentage = 0.3
        text_train, self.y_train, text_test, self.y_test = self.get_dataset(dir_data=dir_data)

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
