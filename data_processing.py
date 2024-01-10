import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
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
            result = torch.mean(output.last_hidden_state.real, dim=1).detach().cpu().numpy()
            results.append(result)
            del encoded_input, output
            torch.cuda.empty_cache()  # Clear GPU memory
        return np.concatenate(results, axis=0)


class DataHolder:

    def __init__(self, target):
        self.test_percentage = 0.3
        texts, embeddings, labels, train_idx, test_idx = self.get_dataset(target=target)

        self.ds_train = NLPDataset(texts[train_idx], embeddings[train_idx], labels[train_idx])
        self.ds_test = NLPDataset(texts[test_idx], embeddings[test_idx], labels[test_idx])

    def get_dataset(self, target: str):
        df = pd.read_csv(f"data/raw/{target}.csv")
        data = np.load(f"data/embeddings/{target}.npy")

        # train-test indices split
        test_idx = df.groupby('label').sample(frac=self.test_percentage).index
        train_idx = df.drop(test_idx).index

        labels = pd.get_dummies(df['label']).values
        embeddings = data[:, 1:]
        texts = df['text'].values
        del df, data

        return texts, embeddings, labels, train_idx, test_idx


class NLPDataset(Dataset):

    def __init__(self, text_data, x_data, y_data, transform=None):
        self.text_data = text_data
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self):
        return self.text_data.shape[0]

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
