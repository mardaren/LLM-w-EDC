import torch
from torch.utils.data import Dataset
import numpy as np

class EmotionDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data/emotion"):
        self.ids = np.load(f"{data_dir}/{split}_ids.npy")
        self.lengths = np.load(f"{data_dir}/{split}_lengths.npy")
        self.labels = np.load(f"{data_dir}/{split}_labels.npy")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.ids[idx].astype(np.int64))
        length = torch.tensor(self.lengths[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, length, y