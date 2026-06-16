import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from models.llm import LLMBasic

# ---------------------------------------------------------
# Dataset: slices fixed-length windows from the tokenized corpus
# ---------------------------------------------------------
class TokenDataset(Dataset):
    # def __init__(self, data_path, context_length):
    #     self.data = np.load(data_path)
    #     self.context_length = context_length

    # def __len__(self):
    #     return len(self.data) - self.context_length

    # def __getitem__(self, idx):
    #     chunk = self.data[idx : idx + self.context_length + 1]
    #     x = torch.from_numpy(chunk[:-1].astype(np.int64))
    #     y = torch.from_numpy(chunk[1:].astype(np.int64))
    #     return x, y
    
    # class TokenDataset(Dataset):
    def __init__(self, data_path, context_length, stride=None, max_tokens=None):
        data = np.load(data_path)
        if max_tokens is not None:
            data = data[:max_tokens]
        self.data = data
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length

    def __len__(self):
        return (len(self.data) - self.context_length) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.data[start : start + self.context_length + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


# ---------------------------------------------------------
# Config — match these to what you used in the sanity check
# ---------------------------------------------------------
VOCAB_SIZE = 8192
D_MODEL = 256
CONTEXT_LENGTH = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 1024

BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 2
EVAL_EVERY = 1000  # steps

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
train_ds = TokenDataset("data/train.npy", CONTEXT_LENGTH)
val_ds = TokenDataset("data/val.npy", CONTEXT_LENGTH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------
# Model + optimizer
# ---------------------------------------------------------
model = LLMBasic(VOCAB_SIZE, D_MODEL, CONTEXT_LENGTH, N_HEADS, N_LAYERS, D_FF).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ---------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------
def train_step(x, y):
    model.train()
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        losses.append(loss.item())
    return sum(losses) / len(losses)

# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
step = 0
for epoch in range(EPOCHS):
    pbar = tqdm(train_loader, desc=f"epoch {epoch}")
    for x, y in pbar:
        loss = train_step(x, y)
        step += 1
        pbar.set_postfix(loss=loss)

        if step % EVAL_EVERY == 0:
            val_loss = evaluate()
            print(f"step {step} | train loss {loss:.4f} | val loss {val_loss:.4f}")

    torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pt")

print("Done.")

# import time

# model.train()
# x, y = next(iter(train_loader))
# x, y = x.to(device), y.to(device)

# # warm up first (first MPS call compiles kernels, skews timing)
# _ = model(x)
# if device.type == "mps":
#     torch.mps.synchronize()

# # time forward only
# start = time.time()
# for _ in range(20):
#     logits = model(x)
# if device.type == "mps":
#     torch.mps.synchronize()
# print("forward only:", (time.time() - start) / 20)

# # time forward + backward
# start = time.time()
# for _ in range(20):
#     logits = model(x)
#     loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
#     loss.backward()
# if device.type == "mps":
#     torch.mps.synchronize()
# print("forward + backward:", (time.time() - start) / 20)

# # time forward + backward + optimizer step
# start = time.time()
# for _ in range(20):
#     optimizer.zero_grad()
#     logits = model(x)
#     loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
#     loss.backward()
#     optimizer.step()
# if device.type == "mps":
#     torch.mps.synchronize()
# print("full step:", (time.time() - start) / 20)