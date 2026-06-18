import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import LLMBasic, EDCLoss, LLMwEDC

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
VOCAB_SIZE = 8192
D_MODEL = 256
CONTEXT_LENGTH = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 1024

HIDDEN_DIM = 128
NUM_CLASSES = ...        # set this to your dataset's number of classes

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20
ANNEALING_EPOCHS = 10

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------------
# Backbone + model
# ---------------------------------------------------------
backbone = LLMBasic(VOCAB_SIZE, D_MODEL, CONTEXT_LENGTH, N_HEADS, N_LAYERS, D_FF)
backbone.load_state_dict(torch.load("checkpoint_epoch1.pt", map_location="cpu"))

model = LLMwEDC(backbone, d_model=D_MODEL, hidden_dim=HIDDEN_DIM,
                 num_classes=NUM_CLASSES).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = EDCLoss(k=NUM_CLASSES, annealing_epochs=ANNEALING_EPOCHS)

# ---------------------------------------------------------
# Data — plug in your classification Dataset/DataLoader here.
# Expected batch shape: (x, lengths, y)
#   x       (batch, seq_len)   token ids, padded
#   lengths (batch,)           true length of each sequence, for pooling
#   y       (batch,)           integer class labels
# ---------------------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------
def train_step(x, lengths, y, epoch):
    model.train()
    x, lengths, y = x.to(device), lengths.to(device), y.to(device)
    y_onehot = F.one_hot(y, num_classes=NUM_CLASSES).float()

    alpha, p, u = model(x, lengths)
    evidence = alpha - 1  # EDCLoss re-adds 1 internally, so undo it here

    loss = criterion(evidence, y_onehot, epoch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(epoch):
    model.eval()
    losses, correct, total, uncertainties = [], 0, 0, []

    for x, lengths, y in val_loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        y_onehot = F.one_hot(y, num_classes=NUM_CLASSES).float()

        alpha, p, u = model(x, lengths)
        evidence = alpha - 1
        loss = criterion(evidence, y_onehot, epoch)
        losses.append(loss.item())

        preds = p.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        uncertainties.append(u.mean().item())

    return (sum(losses) / len(losses),
            correct / total,
            sum(uncertainties) / len(uncertainties))


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
for epoch in range(EPOCHS):
    pbar = tqdm(train_loader, desc=f"epoch {epoch}")
    for x, lengths, y in pbar:
        loss = train_step(x, lengths, y, epoch)
        pbar.set_postfix(loss=loss)

    val_loss, val_acc, val_u = evaluate(epoch)
    print(f"epoch {epoch} | val loss {val_loss:.4f} | val acc {val_acc:.4f} | mean uncertainty {val_u:.4f}")

    torch.save(model.state_dict(), f"classifier_epoch{epoch}.pt")

print("Done.")