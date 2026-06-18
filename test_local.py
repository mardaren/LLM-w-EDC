import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from models import LLMBasic, LLMwEDC
from data.dataset import EmotionDataset

# ---------------------------------------------------------
# Config — must match training
# ---------------------------------------------------------
VOCAB_SIZE = 8192
D_MODEL = 256
CONTEXT_LENGTH = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 1024
HIDDEN_DIM = 128
NUM_CLASSES = 6

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
CHECKPOINT_PATH = "classifier_epoch19.pt"   # set to whichever epoch you want to evaluate
BATCH_SIZE = 32
TOP_K = 10

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------------------------------------------------
# Tokenizer — needed to turn ids back into readable text.
# Must attach the ByteLevel decoder, or output is garbled
# byte-symbols instead of real text.
# ---------------------------------------------------------
tokenizer = Tokenizer.from_file("data/tinystories_tokenizer.json")
tokenizer.decoder = ByteLevelDecoder()

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
backbone = LLMBasic(VOCAB_SIZE, D_MODEL, CONTEXT_LENGTH, N_HEADS, N_LAYERS, D_FF)
model = LLMwEDC(backbone, d_model=D_MODEL, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.to(device)
model.eval()

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
test_dataset = EmotionDataset("test")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------
# Run inference, collect accuracy + (uncertainty, true_label, pred_label, idx)
# ---------------------------------------------------------
results = []
correct = 0
total = 0

with torch.no_grad():
    idx = 0
    for x, lengths, y in test_loader:
        x, lengths = x.to(device), lengths.to(device)
        alpha, p, u = model(x, lengths)

        preds = p.argmax(dim=1).cpu()
        u = u.squeeze().cpu()  # handles (batch,1) or (batch,) shape

        correct += (preds == y).sum().item()
        total += y.size(0)

        for i in range(x.size(0)):
            results.append((u[i].item(), y[i].item(), preds[i].item(), idx))
            idx += 1

accuracy = correct / total

# ---------------------------------------------------------
# Most uncertain first
# ---------------------------------------------------------
results.sort(key=lambda r: r[0], reverse=True)
top_results = results[:TOP_K]

# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------
print(f"Test accuracy: {accuracy:.4f} ({correct}/{total})\n")

print(f"Top {TOP_K} most uncertain test examples:\n")
for rank, (u_val, true_label, pred_label, example_idx) in enumerate(top_results, 1):
    real_len = test_dataset.lengths[example_idx]
    ids = test_dataset.ids[example_idx][:real_len].tolist()
    text = tokenizer.decode(ids)

    print(f"{rank}. uncertainty={u_val:.4f} | true={LABEL_NAMES[true_label]} | predicted={LABEL_NAMES[pred_label]}")
    print(f'   "{text}"\n')