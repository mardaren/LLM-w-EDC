import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import Tokenizer

CONTEXT_LENGTH = 128

# ---------------------------------------------------------
# 0. Load the SAME tokenizer used in pretraining — do not retrain
# ---------------------------------------------------------
tokenizer = Tokenizer.from_file("data/tinystories_tokenizer.json")
eot_id = tokenizer.token_to_id("<|endoftext|>")
pad_id = eot_id  # reuse as padding id — safe, see note below

os.makedirs("data/emotion", exist_ok=True)

# ---------------------------------------------------------
# 1. Load Emotion dataset
# ---------------------------------------------------------
ds = load_dataset("dair-ai/emotion")

# ---------------------------------------------------------
# 2. Tokenize each example, truncate/pad to CONTEXT_LENGTH,
#    track each example's true length and label
# ---------------------------------------------------------
def encode_split(split, batch_size=1000):
    n = len(split)
    all_ids = np.full((n, CONTEXT_LENGTH), pad_id, dtype=np.uint16)
    all_lengths = np.zeros(n, dtype=np.int32)
    all_labels = np.array(split["label"], dtype=np.int64)

    for i in tqdm(range(0, n, batch_size)):
        batch_texts = split[i : i + batch_size]["text"]
        encodings = tokenizer.encode_batch(batch_texts)

        for j, enc in enumerate(encodings):
            ids = enc.ids[:CONTEXT_LENGTH]      # truncate if too long
            length = len(ids)
            all_ids[i + j, :length] = ids
            all_lengths[i + j] = length

    return all_ids, all_lengths, all_labels


train_ids, train_lengths, train_labels = encode_split(ds["train"])
val_ids, val_lengths, val_labels = encode_split(ds["validation"])
test_ids, test_lengths, test_labels = encode_split(ds["test"])

# ---------------------------------------------------------
# 3. Save everything
# ---------------------------------------------------------
for name, ids, lengths, labels in [
    ("train", train_ids, train_lengths, train_labels),
    ("val", val_ids, val_lengths, val_labels),
    ("test", test_ids, test_lengths, test_labels),
]:
    np.save(f"data/emotion/{name}_ids.npy", ids)
    np.save(f"data/emotion/{name}_lengths.npy", lengths)
    np.save(f"data/emotion/{name}_labels.npy", labels)

print(ds)
print("train:", train_ids.shape, "val:", val_ids.shape, "test:", test_ids.shape)