import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

ds = load_dataset("roneneldan/TinyStories", "default")

# ---------------------------------------------------------
# 1. Build and train the tokenizer
# ---------------------------------------------------------
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

trainer = BpeTrainer(
    vocab_size=8192,
    special_tokens=["<|endoftext|>"],
    initial_alphabet=ByteLevel.alphabet(),  # makes sure all 256 byte values are in the base vocab
)

def text_iterator(split, batch_size=1000):
    for i in range(0, len(split), batch_size):
        for text in split[i : i + batch_size]["text"]:
            yield text

tokenizer.train_from_iterator(text_iterator(ds["train"]), trainer=trainer)
tokenizer.save("data/tinystories_tokenizer.json")

eot_id = tokenizer.token_to_id("<|endoftext|>")

# ---------------------------------------------------------
# 2. Tokenize a split into one flat list of token ids
# ---------------------------------------------------------
def tokenize_split(split, batch_size=1000):
    all_ids = []
    for i in tqdm(range(0, len(split), batch_size)):
        batch = split[i : i + batch_size]["text"]
        for enc in tokenizer.encode_batch(batch):
            all_ids.extend(enc.ids)
            all_ids.append(eot_id)
    return all_ids

token_train = tokenize_split(ds["train"])
token_val = tokenize_split(ds["validation"])

# ---------------------------------------------------------
# 3. Convert to numpy and save
# ---------------------------------------------------------
tr_data = np.array(token_train, dtype=np.uint16)
val_data = np.array(token_val, dtype=np.uint16)

np.save("data/pretrain/train.npy", tr_data)
np.save("data/pretrain/val.npy", val_data)

print(ds)
print("train tokens:", len(tr_data))
print("val tokens:", len(val_data))