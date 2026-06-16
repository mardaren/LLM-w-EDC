# Small LLM + Evidential Uncertainty Quantification

## Aim

This project has two goals, combined into a single pipeline:

1. **Train a language model from scratch.** A small GPT-style decoder transformer is implemented from the ground up in plain PyTorch (no pretrained weights, no HuggingFace model classes) and pretrained on raw text using standard next-token prediction.
2. **Demonstrate uncertainty quantification.** The pretrained model's backbone is reused for a text classification task, with the output head replaced by an **evidential deep learning (EDL)** head instead of a standard softmax classifier. This lets the model express calibrated uncertainty about its own predictions — in particular, flagging inputs that are out-of-distribution or ambiguous rather than confidently guessing wrong.

The project is intentionally scoped to train end-to-end on a MacBook (Apple Silicon, MPS backend), so model size, context length, and dataset size are kept modest by design.

## Project structure

| File | Purpose |
|---|---|
| `requirements.txt` | Project dependencies (torch, tokenizers, datasets, numpy, tqdm, matplotlib, scikit-learn). |
| `prepare_tokenizer.py` | Trains a byte-level BPE tokenizer (vocab size 8,192) on the TinyStories corpus and saves it to `tinystories_tokenizer.json`. |
| `prepare_data.py` | Tokenizes the TinyStories train/validation splits using the trained tokenizer and saves them as flat `uint16` numpy arrays (`train.npy`, `val.npy`). |
| `dataset.py` | `TokenDataset` — slices fixed-length, non-overlapping windows out of the tokenized corpus for next-token-prediction training. |
| `model.py` | The model architecture, built from four classes: `CausalSelfAttention`, `MLPBasic`, `TransformerBlock`, and `LLMBasic` (the full model). |
| `train.py` | Pretraining loop: loads the data and model, trains with cross-entropy loss, periodically evaluates on validation data, and saves checkpoints. |
| `checkpoints/` | Saved model weights (`checkpoint_epoch{N}.pt`) from pretraining runs. |
| `tinystories_tokenizer.json` | The trained tokenizer, needed to encode/decode text consistently with the model's vocabulary. |
| `train.npy` / `val.npy` | Tokenized TinyStories corpus, stored as integer arrays. |

*(Stage 2 files — the classification dataset loader, evidential head, and evidential loss — are not yet implemented; see Status below.)*

## Model architecture

A GPT-2-style decoder-only transformer, implemented without any external model libraries:

- Token embeddings + learned positional embeddings (summed)
- Pre-norm transformer blocks: `LayerNorm → causal self-attention → residual add`, then `LayerNorm → MLP → residual add`
- Causal self-attention via `torch.nn.functional.scaled_dot_product_attention`
- MLP with GELU activation, 4x hidden expansion
- Final `LayerNorm` + linear output head (projects to vocabulary size)

Current hyperparameters: vocab size 8,192, `d_model=256`, context length 128, 4 attention heads, 4 transformer blocks, FFN size 1,024 — roughly 8M parameters total.

## Data

- **Pretraining**: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (`roneneldan/TinyStories`), a corpus of simple, short stories designed to let small models learn coherent language quickly.
- **Classification (planned)**: a labeled text classification dataset with a built-in or constructed out-of-distribution set, to evaluate whether the evidential head's uncertainty estimates correctly separate in-distribution from out-of-distribution inputs.

## How to run

```bash
pip install -r requirements.txt
python prepare_tokenizer.py   # trains the tokenizer
python prepare_data.py        # tokenizes train/val splits into .npy files
python train.py               # pretrains the model, saves checkpoints
```

## Status

- [x] Model architecture implemented and shape-verified
- [x] Tokenizer trained on TinyStories
- [x] Data pipeline (tokenization, non-overlapping windowed dataset)
- [x] Pretraining loop running on MPS, loss decreasing as expected
- [ ] Classification dataset + fine-tuning loop
- [ ] Evidential deep learning head and loss
- [ ] Uncertainty evaluation (in-distribution vs. out-of-distribution)

## Notes

Trained locally on a MacBook using PyTorch's MPS backend. Filenames above reflect a suggested project layout — adjust to match how your own code is organized if it differs.