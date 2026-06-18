# Small LLM + Evidential Uncertainty Quantification

## Aim

This project has two goals, combined into a single pipeline:

1. **Train a language model from scratch.** A small GPT-style decoder transformer is implemented from the ground up in plain PyTorch (no pretrained weights, no HuggingFace model classes) and pretrained on raw text using standard next-token prediction.
2. **Demonstrate uncertainty quantification.** The pretrained model's backbone is reused for a text classification task, with the output head replaced by an **evidential deep learning (EDL)** head instead of a standard softmax classifier. This lets the model express calibrated uncertainty about its own predictions — in particular, flagging inputs that are out-of-distribution or ambiguous rather than confidently guessing wrong.

The project is intentionally scoped to train end-to-end on a MacBook (Apple Silicon, MPS backend), so model size, context length, and dataset size are kept modest by design.

## Project structure

| File / folder | Purpose |
|---|---|
| `requirements.txt` | Project dependencies (torch, tokenizers, datasets, numpy, tqdm, matplotlib, scikit-learn). |
| `data/tiny_stories.py` | Trains a byte-level BPE tokenizer (vocab size 8,192) on TinyStories and tokenizes the train/val splits into `data/train.npy` / `data/val.npy`. |
| `data/tinystories_tokenizer.json` | The trained tokenizer, saved here after running `data/tiny_stories.py`. |
| `pretrain.py` | Pretraining loop: `TokenDataset` (windowed next-token prediction with configurable stride), training with AdamW + cross-entropy, periodic validation, and checkpoint saving. |
| `train_local.py` | Classification fine-tuning scaffold: loads a pretrained backbone, attaches `LLMwEDC`, and trains with `EDCLoss`. Plug in a `DataLoader` to use. |
| `models/llm.py` | Core model architecture: `CausalSelfAttention`, `MLPBasic`, `TransformerBlock`, and `LLMBasic`. Exposes `forward_features()` so the backbone can be reused downstream. |
| `models/llmwedc.py` | `EvidentialClassifier` and `LLMwEDC` — wraps the pretrained backbone with an EDL classification head. |
| `models/edc/layer.py` | `EDCLayer` — converts evidence outputs to Dirichlet parameters (α), class probabilities, and a per-sample uncertainty scalar. |
| `models/edc/loss.py` | `EDCLoss` — EDL training loss: MSE-style fitting term over Dirichlet predictions plus an annealed KL divergence penalty. |
| `models/standard_nn_classifiers.py` | `BinaryNNClassifier`, `CategoricalNNClassifier` — standard softmax baselines for comparison. |
| `models/mlp.py` | Standalone `MLP` helper module. |
| `utils.py` | Uncertainty evaluation utilities: `calculate_likelihood`, `calculate_unc_success`, `get_most/least_uncertain_sample`. |
| `test/` | Plotting scripts (`plot.py`, `plot_unc.py`) and saved result figures. |
| `checkpoint_epoch{N}.pt` | Pretrained backbone checkpoints saved after each epoch (gitignored). |
| `data/train.npy`, `data/val.npy` | Tokenized TinyStories corpus (gitignored; regenerate by running `data/tiny_stories.py`). |

## Model architecture

A GPT-2-style decoder-only transformer, implemented without any external model libraries:

- Token embeddings + learned positional embeddings (summed)
- Pre-norm transformer blocks: `LayerNorm → causal self-attention → residual add`, then `LayerNorm → MLP → residual add`
- Causal self-attention via `torch.nn.functional.scaled_dot_product_attention`
- MLP with GELU activation, 4x hidden expansion
- Final `LayerNorm` + linear output head (projects to vocabulary size)
- `forward_features()` exposes the per-token hidden states so the backbone can be reused for downstream classification

Current hyperparameters: vocab size 8,192, `d_model=256`, context length 128, 4 attention heads, 4 transformer blocks, FFN size 1,024 — roughly 8M parameters total.

## Evidential deep learning head

For classification, the LM head is replaced by an `EvidentialClassifier`:

1. A two-layer ReLU MLP projects the pooled hidden state to evidence logits (Softplus-activated, so always positive).
2. `EDCLayer` converts evidence *e* to Dirichlet parameters α = e + 1, class probabilities p = α/S, and per-sample uncertainty u = K/S (where S = Σα, K = number of classes).
3. High uncertainty (u → 1) signals that the model has little evidence; low uncertainty (u → 0) signals a confident, evidence-backed prediction.

`EDCLoss` trains the head with an MSE-style fitting term combined with an annealed KL divergence penalty that shrinks vacuous Dirichlet posteriors toward the uniform prior.

## Data

- **Pretraining**: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (`roneneldan/TinyStories`), a corpus of simple, short stories designed to let small models learn coherent language quickly.
- **Classification**: any labeled text dataset — plug a `Dataset`/`DataLoader` that yields `(x, lengths, y)` batches into `train_local.py`.

## How to run

```bash
pip install -r requirements.txt

# Build tokenizer + tokenize corpus (saves data/tinystories_tokenizer.json, data/train.npy, data/val.npy)
python data/tiny_stories.py

# Pretrain the LM (saves checkpoint_epoch{N}.pt)
python pretrain.py

# Fine-tune with EDL head (plug in your classification DataLoader first)
python train_local.py
```

## Status

- [x] Model architecture implemented and shape-verified
- [x] Tokenizer trained on TinyStories
- [x] Data pipeline (tokenization, windowed dataset with configurable stride)
- [x] Pretraining loop running on MPS, loss decreasing as expected (2 epochs completed)
- [x] Evidential deep learning head (`EDCLayer`) and loss (`EDCLoss`) implemented
- [x] `LLMwEDC` — pretrained backbone + EDL classifier wired together
- [x] Classification fine-tuning loop scaffolded (`train_local.py`)
- [ ] Classification dataset and DataLoader wired in
- [ ] Uncertainty evaluation (in-distribution vs. out-of-distribution analysis)