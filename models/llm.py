from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

# GPT2-like Model ####################################################################

class LLMBasic(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, context_length: int,
             n_heads: int, n_layers: int, d_ff: Optional[int] = None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        # final layernorm and output projection still needed here
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]

        tok = self.token_emb(x)
        positions = torch.arange(seq_len, device=x.device)
        pos = self.pos_emb(positions)
        x = tok + pos

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        x = self.lm_head(x)

        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        qkv = self.qkv_proj(x)

        q, k, v = qkv.split(d_model, dim=-1)

        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        out = self.out_proj(attn_out)

        return out

class MLPBasic(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x
    
class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads=n_heads)
        self.mlp = MLPBasic(d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
# model = LLMBasic(vocab_size=8192, d_model=256, context_length=128,
#                   n_heads=4, n_layers=4, d_ff=1024)
# dummy = torch.randint(0, 8192, (2, 16))  # batch=2, seq_len=16
# out = model(dummy)
# print(out.shape)