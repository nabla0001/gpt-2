"""

GPT
inputs: N, CONTEXT_SIZE
output: N, VOCAB_SIZE

Architecture
  Embedding
  PositionalEncoding
  TransformerBlock
      LayerNorm
      SelfAttention
      LayerNorm
      FFN
  (extra LayerNorm)
  OutputHead

TODO
* test built-in scaled dot attention? https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html


"""
import torch
import torch.nn as nn
from attr import dataclass
from torch import LongTensor, FloatTensor
import math

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    context_size: int = 1024
    embedding_size: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.1

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        transformer_modules = dict(
            token_embedding=nn.Embedding(config.vocab_size, config.embedding_size),
            pos_embedding=nn.Embedding(config.context_size, config.embedding_size),
            embedding_dropout=nn.Dropout(config.dropout),
            transformer_layers=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
            ln=nn.LayerNorm(config.embedding_size)
        )
        self.transformer = nn.ModuleDict(transformer_modules)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        # tie weights between embedding and lm_head projection
        self.transformer.token_embedding.weight = self.lm_head.weight

        # TODO: init weights

    def forward(self, x: LongTensor) -> FloatTensor:
        device = x.device
        N, CONTEXT_SIZE = x.size()

        # position encoding
        pos = torch.arange(0, CONTEXT_SIZE, dtype=torch.long, device=device)

        pos_embed = self.transformer.pos_embedding(pos) # CONTEXT_SIZE, EMBED_D
        token_embed = self.transformer.token_embedding(x) # N, CONTEXT_SIZE, EMBED_D

        out = pos_embed + token_embed
        out = self.transformer.embedding_dropout(out)

        for layer in self.transformer.transformer_layers:
            out = layer(out)

        out = self.transformer.ln(out)

        # N, CONTEXT_SIZE, EMBED_D
        logits = self.lm_head(out[:, -1, :]) # N, VOCAB_SIZE
        return logits


# input:    N, CONTEXT_SIZE, EMBEDDING_SIZE
# output:   N, CONTEXT_SIZE, EMBEDDING_SIZE
class MultiheadSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads

        if config.embedding_size % config.n_heads != 0:
            raise ValueError(f'embedding_size [{config.embedding_size}] must be multiple of n_heads [{config.n_heads}]')

        self.d_attn = config.embedding_size // config.n_heads
        self.scaled_dot_product_factor = 1. / math.sqrt(self.d_attn)

        # each head projects Q, K, V to Dx[D/H]
        # we perform multiplication for all H heads in one go, hence DxD (=Dx[D/H]*H)
        # bias=False because LayerNorm has bias
        self.query = nn.Linear(config.embedding_size, config.embedding_size, bias=False)
        self.key = nn.Linear(config.embedding_size, config.embedding_size, bias=False)
        self.value = nn.Linear(config.embedding_size, config.embedding_size, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)
        self.out_linear = nn.Linear(config.embedding_size, config.embedding_size)

    def forward(self, x: FloatTensor) -> FloatTensor:
        N, T, D = x.size()  # N, CONTEXT_SIZE, EMBEDDING_SIZE
        q = self.query(x)   # N, CONTEXT_SIZE, EMBEDDING_SIZE
        k = self.key(x)     # N, CONTEXT_SIZE, EMBEDDING_SIZE
        v = self.value(x)   # N, CONTEXT_SIZE, EMBEDDING_SIZE

        q = q.view(N, T, self.n_heads, self.d_attn).transpose(1, 2) # N, N_HEADS, CONTEXT_SIZE, D_ATTN
        k = k.view(N, T, self.n_heads, self.d_attn).transpose(1, 2) # N, N_HEADS, CONTEXT_SIZE, D_ATTN
        v = v.view(N, T, self.n_heads, self.d_attn).transpose(1, 2) # N, N_HEADS, CONTEXT_SIZE, D_ATTN

        # scaled dot product attention
        attn = q @ k.transpose(-2, -1) # N, N_HEADS, CONTEXT_SIZE, CONTEXT_SIZE
        attn = attn * self.scaled_dot_product_factor
        # mask out the future
        future_mask = torch.triu(torch.ones_like(attn), diagonal=1).bool()
        attn = attn.masked_fill(future_mask, float('-inf'))

        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        # linear combination of v according to attention weights
        out = attn @ v # N, N_HEADS, CONTEXT_SIZE, D_ATTN
        out = out.transpose(1, 2).reshape(N, T, D)
        out = self.out_linear(out)
        out = self.out_dropout(out)
        return out

# input:    N, CONTEXT_SIZE, EMBEDDING_SIZE
# output:   N, CONTEXT_SIZE, EMBEDDING_SIZE
class FFN(nn.Module):
    """Position-wise Feedforward Network (FFN)"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.embedding_size, 4 * config.embedding_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.embedding_size, config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: FloatTensor) -> FloatTensor:
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out

# input:    N, CONTEXT_SIZE, EMBEDDING_SIZE
# output:   N, CONTEXT_SIZE, EMBEDDING_SIZE
class TransformerBlock(nn.Module):
    """Pre-norm Transformer Block."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.embedding_size)
        self.self_attn = MultiheadSelfAttention(config)
        self.ffn = FFN(config)
        self.ln2 = nn.LayerNorm(config.embedding_size)

    def forward(self, x: FloatTensor) -> FloatTensor:
        out = self.ln1(x)
        out = self.self_attn(out)
        out = out + x
        out = out + self.ffn(self.ln2(out))
        return out