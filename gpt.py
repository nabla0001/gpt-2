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
    dropout: float = 0.0

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.pos_embedding = nn.Embedding(config.context_size, config.embedding_size)
        # self.transformer = nn.ModuleDict(dict())

    def forward(self, x: LongTensor) -> FloatTensor:
        raise NotImplementedError


class MultiheadSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads

        # each head projects Q, K, V to Dx[D/H]
        # we perform multiplication for all H heads in one go, hence DxD (=Dx[D/H]*H)
        # bias=False because LayerNorm has bias
        self.query = nn.Linear(config.embedding_size, config.embedding_size, bias=False)
        self.key = nn.Linear(config.embedding_size, config.embedding_size, bias=False)
        self.value = nn.Linear(config.embedding_size, config.embedding_size, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)
        self.out_linear = nn.Linear(config.embedding_size, config.embedding_size)

    # input:    N, CONTEXT_SIZE, EMBEDDING_SIZE
    # output:   N, CONTEXT_SIZE, EMBEDDING_SIZE
    def forward(self, x: FloatTensor) -> FloatTensor:
        N, T, D = x.size()  # N, CONTEXT_SIZE, EMBEDDING_SIZE
        q = self.query(x)   # N, CONTEXT_SIZE, EMBEDDING_SIZE
        k = self.key(x)     # N, CONTEXT_SIZE, EMBEDDING_SIZE
        v = self.value(x)   # N, CONTEXT_SIZE, EMBEDDING_SIZE

        D_ATTN = D // self.n_heads
        q = q.view(N, T, self.n_heads, D_ATTN).transpose(1, 2) # N, N_HEADS, CONTEXT_SIZE, D_ATTN
        k = k.view(N, T, self.n_heads, D_ATTN).transpose(1, 2) # N, N_HEADS, CONTEXT_SIZE, D_ATTN
        v = v.view(N, T, self.n_heads, D_ATTN).transpose(1, 2) # N, N_HEADS, CONTEXT_SIZE, D_ATTN

        # scaled dot product attention
        attn = q @ k.transpose(-2, -1) # N, N_HEADS, CONTEXT_SIZE, CONTEXT_SIZE
        attn = attn / math.sqrt(D_ATTN)
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
class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embedding_size)

# test
config = GPTConfig()
attn = MultiheadSelfAttention(config)

N = 10
x = torch.rand(N, config.context_size, config.embedding_size)
y = attn(x)
print(y.shape)
print(y.shape == (N, config.context_size, config.embedding_size))
