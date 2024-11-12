import torch
import torch.nn as nn
from torch import LongTensor, FloatTensor
from config import GPTConfig
from typing import Optional
import math

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

        self.apply(self._init_weights)

    def forward(self, x: LongTensor, targets: torch.LongTensor = None) -> tuple[FloatTensor, Optional[float]]:
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
        # logits = self.lm_head(out[:, -1, :]) # N, VOCAB_SIZE
        loss = None
        logits = self.lm_head(out) # N, CONTEXT_LENGTH, VOCAB_SIZE

        if targets is not None:
            targets = targets.view(-1)  # N*CONTEXT_SIZE,
            logits_reshaped = logits.view(-1, logits.size(-1))  # N*CONTEXT_SIZE, VOCAB_SIZE
            loss = nn.functional.cross_entropy(logits_reshaped, targets, ignore_index=-1)

        return logits, loss

    def generate(self,
                 x: torch.LongTensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.FloatTensor:
        """Autoregressive sampling

        Args:
            x:              input token sequence of size N, SEQUENCE_LENGTH
            max_new_tokens: maximum generated sequence length
            temperature:    parameter for dividing logits (the closer to 0, the greedier)
            top_k:
        Returns:
            x+generated tokens
        """
        for i in range(max_new_tokens):

            x_cond = x if x.size(1) <= self.config.context_size else x[:, -self.config.context_size:]

            logits, _ = self(x_cond)
            logits = logits[:, -1, :] # for generation, we only need predict at last token
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # normalize to probability distribution
            p = nn.functional.softmax(logits, dim=-1)
            # sample
            next_token = torch.multinomial(p, num_samples=1) # N, 1
            # add to input for next step
            x = torch.cat((x, next_token), dim=1)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


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