from attr import dataclass
import torch

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    context_size: int = 512
    embedding_size: int = 512
    n_heads: int = 8
    n_layers: int = 8
    dropout: float = 0.1

@dataclass
class Config:
    gpt: GPTConfig = GPTConfig()
    device: str = 'mps'
    dtype: torch.dtype = torch.float16 # 'torch.float16' for mixed precision
    grad_accumulation_batches: int = 1
    batch_size: int = 16
    learning_rate: float = 2.5e-3
    warmup_iters = 2000
    min_lr = 6e-5
    # beta1: float = 0.9
    # beta2: float = 0.95
    n_batches: int = 2000
    log_interval: int = 20
    checkpoint_interval: int = 2000 # 5000
    eval_interval: int = 100
    n_eval_batches: int = 100