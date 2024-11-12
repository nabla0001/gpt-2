from attr import dataclass
import torch

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    context_size: int = 512
    embedding_size: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.1

@dataclass
class Config:
    gpt: GPTConfig = GPTConfig()
    dtype: torch.dtype = torch.float16 # 'torch.float16' for mixed precision
    gradient_accumulation_steps: int = 8
    grad_clip: float = 1.0
    batch_size: int = 8
    learning_rate: float = 6e-4
    warmup_iters = 3000
    min_lr = 6e-5
    beta1: float = 0.9
    beta2: float = 0.95
    # weight_decay: float = 0.1
    n_batches: int = 600000
    log_interval: int = 20
    eval_interval: int = 100
    n_eval_batches: int = 100