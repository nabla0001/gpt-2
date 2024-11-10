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
    device: str = 'mps'
    dtype: torch.dtype = torch.float16 # 'torch.float16' for mixed precision
    gradient_accumulation_steps: int = 8
    batch_size: int = 8
    learning_rate: float = 2.5e-4
    warmup_iters = 2000
    min_lr = 6e-5
    # beta1: float = 0.9
    # beta2: float = 0.95
    n_batches: int = 60000
    log_interval: int = 20
    checkpoint_interval: int = 5000
    eval_interval: int = 100
    n_eval_batches: int = 100