from attr import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 100 # 50304
    context_size: int = 32 # 256
    embedding_size: int = 64 # 256
    n_heads: int = 4 # 8
    n_layers: int = 4 # 8
    dropout: float = 0.1

@dataclass
class Config:
    gpt: GPTConfig = GPTConfig()
    batch_size: int = 32
    learning_rate: float = 2.5e-3
    warmup_iters = 2000
    min_lr = 6e-5
    # beta1: float = 0.9
    # beta2: float = 0.95
    n_batches: int = 10 # 60000
    log_interval: int = 1
    checkpoint_interval: int = 1 # 5000