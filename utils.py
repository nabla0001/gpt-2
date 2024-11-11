import torch
from config import Config
from data import OpenWebTextData
from pathlib import Path
from tqdm import tqdm


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module,
                  data: OpenWebTextData,
                  device: torch.device,
                  n_batches: int,
                  batch_size: int,
                  seq_len: int) -> dict[str, float]:
    model.eval()
    out = {}

    for split in ['train', 'test']:
        losses = torch.empty(n_batches)
        for batch_num in range(n_batches):
            input_tokens, targets = data.get_batch(split, batch_size, seq_len)
            input_tokens = input_tokens.to(device)
            targets = targets.to(device)

            _, loss = model(input_tokens, targets)
            losses[batch_num] = loss.item()
        out[split] = float(losses.mean())
    model.train()
    return out


def save_checkpoint(path: str | Path,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    **kwargs) -> None:
    checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), **kwargs)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: str | torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def get_learning_rate(batch_num: int, config: Config) -> float:
    import math
    lr_decay_iters = config.n_batches
    # 1) linear warmup for warmup_iters steps
    if batch_num < config.warmup_iters:
        return config.learning_rate * batch_num / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if batch_num > lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (batch_num - config.warmup_iters) / (lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)