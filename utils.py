import torch
from pathlib import Path
from typing import Optional

def save_checkpoint(path: str | Path,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    **kwargs) -> None:
    checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), **kwargs)
    torch.save(checkpoint, path)

def load_checkpoint(path: str | Path, device: str | torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)