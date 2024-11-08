import torch
import numpy as np
from pathlib import Path

class OpenWebTextData:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def get_batch(self,
                  split: str = 'train',
                  batch_size: int = 64,
                  seq_len: int  = 512) -> tuple[torch.Tensor, torch.Tensor]:
        if split == 'train':
            data = np.memmap(self.data_dir / 'train.bin', dtype=np.uint16, mode='r')
        elif split == 'test':
            data = np.memmap(self.data_dir / 'test.bin', dtype=np.uint16, mode='r')
        else:
            raise ValueError(f'invalid split: {split}, must be either {'train', 'test'}.')

        ix = torch.randint(len(data) - seq_len, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + seq_len]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + seq_len]).astype(np.int64)) for i in ix])
        return x, y