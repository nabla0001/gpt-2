import datasets
import tiktoken

import numpy as np
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    # creates
    #   data/train.bin  [18GB, 9B tokens]
    #   data/test.bin   [8.7MB, 4M tokens]

    # parameters
    num_proc = 8
    dataset = datasets.load_dataset('openwebtext',
                                    num_proc=num_proc)

    split_dataset = dataset['train'].train_test_split(test_size=0.0005, seed=1234, shuffle=True)

    # tokenize
    enc = tiktoken.get_encoding('gpt2')

    def process(example: dict) -> dict:
        """Tokenize each instance (document) using OpenAI's byte pair encoding (BPE)."""
        tokens = enc.encode_ordinary(example['text'])
        tokens.append(enc.eot_token)
        out = {'tokens': tokens, 'len': len(tokens)}
        return out

    tokenized = split_dataset.map(process,
                                  remove_columns=['text'],
                                  num_proc=num_proc,
                                  desc='tokenizing documents')

    # write out train/test to .bin files contain a single looong sequence of tokens
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    for split, ds in tqdm(tokenized.items(), desc='Writing data to .bin files'):

        set_len = np.sum(ds['len'], dtype=np.uint64)
        filepath = data_dir / f'{split}.bin'
        dtype = np.uint16
        data = np.memmap(filepath, dtype=dtype, mode='w+', shape=(set_len,))

        num_shards = 1024

        start_idx = 0
        for shard_idx in tqdm(range(num_shards), total=num_shards, desc=f'{split}: writing data to .bin'):

            shard = ds.shard(num_shards=num_shards, index=shard_idx, contiguous=True).with_format('numpy')
            shard = np.concatenate(shard['tokens'])

            data[start_idx : start_idx + len(shard)] = shard
            start_idx += len(shard)

        print(f'{split}\t{set_len / 1000}k tokens')
        data.flush()