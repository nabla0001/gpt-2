import pytest

import torch
from config import Config, GPTConfig
from gpt import GPT
from utils import save_checkpoint, load_checkpoint


@pytest.mark.skipif(not torch.cuda.is_available() and not torch.backends.mps.is_available(),
                    reason='no gpu {cuda, mps} detected')
def test_checkpointing(n=10):
    """Save model/optimizer and load on GPU."""
    import pathlib
    import shutil

    device = torch.device('mps:0') if torch.backends.mps.is_available() else torch.device('cuda')

    model_args = dict(context_size=64, embedding_size=32, n_heads=4, vocab_size=100)
    config = GPTConfig(**model_args)
    gpt = GPT(config)
    gpt.to(device)
    gpt.eval()

    optimizer = torch.optim.Adam(gpt.parameters(), lr=0.1)

    x = torch.randint(0, config.vocab_size, size=(n, config.context_size))
    x = x.to(device)
    output, _ = gpt(x)

    tmp_dir = pathlib.Path('temp')
    tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / 'chkpt.pt'

    kwargs = dict(model_args=model_args, batch_num=30, best_loss=1.3, num_tokens_seen=32000)
    save_checkpoint(path, gpt, optimizer, **kwargs)

    checkpoint = load_checkpoint(path, device)

    # check type
    assert type(checkpoint) == dict

    # check checkpoint contains all attributes
    for key in ['model', 'optimizer', 'model_args', 'batch_num', 'best_loss', 'num_tokens_seen']:
        assert key in checkpoint

    # check load model gives same output
    state_dict = checkpoint.get('model')
    model_args = checkpoint.get('model_args')

    config = GPTConfig(**model_args)
    gpt_loaded = GPT(config)
    gpt_loaded.load_state_dict(state_dict)
    gpt_loaded.to(device)
    gpt_loaded.eval()

    output_loaded, _ = gpt_loaded(x)
    assert torch.allclose(output, output_loaded)

    # check model is on GPU
    assert next(gpt_loaded.parameters()).device == device

    # clean up
    shutil.rmtree(tmp_dir)