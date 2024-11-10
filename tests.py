import pytest
import torch

from gpt import GPTConfig, MultiheadSelfAttention, FFN, TransformerBlock, GPT
from data import OpenWebTextData
from utils import save_checkpoint, load_checkpoint, evaluate_loss


@pytest.fixture
def config():
    return GPTConfig(context_size=256, embedding_size=512, n_heads=8)


@torch.no_grad()
def test_self_attention_output(config, n=10):
    attn = MultiheadSelfAttention(config)
    x = torch.rand(n, config.context_size, config.embedding_size)
    output = attn(x)
    assert output.shape == x.shape


def test_self_attention_batch_independence(n=100):
    """Set gradient of all tokens for a single sample to 0 and check if
        (1) gradient wrt to all its input tokens is 0
        (2) gradient of no other samples is 0
    """
    config = GPTConfig(context_size=64, embedding_size=32, n_heads=4)
    attn = MultiheadSelfAttention(config)

    x = torch.rand(n, config.context_size, config.embedding_size, requires_grad=True)

    # forward pass
    attn.eval()
    output = attn(x)
    attn.train()

    # set loss gradient of a single sample to 0
    sample_idx = torch.randint(0, n, ())

    mask = torch.ones_like(output)
    mask[sample_idx] = 0
    output = output * mask

    # backward pass
    loss = output.mean()
    loss.backward()

    # check only the correct sample has 0 input gradients
    for sample_i, grad in enumerate(x.grad):
        if sample_i == sample_idx:
            assert torch.all(grad == 0).item(), f'{sample_i} gradient is not 0\n{grad}'
        else:
            for token_grad in grad:
                assert not torch.all(token_grad == 0).item(), f'{sample_i} gradient is 0\n{grad}'


@torch.no_grad()
def test_ffn(config, n=10):
    ffn = FFN(config)
    batch_input = torch.rand(n, config.context_size, config.embedding_size)
    output = ffn(batch_input)
    assert output.shape == batch_input.shape


@torch.no_grad()
def test_transformer_block(n=10):
    config = GPTConfig(context_size=64, embedding_size=32, n_heads=4)
    transformer = TransformerBlock(config)
    x = torch.rand(n, config.context_size, config.embedding_size)
    output = transformer(x)
    assert output.shape == x.shape


@torch.no_grad()
def test_gpt(config, n=10):
    gpt = GPT(config)
    x = torch.randint(0, config.vocab_size, size=(n, config.context_size))
    output, _ = gpt(x)
    assert output.shape == (n, config.context_size, config.vocab_size)


@torch.no_grad()
def test_gpt_loss_with_targets(config, n=10):
    gpt = GPT(config)
    x = torch.randint(0, config.vocab_size, size=(n, config.context_size))
    targets = torch.randint(0, config.vocab_size, size=(n, config.context_size))
    output, loss = gpt(x, targets)

    assert output.shape == (n, config.context_size, config.vocab_size)
    assert type(loss) == torch.Tensor


@torch.no_grad()
@pytest.mark.skipif(not torch.cuda.is_available() and not torch.backends.mps.is_available(),
                    reason='no gpu {cuda, mps} detected')
def test_gpt_on_gpu(config, n=10):
    """Test if GPT works on GPU"""

    x = torch.randint(0, config.vocab_size, size=(n, config.context_size))

    gpt = GPT(config)
    gpt.eval()
    output, _ = gpt(x)

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda')
    gpt_gpu = gpt.to(device)
    x = x.to(device)
    output_gpu, _ = gpt_gpu(x)


def test_data():
    data = OpenWebTextData('data')
    x, y = data.get_batch('train', 4, 12)

    assert x.shape == (4, 12)
    assert y.shape == (4, 12)

    assert x.dtype == torch.int64
    assert y.dtype == torch.int64

    # y is the next token of x
    assert torch.equal(y[:, :-1], x[:, 1:])


@torch.no_grad
def test_evaluate_loss(config):
    device = torch.device('cpu')

    data = OpenWebTextData('data')

    config = GPTConfig(vocab_size=50304, context_size=32, embedding_size=32, n_heads=2)
    gpt = GPT(config)

    losses = evaluate_loss(gpt, data, device, batch_size=2, n_batches=2, seq_len=config.context_size)

    assert type(losses) == dict
    assert 'train' in losses and 'test' in losses
    assert type(losses['train']) == float
    assert type(losses['test']) == float


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

    kwargs = dict(model_args=model_args, batch_num=30, best_loss=1.3)
    save_checkpoint(path, gpt, optimizer, **kwargs)

    checkpoint = load_checkpoint(path, device)

    # check type
    assert type(checkpoint) == dict

    # check checkpoint contains all attributes
    for key in ['model', 'optimizer', 'model_args', 'batch_num', 'best_loss']:
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

