import pytest
import torch
from gpt import GPTConfig, MultiheadSelfAttention, FFN, TransformerBlock

@pytest.fixture
def config():
    return GPTConfig()

@torch.no_grad()
def test_self_attention_output(config):
    attn = MultiheadSelfAttention(config)

    batch_input = torch.rand(2, config.context_size, config.embedding_size)
    output = attn(batch_input)
    assert output.shape == batch_input.shape

def test_self_attention_batch_independence():
    """Set gradient of all tokens for a single sample to 0 and check if
        (1) gradient wrt to all its input tokens is 0
        (2) gradient of no other samples is 0
    """
    n = 100
    context_size = 64
    embedding_size = 32
    n_heads = 4

    config = GPTConfig(context_size=context_size,
                       embedding_size=embedding_size,
                       n_heads=n_heads)
    attn = MultiheadSelfAttention(config)

    batch_input = torch.rand(n, config.context_size, config.embedding_size, requires_grad=True)

    # forward pass
    attn.eval()
    output = attn(batch_input)
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
    for sample_i, grad in enumerate(batch_input.grad):
        if sample_i == sample_idx:
            assert torch.all(grad == 0).item(), f'{sample_i} gradient is not 0\n{grad}'
        else:
            for token_grad in grad:
                assert not torch.all(token_grad == 0).item(), f'{sample_i} gradient is 0\n{grad}'

def test_ffn(config):
    ffn = FFN(config)
    batch_input = torch.rand(10, config.context_size, config.embedding_size)
    output = ffn(batch_input)
    assert output.shape == batch_input.shape

def test_transformer_block(config):
    transformer = TransformerBlock(config)
    batch_input = torch.rand(10, config.context_size, config.embedding_size)
    output = transformer(batch_input)
    assert output.shape == batch_input.shape

@pytest.mark.skipif(not torch.cuda.is_available() and not torch.backends.mps.is_available(),
                    reason='no gpu {cuda, mps} detected')
def test_gpt_on_gpu():
    """Test if GPT works on GPU and gives equivalent results to CPU."""
    pass