import torch
from data import OpenWebTextData

def test_data():
    data = OpenWebTextData('data')
    x, y = data.get_batch('train', 4, 12)

    assert x.shape == (4, 12)
    assert y.shape == (4, 12)

    assert x.dtype == torch.int64
    assert y.dtype == torch.int64

    # y is the next token of x
    assert torch.equal(y[:, :-1], x[:, 1:])