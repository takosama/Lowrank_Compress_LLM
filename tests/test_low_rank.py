from types import SimpleNamespace

import pytest
import torch
from torch import nn

from low_rank import MyConv1D


def base():
    return SimpleNamespace(
        weight=nn.Parameter(torch.arange(12.0).reshape(3, 4) / 12),
        bias=nn.Parameter(torch.arange(4.0) / 4),
    )


def test_zero_initial_delta_but_adapter_learns_and_base_is_frozen():
    torch.manual_seed(2)
    w = base()
    layer = MyConv1D(w, rank=2)
    x = torch.randn(2, 5, 3)
    expected = x @ w.weight + w.bias
    torch.testing.assert_close(layer(x), expected)
    original = w.weight.detach().clone()
    opt = torch.optim.SGD(layer.parameters(), lr=0.05)
    for _ in range(3):
        opt.zero_grad()
        layer(x).square().mean().backward()
        opt.step()
    assert torch.count_nonzero(layer.u @ layer.v) > 0
    assert w.weight.grad is None and w.bias.grad is None
    torch.testing.assert_close(w.weight, original)
    assert layer.u.shape == (3, 2)


def test_frozen_base_still_propagates_input_gradient_and_noncontiguous_input():
    layer = MyConv1D(base())
    x = torch.randn(2, 3, 5).transpose(1, 2).requires_grad_()
    layer(x).sum().backward()
    torch.testing.assert_close(x.grad, layer.w.sum(1).expand_as(x))


def test_dtype_and_rank():
    w = base()
    w.weight = nn.Parameter(w.weight.double())
    w.bias = nn.Parameter(w.bias.double())
    layer = MyConv1D(w)
    assert layer.u.dtype == layer.v.dtype == layer.b.dtype == torch.float64
    with pytest.raises(ValueError):
        MyConv1D(w, rank=0)


def test_setup_preserves_trained_adapter_and_rejects_shape_mismatch():
    layer = MyConv1D(base(), rank=2)
    with torch.no_grad():
        layer.v.fill_(1.0)
    trained_v = layer.v.detach().clone()
    replacement = base()
    assert layer.setup(replacement) is layer
    assert layer.w is replacement.weight and not layer.w.requires_grad
    torch.testing.assert_close(layer.v, trained_v)
    incompatible = SimpleNamespace(
        weight=nn.Parameter(torch.zeros(4, 4)), bias=nn.Parameter(torch.zeros(4))
    )
    with pytest.raises(ValueError, match="shape"):
        layer.setup(incompatible)
