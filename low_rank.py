"""Small, import-safe low-rank adapter for GPT-style Conv1D weights."""

import torch
from torch import nn


class MyConv1D(nn.Module):
    def __init__(self, w, rank: int = 8):
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        in_features, self.nf = w.weight.shape
        self.w = w.weight
        self.bias = w.bias
        self.w.requires_grad_(False)
        self.bias.requires_grad_(False)
        self.u = nn.Parameter(w.weight.new_empty(in_features, rank))
        self.v = nn.Parameter(w.weight.new_zeros(rank, self.nf))
        self.b = nn.Parameter(w.weight.new_zeros(self.nf))
        # The initial update stays exactly zero, but v receives a gradient.
        nn.init.normal_(self.u, std=0.02)

    def setup(self, w):
        """Retain the original setup API; never reset already-trained adapters."""
        if tuple(w.weight.shape) != tuple(self.w.shape):
            raise ValueError("base weight shape mismatch")
        self.w = w.weight
        self.bias = w.bias
        self.w.requires_grad_(False)
        self.bias.requires_grad_(False)
        return self

    def forward(self, x):
        flat = x.reshape(-1, x.size(-1))
        # Frozen parameters still transmit gradients to earlier layers.
        base = torch.addmm(self.bias, flat, self.w)
        update = torch.addmm(self.b, flat, self.u @ self.v)
        return (base + update).reshape(x.shape[:-1] + (self.nf,))
