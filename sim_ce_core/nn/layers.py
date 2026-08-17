"""Small MLP building blocks."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def mlp(
    n_in: int,
    n_out: int,
    hidden: int,
    *,
    n_hidden: int = 2,
    zero_last: bool = False,
) -> nn.Sequential:
    """Tanh MLP; optional zero last layer so a residual starts at 0."""
    layers: list[nn.Module] = [nn.Linear(n_in, hidden), nn.Tanh()]
    for _ in range(n_hidden - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
    last = nn.Linear(hidden, n_out)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


def time_autograd(outputs: Tensor, times: Tensor) -> Tensor:
    """``d outputs / d times`` for ``outputs`` of shape ``(T, C)``."""
    cols = []
    for idx in range(outputs.shape[1]):
        grad = torch.autograd.grad(
            outputs[:, idx].sum(),
            times,
            create_graph=True,
            retain_graph=True,
        )[0]
        cols.append(grad)
    return torch.stack(cols, dim=1)
