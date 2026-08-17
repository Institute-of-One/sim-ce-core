"""Neural-ODE residual in the Bae-style vector field."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from sim_ce_core.nn.layers import mlp
from sim_ce_core.physio.ode import simulate_ode
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams


class NeuralODEResidual(nn.Module):
    """``r(t, c)`` added to ``dc/dt = A c + b I(t)``. Zero-init last layer."""

    def __init__(self, hidden: int = 32, n_state: int = 3) -> None:
        super().__init__()
        self.net = mlp(1 + n_state, n_state, hidden, zero_last=True)

    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        t_feat = t.reshape(1).to(dtype=state.dtype, device=state.device)
        return self.net(torch.cat([t_feat, state], dim=0))


def simulate_neural_ode(
    params: PhysioParams,
    protocol: InjectionProtocol,
    times_s: Any,
    residual: NeuralODEResidual | None = None,
    **ode_kwargs: Any,
) -> Tensor:
    """Physics ODE, optionally augmented by ``residual(t, c)``."""
    residual_fn = None if residual is None else residual
    return simulate_ode(
        params, protocol, times_s, residual_fn=residual_fn, **ode_kwargs
    )
