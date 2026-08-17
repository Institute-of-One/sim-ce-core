"""Differentiable ODE forward model via torchdiffeq."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torchdiffeq import odeint

from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.physio.system import (
    as_time_tensor,
    gather_states,
    iodine_rate_on_interval,
    params_to_tensors,
    simulation_time_grid,
    system_matrix,
)

ResidualFn = Callable[[Tensor, Tensor], Tensor]


def _constant_input_rhs(
    a_mat: Tensor,
    b_vec: Tensor,
    iodine_rate: float,
    residual_fn: ResidualFn | None = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    drive = b_vec * iodine_rate

    def rhs(t: Tensor, state: Tensor) -> Tensor:
        out = a_mat @ state + drive
        if residual_fn is not None:
            out = out + residual_fn(t, state)
        return out

    return rhs


def simulate_ode(
    params: PhysioParams,
    protocol: InjectionProtocol,
    times_s: Any,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-8,
    method: str = "dopri5",
    residual_fn: ResidualFn | None = None,
) -> Tensor:
    """ODE concentrations ``(T, 3)``: aorta, organ, recirculation (mg I / mL)."""
    theta = params_to_tensors(params, device=device, dtype=dtype)
    return simulate_ode_tensors(
        theta,
        protocol,
        times_s,
        delay_s=params.transit_delay_s,
        dtype=dtype,
        device=device,
        rtol=rtol,
        atol=atol,
        method=method,
        residual_fn=residual_fn,
    )


def simulate_ode_tensors(
    theta: dict[str, Tensor],
    protocol: InjectionProtocol,
    times_s: Any,
    *,
    delay_s: float,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-8,
    method: str = "dopri5",
    residual_fn: ResidualFn | None = None,
) -> Tensor:
    """Tensor-parameter ODE forward (autodiff-friendly in ``theta``)."""
    t_eval = as_time_tensor(times_s, device=device, dtype=dtype)
    if t_eval.numel() == 0:
        raise ValueError("times_s must be non-empty")
    a_mat, b_vec = system_matrix(theta)
    t_grid = simulation_time_grid(t_eval, delay_s, protocol.duration_s)
    state = a_mat.new_zeros(3)
    history = [state]
    for i in range(t_grid.numel() - 1):
        t_a = t_grid[i]
        t_b = t_grid[i + 1]
        dt = float((t_b - t_a).item())
        if dt <= 0.0:
            history.append(state)
            continue
        iodine = iodine_rate_on_interval(
            0.5 * (float(t_a.item()) + float(t_b.item())), protocol, delay_s
        )
        rhs = _constant_input_rhs(a_mat, b_vec, iodine, residual_fn)
        span = torch.stack([t_a, t_b])
        sol = odeint(rhs, state, span, method=method, rtol=rtol, atol=atol)
        state = sol[-1]
        history.append(state)
    states = torch.stack(history, dim=0)
    return gather_states(t_grid, states, t_eval)
