"""Closed-form linear solution via matrix exponential (Van Loan)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.physio.system import (
    as_time_tensor,
    gather_states,
    iodine_rate_on_interval,
    params_to_tensors,
    simulation_time_grid,
    system_matrix,
)


def expm_affine_step(
    a_mat: Tensor,
    b_vec: Tensor,
    state: Tensor,
    dt: float,
    iodine_rate: float,
) -> Tensor:
    """Advance ``dc/dt = A c + b I`` over a constant-input interval of length ``dt``."""
    if dt <= 0.0:
        return state
    n = a_mat.shape[0]
    z_mat = a_mat.new_zeros((n + 1, n + 1))
    z_mat[:n, :n] = a_mat
    z_mat[:n, n] = b_vec * iodine_rate
    exp_z = torch.linalg.matrix_exp(z_mat * dt)
    return exp_z[:n, :n] @ state + exp_z[:n, n]


def simulate_closed_form(
    params: PhysioParams,
    protocol: InjectionProtocol,
    times_s: Any,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> Tensor:
    """Closed-form concentrations ``(T, 3)`` in mg I / mL."""
    theta = params_to_tensors(params, device=device, dtype=dtype)
    return simulate_closed_form_tensors(
        theta,
        protocol,
        times_s,
        delay_s=params.transit_delay_s,
        dtype=dtype,
        device=device,
    )


def simulate_closed_form_tensors(
    theta: dict[str, Tensor],
    protocol: InjectionProtocol,
    times_s: Any,
    *,
    delay_s: float,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> Tensor:
    """Tensor-parameter closed-form forward (autodiff-friendly in ``theta``)."""
    t_eval = as_time_tensor(times_s, device=device, dtype=dtype)
    if t_eval.numel() == 0:
        raise ValueError("times_s must be non-empty")
    a_mat, b_vec = system_matrix(theta)
    t_grid = simulation_time_grid(t_eval, delay_s, protocol.duration_s)
    state = a_mat.new_zeros(3)
    history = [state]
    for i in range(t_grid.numel() - 1):
        t_a = float(t_grid[i].item())
        t_b = float(t_grid[i + 1].item())
        iodine = iodine_rate_on_interval(0.5 * (t_a + t_b), protocol, delay_s)
        state = expm_affine_step(a_mat, b_vec, state, t_b - t_a, iodine)
        history.append(state)
    states = torch.stack(history, dim=0)
    return gather_states(t_grid, states, t_eval)
