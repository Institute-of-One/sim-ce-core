"""Closed-form and ODE Bae-style forwards must agree."""

from __future__ import annotations

import numpy as np
import torch

from sim_ce_core.physio.closed_form import (
    simulate_closed_form,
    simulate_closed_form_tensors,
)
from sim_ce_core.physio.ode import simulate_ode
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.physio.system import params_to_tensors


def test_closed_form_and_ode_agree(
    params: PhysioParams, protocol: InjectionProtocol, times_s: np.ndarray
) -> None:
    closed = simulate_closed_form(params, protocol, times_s)
    ode = simulate_ode(params, protocol, times_s)
    np.testing.assert_allclose(
        closed.detach().cpu().numpy(),
        ode.detach().cpu().numpy(),
        rtol=2e-3,
        atol=1e-3,
    )


def test_pre_arrival_is_zero(params: PhysioParams, protocol: InjectionProtocol) -> None:
    times = np.array([0.0, 2.0, 5.9], dtype=np.float64)
    conc = simulate_closed_form(params, protocol, times).detach().cpu().numpy()
    assert np.max(np.abs(conc)) < 1e-12


def test_closed_form_is_differentiable(
    params: PhysioParams, protocol: InjectionProtocol, times_s: np.ndarray
) -> None:
    theta = params_to_tensors(params)
    q = theta["cardiac_output_ml_s"].detach().clone().requires_grad_(True)
    theta["cardiac_output_ml_s"] = q
    conc = simulate_closed_form_tensors(
        theta, protocol, times_s, delay_s=params.transit_delay_s
    )
    conc.sum().backward()
    assert q.grad is not None
    assert float(q.grad.abs().item()) > 0.0
    assert torch.isfinite(q.grad)
