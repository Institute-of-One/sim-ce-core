"""PINN hybrid fit and Neural-ODE residual (synthetic, no network)."""

from __future__ import annotations

import numpy as np
import torch

from sim_ce_core.data.synthetic import generate_synthetic
from sim_ce_core.nn.neural_ode import NeuralODEResidual, simulate_neural_ode
from sim_ce_core.nn.pinn import fit_pinn
from sim_ce_core.physio.ode import simulate_ode
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.validate.metrics import nrmse


def test_pinn_hybrid_fits_noiseless(
    params: PhysioParams, protocol: InjectionProtocol
) -> None:
    times = np.linspace(0.0, 60.0, 31)
    series = generate_synthetic(params, protocol, times, seed=0)
    result = fit_pinn(
        series.times_s,
        series.curves_hu[:, :2],
        protocol,
        params,
        mode="hybrid",
        init={
            "central_blood_volume_ml": params.central_blood_volume_ml * 1.2,
            "cardiac_output_ml_s": params.cardiac_output_ml_s * 0.85,
        },
        hidden=16,
        n_steps=40,
        lr=3e-2,
        seed=0,
    )
    pred = result.predict_hu(times, protocol)
    assert nrmse(pred[:, :2], series.curves_hu[:, :2]) < 0.08
    assert result.physics_loss < 1.0


def test_neural_ode_zero_residual_matches_physics(
    params: PhysioParams, protocol: InjectionProtocol
) -> None:
    times = np.linspace(0.0, 40.0, 21)
    residual = NeuralODEResidual(hidden=8).double()
    phys = simulate_ode(params, protocol, times)
    aug = simulate_neural_ode(params, protocol, times, residual=residual)
    np.testing.assert_allclose(
        phys.detach().cpu().numpy(),
        aug.detach().cpu().numpy(),
        rtol=2e-3,
        atol=1e-3,
    )


def test_neural_ode_nonzero_residual_changes_curve(
    params: PhysioParams, protocol: InjectionProtocol
) -> None:
    times = np.linspace(0.0, 40.0, 21)
    residual = NeuralODEResidual(hidden=8).double()
    with torch.no_grad():
        residual.net[-1].bias.fill_(0.05)
    phys = simulate_ode(params, protocol, times)
    aug = simulate_neural_ode(params, protocol, times, residual=residual)
    assert not np.allclose(
        phys.detach().cpu().numpy(),
        aug.detach().cpu().numpy(),
        rtol=1e-4,
        atol=1e-4,
    )
