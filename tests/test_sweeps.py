"""Degradation helpers and a one-cell robustness sweep."""

from __future__ import annotations

import numpy as np

from sim_ce_core.data.config import DatasetConfig
from sim_ce_core.data.synthetic import generate_synthetic
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.validate.degrade import Degradation, apply_degradation
from sim_ce_core.validate.sweeps import run_robustness_sweep


def test_degrade_is_seeded(params: PhysioParams, protocol: InjectionProtocol) -> None:
    times = np.linspace(0.0, 40.0, 21)
    clean = generate_synthetic(params, protocol, times, seed=0)
    deg = Degradation(noise_sd_hu=12.0, subsample_stride=2, dose_scale=0.5)
    a = apply_degradation(clean, deg, seed=3)
    b = apply_degradation(clean, deg, seed=3)
    np.testing.assert_allclose(a.curves_hu, b.curves_hu)
    assert a.times_s.size == 11


def test_sweep_one_cell_returns_methods(
    params: PhysioParams, protocol: InjectionProtocol
) -> None:
    times = np.linspace(0.0, 50.0, 26)
    clean = generate_synthetic(params, protocol, times, seed=0)
    rows = run_robustness_sweep(
        clean,
        protocol,
        params,
        [Degradation(noise_sd_hu=5.0, subsample_stride=2, dose_scale=1.0)],
        pinn_hidden=8,
        pinn_steps=15,
        seed=0,
        amortized=None,
    )
    methods = {row["method"] for row in rows}
    assert "closed_form" in methods
    assert "deconvolution" in methods
    assert "pinn_hybrid" in methods
    for row in rows:
        assert np.isfinite(row["curve_nrmse"])


def test_dataset_config_defaults_to_synthetic() -> None:
    assert DatasetConfig().primary == "synthetic"
