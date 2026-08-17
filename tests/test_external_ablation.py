"""External-case eval and a tiny ablation (synthetic fixtures only)."""

from __future__ import annotations

import numpy as np

from sim_ce_core.data.synthetic import generate_synthetic
from sim_ce_core.data.types import EnhancementSeries
from sim_ce_core.experiments.ablation import _synthetic_ablation_cohort
from sim_ce_core.experiments.external import evaluate_external_case
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.validate.degrade import Degradation, apply_degradation


def test_evaluate_external_case_returns_methods(
    params: PhysioParams, protocol: InjectionProtocol
) -> None:
    times = np.linspace(0.0, 50.0, 26)
    clean = generate_synthetic(params, protocol, times, seed=0)
    observed = apply_degradation(
        clean, Degradation(noise_sd_hu=5.0, subsample_stride=2), seed=0
    )
    series = EnhancementSeries(
        times_s=observed.times_s,
        curves_hu=observed.curves_hu,
        region_names=observed.region_names,
        aif_hu=observed.aif_hu,
        metadata={
            "case_id": "ext_001",
            "dataset": "ctp_brain",
            "source": "fixture",
            "injection": protocol.model_dump(),
            "physiology": params.model_dump(),
        },
    )
    rows = evaluate_external_case(
        series,
        params,
        protocol,
        free_params=["central_blood_volume_ml", "cardiac_output_ml_s"],
        pinn_hidden=8,
        pinn_steps=12,
        physics_weight=1.0,
        seed=0,
    )
    methods = {row["method"] for row in rows}
    assert {"closed_form", "pinn_hybrid", "deconvolution"} <= methods
    for row in rows:
        assert np.isfinite(row["curve_nrmse"])


def test_ablation_cohort_is_local(
    params: PhysioParams, protocol: InjectionProtocol
) -> None:
    cohort = _synthetic_ablation_cohort(
        params, protocol, n_cases=2, t_end_s=40.0, dt_s=2.0, seed=0
    )
    assert len(cohort) == 2
    assert cohort[0].metadata["source"] == "synthetic"
    assert "physiology" in cohort[0].metadata
