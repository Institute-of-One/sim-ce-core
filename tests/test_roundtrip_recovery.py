"""Round-trip recovery of known physiology from synthetic curves."""

from __future__ import annotations

import numpy as np

from sim_ce_core.data.synthetic import generate_synthetic
from sim_ce_core.physio.fit import recover_parameters
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.validate.metrics import nrmse, relative_error


def test_roundtrip_recovers_volume_and_cardiac_output(
    params: PhysioParams, protocol: InjectionProtocol, times_s: np.ndarray
) -> None:
    series = generate_synthetic(
        params, protocol, times_s, backend="closed_form", noise_sd_hu=0.0, seed=0
    )
    init = {
        "central_blood_volume_ml": params.central_blood_volume_ml * 1.4,
        "cardiac_output_ml_s": params.cardiac_output_ml_s * 0.7,
    }
    fitted, _info = recover_parameters(
        series.times_s,
        series.curves_hu[:, :2],
        protocol,
        params,
        free_params=("central_blood_volume_ml", "cardiac_output_ml_s"),
        init=init,
        region_names=("aorta", "organ"),
    )
    assert (
        relative_error(fitted.central_blood_volume_ml, params.central_blood_volume_ml)
        < 0.05
    )
    assert relative_error(fitted.cardiac_output_ml_s, params.cardiac_output_ml_s) < 0.05

    recovered = generate_synthetic(
        fitted, protocol, times_s, backend="closed_form", noise_sd_hu=0.0, seed=0
    )
    assert nrmse(recovered.curves_hu, series.curves_hu) < 0.02


def test_synthetic_seed_is_reproducible(
    params: PhysioParams, protocol: InjectionProtocol, times_s: np.ndarray
) -> None:
    a = generate_synthetic(
        params, protocol, times_s, noise_sd_hu=8.0, seed=7, backend="closed_form"
    )
    b = generate_synthetic(
        params, protocol, times_s, noise_sd_hu=8.0, seed=7, backend="closed_form"
    )
    np.testing.assert_allclose(a.curves_hu, b.curves_hu)
