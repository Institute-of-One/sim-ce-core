"""Amortized inference trains and returns positive physiology."""

from __future__ import annotations

import numpy as np

from sim_ce_core.data.synthetic import generate_synthetic
from sim_ce_core.nn.amortized import train_amortized
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.validate.degrade import Degradation, apply_degradation


def test_amortized_trains_and_infers_positive(
    params: PhysioParams, protocol: InjectionProtocol
) -> None:
    model = train_amortized(
        params,
        protocol,
        t_end_s=60.0,
        n_times=32,
        hidden=16,
        n_train=8,
        n_epochs=3,
        batch_size=4,
        use_aif=True,
        seed=0,
        degradations=[Degradation(noise_sd_hu=5.0, subsample_stride=2)],
    )
    times = np.linspace(0.0, 60.0, 32)
    series = generate_synthetic(params, protocol, times, seed=1)
    observed = apply_degradation(
        series, Degradation(noise_sd_hu=5.0, subsample_stride=2), seed=1
    )
    fitted = model.infer(observed)
    assert fitted.central_blood_volume_ml > 0.0
    assert fitted.cardiac_output_ml_s > 0.0
    assert np.isfinite(fitted.central_blood_volume_ml)
    assert np.isfinite(fitted.cardiac_output_ml_s)


def test_amortized_aif_free_shape(
    params: PhysioParams, protocol: InjectionProtocol
) -> None:
    model = train_amortized(
        params,
        protocol,
        t_end_s=40.0,
        n_times=16,
        hidden=8,
        n_train=4,
        n_epochs=1,
        batch_size=2,
        use_aif=False,
        seed=0,
    )
    assert model.net.n_channels == 1
    times = np.linspace(0.0, 40.0, 16)
    series = generate_synthetic(params, protocol, times, seed=0)
    fitted = model.infer(series)
    assert fitted.cardiac_output_ml_s > 0.0
