"""Tikhonov deconvolution reconstructs a noiseless organ curve."""

from __future__ import annotations

import numpy as np

from sim_ce_core.data.synthetic import generate_synthetic
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.validate.deconvolution import reconstruct_organ
from sim_ce_core.validate.metrics import nrmse


def test_deconvolution_reconstructs_noiseless_organ(
    params: PhysioParams, protocol: InjectionProtocol
) -> None:
    times = np.linspace(0.0, 60.0, 61)
    series = generate_synthetic(params, protocol, times, seed=0)
    organ_hat = reconstruct_organ(series.aif_hu, series.region("organ"), lam=0.05)
    assert nrmse(organ_hat, series.region("organ")) < 0.2
