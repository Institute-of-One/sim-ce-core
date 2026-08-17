"""Synthetic degradations: noise, temporal sparsity, dose / CNR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim_ce_core.data.types import EnhancementSeries
from sim_ce_core.physio.params import InjectionProtocol


@dataclass(frozen=True)
class Degradation:
    """Observation-model stress. Dose scale is linear in this forward model."""

    noise_sd_hu: float = 0.0
    subsample_stride: int = 1
    dose_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.noise_sd_hu < 0.0:
            raise ValueError("noise_sd_hu must be >= 0")
        if self.subsample_stride < 1:
            raise ValueError("subsample_stride must be >= 1")
        if self.dose_scale <= 0.0:
            raise ValueError("dose_scale must be > 0")


def scale_protocol(protocol: InjectionProtocol, dose_scale: float) -> InjectionProtocol:
    """Known injected volume after a dose reduction."""
    return protocol.model_copy(update={"volume_ml": protocol.volume_ml * dose_scale})


def apply_degradation(
    series: EnhancementSeries,
    degradation: Degradation,
    *,
    seed: int = 0,
) -> EnhancementSeries:
    """Scale dose, add Gaussian noise, then subsample time."""
    curves = np.asarray(series.curves_hu, dtype=np.float64) * degradation.dose_scale
    if degradation.noise_sd_hu > 0.0:
        rng = np.random.default_rng(seed)
        curves = curves + rng.normal(0.0, degradation.noise_sd_hu, size=curves.shape)
    stride = degradation.subsample_stride
    times = series.times_s[::stride]
    curves = curves[::stride]
    metadata = dict(series.metadata)
    metadata["degradation"] = {
        "noise_sd_hu": degradation.noise_sd_hu,
        "subsample_stride": degradation.subsample_stride,
        "dose_scale": degradation.dose_scale,
        "seed": seed,
    }
    return EnhancementSeries(
        times_s=times,
        curves_hu=curves,
        region_names=series.region_names,
        aif_hu=curves[:, 0].copy(),
        metadata=metadata,
    )
