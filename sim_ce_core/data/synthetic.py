"""Synthetic ground-truth generator (no download, no patient data)."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim_ce_core.data.types import EnhancementSeries
from sim_ce_core.physio.forward import Backend, simulate_hu
from sim_ce_core.physio.params import REGION_NAMES, InjectionProtocol, PhysioParams
from sim_ce_core.repro import seed_everything


def default_times_s(t_end_s: float = 120.0, dt_s: float = 0.5) -> np.ndarray:
    """Inclusive time grid from 0 to ``t_end_s``."""
    n = int(round(t_end_s / dt_s)) + 1
    return np.linspace(0.0, t_end_s, n, dtype=np.float64)


def generate_synthetic(
    params: PhysioParams,
    protocol: InjectionProtocol,
    times_s: np.ndarray | None = None,
    *,
    backend: Backend = "closed_form",
    noise_sd_hu: float = 0.0,
    seed: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> EnhancementSeries:
    """Forward-simulate enhancement curves from known physiology θ.

    When ``noise_sd_hu > 0``, Gaussian noise is added with ``seed``. Ground-truth
    parameters are stored in ``metadata`` for round-trip tests.
    """
    if seed is not None:
        seed_everything(seed)
    times = (
        default_times_s() if times_s is None else np.asarray(times_s, dtype=np.float64)
    )
    hu = simulate_hu(params, protocol, times, backend=backend).detach().cpu().numpy()
    if noise_sd_hu > 0.0:
        rng = np.random.default_rng(seed)
        hu = hu + rng.normal(0.0, noise_sd_hu, size=hu.shape)
    metadata: dict[str, Any] = {
        "backend": backend,
        "noise_sd_hu": noise_sd_hu,
        "seed": seed,
        "physiology": params.model_dump(),
        "injection": protocol.model_dump(),
        "ground_truth": True,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return EnhancementSeries(
        times_s=times,
        curves_hu=hu,
        region_names=REGION_NAMES,
        aif_hu=hu[:, 0].copy(),
        metadata=metadata,
    )
