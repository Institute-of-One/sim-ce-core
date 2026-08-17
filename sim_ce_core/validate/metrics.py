"""Recovery and curve-fit metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sim_ce_core.physio.params import PhysioParams


def nrmse(pred: np.ndarray, obs: np.ndarray, eps: float = 1e-12) -> float:
    """Root-mean-square error normalized by the RMS of ``obs``."""
    pred_arr = np.asarray(pred, dtype=np.float64)
    obs_arr = np.asarray(obs, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((pred_arr - obs_arr) ** 2)))
    scale = float(np.sqrt(np.mean(obs_arr**2)))
    return rmse / (scale + eps)


def relative_error(recovered: float, truth: float, eps: float = 1e-12) -> float:
    """Absolute relative error ``|hat - true| / |true|``."""
    return abs(recovered - truth) / (abs(truth) + eps)


def parameter_rel_errors(
    fitted: PhysioParams,
    truth: PhysioParams,
    names: Sequence[str],
) -> dict[str, float]:
    """Per-parameter relative errors."""
    return {
        name: relative_error(float(getattr(fitted, name)), float(getattr(truth, name)))
        for name in names
    }


def mean_relative_error(
    fitted: PhysioParams,
    truth: PhysioParams,
    names: Sequence[str],
) -> float:
    """Mean relative error over ``names``."""
    errors = parameter_rel_errors(fitted, truth, names)
    return float(np.mean(list(errors.values())))
