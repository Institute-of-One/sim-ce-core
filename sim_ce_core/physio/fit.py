"""Recover known physiology from observed enhancement curves."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import least_squares

from sim_ce_core.physio.closed_form import simulate_closed_form
from sim_ce_core.physio.params import REGION_NAMES, InjectionProtocol, PhysioParams
from sim_ce_core.physio.system import concentrations_to_hu

DEFAULT_FREE: tuple[str, ...] = (
    "central_blood_volume_ml",
    "cardiac_output_ml_s",
)


def recover_parameters(
    times_s: np.ndarray,
    observed_hu: np.ndarray,
    protocol: InjectionProtocol,
    template: PhysioParams,
    *,
    free_params: Sequence[str] = DEFAULT_FREE,
    init: dict[str, float] | None = None,
    region_names: Sequence[str] = ("aorta", "organ"),
    max_nfev: int | None = None,
) -> tuple[PhysioParams, dict[str, float]]:
    """Least-squares recovery of a subset of ``PhysioParams`` (log-space).

    ``observed_hu`` is ``(T, R)`` aligned with ``region_names``. Frozen fields
    stay at ``template`` values (injection protocol is always known).
    """
    free = tuple(free_params)
    for name in free:
        if name not in template.model_fields:
            raise ValueError(f"Unknown parameter: {name}")
    col_index = [REGION_NAMES.index(name) for name in region_names]
    obs = np.asarray(observed_hu, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] != len(region_names):
        raise ValueError("observed_hu must have shape (T, len(region_names))")
    x0 = np.log(
        np.array(
            [
                (init[name] if init is not None else getattr(template, name))
                for name in free
            ],
            dtype=np.float64,
        )
    )

    def residual(theta_log: np.ndarray) -> np.ndarray:
        updates = {
            name: float(np.exp(val)) for name, val in zip(free, theta_log, strict=True)
        }
        trial = template.model_copy(update=updates)
        conc = simulate_closed_form(trial, protocol, times_s)
        hu = concentrations_to_hu(conc, trial.iodine_to_hu).detach().cpu().numpy()
        return (hu[:, col_index] - obs).ravel()

    result = least_squares(
        residual,
        x0,
        method="trf",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=max_nfev,
    )
    recovered_vals = {
        name: float(np.exp(val)) for name, val in zip(free, result.x, strict=True)
    }
    fitted = template.model_copy(update=recovered_vals)
    info = {
        "cost": float(result.cost),
        "nfev": float(result.nfev),
        "success": float(result.success),
        **recovered_vals,
    }
    return fitted, info
