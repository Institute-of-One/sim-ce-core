"""Fisher information of a sampling design: ``F(S) = sum_i J_i^T Sigma_i^-1 J_i``.

For Gaussian measurement noise this is the whole of what a set of acquisition times
determines about physiology. Its inverse bounds the covariance of any unbiased
estimator, so a design singular in some direction is one no estimator can recover in
that direction -- which is the claim this package now sets out to test.

Scaling is not a detail here. The parameters carry units that differ by four orders of
magnitude: volumes in millilitres run to 25 000, the organ flow fraction is 0.25,
and the attenuation constant is 26. A Fisher matrix built from raw derivatives has
a condition number reporting mostly the choice of units, and would call the design
ill-conditioned in litres and well-conditioned in millilitres. Every quantity is
therefore computed in **log-parameter space** by default, ``theta * dC/dtheta``,
which is dimensionless and turns the Cramer-Rao bound into a relative standard
error a reader can act on: "cardiac output to within 12%".
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from sim_ce_core.design.sensitivity import Sensitivity


def _noise_vector(sigma_hu: Any, n_rows: int, dtype: torch.dtype) -> Tensor:
    sigma = torch.as_tensor(sigma_hu, dtype=dtype).reshape(-1)
    if sigma.numel() == 1:
        sigma = sigma.expand(n_rows)
    if sigma.numel() != n_rows:
        raise ValueError(
            f"sigma_hu has {sigma.numel()} entries for {n_rows} measurements"
        )
    if bool((sigma <= 0).any()):
        raise ValueError("sigma_hu must be positive")
    return sigma


def scaled_jacobian(
    sensitivity: Sensitivity,
    parameter_values: dict[str, float],
    *,
    log_scale: bool = True,
) -> Tensor:
    """``(M, P)`` sensitivities, in log-parameter space unless asked otherwise.

    A parameter whose value is zero cannot be put on a log scale; its column is left
    in absolute units and the caller is expected to know, because dropping it would
    remove a direction from the analysis without saying so.
    """
    flat = sensitivity.flat
    if not log_scale:
        return flat
    scale = torch.tensor(
        [float(parameter_values[name]) for name in sensitivity.parameter_names],
        dtype=flat.dtype,
    )
    scale = torch.where(scale.abs() > 0, scale, torch.ones_like(scale))
    return flat * scale


def fisher_information(
    sensitivity: Sensitivity,
    parameter_values: dict[str, float],
    *,
    sigma_hu: Any = 10.0,
    log_scale: bool = True,
) -> Tensor:
    """``(P, P)`` Fisher information for independent Gaussian noise of ``sigma_hu``."""
    jac = scaled_jacobian(sensitivity, parameter_values, log_scale=log_scale)
    sigma = _noise_vector(sigma_hu, jac.shape[0], jac.dtype)
    weighted = jac / sigma.reshape(-1, 1)
    return weighted.T @ weighted


def add_measurement(
    fisher: Tensor,
    sensitivity: Sensitivity,
    parameter_values: dict[str, float],
    *,
    sigma_hu: Any = 10.0,
    log_scale: bool = True,
) -> Tensor:
    """``F(S union {t})`` -- information is additive over independent measurements.

    Phase selection is a search over candidate times, and rebuilding the whole matrix
    for each candidate is the obvious way to get it wrong by counting a time twice.
    """
    return fisher + fisher_information(
        sensitivity, parameter_values, sigma_hu=sigma_hu, log_scale=log_scale
    )
