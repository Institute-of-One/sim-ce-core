"""Linear 3-compartment system matrix (central blood, organ, recirculation)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from sim_ce_core.physio.params import InjectionProtocol, PhysioParams

TENSOR_PARAM_KEYS: tuple[str, ...] = (
    "central_blood_volume_ml",
    "organ_volume_ml",
    "recirculation_volume_ml",
    "cardiac_output_ml_s",
    "organ_flow_fraction",
    "elimination_rate_1_s",
    "iodine_to_hu",
)


def params_to_tensors(
    params: PhysioParams,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor]:
    """Lift pydantic physiology fields to 0-d tensors."""
    dev = device if device is not None else torch.device("cpu")
    return {
        key: torch.tensor(getattr(params, key), device=dev, dtype=dtype)
        for key in TENSOR_PARAM_KEYS
    }


def system_matrix(theta: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Return ``(A, b)`` for ``dc/dt = A c + b I(t)``.

    State order: central (aorta), organ, recirculation. ``I(t)`` is iodine
    mass rate (mg I / s) into the central compartment.
    """
    v_c = theta["central_blood_volume_ml"]
    v_o = theta["organ_volume_ml"]
    v_r = theta["recirculation_volume_ml"]
    q = theta["cardiac_output_ml_s"]
    f_o = theta["organ_flow_fraction"]
    k_el = theta["elimination_rate_1_s"]
    q_o = q * f_o
    q_r = q * (1.0 - f_o)

    dtype = v_c.dtype
    device = v_c.device
    a00 = -q / v_c
    a01 = q_o / v_c
    a02 = q_r / v_c
    a10 = q_o / v_o
    a11 = -q_o / v_o
    a20 = q_r / v_r
    a22 = -q_r / v_r - k_el

    row0 = torch.stack([a00, a01, a02])
    row1 = torch.stack([a10, a11, torch.zeros((), dtype=dtype, device=device)])
    row2 = torch.stack([a20, torch.zeros((), dtype=dtype, device=device), a22])
    a_mat = torch.stack([row0, row1, row2])
    b_vec = torch.stack(
        [
            1.0 / v_c,
            torch.zeros((), dtype=dtype, device=device),
            torch.zeros((), dtype=dtype, device=device),
        ]
    )
    return a_mat, b_vec


def gather_states(t_grid: Tensor, states: Tensor, t_eval: Tensor) -> Tensor:
    """Select states at ``t_eval`` by nearest grid time."""
    diffs = (t_grid.unsqueeze(0) - t_eval.reshape(-1, 1)).abs()
    idx = diffs.argmin(dim=1)
    return states[idx]


def unique_sorted_times(times: Tensor, eps: float = 1e-9) -> Tensor:
    """Strictly increasing 1-d time grid."""
    t_sorted, _ = torch.sort(times.reshape(-1))
    if t_sorted.numel() <= 1:
        return t_sorted
    keep = torch.ones(t_sorted.numel(), dtype=torch.bool, device=t_sorted.device)
    keep[1:] = torch.diff(t_sorted) > eps
    return t_sorted[keep]


def simulation_time_grid(
    eval_times: Tensor,
    delay_s: float,
    duration_s: float,
) -> Tensor:
    """Eval times union bolus events, starting at t=0."""
    events = eval_times.new_tensor([0.0, delay_s, delay_s + duration_s])
    merged = torch.cat([events, eval_times.reshape(-1)])
    t_max = float(eval_times.reshape(-1)[-1].item())
    merged = merged[(merged >= 0.0) & (merged <= t_max + 1e-9)]
    return unique_sorted_times(merged)


def iodine_rate_on_interval(
    t_mid: float,
    protocol: InjectionProtocol,
    delay_s: float,
) -> float:
    """Constant iodine rate (mg I / s) on a segment that does not cross events."""
    t0 = delay_s
    t1 = delay_s + protocol.duration_s
    if t0 <= t_mid < t1:
        return protocol.iodine_rate_mgi_s
    return 0.0


def as_time_tensor(
    times_s: Any,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Convert a time vector to a 1-d CPU/GPU tensor."""
    if isinstance(times_s, Tensor):
        t = times_s.to(dtype=dtype)
        if device is not None:
            t = t.to(device)
        return t.reshape(-1)
    dev = device if device is not None else torch.device("cpu")
    return torch.as_tensor(times_s, device=dev, dtype=dtype).reshape(-1)


def concentrations_to_hu(conc_mgi_ml: Tensor, iodine_to_hu: Tensor | float) -> Tensor:
    """Convert iodine concentration (mg I / mL) to HU."""
    return conc_mgi_ml * iodine_to_hu


def compartment_masses_mgi(conc_mgi_ml: Tensor, theta: dict[str, Tensor]) -> Tensor:
    """Iodine mass in each compartment, shape ``(..., 3)``."""
    volumes = torch.stack(
        [
            theta["central_blood_volume_ml"],
            theta["organ_volume_ml"],
            theta["recirculation_volume_ml"],
        ]
    )
    return conc_mgi_ml * volumes
