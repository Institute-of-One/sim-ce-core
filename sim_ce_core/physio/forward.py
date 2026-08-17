"""Unified Bae-style forward simulator."""

from __future__ import annotations

from typing import Any, Literal

from torch import Tensor

from sim_ce_core.physio.closed_form import simulate_closed_form
from sim_ce_core.physio.ode import simulate_ode
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.physio.system import concentrations_to_hu

Backend = Literal["closed_form", "ode"]


def simulate(
    params: PhysioParams,
    protocol: InjectionProtocol,
    times_s: Any,
    *,
    backend: Backend = "closed_form",
) -> Tensor:
    """Concentrations ``(T, 3)`` in mg I / mL."""
    if backend == "closed_form":
        return simulate_closed_form(params, protocol, times_s)
    if backend == "ode":
        return simulate_ode(params, protocol, times_s)
    raise ValueError(f"Unknown backend: {backend!r}")


def simulate_hu(
    params: PhysioParams,
    protocol: InjectionProtocol,
    times_s: Any,
    *,
    backend: Backend = "closed_form",
) -> Tensor:
    """Enhancement ``(T, 3)`` in HU (aorta, organ, recirculation)."""
    conc = simulate(params, protocol, times_s, backend=backend)
    return concentrations_to_hu(conc, params.iodine_to_hu)
