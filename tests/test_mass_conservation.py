"""Iodine mass is conserved when elimination is off."""

from __future__ import annotations

import numpy as np

from sim_ce_core.physio.closed_form import simulate_closed_form
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.physio.system import compartment_masses_mgi, params_to_tensors


def _injected_mass(
    times: np.ndarray, protocol: InjectionProtocol, delay_s: float
) -> np.ndarray:
    t0 = delay_s
    t1 = delay_s + protocol.duration_s
    mass = np.zeros_like(times)
    during = (times >= t0) & (times < t1)
    mass[during] = protocol.iodine_rate_mgi_s * (times[during] - t0)
    mass[times >= t1] = protocol.iodine_mass_mgi
    return mass


def test_mass_conserved_without_elimination(
    params: PhysioParams, protocol: InjectionProtocol, times_s: np.ndarray
) -> None:
    conc = simulate_closed_form(params, protocol, times_s)
    theta = params_to_tensors(params)
    total = compartment_masses_mgi(conc, theta).sum(dim=1).detach().cpu().numpy()
    expected = _injected_mass(times_s, protocol, params.transit_delay_s)
    np.testing.assert_allclose(total, expected, rtol=1e-6, atol=1e-4)
