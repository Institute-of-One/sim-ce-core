"""Shared physiology fixtures (synthetic only, no network)."""

from __future__ import annotations

import numpy as np
import pytest

from sim_ce_core.physio.params import InjectionProtocol, PhysioParams


@pytest.fixture
def protocol() -> InjectionProtocol:
    return InjectionProtocol(
        concentration_mgi_ml=350.0,
        volume_ml=100.0,
        duration_s=25.0,
    )


@pytest.fixture
def params() -> PhysioParams:
    return PhysioParams(
        central_blood_volume_ml=1000.0,
        organ_volume_ml=400.0,
        recirculation_volume_ml=2500.0,
        cardiac_output_ml_s=108.3,
        organ_flow_fraction=0.25,
        elimination_rate_1_s=0.0,
        iodine_to_hu=26.0,
        transit_delay_s=6.0,
    )


@pytest.fixture
def times_s() -> np.ndarray:
    return np.linspace(0.0, 90.0, 91)
