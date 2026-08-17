"""Local extract loaders. No network, no real patient files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim_ce_core.data.catalog import MissingLocalDataError, load_cohort
from sim_ce_core.data.config import DatasetConfig
from sim_ce_core.data.ctp import load_ctp_case
from sim_ce_core.data.io import load_case, write_case
from sim_ce_core.data.mphase import liver_protocol, load_mphase_case
from sim_ce_core.data.synthetic import generate_synthetic
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams


def test_npz_roundtrip(
    tmp_path: Path, params: PhysioParams, protocol: InjectionProtocol
) -> None:
    times = np.linspace(0.0, 40.0, 21)
    series = generate_synthetic(params, protocol, times, seed=0)
    case_dir = tmp_path / "case_001"
    write_case(
        case_dir,
        series,
        metadata={"case_id": "case_001", "dataset": "ctp_brain", "source": "fixture"},
    )
    loaded = load_case(case_dir)
    np.testing.assert_allclose(loaded.times_s, series.times_s)
    np.testing.assert_allclose(loaded.curves_hu, series.curves_hu)
    assert loaded.metadata["case_id"] == "case_001"


def test_ctp_and_mphase_wrappers(
    tmp_path: Path, params: PhysioParams, protocol: InjectionProtocol
) -> None:
    times = np.linspace(0.0, 30.0, 16)
    series = generate_synthetic(params, protocol, times, seed=0)
    write_case(tmp_path / "ctp", series, metadata={"dataset": "ctp_brain"})
    write_case(
        tmp_path / "liver",
        series,
        metadata={"dataset": "mphase_liver", "body_weight_kg": 80.0},
    )
    ctp = load_ctp_case(tmp_path / "ctp")
    liver = load_mphase_case(tmp_path / "liver")
    assert ctp.metadata["dataset"] == "ctp_brain"
    assert liver.metadata["body_weight_kg"] == 80.0
    proto = liver_protocol(80.0)
    assert proto.volume_ml == 80.0
    assert proto.rate_ml_s == pytest.approx(3.0)


def test_load_cohort_proxy_when_raw_empty(
    tmp_path: Path, params: PhysioParams, protocol: InjectionProtocol
) -> None:
    cfg = DatasetConfig(
        primary="ctp_brain",
        root=str(tmp_path / "raw"),
        proxy_root=str(tmp_path / "proxy"),
        max_cases=3,
        allow_proxy=True,
    )
    cohort = load_cohort(cfg, template=params, protocol=protocol, seed=0)
    assert len(cohort) == 3
    assert all(case.metadata["source"] == "synthetic_proxy" for case in cohort)


def test_load_cohort_missing_raises(
    tmp_path: Path, params: PhysioParams, protocol: InjectionProtocol
) -> None:
    cfg = DatasetConfig(
        primary="ctp_brain",
        root=str(tmp_path / "raw"),
        proxy_root=str(tmp_path / "proxy"),
        allow_proxy=False,
    )
    with pytest.raises(MissingLocalDataError):
        load_cohort(cfg, template=params, protocol=protocol)
