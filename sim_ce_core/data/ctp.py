"""UniToBrain-style brain CT perfusion loader (local extracts only)."""

from __future__ import annotations

from pathlib import Path

from sim_ce_core.data.io import discover_case_dirs, load_case
from sim_ce_core.data.types import EnhancementSeries
from sim_ce_core.physio.params import InjectionProtocol

# UniToBrain acquisition (IEEE DataPort / Zenodo description)
UNITOBRAIN_PROTOCOL = InjectionProtocol(
    concentration_mgi_ml=300.0,
    volume_ml=40.0,
    duration_s=10.0,
)


def load_ctp_case(case_dir: Path) -> EnhancementSeries:
    """Load one CTP extract. Metadata should record ``dataset: ctp_brain``."""
    series = load_case(case_dir)
    meta = dict(series.metadata)
    meta.setdefault("dataset", "ctp_brain")
    meta.setdefault("injection", UNITOBRAIN_PROTOCOL.model_dump())
    return EnhancementSeries(
        times_s=series.times_s,
        curves_hu=series.curves_hu,
        region_names=series.region_names,
        aif_hu=series.aif_hu,
        metadata=meta,
    )


def load_ctp_cohort(root: Path, *, max_cases: int = 30) -> list[EnhancementSeries]:
    """Load up to ``max_cases`` CTP extracts from ``root``."""
    dirs = discover_case_dirs(root)[:max_cases]
    return [load_ctp_case(path) for path in dirs]
