"""DCE-MRI loader (H3 only). Same NPZ schema; no MR signal-model swap yet."""

from __future__ import annotations

from pathlib import Path

from sim_ce_core.data.io import discover_case_dirs, load_case
from sim_ce_core.data.types import EnhancementSeries


def load_dce_case(case_dir: Path) -> EnhancementSeries:
    series = load_case(case_dir)
    meta = dict(series.metadata)
    meta.setdefault("dataset", "dce_mri")
    meta.setdefault("signal_model", "identity_enhancement")
    return EnhancementSeries(
        times_s=series.times_s,
        curves_hu=series.curves_hu,
        region_names=series.region_names,
        aif_hu=series.aif_hu,
        metadata=meta,
    )


def load_dce_cohort(root: Path, *, max_cases: int = 30) -> list[EnhancementSeries]:
    dirs = discover_case_dirs(root)[:max_cases]
    return [load_dce_case(path) for path in dirs]
