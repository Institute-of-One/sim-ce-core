"""On-disk EnhancementSeries schema (NPZ + JSON). No network I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sim_ce_core.data.types import EnhancementSeries
from sim_ce_core.physio.params import REGION_NAMES, InjectionProtocol, PhysioParams

SERIES_NPZ = "series.npz"
METADATA_JSON = "metadata.json"


def write_case(
    case_dir: Path,
    series: EnhancementSeries,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write ``series.npz`` + ``metadata.json`` under ``case_dir``."""
    case_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        case_dir / SERIES_NPZ,
        times_s=np.asarray(series.times_s, dtype=np.float64),
        curves_hu=np.asarray(series.curves_hu, dtype=np.float64),
        aif_hu=np.asarray(
            series.aif_hu if series.aif_hu is not None else series.curves_hu[:, 0],
            dtype=np.float64,
        ),
        region_names=np.asarray(series.region_names),
    )
    payload = dict(series.metadata)
    if metadata:
        payload.update(metadata)
    payload.setdefault("region_names", list(series.region_names))
    (case_dir / METADATA_JSON).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return case_dir


def read_metadata(case_dir: Path) -> dict[str, Any]:
    path = case_dir / METADATA_JSON
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"metadata.json must be an object: {path}")
    return raw


def load_case(case_dir: Path) -> EnhancementSeries:
    """Load one extracted case. Requires ``series.npz``."""
    npz_path = case_dir / SERIES_NPZ
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing {SERIES_NPZ} in {case_dir}")
    with np.load(npz_path, allow_pickle=False) as handle:
        times = np.asarray(handle["times_s"], dtype=np.float64)
        curves = np.asarray(handle["curves_hu"], dtype=np.float64)
        aif = (
            np.asarray(handle["aif_hu"], dtype=np.float64)
            if "aif_hu" in handle
            else None
        )
        names = (
            tuple(str(x) for x in handle["region_names"].tolist())
            if "region_names" in handle
            else REGION_NAMES
        )
    meta = read_metadata(case_dir)
    meta.setdefault("case_id", case_dir.name)
    meta.setdefault("case_dir", str(case_dir))
    return EnhancementSeries(
        times_s=times,
        curves_hu=curves,
        region_names=names,
        aif_hu=aif,
        metadata=meta,
    )


def discover_case_dirs(root: Path) -> list[Path]:
    """Case folders that contain ``series.npz``, sorted by name."""
    if not root.is_dir():
        return []
    found = [path.parent for path in sorted(root.glob(f"*/{SERIES_NPZ}"))]
    return found


def protocol_from_metadata(
    metadata: dict[str, Any],
    default: InjectionProtocol,
) -> InjectionProtocol:
    raw = metadata.get("injection")
    if not raw:
        return default
    return InjectionProtocol.model_validate(raw)


def physio_from_metadata(
    metadata: dict[str, Any],
    default: PhysioParams,
) -> PhysioParams | None:
    raw = metadata.get("physiology")
    if not raw:
        return None
    merged = default.model_dump()
    merged.update(raw)
    return PhysioParams.model_validate(merged)
