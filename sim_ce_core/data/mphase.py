"""MCT-LTDiag-style multi-phase liver CT loader (local extracts only)."""

from __future__ import annotations

from pathlib import Path

from sim_ce_core.data.io import discover_case_dirs, load_case
from sim_ce_core.data.types import EnhancementSeries
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams

DEFAULT_PHASES_S: dict[str, float] = {
    "nc": 0.0,
    "ap": 25.0,
    "pvp": 60.0,
    "dp": 180.0,
}


def liver_protocol(
    body_weight_kg: float, *, rate_ml_s: float = 3.0
) -> InjectionProtocol:
    """MCT-LTDiag protocol: 1 mL/kg (max 100 mL) at 3 mL/s, 300 mg I/mL."""
    volume = min(max(body_weight_kg, 1.0), 100.0)
    return InjectionProtocol(
        concentration_mgi_ml=300.0,
        volume_ml=volume,
        duration_s=volume / rate_ml_s,
    )


def scale_physio_for_weight(
    template: PhysioParams,
    body_weight_kg: float,
    *,
    reference_kg: float = 70.0,
) -> PhysioParams:
    """Scale blood / organ volumes and cardiac output with body weight."""
    scale = body_weight_kg / reference_kg
    return template.model_copy(
        update={
            "central_blood_volume_ml": template.central_blood_volume_ml * scale,
            "organ_volume_ml": template.organ_volume_ml * scale,
            "recirculation_volume_ml": template.recirculation_volume_ml * scale,
            "cardiac_output_ml_s": template.cardiac_output_ml_s * scale,
        }
    )


def load_mphase_case(case_dir: Path) -> EnhancementSeries:
    series = load_case(case_dir)
    meta = dict(series.metadata)
    meta.setdefault("dataset", "mphase_liver")
    meta.setdefault("phases_s", DEFAULT_PHASES_S)
    weight = float(meta.get("body_weight_kg", 70.0))
    meta.setdefault("body_weight_kg", weight)
    meta.setdefault("injection", liver_protocol(weight).model_dump())
    return EnhancementSeries(
        times_s=series.times_s,
        curves_hu=series.curves_hu,
        region_names=series.region_names,
        aif_hu=series.aif_hu,
        metadata=meta,
    )


def load_mphase_cohort(root: Path, *, max_cases: int = 30) -> list[EnhancementSeries]:
    dirs = discover_case_dirs(root)[:max_cases]
    return [load_mphase_case(path) for path in dirs]
