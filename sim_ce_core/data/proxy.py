"""Synthetic proxy cohort in the real-data NPZ layout (no download, no PHI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim_ce_core.data.io import write_case
from sim_ce_core.data.mphase import DEFAULT_PHASES_S, liver_protocol
from sim_ce_core.data.synthetic import default_times_s, generate_synthetic
from sim_ce_core.data.types import EnhancementSeries
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.validate.degrade import Degradation, apply_degradation


def _jitter_params(
    template: PhysioParams, rng: np.random.Generator, sigma: float = 0.2
) -> PhysioParams:
    updates = {
        "central_blood_volume_ml": float(
            np.exp(rng.normal(np.log(template.central_blood_volume_ml), sigma))
        ),
        "cardiac_output_ml_s": float(
            np.exp(rng.normal(np.log(template.cardiac_output_ml_s), sigma))
        ),
        "organ_flow_fraction": float(
            np.clip(template.organ_flow_fraction + rng.normal(0.0, 0.04), 0.08, 0.55)
        ),
    }
    return template.model_copy(update=updates)


def write_proxy_cohort(
    root: Path,
    template: PhysioParams,
    protocol: InjectionProtocol,
    *,
    n_cases: int = 20,
    t_end_s: float = 90.0,
    dt_s: float = 1.0,
    noise_sd_hu: float = 8.0,
    subsample_stride: int = 2,
    seed: int = 0,
    dataset_id: str = "ctp_brain",
) -> list[EnhancementSeries]:
    """Write ``n_cases`` proxy extracts and return the loaded-style series."""
    rng = np.random.default_rng(seed)
    times = default_times_s(t_end_s, dt_s)
    cohort: list[EnhancementSeries] = []
    for i in range(n_cases):
        trial = _jitter_params(template, rng)
        clean = generate_synthetic(
            trial, protocol, times, backend="closed_form", seed=None
        )
        observed = apply_degradation(
            clean,
            Degradation(noise_sd_hu=noise_sd_hu, subsample_stride=subsample_stride),
            seed=seed + 11 * i,
        )
        case_id = f"proxy_{i:03d}"
        meta = {
            "case_id": case_id,
            "dataset": dataset_id,
            "source": "synthetic_proxy",
            "injection": protocol.model_dump(),
            "physiology": trial.model_dump(),
            "ground_truth": True,
        }
        series = EnhancementSeries(
            times_s=observed.times_s,
            curves_hu=observed.curves_hu,
            region_names=observed.region_names,
            aif_hu=observed.aif_hu,
            metadata=meta,
        )
        write_case(root / case_id, series, metadata=meta)
        cohort.append(series)
    return cohort


def write_mphase_proxy_cohort(
    root: Path,
    template: PhysioParams,
    *,
    n_cases: int = 12,
    seed: int = 0,
) -> list[EnhancementSeries]:
    """Four-phase liver-style proxy cases with recorded weight and protocol."""
    rng = np.random.default_rng(seed)
    times = np.array(list(DEFAULT_PHASES_S.values()), dtype=np.float64)
    cohort: list[EnhancementSeries] = []
    for i in range(n_cases):
        weight = float(np.clip(rng.normal(70.0, 12.0), 45.0, 110.0))
        protocol = liver_protocol(weight)
        scale = weight / 70.0
        trial = template.model_copy(
            update={
                "central_blood_volume_ml": template.central_blood_volume_ml * scale,
                "organ_volume_ml": template.organ_volume_ml * scale,
                "recirculation_volume_ml": template.recirculation_volume_ml * scale,
                "cardiac_output_ml_s": template.cardiac_output_ml_s * scale,
            }
        )
        clean = generate_synthetic(
            trial, protocol, times, backend="closed_form", seed=None
        )
        observed = apply_degradation(
            clean, Degradation(noise_sd_hu=6.0), seed=seed + 5 * i
        )
        case_id = f"mphase_proxy_{i:03d}"
        meta = {
            "case_id": case_id,
            "dataset": "mphase_liver",
            "source": "synthetic_proxy",
            "body_weight_kg": weight,
            "phases_s": DEFAULT_PHASES_S,
            "injection": protocol.model_dump(),
            "physiology": trial.model_dump(),
            "ground_truth": True,
        }
        series = EnhancementSeries(
            times_s=observed.times_s,
            curves_hu=observed.curves_hu,
            region_names=observed.region_names,
            aif_hu=observed.aif_hu,
            metadata=meta,
        )
        write_case(root / case_id, series, metadata=meta)
        cohort.append(series)
    return cohort
