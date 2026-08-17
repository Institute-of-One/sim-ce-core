"""M3 ablations: physics-only vs neural-only vs PINN hybrid, ±AIF."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from sim_ce_core.data.catalog import load_cohort
from sim_ce_core.data.io import physio_from_metadata, protocol_from_metadata
from sim_ce_core.data.synthetic import default_times_s, generate_synthetic
from sim_ce_core.data.types import EnhancementSeries
from sim_ce_core.nn.pinn import PinnMode, fit_pinn
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.report.figures import save_bar_plot, save_rows_csv
from sim_ce_core.repro import seed_everything
from sim_ce_core.validate.degrade import Degradation, apply_degradation
from sim_ce_core.validate.metrics import mean_relative_error, nrmse
from sim_ce_core.validate.sweeps import default_ls_init


def _synthetic_ablation_cohort(
    template: PhysioParams,
    protocol: InjectionProtocol,
    *,
    n_cases: int,
    t_end_s: float,
    dt_s: float,
    seed: int,
) -> list[EnhancementSeries]:
    rng = np.random.default_rng(seed)
    times = default_times_s(t_end_s, dt_s)
    cohort: list[EnhancementSeries] = []
    for i in range(n_cases):
        trial = template.model_copy(
            update={
                "central_blood_volume_ml": float(
                    np.exp(rng.normal(np.log(template.central_blood_volume_ml), 0.18))
                ),
                "cardiac_output_ml_s": float(
                    np.exp(rng.normal(np.log(template.cardiac_output_ml_s), 0.18))
                ),
            }
        )
        clean = generate_synthetic(trial, protocol, times, seed=None)
        observed = apply_degradation(
            clean, Degradation(noise_sd_hu=10.0, subsample_stride=2), seed=seed + i
        )
        meta = dict(observed.metadata)
        meta.update(
            {
                "case_id": f"ablation_{i:03d}",
                "source": "synthetic",
                "physiology": trial.model_dump(),
                "injection": protocol.model_dump(),
                "ground_truth": True,
            }
        )
        cohort.append(
            EnhancementSeries(
                times_s=observed.times_s,
                curves_hu=observed.curves_hu,
                region_names=observed.region_names,
                aif_hu=observed.aif_hu,
                metadata=meta,
            )
        )
    return cohort


def run_ablation_experiment(cfg: Any, output_dir: Path) -> dict[str, Any]:
    seed_everything(cfg.seed)
    if cfg.dataset.primary == "synthetic":
        cohort = _synthetic_ablation_cohort(
            cfg.physiology,
            cfg.injection,
            n_cases=min(cfg.dataset.max_cases, 8),
            t_end_s=cfg.t_end_s,
            dt_s=cfg.dt_s,
            seed=cfg.seed,
        )
    else:
        cohort = load_cohort(
            cfg.dataset, template=cfg.physiology, protocol=cfg.injection, seed=cfg.seed
        )[: min(cfg.dataset.max_cases, 8)]

    modes: tuple[PinnMode, ...] = ("physics_only", "neural_only", "hybrid")
    aif_flags = (True, False)
    rows: list[dict[str, Any]] = []
    for i, series in enumerate(cohort):
        protocol = protocol_from_metadata(series.metadata, cfg.injection)
        truth = physio_from_metadata(series.metadata, cfg.physiology)
        init = default_ls_init(cfg.physiology, cfg.free_params)
        for mode in modes:
            for use_aif in aif_flags:
                regions = ("aorta", "organ") if use_aif else ("organ",)
                obs = np.column_stack([series.region(name) for name in regions])
                result = fit_pinn(
                    series.times_s,
                    obs,
                    protocol,
                    cfg.physiology,
                    mode=mode,
                    free_params=cfg.free_params,
                    init=init,
                    hidden=cfg.pinn.hidden,
                    n_steps=cfg.pinn.n_steps,
                    physics_weight=(
                        0.0 if mode == "neural_only" else cfg.pinn.physics_weight
                    ),
                    region_names=regions,
                    seed=cfg.seed + i,
                )
                pred = result.predict_hu(series.times_s, protocol)
                pred_obs = np.column_stack(
                    [pred[:, 0 if name == "aorta" else 1] for name in regions]
                )
                param_mre = (
                    float("nan")
                    if mode == "neural_only" or truth is None
                    else mean_relative_error(result.params, truth, cfg.free_params)
                )
                rows.append(
                    {
                        "case_id": series.metadata.get("case_id"),
                        "mode": mode,
                        "use_aif": use_aif,
                        "curve_nrmse": nrmse(pred_obs, obs),
                        "param_mre": param_mre,
                    }
                )

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        label = f"{row['mode']}/{'AIF' if row['use_aif'] else 'AIF-free'}"
        grouped[label].append(float(row["curve_nrmse"]))
    labels = sorted(grouped)
    means = [float(np.mean(grouped[k])) for k in labels]
    stds = [float(np.std(grouped[k])) for k in labels]

    output_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(rows, output_dir / "ablation.csv")
    save_bar_plot(
        labels,
        means,
        output_dir / "fig3b_ablation.png",
        title="Ablation: physics / neural / hybrid × AIF",
        ylabel="Curve NRMSE",
        yerr=stds,
    )
    return {
        "n_cases": len(cohort),
        "n_rows": len(rows),
        "mean_curve_nrmse": dict(zip(labels, means, strict=True)),
        "csv": str(output_dir / "ablation.csv"),
        "fig": str(output_dir / "fig3b_ablation.png"),
    }
