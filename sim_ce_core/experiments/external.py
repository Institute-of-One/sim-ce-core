"""M3 external validation (Fig 3) on a local or proxy cohort."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from sim_ce_core.data.catalog import load_cohort
from sim_ce_core.data.io import physio_from_metadata, protocol_from_metadata
from sim_ce_core.data.mphase import scale_physio_for_weight
from sim_ce_core.data.types import EnhancementSeries
from sim_ce_core.nn.pinn import fit_pinn
from sim_ce_core.physio.fit import recover_parameters
from sim_ce_core.physio.forward import simulate_hu
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.report.figures import (
    save_overlay_plot,
    save_paired_case_plot,
    save_rows_csv,
)
from sim_ce_core.repro import seed_everything
from sim_ce_core.validate.deconvolution import reconstruct_organ
from sim_ce_core.validate.metrics import mean_relative_error, nrmse
from sim_ce_core.validate.sweeps import default_ls_init, interpolate_to

#: Below this a curve fit has not fitted, it has interpolated. On two-phase cases the
#: closed form passes exactly through the one informative measurement, and a residual at
#: machine precision is the signature of that rather than of accuracy.
EXACT_FIT_NRMSE = 1e-9


def _observed_regions(use_aif: bool) -> tuple[str, ...]:
    return ("aorta", "organ") if use_aif else ("organ",)


def _obs_matrix(series: EnhancementSeries, regions: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([series.region(name) for name in regions])


def evaluate_external_case(
    series: EnhancementSeries,
    template: PhysioParams,
    default_protocol: InjectionProtocol,
    *,
    free_params: list[str],
    pinn_hidden: int,
    pinn_steps: int,
    physics_weight: float,
    seed: int,
    use_aif: bool = True,
) -> list[dict[str, Any]]:
    protocol = protocol_from_metadata(series.metadata, default_protocol)
    truth = physio_from_metadata(series.metadata, template)
    regions = _observed_regions(use_aif)
    obs = _obs_matrix(series, regions)
    init = default_ls_init(template, free_params)
    rows: list[dict[str, Any]] = []

    ls_fit, _ = recover_parameters(
        series.times_s,
        obs,
        protocol,
        template,
        free_params=free_params,
        init=init,
        region_names=regions,
        max_nfev=80,
    )
    ls_pred = simulate_hu(ls_fit, protocol, series.times_s).detach().cpu().numpy()
    ls_nrmse = nrmse(
        np.column_stack([ls_pred[:, 0 if r == "aorta" else 1] for r in regions]),
        obs,
    )
    rows.append(
        {
            "case_id": series.metadata.get("case_id"),
            "method": "closed_form",
            "curve_nrmse": ls_nrmse,
            "param_mre": (
                mean_relative_error(ls_fit, truth, free_params)
                if truth
                else float("nan")
            ),
            "source": series.metadata.get("source", "unknown"),
            "dataset": series.metadata.get("dataset"),
        }
    )

    pinn = fit_pinn(
        series.times_s,
        obs,
        protocol,
        template,
        mode="hybrid",
        free_params=free_params,
        init=init,
        hidden=pinn_hidden,
        n_steps=pinn_steps,
        physics_weight=physics_weight,
        region_names=regions,
        seed=seed,
    )
    pinn_pred = pinn.predict_hu(series.times_s, protocol)
    pinn_nrmse = nrmse(
        np.column_stack([pinn_pred[:, 0 if r == "aorta" else 1] for r in regions]),
        obs,
    )
    rows.append(
        {
            "case_id": series.metadata.get("case_id"),
            "method": "pinn_hybrid",
            "curve_nrmse": pinn_nrmse,
            "param_mre": (
                mean_relative_error(pinn.params, truth, free_params)
                if truth
                else float("nan")
            ),
            "source": series.metadata.get("source", "unknown"),
            "dataset": series.metadata.get("dataset"),
        }
    )

    if (
        series.aif_hu is not None
        and "organ" in series.region_names
        and series.times_s.size >= 8
    ):
        organ_hat = reconstruct_organ(series.aif_hu, series.region("organ"))
        organ_cmp = interpolate_to(series.times_s, organ_hat, series.times_s)
        rows.append(
            {
                "case_id": series.metadata.get("case_id"),
                "method": "deconvolution",
                "curve_nrmse": nrmse(organ_cmp, series.region("organ")),
                "param_mre": float("nan"),
                "source": series.metadata.get("source", "unknown"),
                "dataset": series.metadata.get("dataset"),
            }
        )
    return rows


def _mphase_phase_rows(
    series: EnhancementSeries,
    template: PhysioParams,
    default_protocol: InjectionProtocol,
) -> list[dict[str, Any]]:
    """Predict phase HU from weight + recorded protocol (Bae digital-twin check)."""
    protocol = protocol_from_metadata(series.metadata, default_protocol)
    weight = float(series.metadata.get("body_weight_kg", 70.0))
    scaled = scale_physio_for_weight(template, weight)
    pred = simulate_hu(scaled, protocol, series.times_s).detach().cpu().numpy()
    phases = series.metadata.get("phases_s") or {}
    rows: list[dict[str, Any]] = []
    for i, time_s in enumerate(series.times_s):
        phase_name = next(
            (
                name
                for name, t in phases.items()
                if abs(float(t) - float(time_s)) < 1e-6
            ),
            f"t{int(time_s)}",
        )
        rows.append(
            {
                "case_id": series.metadata.get("case_id"),
                "phase": phase_name,
                "time_s": float(time_s),
                "obs_aorta_hu": float(series.region("aorta")[i]),
                "pred_aorta_hu": float(pred[i, 0]),
                "obs_organ_hu": float(series.region("organ")[i]),
                "pred_organ_hu": float(pred[i, 1]),
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(key)
        if val is None or not np.isfinite(val):
            continue
        grouped[str(row["method"])].append(float(val))
    return {
        method: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": float(len(vals)),
        }
        for method, vals in grouped.items()
    }


def run_external_experiment(cfg: Any, output_dir: Path) -> dict[str, Any]:
    seed_everything(cfg.seed)
    if cfg.dataset.primary == "synthetic":
        raise ValueError("external validation needs dataset.primary != synthetic")
    cohort = load_cohort(
        cfg.dataset, template=cfg.physiology, protocol=cfg.injection, seed=cfg.seed
    )
    rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for i, series in enumerate(cohort):
        rows.extend(
            evaluate_external_case(
                series,
                cfg.physiology,
                cfg.injection,
                free_params=cfg.free_params,
                pinn_hidden=cfg.pinn.hidden,
                pinn_steps=cfg.pinn.n_steps,
                physics_weight=cfg.pinn.physics_weight,
                seed=cfg.seed + i,
                use_aif=cfg.amortized.use_aif,
            )
        )
        if series.metadata.get("dataset") == "mphase_liver":
            phase_rows.extend(_mphase_phase_rows(series, cfg.physiology, cfg.injection))

    output_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(rows, output_dir / "fig3_external.csv")
    curve_stats = _summarize(rows, "curve_nrmse")
    methods = sorted(curve_stats)
    by_case: dict[str, dict[str, float]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), {})[str(row["method"])] = float(
            row["curve_nrmse"]
        )
    cases = sorted(by_case)
    # A fit whose residual is at machine precision has not fitted well; it has
    # interpolated. Two-phase cases carry one informative measurement -- the
    # pre-contrast phase has no contrast in it -- against two free parameters, so the
    # model passes through the data exactly and scores zero error. Averaging those in
    # with the constrained cases is how the closed form came to look better than it is.
    exact = {
        method: sorted(
            case
            for case in cases
            if abs(by_case[case].get(method, float("nan"))) < EXACT_FIT_NRMSE
        )
        for method in methods
    }
    constrained = {
        method: [
            by_case[case][method]
            for case in cases
            if method in by_case[case] and abs(by_case[case][method]) >= EXACT_FIT_NRMSE
        ]
        for method in methods
    }
    save_paired_case_plot(
        cases,
        {
            method: [by_case[case].get(method, float("nan")) for case in cases]
            for method in methods
        },
        output_dir / "fig3_external_nrmse.png",
        title="External validation: curve NRMSE per case",
        ylabel="Curve NRMSE",
    )
    underdetermined = {
        "threshold_nrmse": EXACT_FIT_NRMSE,
        "exact_fits": {method: len(cases_) for method, cases_ in exact.items()},
        "n_cases": len(cases),
        "constrained_mean_nrmse": {
            method: (float(np.mean(values)) if values else None)
            for method, values in constrained.items()
        },
        "n_constrained": {
            method: len(values) for method, values in constrained.items()
        },
    }

    example = cohort[0]
    example_pinn = next(
        r
        for r in rows
        if r["case_id"] == example.metadata.get("case_id")
        and r["method"] == "pinn_hybrid"
    )
    protocol = protocol_from_metadata(example.metadata, cfg.injection)
    pop = simulate_hu(cfg.physiology, protocol, example.times_s).detach().cpu().numpy()
    save_overlay_plot(
        example.times_s,
        {
            "observed organ": example.region("organ"),
            "population Bae": pop[:, 1],
        },
        output_dir / "fig3_example.png",
        title=f"External case {example.metadata.get('case_id')}",
    )
    if phase_rows:
        save_rows_csv(phase_rows, output_dir / "fig3_mphase_phases.csv")

    sources = sorted({str(s.metadata.get("source", "unknown")) for s in cohort})
    return {
        "n_cases": len(cohort),
        "dataset": cfg.dataset.primary,
        "sources": sources,
        "curve_nrmse": curve_stats,
        "param_mre": _summarize(rows, "param_mre"),
        "csv": str(output_dir / "fig3_external.csv"),
        "fig3": str(output_dir / "fig3_external_nrmse.png"),
        "example_pinn_nrmse": example_pinn["curve_nrmse"],
        "n_phase_rows": len(phase_rows),
        "underdetermined": underdetermined,
    }
