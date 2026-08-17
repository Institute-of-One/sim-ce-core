"""M2 robustness sweep: closed-form, deconvolution, PINN, amortized."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sim_ce_core.data.synthetic import default_times_s, generate_synthetic
from sim_ce_core.nn.amortized import train_amortized
from sim_ce_core.nn.pinn import fit_pinn
from sim_ce_core.physio.fit import recover_parameters
from sim_ce_core.physio.forward import simulate_hu
from sim_ce_core.report.figures import (
    save_overlay_plot,
    save_rows_csv,
    save_scatter_plot,
    save_sweep_plot,
)
from sim_ce_core.repro import seed_everything
from sim_ce_core.validate.deconvolution import reconstruct_organ
from sim_ce_core.validate.degrade import Degradation, apply_degradation, scale_protocol
from sim_ce_core.validate.sweeps import (
    default_ls_init,
    interpolate_to,
    run_robustness_sweep,
)


def _degradations_from_cfg(cfg: Any) -> list[Degradation]:
    return [
        Degradation(noise_sd_hu=noise, subsample_stride=stride, dose_scale=dose)
        for noise in cfg.sweep.noise_sd_hu
        for stride in cfg.sweep.subsample_stride
        for dose in cfg.sweep.dose_scale
    ]


def _example_overlay(
    cfg: Any,
    clean_times: np.ndarray,
    clean_organ: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    """Fig 1: stressed cell, organ reconstruction vs ground truth."""
    deg = Degradation(
        noise_sd_hu=max(cfg.sweep.noise_sd_hu),
        subsample_stride=max(cfg.sweep.subsample_stride),
        dose_scale=min(cfg.sweep.dose_scale),
    )
    clean = generate_synthetic(
        cfg.physiology,
        cfg.injection,
        clean_times,
        backend="closed_form",
        seed=cfg.seed,
    )
    observed = apply_degradation(clean, deg, seed=cfg.seed + 99)
    protocol = scale_protocol(cfg.injection, deg.dose_scale)
    init = default_ls_init(cfg.physiology, cfg.free_params)
    ls_fit, _ = recover_parameters(
        observed.times_s,
        observed.curves_hu[:, :2],
        protocol,
        cfg.physiology,
        free_params=cfg.free_params,
        init=init,
        max_nfev=80,
    )
    ls_hu = simulate_hu(ls_fit, protocol, clean_times).detach().cpu().numpy()
    pinn = fit_pinn(
        observed.times_s,
        observed.curves_hu[:, :2],
        protocol,
        cfg.physiology,
        mode=cfg.pinn.mode,
        free_params=cfg.free_params,
        init=init,
        hidden=cfg.pinn.hidden,
        n_steps=cfg.pinn.n_steps,
        physics_weight=cfg.pinn.physics_weight,
        seed=cfg.seed,
    )
    pinn_hu = pinn.predict_hu(clean_times, protocol)
    deconv = reconstruct_organ(observed.aif_hu, observed.region("organ"))
    deconv_full = interpolate_to(observed.times_s, deconv, clean_times)
    save_overlay_plot(
        clean_times,
        {
            "ground truth": clean_organ,
            "closed-form Bae": ls_hu[:, 1],
            "PINN hybrid": pinn_hu[:, 1],
            "deconvolution": deconv_full,
        },
        output_dir / "fig1_reconstruction.png",
        title="Organ reconstruction under sparse / noisy / low-dose sampling",
    )
    return {
        "example_noise_sd_hu": deg.noise_sd_hu,
        "example_stride": deg.subsample_stride,
        "example_dose_scale": deg.dose_scale,
    }


def _calibration_scatter(
    cfg: Any,
    model: Any,
    output_dir: Path,
    n_holdout: int = 24,
) -> None:
    rng = np.random.default_rng(cfg.seed + 3)
    t_grid = model.t_grid
    true_q: list[float] = []
    pred_q: list[float] = []
    for i in range(n_holdout):
        q = float(np.exp(rng.normal(np.log(cfg.physiology.cardiac_output_ml_s), 0.25)))
        vc = float(
            np.exp(rng.normal(np.log(cfg.physiology.central_blood_volume_ml), 0.25))
        )
        trial = cfg.physiology.model_copy(
            update={"cardiac_output_ml_s": q, "central_blood_volume_ml": vc}
        )
        series = generate_synthetic(
            trial, cfg.injection, t_grid, backend="closed_form", seed=None
        )
        deg = Degradation(noise_sd_hu=10.0, subsample_stride=2, dose_scale=1.0)
        observed = apply_degradation(series, deg, seed=cfg.seed + 200 + i)
        fitted = model.infer(observed)
        true_q.append(q)
        pred_q.append(fitted.cardiac_output_ml_s)
    save_scatter_plot(
        np.asarray(true_q),
        np.asarray(pred_q),
        output_dir / "fig2b_calibration_q.png",
        title="Amortized cardiac output (hold-out synthetic)",
        xlabel="True Q (mL/s)",
        ylabel="Predicted Q (mL/s)",
    )


def run_robustness_experiment(cfg: Any, output_dir: Path) -> dict[str, Any]:
    seed_everything(cfg.seed)
    times = default_times_s(cfg.t_end_s, cfg.dt_s)
    clean = generate_synthetic(
        cfg.physiology,
        cfg.injection,
        times,
        backend="closed_form",
        noise_sd_hu=0.0,
        seed=cfg.seed,
    )
    degradations = _degradations_from_cfg(cfg)
    amortized = train_amortized(
        cfg.physiology,
        cfg.injection,
        free_params=cfg.free_params,
        t_end_s=cfg.t_end_s,
        n_times=cfg.amortized.n_times,
        hidden=cfg.amortized.hidden,
        n_train=cfg.amortized.n_train,
        n_epochs=cfg.amortized.n_epochs,
        batch_size=cfg.amortized.batch_size,
        lr=cfg.amortized.lr,
        use_aif=cfg.amortized.use_aif,
        degradations=degradations,
        seed=cfg.seed,
    )
    rows = run_robustness_sweep(
        clean,
        cfg.injection,
        cfg.physiology,
        degradations,
        free_params=cfg.free_params,
        amortized=amortized,
        pinn_mode=cfg.pinn.mode,
        pinn_hidden=cfg.pinn.hidden,
        pinn_steps=cfg.pinn.n_steps,
        physics_weight=cfg.pinn.physics_weight,
        seed=cfg.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(rows, output_dir / "robustness_sweep.csv")
    save_sweep_plot(
        rows,
        output_dir / "fig2_curve_nrmse.png",
        y_key="curve_nrmse",
        title="Curve NRMSE vs noise (dose=1)",
        ylabel="Curve NRMSE",
        dose_scale=1.0,
    )
    save_sweep_plot(
        rows,
        output_dir / "fig2_param_mre.png",
        y_key="param_mre",
        title="Parameter mean relative error vs noise (dose=1)",
        ylabel="Mean relative error (θ)",
        dose_scale=1.0,
    )
    example = _example_overlay(cfg, times, clean.region("organ"), output_dir)
    _calibration_scatter(cfg, amortized, output_dir)

    stressed = [
        row
        for row in rows
        if row["noise_sd_hu"] == max(cfg.sweep.noise_sd_hu)
        and row["subsample_stride"] == max(cfg.sweep.subsample_stride)
        and row["dose_scale"] == min(cfg.sweep.dose_scale)
    ]
    summary = {
        "n_cells": len(degradations),
        "n_rows": len(rows),
        "csv": str(output_dir / "robustness_sweep.csv"),
        "fig1": str(output_dir / "fig1_reconstruction.png"),
        "fig2_curve": str(output_dir / "fig2_curve_nrmse.png"),
        "fig2_param": str(output_dir / "fig2_param_mre.png"),
        "stressed_cell": stressed,
        **example,
    }
    return summary
