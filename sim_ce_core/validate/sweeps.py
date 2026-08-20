"""Robustness sweeps: recovery vs noise / sparsity / dose."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from sim_ce_core.data.types import EnhancementSeries
from sim_ce_core.nn.amortized import AmortizedModel
from sim_ce_core.nn.pinn import PinnMode, fit_pinn
from sim_ce_core.physio.fit import DEFAULT_FREE, recover_parameters
from sim_ce_core.physio.forward import simulate_hu
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.validate.deconvolution import reconstruct_organ
from sim_ce_core.validate.degrade import Degradation, apply_degradation, scale_protocol
from sim_ce_core.validate.metrics import mean_relative_error, nrmse

LS_INIT_SCALE = {
    "central_blood_volume_ml": 1.35,
    "cardiac_output_ml_s": 0.72,
}


def default_ls_init(
    template: PhysioParams, free_params: Sequence[str]
) -> dict[str, float]:
    """Fixed off-center start so LS is not initialized at the truth."""
    init: dict[str, float] = {}
    for name in free_params:
        scale = LS_INIT_SCALE.get(name, 1.25)
        init[name] = float(getattr(template, name) * scale)
    return init


def interpolate_to(
    times_src: np.ndarray, values: np.ndarray, times_dst: np.ndarray
) -> np.ndarray:
    """Linear interpolation of a 1-d or 2-d curve onto ``times_dst``."""
    src = np.asarray(times_src, dtype=np.float64)
    dst = np.asarray(times_dst, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return np.interp(dst, src, arr)
    return np.column_stack(
        [np.interp(dst, src, arr[:, i]) for i in range(arr.shape[1])]
    )


def _row(
    method: str,
    degradation: Degradation,
    *,
    curve_nrmse: float,
    param_mre: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "method": method,
        "noise_sd_hu": degradation.noise_sd_hu,
        "subsample_stride": degradation.subsample_stride,
        "dose_scale": degradation.dose_scale,
        "curve_nrmse": curve_nrmse,
        "param_mre": float("nan") if param_mre is None else param_mre,
    }
    if extra:
        payload.update(extra)
    return payload


def evaluate_closed_form(
    clean: EnhancementSeries,
    observed: EnhancementSeries,
    protocol: InjectionProtocol,
    template: PhysioParams,
    free_params: Sequence[str],
    degradation: Degradation,
) -> dict[str, Any]:
    init = default_ls_init(template, free_params)
    fitted, _info = recover_parameters(
        observed.times_s,
        observed.curves_hu[:, :2],
        protocol,
        template,
        free_params=free_params,
        init=init,
        region_names=("aorta", "organ"),
        max_nfev=80,
    )
    pred = simulate_hu(fitted, protocol, clean.times_s).detach().cpu().numpy()
    return _row(
        "closed_form",
        degradation,
        curve_nrmse=nrmse(pred[:, :2], clean.curves_hu[:, :2]),
        param_mre=mean_relative_error(fitted, template, free_params),
    )


def evaluate_deconvolution(
    clean: EnhancementSeries,
    observed: EnhancementSeries,
    degradation: Degradation,
) -> dict[str, Any]:
    organ_hat = reconstruct_organ(observed.aif_hu, observed.region("organ"))
    organ_full = interpolate_to(observed.times_s, organ_hat, clean.times_s)
    return _row(
        "deconvolution",
        degradation,
        curve_nrmse=nrmse(organ_full, clean.region("organ")),
        param_mre=None,
    )


def evaluate_pinn(
    clean: EnhancementSeries,
    observed: EnhancementSeries,
    protocol: InjectionProtocol,
    template: PhysioParams,
    free_params: Sequence[str],
    degradation: Degradation,
    *,
    mode: PinnMode = "hybrid",
    hidden: int = 32,
    n_steps: int = 120,
    physics_weight: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    init = default_ls_init(template, free_params)
    result = fit_pinn(
        observed.times_s,
        observed.curves_hu[:, :2],
        protocol,
        template,
        mode=mode,
        free_params=free_params,
        init=init,
        hidden=hidden,
        n_steps=n_steps,
        physics_weight=physics_weight,
        seed=seed,
    )
    pred = result.predict_hu(clean.times_s, protocol)
    param_mre = (
        None
        if mode == "neural_only"
        else mean_relative_error(result.params, template, free_params)
    )
    return _row(
        f"pinn_{mode}",
        degradation,
        curve_nrmse=nrmse(pred[:, :2], clean.curves_hu[:, :2]),
        param_mre=param_mre,
        extra={"data_loss": result.data_loss, "physics_loss": result.physics_loss},
    )


def evaluate_amortized(
    clean: EnhancementSeries,
    observed: EnhancementSeries,
    protocol: InjectionProtocol,
    template: PhysioParams,
    free_params: Sequence[str],
    degradation: Degradation,
    model: AmortizedModel,
) -> dict[str, Any]:
    fitted = model.infer(observed)
    pred = simulate_hu(fitted, protocol, clean.times_s).detach().cpu().numpy()
    return _row(
        "amortized",
        degradation,
        curve_nrmse=nrmse(pred[:, :2], clean.curves_hu[:, :2]),
        param_mre=mean_relative_error(fitted, template, free_params),
    )


def run_robustness_sweep(
    clean: EnhancementSeries,
    base_protocol: InjectionProtocol,
    template: PhysioParams,
    degradations: Sequence[Degradation],
    *,
    free_params: Sequence[str] = DEFAULT_FREE,
    amortized: AmortizedModel | None = None,
    pinn_mode: PinnMode = "hybrid",
    pinn_hidden: int = 32,
    pinn_steps: int = 120,
    physics_weight: float = 1.0,
    seed: int = 0,
    include_neural_only: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate baselines and neural methods on each degradation cell."""
    rows: list[dict[str, Any]] = []
    for i, deg in enumerate(degradations):
        observed = apply_degradation(clean, deg, seed=seed + 17 * i)
        protocol = scale_protocol(base_protocol, deg.dose_scale)
        rows.append(
            evaluate_closed_form(clean, observed, protocol, template, free_params, deg)
        )
        rows.append(evaluate_deconvolution(clean, observed, deg))
        rows.append(
            evaluate_pinn(
                clean,
                observed,
                protocol,
                template,
                free_params,
                deg,
                mode=pinn_mode,
                hidden=pinn_hidden,
                n_steps=pinn_steps,
                physics_weight=physics_weight,
                seed=seed + i,
            )
        )
        if include_neural_only:
            rows.append(
                evaluate_pinn(
                    clean,
                    observed,
                    protocol,
                    template,
                    free_params,
                    deg,
                    mode="neural_only",
                    hidden=pinn_hidden,
                    n_steps=pinn_steps,
                    physics_weight=0.0,
                    seed=seed + i,
                )
            )
        if amortized is not None:
            rows.append(
                evaluate_amortized(
                    clean, observed, protocol, template, free_params, deg, amortized
                )
            )
    return rows


def aggregate_cells(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated realisations of each cell into one row per method.

    The parameter error is reported as a **root-mean-square** over realisations, because
    that is the quantity a Cramer-Rao bound constrains. A mean absolute error is not
    comparable with a standard deviation, and the difference is not cosmetic: the two
    diverge exactly where the error distribution is skewed, which is where a fit is
    starting to fail and where the comparison matters most.
    """
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["method"],
            row["noise_sd_hu"],
            row["subsample_stride"],
            row["dose_scale"],
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (method, noise, stride, dose), members in sorted(grouped.items(), key=str):
        errors = np.array(
            [float(m["param_mre"]) for m in members if m["param_mre"] is not None],
            dtype=np.float64,
        )
        errors = errors[np.isfinite(errors)]
        curves = np.array([float(m["curve_nrmse"]) for m in members], dtype=np.float64)
        curves = curves[np.isfinite(curves)]
        out.append(
            {
                "method": method,
                "noise_sd_hu": noise,
                "subsample_stride": stride,
                "dose_scale": dose,
                "n_realisations": len(members),
                "param_rmse": (
                    float(np.sqrt((errors**2).mean())) if errors.size else None
                ),
                "param_mre_mean": float(errors.mean()) if errors.size else None,
                "param_mre_sd": float(errors.std(ddof=1)) if errors.size > 1 else None,
                "curve_nrmse_mean": float(curves.mean()) if curves.size else None,
                "curve_nrmse_sd": (
                    float(curves.std(ddof=1)) if curves.size > 1 else None
                ),
            }
        )
    return out
