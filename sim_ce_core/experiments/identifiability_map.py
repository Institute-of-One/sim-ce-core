"""Does the Fisher information of a sampling design predict what can be recovered?

This is the study's primary endpoint, and it is falsifiable: if the Cramer-Rao bound of
an acquisition schedule does not track the error that estimators actually make on data
from that schedule, the identifiability account is wrong and the paper says so.

It is tested against the robustness sweep rather than against anything new. Each cell of
that sweep is a sampling design -- a noise level, a temporal stride and a dose -- run
through twenty independent noise draws, so the measured quantity is a spread and can be
compared with a bound on a spread. The v1 sweep ran one draw per cell, which cannot be.

Like for like matters here and is easy to get wrong. The reported parameter error is a
mean absolute relative error over the free parameters; the bound is a standard error per
parameter. For an efficient unbiased estimator the absolute error of a Gaussian has
expectation ``sqrt(2/pi)`` times its standard deviation, so the comparator is the
mean of the per-parameter bounds scaled by that factor. Comparing the mean error against
the largest bound, as a first pass did, makes every estimator look better than it is.

    python -m sim_ce_core.experiments.identifiability_map
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sim_ce_core.design.fisher import fisher_information
from sim_ce_core.design.identifiability import analyse
from sim_ce_core.design.sensitivity import DESIGN_PARAMS, jacobian
from sim_ce_core.physio.fit import DEFAULT_FREE
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams

#: E|X| = sqrt(2/pi) * sd for a zero-mean Gaussian. See the module docstring.
FOLDED_NORMAL = float(np.sqrt(2.0 / np.pi))

#: The acquisition schedules a routine abdominal CT actually uses, in seconds after the
#: start of injection. Named rather than swept because the paper's question is about
#: these, not about an arbitrary grid.
CLINICAL_DESIGNS: dict[str, list[float]] = {
    "two-phase (pre, portal venous)": [0.0, 70.0],
    "three-phase (pre, arterial, portal venous)": [0.0, 35.0, 70.0],
    "four-phase (adds delayed)": [0.0, 35.0, 70.0, 180.0],
    "dense (every 10 s to 190 s)": [float(t) for t in range(0, 200, 10)],
}


def _sweep_geometry(stride: int, dose: float, protocol: InjectionProtocol):
    """The times and injection the robustness sweep used for one cell."""
    grid = np.arange(0.0, 90.0 + 1e-9, 1.0)
    return (
        grid[::stride],
        protocol.model_copy(update={"volume_ml": protocol.volume_ml * dose}),
    )


def bound_for(
    physiology: PhysioParams,
    protocol: InjectionProtocol,
    times: Any,
    *,
    sigma_hu: float,
    parameter_names: tuple[str, ...] = DEFAULT_FREE,
) -> dict[str, Any]:
    """Cramer-Rao bounds for one design, and the mean-absolute-error comparator."""
    values = {name: float(getattr(physiology, name)) for name in parameter_names}
    sensitivity = jacobian(
        physiology, protocol, times, parameter_names=parameter_names
    )
    result = analyse(
        fisher_information(sensitivity, values, sigma_hu=sigma_hu), parameter_names
    )
    crlb = result.crlb.numpy()
    return {
        "crlb": {
            name: float(value)
            for name, value in zip(parameter_names, crlb, strict=True)
        },
        "crlb_mean": float(crlb.mean()),
        "expected_absolute_error": FOLDED_NORMAL * float(crlb.mean()),
        "numerical_rank": result.numerical_rank,
        "condition_number": result.condition_number,
    }


def clinical_map(
    physiology: PhysioParams, protocol: InjectionProtocol, *, sigma_hu: float = 25.0
) -> list[dict[str, Any]]:
    """What each routine phase pattern determines, for the full physiology and for the
    two parameters the inverse actually frees."""
    rows = []
    for label, times in CLINICAL_DESIGNS.items():
        full = analyse(
            fisher_information(
                jacobian(physiology, protocol, times),
                {name: float(getattr(physiology, name)) for name in DESIGN_PARAMS},
                sigma_hu=sigma_hu,
            ),
            DESIGN_PARAMS,
        )
        fitted = bound_for(
            physiology, protocol, times, sigma_hu=sigma_hu, parameter_names=DEFAULT_FREE
        )
        rows.append(
            {
                "design": label,
                "n_phases": len(times),
                "times_s": list(times),
                "full_model_rank": full.numerical_rank,
                "full_model_parameters": len(DESIGN_PARAMS),
                "not_separable": list(full.not_separable),
                **{f"fitted_{key}": value for key, value in fitted.items()},
            }
        )
    return rows


def bound_versus_error(
    physiology: PhysioParams,
    protocol: InjectionProtocol,
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    """The primary endpoint: bound against measured error, per estimator."""
    from scipy.stats import spearmanr  # noqa: PLC0415 - optional at import time

    grouped: dict[tuple[float, int, float], dict[str, Any]] = {}
    for cell in cells:
        key = (cell["noise_sd_hu"], cell["subsample_stride"], cell["dose_scale"])
        grouped.setdefault(key, {})[cell["method"]] = cell

    rows = []
    for (noise, stride, dose), methods in sorted(grouped.items()):
        if noise <= 0.0:
            continue  # noiseless control: the bound is zero and recovery is exact
        times, scaled = _sweep_geometry(stride, dose, protocol)
        bound = bound_for(physiology, scaled, times, sigma_hu=noise)
        rows.append(
            {
                "noise_sd_hu": noise,
                "subsample_stride": stride,
                "dose_scale": dose,
                "n_times": len(times),
                "expected_absolute_error": bound["expected_absolute_error"],
                **{
                    method: methods[method]["param_mre_mean"]
                    for method in ("closed_form", "pinn_hybrid", "amortized")
                    if method in methods
                },
            }
        )

    bounds = np.array([row["expected_absolute_error"] for row in rows])
    per_method = {}
    for method in ("closed_form", "pinn_hybrid", "amortized"):
        errors = np.array([row[method] for row in rows], dtype=np.float64)
        stat = spearmanr(bounds, errors)
        ratio = errors / bounds
        per_method[method] = {
            "spearman": float(stat.statistic),
            "p_value": float(stat.pvalue),
            "efficiency_ratio_median": float(np.median(ratio)),
            "efficiency_ratio_min": float(ratio.min()),
            "efficiency_ratio_max": float(ratio.max()),
            "cells_below_bound": int((errors < bounds).sum()),
        }
    return {"n_cells": len(rows), "rows": rows, "by_method": per_method}


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parent.parent.parent
    summary_path = repo / "outputs" / "m2_robustness" / "summary.json"
    if not summary_path.exists():
        print(f"{summary_path} is missing: run configs/m2_robustness.yaml first")
        return 2
    sweep = json.loads(summary_path.read_text(encoding="utf-8"))
    if "cells" not in sweep:
        print("the sweep summary predates per-cell aggregation; re-run it")
        return 2

    physiology = PhysioParams(
        central_blood_volume_ml=1000.0,
        organ_volume_ml=400.0,
        recirculation_volume_ml=2500.0,
        cardiac_output_ml_s=108.3,
        organ_flow_fraction=0.25,
        elimination_rate_1_s=0.0,
        iodine_to_hu=26.0,
        transit_delay_s=6.0,
    )
    protocol = InjectionProtocol(
        concentration_mgi_ml=350.0, volume_ml=100.0, duration_s=25.0
    )

    payload = {
        "n_realisations": sweep.get("n_realisations"),
        "clinical_designs": clinical_map(physiology, protocol),
        "primary_endpoint": bound_versus_error(physiology, protocol, sweep["cells"]),
    }
    out_dir = repo / "outputs" / "m35_identifiability"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    endpoint = payload["primary_endpoint"]
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"  primary endpoint over {endpoint['n_cells']} cells:")
    for method, stats in endpoint["by_method"].items():
        print(
            f"    {method:12s} Spearman={stats['spearman']:+.3f} "
            f"p={stats['p_value']:.4f}  error/bound median "
            f"{stats['efficiency_ratio_median']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
