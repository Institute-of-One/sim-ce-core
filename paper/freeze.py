"""Copy the runs the paper cites into ``paper/frozen/``, and derive the manifest.

    python paper/freeze.py            # -> paper/frozen/
    python paper/freeze.py --check    # fail if the freeze is stale

``paper/frozen/`` is what the manuscript resolves its numbers against, and until now it
was populated by hand. A hand-copied freeze can differ from the run that produced it in
exactly the way a typed number can differ from the run that produced it, and neither
fails loudly. This makes the copy an operation with a check attached.

The manifest's ``metrics`` block is **derived** from the copied files rather than
restated beside them. A summary that repeats a number is a second place for it to be
wrong.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
FROZEN = PAPER / "frozen"

RUN = "python -m sim_ce_core.experiments.run configs/{}.yaml"
DESIGN = "python -m sim_ce_core.experiments.identifiability_map"

#: ``destination -> (source, producing command)``. The command is recorded rather than
#: inferred: a check that matches directory names against command strings passes on a
#: coincidence and fails on a rename, and it did both while this was being written.
#: Only what the manuscript cites is frozen -- an artefact nobody reads is a file that
#: goes stale without anyone noticing.
SOURCES: dict[str, tuple[str, str]] = {
    "m1_summary.json": (
        "outputs/m1_synthetic/summary.json",
        RUN.format("m1_synthetic"),
    ),
    "m2_summary.json": (
        "outputs/m2_robustness/summary.json",
        RUN.format("m2_robustness"),
    ),
    "m2_robustness_sweep.csv": (
        "outputs/m2_robustness/robustness_sweep.csv",
        RUN.format("m2_robustness"),
    ),
    "m3_ablation_summary.json": (
        "outputs/m3_ablation/summary.json",
        RUN.format("m3_ablation"),
    ),
    "m3_ablation.csv": ("outputs/m3_ablation/ablation.csv", RUN.format("m3_ablation")),
    "m3_tcia_summary.json": ("outputs/m3_tcia/summary.json", RUN.format("m3_tcia")),
    "m3_tcia_external.csv": (
        "outputs/m3_tcia/fig3_external.csv",
        RUN.format("m3_tcia"),
    ),
    "m3_tcia_phases.csv": (
        "outputs/m3_tcia/fig3_mphase_phases.csv",
        RUN.format("m3_tcia"),
    ),
    "m35_identifiability.json": ("outputs/m35_identifiability/summary.json", DESIGN),
}


def _reproduce_order() -> list[str]:
    """Each producing command once, in the order the freeze needs them, freeze last."""
    seen: list[str] = []
    for _source, command in SOURCES.values():
        if command not in seen:
            seen.append(command)
    return [*seen, "python paper/freeze.py"]


def _stressed(summary: dict[str, Any], method: str, key: str) -> Any:
    for row in summary.get("stressed_cell", []):
        if row["method"] == method:
            return row.get(key)
    return None


def derive_metrics(loaded: dict[str, Any]) -> dict[str, Any]:
    """Every number the manuscript quotes, read back out of the frozen files."""
    m1, m2 = loaded["m1_summary.json"], loaded["m2_summary.json"]
    ablation, tcia = loaded["m3_ablation_summary.json"], loaded["m3_tcia_summary.json"]
    design = loaded["m35_identifiability.json"]
    endpoint = design["primary_endpoint"]["by_method"]

    metrics: dict[str, Any] = {
        "m1_closed_form_ode_nrmse": m1["closed_form_ode_nrmse"],
        "m1_peak_aorta_hu": m1["peak_aorta_hu"],
        "m1_peak_organ_hu": m1["peak_organ_hu"],
        "m2_n_realisations": m2["n_realisations"],
        "m3_tcia_n_cases": tcia["n_cases"],
        "m3_tcia_source": tcia["sources"][0],
        "m3_tcia_n_phase_rows": tcia["n_phase_rows"],
        "m3_tcia_closed_form_nrmse_mean": tcia["curve_nrmse"]["closed_form"]["mean"],
        "m3_tcia_pinn_nrmse_mean": tcia["curve_nrmse"]["pinn_hybrid"]["mean"],
        "m3_ablation_physics_aif": ablation["mean_curve_nrmse"]["physics_only/AIF"],
        "m3_ablation_hybrid_aif": ablation["mean_curve_nrmse"]["hybrid/AIF"],
        "m3_ablation_neural_aif": ablation["mean_curve_nrmse"]["neural_only/AIF"],
        "amortized_calibration_correlation": m2["amortized_calibration"]["correlation"],
        "amortized_calibration_sd_ratio": m2["amortized_calibration"]["sd_ratio"],
        "amortized_n_train": m2["amortized_budget"]["n_train"],
        "amortized_n_epochs": m2["amortized_budget"]["n_epochs"],
        "endpoint_n_cells": design["primary_endpoint"]["n_cells"],
    }
    for method in ("closed_form", "pinn_hybrid", "amortized"):
        metrics[f"m2_stressed_{method}_param_rmse"] = _stressed(
            m2, method, "param_rmse"
        )
        metrics[f"m2_stressed_{method}_curve_nrmse"] = _stressed(
            m2, method, "curve_nrmse_mean"
        )
        metrics[f"endpoint_{method}_spearman"] = endpoint[method]["spearman"]
        metrics[f"endpoint_{method}_p"] = endpoint[method]["p_value"]
        metrics[f"endpoint_{method}_efficiency"] = endpoint[method][
            "efficiency_ratio_median"
        ]
    return metrics


def build(*, check: bool = False) -> int:
    missing = [src for src, _cmd in SOURCES.values() if not (REPO / src).exists()]
    if missing:
        print("cannot freeze; these runs have not been produced:")
        for src in missing:
            print(f"  {src}")
        return 2

    FROZEN.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    loaded: dict[str, Any] = {}
    for name, (source, _command) in SOURCES.items():
        src, dst = REPO / source, FROZEN / name
        payload = src.read_bytes()
        if check and (not dst.exists() or dst.read_bytes() != payload):
            stale.append(name)
        elif not check:
            shutil.copyfile(src, dst)
        if name.endswith(".json"):
            loaded[name] = json.loads(payload.decode("utf-8"))

    manifest = {
        "hypothesis": "H1",
        "reproduce_figures": _reproduce_order(),
        "primary_endpoint": (
            "Whether the Fisher information of a sampling design predicts which "
            "physiological parameters are recoverable from it."
        ),
        "metrics": derive_metrics(loaded),
        "files": sorted(f"paper/frozen/{name}" for name in SOURCES),
        "sources": {
            name: {"path": source, "command": command}
            for name, (source, command) in sorted(SOURCES.items())
        },
    }
    rendered = json.dumps(manifest, indent=2) + "\n"
    manifest_path = FROZEN / "manifest.json"
    if check:
        current = (
            manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else ""
        )
        if current != rendered:
            stale.append("manifest.json")
        if stale:
            print("the freeze is out of date; run python paper/freeze.py")
            for name in stale:
                print(f"  {name}")
            return 1
        print(f"paper/frozen/ is current ({len(SOURCES)} files)")
        return 0

    manifest_path.write_text(rendered, encoding="utf-8")
    print(f"froze {len(SOURCES)} files and derived {len(manifest['metrics'])} metrics")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify without copying")
    return build(check=parser.parse_args(argv).check)


if __name__ == "__main__":
    sys.exit(main())
