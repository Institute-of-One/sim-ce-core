"""The two figures the identifiability result needs, drawn from the frozen runs.

Both are drawn at the width they print into and with every text size set explicitly.
A figure judged on screen and reduced to a journal column arrives with its axis labels
under six point, which is the shape a companion paper shipped twice before anyone
noticed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

#: One column of the journal's text block. Figures are drawn at this width rather than
#: drawn large and shrunk.
COLUMN_INCHES = 6.5

_STYLE: dict[str, Any] = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}

#: Estimator labels as the manuscript names them, in the order they are discussed.
METHODS: tuple[tuple[str, str], ...] = (
    ("closed_form", "closed-form Bae"),
    ("pinn_hybrid", "PINN hybrid"),
    ("amortized", "amortized"),
)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def figure_identifiability_map(path: Path, design: dict[str, Any]) -> Path:
    """What each routine phase pattern determines, against what the model allows.

    Two panels because two different things limit recovery and conflating them is the
    error the paper exists to correct. Left: the rank of the full seven-parameter
    physiology, which never reaches seven however densely the curve is sampled, because
    the model has an exact scale symmetry. Right: the Cramer-Rao bound on the two
    parameters an inverse actually frees, which does improve with sampling and is what a
    clinical phase count decides.
    """
    rows = design["clinical_designs"]["designs"]
    labels = [row["design"].split(" (")[0] for row in rows]
    positions = range(len(rows))

    with plt.rc_context(_STYLE):
        fig, (left, right) = plt.subplots(
            1, 2, figsize=(COLUMN_INCHES, 3.4), constrained_layout=True
        )
        total = rows[0]["full_model_parameters"]
        left.bar(
            positions,
            [row["full_model_rank"] for row in rows],
            color="#4C78A8",
            alpha=0.85,
        )
        left.axhline(
            total,
            color="0.35",
            ls="--",
            lw=1.0,
            label=f"{total} free parameters",
        )
        left.set_ylim(0, total + 0.6)
        left.set_ylabel("directions the design constrains")
        left.set_title("Full physiology: rank of the design", fontsize=9)
        left.legend(fontsize=7.5, loc="lower right")

        bounds = [row["fitted_expected_absolute_error"] for row in rows]
        right.bar(positions, bounds, color="#E45756", alpha=0.85)
        right.axhline(0.20, color="0.35", ls=":", lw=1.0, label="20% relative error")
        right.set_yscale("log")
        right.set_ylabel("Cramer-Rao bound (relative)")
        right.set_title("Two fitted parameters: the bound", fontsize=9)
        right.legend(fontsize=7.5, loc="upper right")

        for axis in (left, right):
            axis.set_xticks(list(positions))
            axis.set_xticklabels(labels, fontsize=7.5, rotation=20, ha="right")
        return _save(fig, path)


def figure_bound_versus_error(path: Path, design: dict[str, Any]) -> Path:
    """The primary endpoint: the bound of a design against the error estimators make.

    The identity line is where an efficient unbiased estimator sits. Distance above it
    is the estimator's own cost, and it is the whole of what a better estimator could
    recover -- which on this evidence is nothing for the closed form and about half a
    bound's worth for each neural method.
    """
    endpoint = design["primary_endpoint"]
    rows = endpoint["rows"]
    bounds = [row["expected_absolute_error"] for row in rows]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(COLUMN_INCHES, 4.0), constrained_layout=True)
        low = min(bounds) * 0.6
        high = max(max(row[key] for key, _ in METHODS) for row in rows) * 1.5
        ax.plot(
            [low, high],
            [low, high],
            "k--",
            lw=1.0,
            label="efficient estimator (error = bound)",
        )
        for key, label in METHODS:
            ratio = endpoint["by_method"][key]["efficiency_ratio_median"]
            ax.plot(
                bounds,
                [row[key] for row in rows],
                marker="o",
                ms=5,
                ls="none",
                alpha=0.85,
                label=f"{label} (median {ratio:.2f}x)",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("Cramer-Rao bound of the sampling design (relative)")
        ax.set_ylabel("measured error, mean over 20 realisations")
        spearman = endpoint["by_method"]["closed_form"]["spearman"]
        ax.set_title(
            f"The design's bound predicts the error (Spearman {spearman:.2f})",
            fontsize=9.5,
        )
        ax.legend(fontsize=7.5, loc="upper left")
        return _save(fig, path)


def make_design_figures(design_summary: Path, figures: Path) -> dict[str, Path]:
    design = json.loads(design_summary.read_text(encoding="utf-8"))
    return {
        "identifiability_map": figure_identifiability_map(
            figures / "fig_identifiability_map.png", design
        ),
        "bound_versus_error": figure_bound_versus_error(
            figures / "fig_bound_vs_error.png", design
        ),
    }
