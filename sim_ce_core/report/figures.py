"""Figure and CSV export (accessible matplotlib defaults)."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim_ce_core.data.types import EnhancementSeries

# Okabe–Ito-ish, colorblind-friendly
_PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")


def save_enhancement_csv(series: EnhancementSeries, path: Path) -> None:
    """Write times + named HU columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "time_s," + ",".join(series.region_names)
    n_regions = series.curves_hu.shape[1]
    cols = [series.times_s, *[series.curves_hu[:, i] for i in range(n_regions)]]
    stacked = np.column_stack(cols)
    np.savetxt(path, stacked, delimiter=",", header=header, comments="")


def save_enhancement_plot(
    series: EnhancementSeries,
    path: Path,
    *,
    title: str = "Synthetic contrast enhancement",
) -> None:
    """Plot time–enhancement curves and save PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    for i, name in enumerate(series.region_names):
        ax.plot(
            series.times_s,
            series.curves_hu[:, i],
            color=_PALETTE[i % len(_PALETTE)],
            lw=2.0,
            label=name,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Enhancement (HU)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_rows_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write a list of homogeneous dicts as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_overlay_plot(
    times_s: np.ndarray,
    curves: Mapping[str, np.ndarray],
    path: Path,
    *,
    title: str,
    ylabel: str = "Enhancement (HU)",
) -> None:
    """Overlay named 1-d curves (Fig 1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    for i, (label, values) in enumerate(curves.items()):
        style = "-" if i == 0 else "--"
        ax.plot(
            times_s,
            values,
            color=_PALETTE[i % len(_PALETTE)],
            lw=2.0,
            ls=style,
            label=label,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_sweep_plot(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    y_key: str,
    title: str,
    ylabel: str,
    dose_scale: float = 1.0,
) -> None:
    """Fig 2: metric vs noise, one line per method (fixed dose, all strides)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [row for row in rows if float(row["dose_scale"]) == dose_scale]
    methods = sorted({str(row["method"]) for row in selected})
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    for i, method in enumerate(methods):
        method_rows = [row for row in selected if row["method"] == method]
        strides = sorted({int(row["subsample_stride"]) for row in method_rows})
        for j, stride in enumerate(strides):
            pts = [row for row in method_rows if int(row["subsample_stride"]) == stride]
            pts = sorted(pts, key=lambda row: float(row["noise_sd_hu"]))
            xs = [float(row["noise_sd_hu"]) for row in pts]
            ys = [float(row[y_key]) for row in pts]
            if any(np.isnan(ys)):
                continue
            ax.plot(
                xs,
                ys,
                color=_PALETTE[i % len(_PALETTE)],
                ls="-" if j == 0 else "--",
                marker="o",
                lw=2.0,
                label=f"{method} (stride={stride})",
            )
    ax.set_xlabel("Noise SD (HU)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Calibration scatter (predicted vs true)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), layout="constrained")
    ax.scatter(x, y, c=_PALETTE[0], s=28, alpha=0.8, edgecolors="none")
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=1.0, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_bar_plot(
    labels: Sequence[str],
    values: Sequence[float],
    path: Path,
    *,
    title: str,
    ylabel: str,
    yerr: Sequence[float] | None = None,
) -> None:
    """Single-series bar chart (Fig 3 / ablations)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]
    ax.bar(
        list(labels),
        list(values),
        yerr=None if yerr is None else list(yerr),
        color=colors,
        edgecolor="none",
        capsize=4,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)
