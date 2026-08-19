"""Gather the figures the manuscript places, and hold their captions in one file.

    python paper/collect_figures.py            # -> paper/figures/
    python paper/collect_figures.py --check    # fail if any is stale or missing

Fourteen PNGs live under ``outputs/``, of which the paper places eight. Copying the
eight here means a figure the manuscript names cannot quietly become the one a rerun
last wrote to a different directory, and it makes the set that ships visible as a
directory listing rather than as a claim.

Captions live in ``paper/README.md``, once. A caption written twice -- as image alt text
and again as a numbered paragraph -- prints twice in the converted document, in two
wordings, which is a defect a companion paper shipped.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
FIGURES = PAPER / "figures"
README = PAPER / "README.md"

#: ``figure number -> (destination, source)``, in the order the manuscript cites them.
FIGURE_SOURCES: dict[int, tuple[str, str]] = {
    1: (
        "fig1_identifiability_map.png",
        "outputs/m35_identifiability/fig_identifiability_map.png",
    ),
    2: (
        "fig2_bound_vs_error.png",
        "outputs/m35_identifiability/fig_bound_vs_error.png",
    ),
    3: ("fig3_forward_model.png", "outputs/m1_synthetic/enhancement.png"),
    4: ("fig4_reconstruction.png", "outputs/m2_robustness/fig1_reconstruction.png"),
    5: ("fig5_param_error.png", "outputs/m2_robustness/fig2_param_mre.png"),
    6: (
        "fig6_amortized_calibration.png",
        "outputs/m2_robustness/fig2b_calibration_q.png",
    ),
    7: ("fig7_ablation.png", "outputs/m3_ablation/fig3b_ablation.png"),
    8: ("fig8_real_multiphase.png", "outputs/m3_tcia/fig3_external_nrmse.png"),
}

_CAPTION_ROW = re.compile(r"^\|\s*Fig\s*(\d+)\s*\|\s*(.+?)\s*\|\s*$", re.M)


def captions() -> dict[int, str]:
    """The caption table in ``paper/README.md``, as ``number -> text``."""
    if not README.exists():
        return {}
    return {
        int(number): text
        for number, text in _CAPTION_ROW.findall(README.read_text(encoding="utf-8"))
    }


def collect(*, check: bool = False) -> int:
    missing = [
        src for _dst, src in FIGURE_SOURCES.values() if not (REPO / src).exists()
    ]
    if missing:
        print("these figures have not been generated:")
        for src in missing:
            print(f"  {src}")
        return 2

    FIGURES.mkdir(parents=True, exist_ok=True)
    stale = []
    for number, (name, source) in sorted(FIGURE_SOURCES.items()):
        src, dst = REPO / source, FIGURES / name
        if check:
            if not dst.exists() or dst.read_bytes() != src.read_bytes():
                stale.append(f"Figure {number}: {name}")
        else:
            shutil.copyfile(src, dst)

    written = captions()
    for number in FIGURE_SOURCES:
        if number not in written:
            stale.append(f"Figure {number} has no caption in paper/README.md")
    extra = set(written) - set(FIGURE_SOURCES)
    for number in sorted(extra):
        stale.append(f"paper/README.md captions Figure {number}, which is not placed")

    if stale:
        verb = "is out of date" if check else "needs attention"
        print(f"the figure set {verb}:")
        for item in stale:
            print(f"  {item}")
        return 1
    print(f"{len(FIGURE_SOURCES)} figures collected, each with one caption")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify without copying")
    return collect(check=parser.parse_args(argv).check)


if __name__ == "__main__":
    sys.exit(main())
