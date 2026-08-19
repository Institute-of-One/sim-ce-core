"""The frozen inventory the manuscript resolves its numbers against.

``paper/frozen/`` used to be populated by hand. It is now produced by
``paper/freeze.py`` from the runs themselves, and the manifest's metrics are derived
from the copied files rather than restated beside them, so each number has one home
rather than two.
"""

from __future__ import annotations

import json
from pathlib import Path

from sim_ce_core.experiments.repro_check import _check_frozen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper" / "frozen" / "manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_every_frozen_file_exists() -> None:
    for rel in _manifest()["files"]:
        assert (ROOT / rel).is_file(), rel


def test_the_external_arm_is_real_and_not_the_proxy() -> None:
    """The one metric whose wrong value would misrepresent the study.

    A synthetic proxy cohort is available to the same loaders and is tagged
    ``synthetic_proxy``. If that tag ever reaches the freeze, the paper would be
    reporting simulated data as external validation.
    """
    metrics = _manifest()["metrics"]
    assert metrics["m3_tcia_source"] == "tcia_hcc_tace_seg"
    assert metrics["m3_tcia_n_cases"] == 20


def test_the_manifest_says_how_to_regenerate_itself() -> None:
    """Every run the freeze copies, plus the freeze, named as commands."""
    manifest = _manifest()
    commands = manifest["reproduce_figures"]
    assert commands, "manifest.reproduce_figures is empty"
    assert any("freeze" in command for command in commands), (
        "the manifest lists the experiments but not the step that freezes them"
    )
    for name, entry in manifest["sources"].items():
        assert entry["command"] in commands, (
            f"{name} names a producing command that reproduce_figures omits"
        )


def test_check_frozen_passes_at_repo_root() -> None:
    assert _check_frozen(ROOT) == []
