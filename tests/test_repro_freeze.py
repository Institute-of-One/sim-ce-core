"""Frozen figure/CSV inventory (no network, no experiment rerun)."""

from __future__ import annotations

import json
from pathlib import Path

from sim_ce_core.experiments.repro_check import _check_frozen

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_manifest_lists_existing_files() -> None:
    manifest_path = ROOT / "paper" / "frozen" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["metrics"]["m3_tcia_n_cases"] == 20
    assert manifest["metrics"]["m3_tcia_source"] == "tcia_hcc_tace_seg"
    assert len(manifest["reproduce_figures"]) == 4
    for rel in manifest["files"]:
        assert (ROOT / rel).is_file(), rel


def test_check_frozen_passes_at_repo_root() -> None:
    assert _check_frozen(ROOT) == []
