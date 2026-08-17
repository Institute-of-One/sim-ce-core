"""v1 reproducibility check: lint, tests, and frozen figure/CSV inventory.

Usage:
    python -m sim_ce_core.experiments.repro_check
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FROZEN_MANIFEST = Path("paper/frozen/manifest.json")


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _check_frozen(root: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = root / FROZEN_MANIFEST
    if not manifest_path.is_file():
        return [f"missing {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel in manifest.get("files", []):
        path = root / rel
        if not path.is_file():
            errors.append(f"missing frozen file: {rel}")
    required_cmds = manifest.get("reproduce_figures", [])
    if not required_cmds:
        errors.append("manifest.reproduce_figures is empty")
    return errors


def main(argv: list[str] | None = None) -> int:
    del argv
    root = Path.cwd()
    py = sys.executable
    steps = [
        [py, "-m", "ruff", "check", "."],
        [py, "-m", "black", "--check", "."],
        [py, "-m", "pytest", "-q"],
    ]
    failed = False
    for cmd in steps:
        if _run(cmd) != 0:
            failed = True
    freeze_errors = _check_frozen(root)
    for err in freeze_errors:
        print(f"FREEZE: {err}")
        failed = True
    if not freeze_errors:
        print("FREEZE: paper/frozen/manifest.json OK")
    print("To regenerate figures:")
    if (root / FROZEN_MANIFEST).is_file():
        manifest = json.loads((root / FROZEN_MANIFEST).read_text(encoding="utf-8"))
        for cmd in manifest.get("reproduce_figures", []):
            print(" ", cmd)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
