"""Rebuild every submission artefact in dependency order, then check them.

    python paper/build_all.py

There are eight artefacts and they depend on each other: the frozen results feed the
manuscript, the manuscript feeds the Word file and the portal fields, the figures feed
both, and the cover letter and the kit quote all of it. Rebuilding one and forgetting
another is how a copy goes stale, and every copy in this programme has gone stale at
least once -- a cover letter that kept a withdrawn claim, a submission form three weeks
behind the abstract, a kit naming a title the paper no longer had.

Running the steps by hand in the right order is a thing to get wrong. This is the order.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent

#: In dependency order. Each is skipped if the experiments it needs have not been run;
#: the checks at the end fail loudly rather than letting a gap pass as success.
STEPS: tuple[tuple[str, list[str]], ...] = (
    ("freeze the runs the paper cites", ["paper/freeze.py"]),
    ("collect the figures", ["paper/collect_figures.py"]),
    ("resolve the manuscript's numbers", ["paper/build_manuscript.py"]),
    ("render the manuscript to Word", ["paper/build_docx.py"]),
    (
        "build highlights, declaration and cover letter",
        ["paper/build_submission_files.py"],
    ),
    ("write the portal's title and abstract", ["paper/make_portal_fields.py"]),
    ("check everything", ["paper/presubmission_check.py", "--strict"]),
)


def main(argv: list[str] | None = None) -> int:
    for description, command in STEPS:
        print(f"\n=== {description}")
        result = subprocess.run(
            [sys.executable, *command], cwd=REPO, check=False, text=True
        )
        if result.returncode != 0:
            print(f"\nstopped at: {description}")
            return result.returncode
    print("\nevery artefact rebuilt and checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
