"""Build the four files the submission portal asks for, as Word documents.

    python paper/build_submission_files.py

Elsevier's upload step requires a declaration of competing interests, highlights, a
cover letter and the manuscript. Three were prose files in whatever form they happened
to be in; the portal wants documents, and one of them has a hard limit that is easy to
breach and invisible until an editor counts.

Highlights are three to five bullets of **at most 85 characters including spaces**.
Their numbers are markers resolved from ``paper/frozen/``, for the same reason the
manuscript's are: a highlight is a fourth copy of the results, and copies drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER))

import build_manuscript  # noqa: E402

BUILD = PAPER / "build"

#: Elsevier's limit. Counted after the markers resolve, because that is the string an
#: editor sees.
HIGHLIGHT_CHARACTERS = 85
HIGHLIGHT_MIN, HIGHLIGHT_MAX = 3, 5

#: ``source -> output``. The manuscript is built separately; it carries figures.
DOCUMENTS: dict[str, str] = {
    "declaration_of_interest.md": "declaration_of_interest.docx",
    "highlights.md": "highlights.docx",
}


def _resolved(name: str) -> str:
    return build_manuscript.render((PAPER / name).read_text(encoding="utf-8"))


def check_highlights(text: str) -> list[str]:
    """Return the bullets, having refused any that breaches the limit."""
    bullets = [
        line[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")
    ]
    problems = [
        f"{len(bullet)} characters, over {HIGHLIGHT_CHARACTERS}: {bullet}"
        for bullet in bullets
        if len(bullet) > HIGHLIGHT_CHARACTERS
    ]
    if not HIGHLIGHT_MIN <= len(bullets) <= HIGHLIGHT_MAX:
        problems.append(
            f"{len(bullets)} highlights; Elsevier wants "
            f"{HIGHLIGHT_MIN} to {HIGHLIGHT_MAX}"
        )
    if problems:
        raise SystemExit("highlights:\n  " + "\n  ".join(problems))
    return bullets


def main(argv: list[str] | None = None) -> int:
    import pypandoc  # noqa: PLC0415

    BUILD.mkdir(parents=True, exist_ok=True)

    bullets = check_highlights(_resolved("highlights.md"))
    for name, output in DOCUMENTS.items():
        pypandoc.convert_text(
            _resolved(name),
            to="docx",
            format="markdown",
            outputfile=str(BUILD / output),
        )
        print(f"wrote {BUILD / output}")

    letter = (PAPER / "cover_letter_cmpb.txt").read_text(encoding="utf-8")
    pypandoc.convert_text(
        letter,
        to="docx",
        format="markdown",
        outputfile=str(BUILD / "cover_letter.docx"),
    )
    print(f"wrote {BUILD / 'cover_letter.docx'}")

    print(
        f"\n{len(bullets)} highlights, longest {max(len(b) for b in bullets)}/"
        f"{HIGHLIGHT_CHARACTERS} characters:"
    )
    for bullet in bullets:
        print(f"  [{len(bullet):2d}] {bullet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
