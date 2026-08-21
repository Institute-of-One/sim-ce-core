"""The manuscript may not contain a typed number.

Every figure in the prose is a marker resolved from ``paper/frozen/`` at build time.
The v1 draft carried all seventeen of its metrics as typed digits, and a typed number
can drift from the run that produced it and be wrong without anything failing. Both
companion papers shipped exactly that, and in both cases a person reading the page
caught it rather than any check.

Three things are asserted here:

1. the committed build is current, so a stale figure cannot reach a reader;
2. a marker naming something that does not exist is an error, not a string passed
   through in silence;
3. **no frozen metric appears in the source as a typed literal**, which is the property
   that actually stops hand-typing, rather than merely making markers available.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper"
FROZEN = PAPER / "frozen"
SOURCE = PAPER / "manuscript.md"
BUILT = PAPER / "build" / "manuscript.md"

sys.path.insert(0, str(PAPER))
import build_manuscript  # noqa: E402


def test_the_committed_build_is_current():
    """A rebuild must reproduce the committed file exactly."""
    rendered = build_manuscript.render(SOURCE.read_text(encoding="utf-8"), FROZEN)
    assert BUILT.exists(), "paper/build/manuscript.md has never been built"
    assert (
        BUILT.read_text(encoding="utf-8") == rendered
    ), "paper/build/manuscript.md is out of date: run python paper/build_manuscript.py"


def test_the_build_resolves_every_marker():
    # The file documents its own notation in a comment at the top, and that example is
    # deliberately left unresolved -- it names no real file. Counting it as a failure is
    # the checker being wrong about the manuscript rather than the other way round.
    built = re.sub(r"<!--.*?-->", "", BUILT.read_text(encoding="utf-8"), flags=re.S)
    # A malformed marker passes through as literal text and a well-formed-only search
    # would not see it, so openings are counted against closings.
    opened = len(re.findall(r"\[\[results:", built))
    closed = len(re.findall(r"\[\[results:[^\]]*\]\]", built))
    assert (
        opened == closed == 0
    ), f"{opened} unresolved marker(s) in the built manuscript"


def test_a_marker_that_names_nothing_is_an_error():
    with pytest.raises(build_manuscript.ResolutionError):
        build_manuscript.render(
            "[[results:manifest.json:metrics.not_a_real_metric]]", FROZEN
        )
    with pytest.raises(build_manuscript.ResolutionError):
        build_manuscript.render("[[results:no_such_file.json:x]]", FROZEN)


def _frozen_numbers() -> dict[str, float]:
    """Every float recorded anywhere under paper/frozen/, by dotted path."""
    found: dict[str, float] = {}

    def walk(node, trail: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{trail}.{key}" if trail else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{trail}[{index}]")
        elif isinstance(node, float):
            found[trail] = node

    for path in sorted(FROZEN.glob("*.json")):
        walk(json.loads(path.read_text(encoding="utf-8")), path.name)
    return found


def test_no_frozen_metric_is_typed_into_the_manuscript():
    """The property that stops hand-typing, rather than merely allowing markers.

    Only values with a fractional part are checked, at two and three decimals. An
    integer such as the case count appears in prose ("20 cases") in ways a literal
    search cannot distinguish from a typed metric, and the marker for it is cheap enough
    that the weaker check is not worth the false positives.
    """
    source = build_manuscript.MARKER.sub("", SOURCE.read_text(encoding="utf-8"))
    source = re.sub(r"<!--.*?-->", "", source, flags=re.S)
    # References carry volume numbers, years and DOIs that collide with rounded metrics.
    source = source.split("## References")[0]

    typed: list[str] = []
    for trail, value in _frozen_numbers().items():
        if value == int(value):
            continue
        for digits in (2, 3):
            literal = f"{abs(value):.{digits}f}"
            if re.search(rf"(?<![\d.]){re.escape(literal)}(?![\d])", source):
                typed.append(f"{literal} is {trail}")
    assert not typed, "typed instead of resolved from paper/frozen/:\n  " + "\n  ".join(
        sorted(set(typed))
    )
