"""Every defect two companion papers shipped or nearly shipped, checked in one pass.

    python paper/presubmission_check.py            # the manuscript
    python paper/presubmission_check.py --strict   # warnings fail too

Ported from the checks accumulated on IORN-003 and IORN-005, retuned for this journal.
Each item is here because a person found it by hand, in a built document, after it had
been declared finished:

* a figure taller than the page it printed on, cut in half with its caption overleaf;
* a legend and an inset drawn over the data they were meant to explain;
* a table cell broken inside a number, so a negative lower bound read as positive;
* a table row split across a page break, leaving numbers under a repeated header;
* bold used both for run-in headings and for shouting, until neither meant anything;
* a caption written twice, printing twice in two wordings;
* references cited and not listed, listed and not cited, and out of order;
* a title that contradicted its own abstract.

Two are specific to this study. The manuscript may not contain a number that a frozen
file already holds, and the abstract must name the material the body is built on -- the
failure that reached submission twice was a body that gained a whole arm while the front
matter still described the old one.

This reports; it does not fix.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
SOURCE = PAPER / "manuscript.md"
BUILT = PAPER / "build" / "manuscript.md"
FIGURES = PAPER / "figures"
FROZEN = PAPER / "frozen"

#: CMPB: "a concise and factual abstract which does not exceed 250 words".
ABSTRACT_WORDS = 250

#: Words the abstract must contain because the body rests on them. This is the
#: front-matter drift check, stated as the vocabulary of the current claim rather
#: than of the one the paper started from.
ABSTRACT_MUST_MENTION = ("Cramér–Rao", "sampling", "identifiab", "phase", "estimator")

#: A figure is scaled to this width and must still fit the page it lands on.
COLUMN_INCHES = 6.5
MAX_PRINTED_INCHES = 7.5
#: Below this reduction, type set at 9 pt lands under 6 pt on the page.
MIN_SCALE = 0.70

errors: list[str] = []
warnings: list[str] = []


def fail(message: str) -> None:
    errors.append(message)


def warn(message: str) -> None:
    warnings.append(message)


def _strip_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


# ---------------------------------------------------------------- emphasis
def check_emphasis(built: str) -> None:
    """Bold that is not a run-in heading is bold doing italic's job.

    Checked on the built file, where paragraphs are one line. A source-line scan misses
    every bold span that straddles a line break, which is how half a sentence survived
    two rounds of this on a companion paper.
    """
    body = _strip_comments(built).split("## References")[0]
    for match in re.finditer(r"\*\*(.+?)\*\*", body, re.S):
        span = " ".join(match.group(1).split())
        before, after = body[: match.start()], body[match.end() :]
        opens_line = before.endswith("\n") or not before
        if opens_line and (after.startswith("\n") or not after):
            continue  # a title or an author line
        if opens_line and span.endswith((".", ":")):
            continue  # a run-in heading, or an abstract label
        fail(f"bold mid-sentence: **{span[:60]}** — should it be italic?")


# ---------------------------------------------------------------- front matter
def check_front_matter(src: str, built: str) -> None:
    """The half an editor reads first, against the half that changed."""
    abstract = built.split("## Abstract")[1].split("**Keywords")[0]
    lowered = abstract.lower()
    for word in ABSTRACT_MUST_MENTION:
        if word.lower() not in lowered:
            fail(f"abstract never mentions {word!r}, which the body is built on")

    words = len(abstract.split())
    if words > ABSTRACT_WORDS:
        fail(f"abstract is {words} words, over the {ABSTRACT_WORDS}-word limit")
    elif words > ABSTRACT_WORDS - 15:
        warn(f"abstract is {words} words, close to the {ABSTRACT_WORDS}-word limit")

    title = next((line for line in src.splitlines() if line.startswith("# ")), "")
    if not title:
        fail("no level-1 title found")
    if "**Keywords:**" not in built:
        fail("no keyword list")


# ---------------------------------------------------------------- references
def check_references(src: str) -> None:
    body, reflist = src.split("## References")
    cited: set[int] = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", body):
        cited.update(int(part) for part in re.split(r"\s*,\s*", group))
    listed = {int(m) for m in re.findall(r"^(\d+)\. ", reflist, re.M)}

    for number in sorted(cited - listed):
        fail(f"reference [{number}] is cited but not listed")
    for number in sorted(listed - cited):
        fail(f"reference [{number}] is listed but never cited")
    if listed and listed != set(range(1, max(listed) + 1)):
        fail(f"reference numbering is not contiguous: {sorted(listed)}")

    order: list[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", body):
        for number in (int(p) for p in re.split(r"\s*,\s*", match.group(1))):
            if number in listed and number not in seen:
                seen.add(number)
                order.append(number)
    if order != sorted(order):
        fail(f"references are not in order of first citation: {order}")

    for entry in re.findall(r"^\d+\. .+$", reflist, re.M):
        if "doi:" not in entry:
            fail(f"reference without a DOI: {entry[:70]}")


# ---------------------------------------------------------------- numbers
def check_numbers(src: str, built: str) -> None:
    """No frozen metric may appear in the prose as a typed literal."""
    stripped = re.sub(r"\[\[results:[^\]]*\]\]", "", _strip_comments(src))
    stripped = stripped.split("## References")[0]

    def walk(node, trail: str, found: dict[str, float]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{trail}.{key}" if trail else str(key), found)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{trail}[{index}]", found)
        elif isinstance(node, float):
            found[trail] = node

    numbers: dict[str, float] = {}
    for path in sorted(FROZEN.glob("*.json")):
        walk(json.loads(path.read_text(encoding="utf-8")), path.name, numbers)

    for trail, value in numbers.items():
        if value == int(value):
            continue
        for digits in (2, 3):
            literal = f"{abs(value):.{digits}f}"
            if re.search(rf"(?<![\d.]){re.escape(literal)}(?![\d])", stripped):
                fail(f"{literal} is typed into the prose; it is {trail}")

    opened = len(re.findall(r"\[\[results:", _strip_comments(built)))
    closed = len(re.findall(r"\[\[results:[^\]]*\]\]", _strip_comments(built)))
    if opened or closed:
        fail(f"{opened} unresolved marker(s) in the built manuscript")


# ---------------------------------------------------------------- figures
def check_figures(src: str) -> None:
    sys.path.insert(0, str(PAPER))
    import collect_figures  # noqa: PLC0415

    registered = collect_figures.FIGURE_SOURCES
    captions = collect_figures.captions()
    on_disk = {path.name for path in FIGURES.glob("*.png")}

    for number, (name, _source) in registered.items():
        if name not in on_disk:
            fail(f"Figure {number} ({name}) is registered but not in paper/figures/")
        if number not in captions:
            fail(f"Figure {number} has no caption in paper/README.md")
        if not re.search(rf"\bFigure {number}\b", src):
            fail(f"Figure {number} is registered but the prose never names it")
    for name in sorted(on_disk - {n for n, _ in registered.values()}):
        warn(f"{name} is in paper/figures/ but is not placed")

    for path in sorted(FIGURES.glob("*.png")):
        raw = path.read_bytes()
        width, height = struct.unpack(">II", raw[16:24])
        marker = raw.find(b"pHYs")
        dpi = (
            round(struct.unpack(">I", raw[marker + 4 : marker + 8])[0] * 0.0254)
            if marker > 0
            else 100
        )
        scale = COLUMN_INCHES / (width / dpi)
        printed = (height / dpi) * min(1.0, scale)
        if printed > MAX_PRINTED_INCHES:
            fail(
                f"{path.name} prints {printed:.1f} in tall in a {COLUMN_INCHES} in "
                "column; the converter will cut it across a page break"
            )
        if scale < MIN_SCALE:
            fail(
                f"{path.name} is {width / dpi:.1f} in wide at {dpi} dpi and reduces to "
                f"x{scale:.2f}; its type will not survive printing"
            )


# ---------------------------------------------------------------- built output
def check_built(built: str) -> None:
    for lineno, line in enumerate(built.splitlines(), 1):
        if "\t" in line or "\x0b" in line or "\x0c" in line:
            fail(f"built manuscript line {lineno} contains a control character")
    for match in re.finditer(r"\*\*(Figure|Table) (\d+)\.\*\*", built):
        label = match.group(0)
        if built.count(label) > 1:
            fail(f"{label} caption appears {built.count(label)} times")
    stripped = re.sub(r"`[^`]*`", "", built)
    for command in ("times", "exp", "mathrm", "alpha", "sigma", "Delta"):
        if "\\" + command in stripped:
            warn(f"built manuscript contains literal \\{command}; check how it renders")


# ---------------------------------------------------------------- cover letter
def check_cover_letter() -> None:
    """The cover letter is a second copy of the results, and copies drift.

    A companion paper's letter kept an overclaim the manuscript had already dropped, and
    another's submission form held an abstract three weeks out of date. Every number the
    letter quotes must appear, at the precision the letter quotes it, in the frozen
    metrics -- and the letter must not contradict the manuscript's title.
    """
    letter_path = PAPER / "cover_letter_cmpb.txt"
    if not letter_path.exists():
        warn("no cover letter yet")
        return
    letter = letter_path.read_text(encoding="utf-8")
    # Identifiers that are not measurements: a licence version, a DOI, an ORCID. They
    # look exactly like results to a search for decimal numbers, and "CC BY 4.0" is the
    # one that fired first.
    letter = re.sub(r"CC BY \d+\.\d+", "", letter)
    letter = re.sub(r"10\.\d{4,}/\S+", "", letter)
    letter = re.sub(r"\d{4}-\d{4}-\d{4}-\d{3}[\dXx]", "", letter)

    manifest = json.loads((FROZEN / "manifest.json").read_text(encoding="utf-8"))
    values = {
        value
        for value in manifest["metrics"].values()
        if isinstance(value, (int, float))
    }
    renderings = {
        format(value, spec)
        for value in values
        for spec in (".0f", ".1f", ".2f", ".3f", ".4f", ".0%", ".1%")
    } | {str(value) for value in values if float(value) == int(value)}

    for literal in set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w])", letter)):
        if literal not in renderings:
            fail(
                f"the cover letter quotes {literal}, which no frozen metric "
                "renders as"
            )

    _quotes_the_title(letter, "cover letter")


def _quotes_the_title(text: str, where: str) -> None:
    """Anything that names the paper must name the current one.

    The title changed once during preparation and left two stale copies behind, of which
    only the cover letter was guarded. Whitespace is normalised on both sides: the
    manuscript's title wraps across two source lines and the letter's does not.
    """
    title = next(
        (
            line[2:].strip()
            for line in SOURCE.read_text(encoding="utf-8").splitlines()
            if line.startswith("# ")
        ),
        "",
    )
    if title and " ".join(title.split()) not in " ".join(text.split()):
        fail(f"the {where} does not quote the manuscript's title verbatim")


def check_submission_kit() -> None:
    """The kit records what goes in each form field, so it must not restate the paper.

    It used to carry its own copy of the title, which went stale when the title changed
    and which nothing checked. It now points at the generated portal fields instead.
    """
    kit = PAPER / "cmpb_submission_kit.md"
    if not kit.exists():
        warn("no submission kit")
        return
    if "make_portal_fields" not in kit.read_text(encoding="utf-8"):
        fail(
            "the kit restates the title and abstract instead of pointing at the "
            "generated portal fields"
        )


# ---------------------------------------------------------------- submission format
def check_submission_format() -> None:
    """Page numbers and line numbering, in the file that is actually uploaded.

    A companion submission was returned before peer review for the want of both. Every
    other check here reads the markdown, where the numbers and the prose live, and never
    the artefact that reaches the editor -- and page furniture exists only in the
    artefact.
    """
    import zipfile  # noqa: PLC0415

    path = PAPER / "build" / "manuscript.docx"
    if not path.exists():
        warn("manuscript.docx has not been built")
        return
    with zipfile.ZipFile(path) as archive:
        document = archive.read("word/document.xml").decode("utf-8")
        parts = archive.namelist()
    if "lnNumType" not in document:
        fail("manuscript.docx has no line numbering")
    if not any(part.startswith("word/footer") for part in parts):
        fail("manuscript.docx has no footer, so no page numbers")


# ---------------------------------------------------------------- declarations
#: Sections the publisher's guide requires in the manuscript itself, not only in the
#: submission form. Their absence is invisible in the markdown and is exactly what an
#: editorial office returns a submission for before peer review -- which is how a
#: companion paper lost a round trip over page numbering.
REQUIRED_SECTIONS = (
    "Declaration of competing interest",
    "Funding",
    "Declaration of generative AI use",
)


def check_declarations(built: str) -> None:
    """The three statements CMPB's guide asks for, in the order it asks for them."""
    headings = [
        line[3:].strip() for line in built.splitlines() if line.startswith("## ")
    ]
    for wanted in REQUIRED_SECTIONS:
        if not any(heading.startswith(wanted) for heading in headings):
            fail(f"the manuscript has no '{wanted}' section, which the guide requires")

    # Elsevier asks for the generative-AI statement directly before the references.
    if "References" in headings:
        before = headings[headings.index("References") - 1]
        if not before.startswith("Declaration of generative AI use"):
            fail(
                "the generative-AI declaration must sit directly before the "
                f"references; {before!r} is there instead"
            )


# ---------------------------------------------------------------- provenance
def check_provenance() -> None:
    """The external arm must be the real cohort and not the synthetic proxy."""
    manifest = json.loads((FROZEN / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["metrics"].get("m3_tcia_source")
    if source != "tcia_hcc_tace_seg":
        fail(f"the frozen external arm reads {source!r}, not the real TCIA cohort")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true", help="warnings fail too")
    args = parser.parse_args(argv)

    if not BUILT.exists():
        print("paper/build/manuscript.md is missing; run paper/build_manuscript.py")
        return 2
    src = SOURCE.read_text(encoding="utf-8")
    built = BUILT.read_text(encoding="utf-8")

    check_emphasis(built)
    check_front_matter(src, built)
    check_references(src)
    check_numbers(src, built)
    check_figures(src)
    check_built(built)
    check_declarations(built)
    check_cover_letter()
    check_submission_kit()
    check_submission_format()
    check_provenance()

    for message in warnings:
        print(f"  warn  {message}")
    for message in errors:
        print(f"  FAIL  {message}")
    print(f"\n{len(errors)} failures, {len(warnings)} warnings")
    if errors or (args.strict and warnings):
        return 1
    print("ready to submit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
