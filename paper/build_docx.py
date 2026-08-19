"""Render the built manuscript to the ``.docx`` Editorial Manager takes.

    python paper/build_docx.py              # -> paper/build/manuscript.docx
    python paper/build_docx.py --output some/where.docx

A derivative, never a source. It re-renders ``paper/manuscript.md`` and refuses to
continue if the committed build is out of date, so a stale number cannot be baked into a
circulated document. Figures are placed after the paragraph that first names them and
carry the caption from ``paper/README.md``, written once.

Two Word defaults are wrong for a results table and are patched after pandoc writes the
file: a row may break across a page, and a table may break between rows even when the
whole of it would fit overleaf. Neither is reachable from pandoc or from a reference
document, because one is a row property and the other has to be set paragraph by
paragraph. A companion paper shipped both.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER))

import build_manuscript  # noqa: E402
import collect_figures  # noqa: E402

DEFAULT_OUTPUT = PAPER / "build" / "manuscript.docx"

_FIGURE_MENTION = re.compile(r"\bFigure (\d+)\b")


def _figure_block(number: int, caption: str) -> str:
    """One figure as a captioned image. The alt text is left empty deliberately.

    Pandoc renders alt text as a caption of its own, so text placed there *and* in a
    numbered paragraph prints twice, in two wordings.
    """
    path = collect_figures.FIGURES / collect_figures.FIGURE_SOURCES[number][0]
    return f"![]({path.as_posix()})\n\n**Figure {number}.** {caption}\n"


def assemble(built: str) -> str:
    """The built prose, each figure placed after the paragraph that first names it."""
    captions = collect_figures.captions()
    placed: set[int] = set()
    out: list[str] = []
    for block in built.split("\n\n"):
        out.append(block)
        wanted = sorted(
            {
                int(number)
                for number in _FIGURE_MENTION.findall(block)
                if int(number) in collect_figures.FIGURE_SOURCES
            }
            - placed
        )
        for number in wanted:
            out.append(_figure_block(number, captions[number]))
            placed.add(number)
    missing = sorted(set(collect_figures.FIGURE_SOURCES) - placed)
    if missing:
        raise SystemExit(
            f"figures never named in the prose, so never placed: {missing}"
        )
    return "\n\n".join(out)


def _set_property(fragment: str, container: str, element: str, prop: str) -> str:
    """Add ``prop`` to every ``container``'s properties, whatever form they take.

    Three forms occur and all three must be handled: present with children, present and
    self-closing, absent. Missing the middle one appends a *second* properties element,
    which the schema forbids and Word silently ignores, so the setting reads as
    applied and does nothing.
    """
    opened = f"<{element}>"
    fragment = re.sub(re.escape(opened), f"{opened}{prop}", fragment)
    fragment = re.sub(
        rf"<{re.escape(element)}\s*/>", f"{opened}{prop}</{element}>", fragment
    )
    return re.sub(
        rf"(<{re.escape(container)}\b[^>]*>)(?<!/>)(?!<{re.escape(element)})",
        rf"\1{opened}{prop}</{element}>",
        fragment,
    )


def keep_tables_whole(path: Path) -> Path:
    """``cantSplit`` on every row; ``keepNext`` on the caption and all but the last."""
    entry = "word/document.xml"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        contents = {name: archive.read(name) for name in names}

    document = contents[entry].decode("utf-8")
    document = _set_property(document, "w:tr", "w:trPr", "<w:cantSplit/>")

    def bind(match: re.Match) -> str:
        caption, table = match.group(1), match.group(2)
        rows = re.findall(r"<w:tr\b.*?</w:tr>", table, re.S)
        caption = _set_property(caption, "w:p", "w:pPr", "<w:keepNext/>")
        # The last row is left free: binding it drags the paragraph after the table onto
        # the same page, which trades one defect for another.
        for row in rows[:-1]:
            table = table.replace(
                row, _set_property(row, "w:p", "w:pPr", "<w:keepNext/>"), 1
            )
        return caption + table

    document = re.sub(
        r"(<w:p\b(?:(?!<w:p\b).)*?</w:p>\s*)(<w:tbl>.*?</w:tbl>)",
        bind,
        document,
        flags=re.S,
    )
    contents[entry] = document.encode("utf-8")

    temporary = path.with_suffix(".docx.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.writestr(name, contents[name])
    shutil.move(str(temporary), str(path))
    return path


def build(output: Path = DEFAULT_OUTPUT) -> int:
    import pypandoc  # noqa: PLC0415

    source = build_manuscript.DEFAULT_SOURCE.read_text(encoding="utf-8")
    rendered = build_manuscript.render(source)
    committed = build_manuscript.DEFAULT_OUTPUT
    if not committed.exists() or committed.read_text(encoding="utf-8") != rendered:
        print("paper/build/manuscript.md is out of date; run paper/build_manuscript.py")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    pypandoc.convert_text(
        assemble(rendered),
        to="docx",
        format="markdown+implicit_figures",
        outputfile=str(output),
        extra_args=["--resource-path", str(PAPER)],
    )
    keep_tables_whole(output)
    print(f"wrote {output} ({output.stat().st_size // 1024} KB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return build(parser.parse_args(argv).output)


if __name__ == "__main__":
    raise SystemExit(main())
