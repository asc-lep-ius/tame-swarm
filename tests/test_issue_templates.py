"""The issue templates are markdown a renderer reads, so a renderer is what checks them (#56).

`.gitlab/issue_templates/measurement.md` was broken by hand in two of the three
commits that touched it, both times by a defect no reader of the diff would see:
a dropped `-->` turned two rows into one comment that swallowed the markup
between them, and an unescaped `|t|` truncated a row's guidance at the pipe.
Both render as *missing text*, never as an error.

The first version of this file asserted properties of each row by hand, and a
reviewer broke it four ways in ten minutes -- a comment dropped one line *above*
a table swallows the whole table, a row indented four spaces becomes a code
block, a row missing its leading pipe still renders, and a delimiter row of the
wrong width stops the table rendering at all. Every one of them was a place the
hand-written model and a real renderer disagreed. So the model is gone: these
render the file with `markdown-it-py` in its GitHub-flavoured mode and assert
what came out, which is the only way to stop discovering the next disagreement
the way the last four were discovered.

Two invariants do the work. **Every row the source writes is a row the renderer
emits** -- a table's delimiter row is the one that is not, so `T` tables account
for exactly `T` of the difference -- which catches anything swallowed, indented
into a code block, or lost because its table stopped parsing. And **the tail of
every row's guidance survives into the output**, which catches the truncation an
unescaped pipe causes, where the row still renders and its text does not.
"""

import re
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

TEMPLATES = sorted((Path(__file__).parent.parent / ".gitlab" / "issue_templates").glob("*.md"))
# GitLab renders GFM. `html` so an HTML comment is passed through as one rather
# than escaped into text, and linkify off because it is an optional dependency
# this project does not carry and it changes nothing a table row asserts.
RENDERER = MarkdownIt("gfm-like", {"html": True}).disable("linkify")
# The last of a row's guidance, which is what a truncated cell loses first.
TAIL = 40


def render(path: Path) -> str:
    return RENDERER.render(path.read_text())


def source_rows(path: Path) -> list[tuple[int, str]]:
    """Every line the source writes as a table row, numbered."""
    return [
        (number, line)
        for number, line in enumerate(path.read_text().split("\n"), 1)
        if line.strip().startswith("|")
    ]


def normalised(text: str) -> str:
    """Collapsed whitespace, and `\\|` read as the pipe the renderer resolves it to."""
    return re.sub(r"\s+", " ", text.replace("\\|", "|")).strip()


def guidance(line: str) -> str | None:
    """What a row's comment says, or None when the row carries no comment."""
    if "<!--" not in line or "-->" not in line:
        return None
    return normalised(line[line.index("<!--") + 4 : line.rindex("-->")])


def test_there_are_templates_to_check():
    """A glob that matches nothing passes every parametrised case below."""
    assert TEMPLATES
    assert any(path.name == "measurement.md" for path in TEMPLATES)
    assert all(source_rows(path) for path in TEMPLATES)


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_every_row_the_source_writes_is_a_row_the_renderer_emits(path: Path):
    """The delimiter row is the only one that is not, so the arithmetic is exact.

    A comment left open anywhere above a table, a row indented into a code block,
    a row that lost its leading pipe, a delimiter row of the wrong width -- each
    changes this count, and none of them is visible in a diff.
    """
    html = render(path)
    tables = len(re.findall(r"<table>", html))
    emitted = len(re.findall(r"<tr>", html))
    written = len(source_rows(path))

    assert tables > 0, f"{path.name} renders no table at all"
    assert emitted == written - tables, (
        f"{path.name} writes {written} table rows across {tables} tables, so the renderer "
        f"should emit {written - tables} and emits {emitted}. A row is being swallowed by an "
        "unclosed comment, indented into a code block, or dropped with its table"
    )


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_the_end_of_every_rows_guidance_survives_into_the_output(path: Path):
    """An unescaped `|` truncates the cell there: the row renders, its text does not."""
    html = normalised(render(path))
    for number, line in source_rows(path):
        said = guidance(line)
        if said is None:
            continue
        assert said[-TAIL:] in html, (
            f"{path.name}:{number} loses the end of its guidance in the rendered issue, "
            f"which is what an unescaped `|` does -- write `\\|`. Missing: {said[-TAIL:]!r}"
        )


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_no_comment_is_left_open(path: Path):
    """Read before the renderer, because this is the defect that hides the others.

    An open comment swallows whatever follows it, so the count above reports it
    as missing rows and says nothing about where it started. This names the line.
    """
    open_at: tuple[int, str] | None = None
    for number, line in enumerate(path.read_text().split("\n"), 1):
        for token in re.findall(r"<!--|-->", line):
            if token == "<!--":
                assert open_at is None, (
                    f"{path.name}:{number} opens a comment inside one opened at line "
                    f"{open_at[0] if open_at else '?'}"
                )
                open_at = (number, line)
            else:
                assert open_at is not None, (
                    f"{path.name}:{number} closes a comment that was never opened"
                )
                open_at = None
    assert open_at is None, (
        f"{path.name}:{open_at[0]} leaves a comment open to the end of the file, so everything "
        f"below it is swallowed:\n{open_at[1][:120]}"
    )
