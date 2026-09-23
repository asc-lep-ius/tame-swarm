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
wrong width stops the table rendering at all. Every one was a place the model
and a real renderer disagreed. So the structural half is rendered rather than
modelled, with `markdown-it-py` in its GitHub-flavoured mode: every row the
source writes is a row the renderer emits, and each row's guidance survives into
**that row's own cell**.

Two things a renderer cannot check are still read off the source, and the reason
is the same for both: GFM's response to them is silence rather than a
difference. **A surplus cell is dropped**, so a row carrying one delimiter too
many renders as a well-formed row with a cell missing and nothing to compare it
against -- the width check is the only evidence, and it lives in the source.
**An unterminated comment inside a cell is escaped rather than swallowed**,
because both renderers split a row into cells before parsing its inline
content, so the row count is identical on a file with the defect and a file
without it. `test_no_comment_is_left_open` is therefore load-bearing and not a
locator: it is the only thing catching one of the two defects this file exists
for, which is worth knowing before anyone tidies it away as redundant.
"""

import re
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

TEMPLATES = sorted((Path(__file__).parent.parent / ".gitlab" / "issue_templates").glob("*.md"))
# GitLab renders GFM. `html` so an HTML comment is passed through as one rather
# than escaped into text, which is what the guidance check reads; linkify off
# because it is an optional extra this project does not carry, and without the
# disable the preset raises rather than differing. A URL-shaped defect in a cell
# is outside what this file sees, and that is the price.
RENDERER = MarkdownIt("gfm-like", {"html": True}).disable("linkify")
# `|---|:--|` and friends: the one row of a table that emits no `<tr>`. Matched
# against the stripped line, because GFM takes a delimiter row with trailing
# space or up to three of indent and this would have called one a written row.
DELIMITER = re.compile(r"^\|[\s\-:|]+\|$")
# The opener is captured, not just detected: CommonMark closes a fence only with
# the same character, so a ``` inside a ~~~ block closes nothing. A bare toggle
# desynchronised there and hid every table below it from all four checks.
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
# The last of a row's guidance, which is what a truncated cell loses first.
TAIL = 40


def unescaped_pipes(line: str) -> int:
    """Delimiters GFM will read, counting `\\|` as a literal and `\\\\` as a backslash.

    Not a lookbehind: ``(?<!\\\\)\\|`` reads the pipe in ``\\\\|`` as escaped, when
    that is an escaped *backslash* followed by a real delimiter.
    """
    return line.replace("\\\\", "").replace("\\|", "").count("|")


def source_rows(path: Path) -> list[tuple[int, str]]:
    """Every line the source writes as a table row, numbered, fences excluded.

    A ``|`` line inside a fenced block is prose about a table rather than a
    table, and counting it reports a defect with a message describing the
    opposite of what happened.
    """
    rows: list[tuple[int, str]] = []
    opener: str | None = None
    for number, line in enumerate(path.read_text().split("\n"), 1):
        fence = FENCE.match(line)
        if fence and opener is None:
            opener = fence.group(1)[0]
        elif fence and fence.group(1)[0] == opener:
            opener = None
        elif opener is None and line.strip().startswith("|"):
            rows.append((number, line))
    # Without this the file simply ends inside the fence, every table below it
    # vanishes from the check, and the renderer agrees at zero -- so nothing
    # raises while the issue renders its whole table as a code block.
    assert opener is None, f"{path.name} leaves a `{opener}` code fence open to the end of the file"
    return rows


def tables(path: Path) -> list[list[tuple[int, str]]]:
    """Runs of consecutive table rows: each one's first row is its header."""
    grouped: list[list[tuple[int, str]]] = []
    previous = -2
    for number, line in source_rows(path):
        if number == previous + 1:
            grouped[-1].append((number, line))
        else:
            grouped.append([(number, line)])
        previous = number
    return grouped


def emitted_tables(path: Path) -> list[list[str]]:
    """Each rendered table's `<tr>`s, in document order."""
    html = RENDERER.render(path.read_text())
    return [
        re.findall(r"<tr>.*?</tr>", table, re.S)
        for table in re.findall(r"<table>.*?</table>", html, re.S)
    ]


def written_tables(path: Path) -> list[list[tuple[int, str]]]:
    """Each source table's rows that should become a `<tr>`: every one but the delimiters."""
    return [
        [(number, line) for number, line in rows if not DELIMITER.match(line.strip())]
        for rows in tables(path)
    ]


def paired(path: Path) -> list[tuple[int, str, str]]:
    """Every source row beside the `<tr>` it produced, per table rather than per file.

    Per table because a file-level count is a sum and two defects cancel it: a
    row indented into a code block and a pair of adjacent tables elsewhere leave
    the totals equal while every row after the first defect is paired with the
    wrong `<tr>` -- including, in the construction that found this, a source row
    against a *rendered delimiter row*. Grouping first localises the mismatch to
    the table it happened in, which is also the only way to name it.
    """
    written, emitted = written_tables(path), emitted_tables(path)
    assert len(written) == len(emitted), (
        f"{path.name} writes {len(written)} runs of table rows and the renderer makes "
        f"{len(emitted)} of them into tables. A run wrote no table at all -- a missing "
        "delimiter row, or rows the renderer read into a neighbouring table"
    )
    rows: list[tuple[int, str, str]] = []
    for source, rendered in zip(written, emitted, strict=True):
        assert len(source) == len(rendered), (
            f"{path.name}: the table starting at line {source[0][0] if source else '?'} writes "
            f"{len(source)} rows the renderer should emit and it emits {len(rendered)}. A row "
            "is being swallowed by an unclosed comment, indented into a code block, dropped "
            "with its table, or read into a table it does not belong to"
        )
        rows += [
            (number, line, html) for (number, line), html in zip(source, rendered, strict=True)
        ]
    return rows


def normalised(text: str) -> str:
    """Collapsed whitespace, and `\\|` read as the pipe the renderer resolves it to."""
    return re.sub(r"\s+", " ", text.replace("\\|", "|")).strip()


def guidance(line: str) -> str | None:
    """What a row's one comment says, or None when it carries none, or more than one."""
    if line.count("<!--") != 1 or line.count("-->") != 1:
        return None
    return normalised(line[line.index("<!--") + 4 : line.rindex("-->")])


def test_there_are_templates_to_check():
    """A glob that matches nothing passes every parametrised case below."""
    assert TEMPLATES
    assert any(path.name == "measurement.md" for path in TEMPLATES)
    assert all(tables(path) for path in TEMPLATES)


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_every_row_the_source_writes_is_a_row_the_renderer_emits(path: Path):
    """A comment left open above a table, a row indented into a code block, a row
    that lost its leading pipe, a delimiter row of the wrong width -- each
    changes this count, and none of them is visible in a diff.

    `paired` is what asserts it, table by table, and the guidance check below
    reads the same pairing -- so a mismatch is a failure there too rather than a
    bare `zip() argument 2 is shorter` with no file and no line in it.
    """
    assert paired(path), f"{path.name} writes no table rows"


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_the_end_of_every_rows_guidance_survives_into_that_rows_own_cell(path: Path):
    """An unescaped `|` truncates the cell there: the row renders, its text does not.

    Against the row's own `<tr>` rather than the whole document, or a row whose
    guidance a sibling repeats asserts nothing about itself -- which was true of
    every `Dependencies` row in the repository, their `#N or n/a` supplied by
    the row below.
    """
    # `paired` raises before this runs when the rows cannot be lined up, so a
    # row is never checked against a `<tr>` that is not its own.
    for number, line, rendered in paired(path):
        said = guidance(line)
        if said is None:
            continue
        assert said[-TAIL:] in normalised(rendered), (
            f"{path.name}:{number} loses the end of its guidance in the row the renderer "
            f"emits, which is what an unescaped `|` does -- write `\\|`. "
            f"Missing: {said[-TAIL:]!r}"
        )


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_no_row_carries_more_delimiters_than_its_header_declares(path: Path):
    """Read off the source, because GFM's answer to a surplus cell is to drop it.

    The row still renders, with one cell fewer than it was written with and
    nothing in the output to compare against -- so a renderer cannot see this
    and the source is the only evidence there is.
    """
    for rows in tables(path):
        (_, header), width = rows[0], unescaped_pipes(rows[0][1])
        for number, line in rows:
            found = unescaped_pipes(line)
            assert found == width, (
                f"{path.name}:{number} has {found} unescaped pipes where its header declares "
                f"{width}, so a cell is dropped or missing. Write `\\|` for a pipe inside a "
                f"cell:\n  header  {header[:100]}\n  row     {line[:100]}"
            )


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_no_comment_is_left_open(path: Path):
    """The only thing catching an unterminated comment inside a table row.

    Both renderers split a row into cells before parsing its inline content, so
    an unterminated `<!--` is escaped within its own cell rather than swallowing
    what follows: the row count is identical with the defect and without it.
    The count catches an unterminated comment in *prose*; this catches the one
    that broke this file twice.
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


MEASUREMENT = next(path for path in TEMPLATES if path.name == "measurement.md")


def text_of(html: str) -> str:
    """A rendered fragment with its tags stripped, whitespace collapsed."""
    return normalised(re.sub(r"<[^>]+>", "", html))


def field_names(path: Path) -> list[list[str]]:
    """Each rendered table's first-cell texts, in row order: what a reader sees as the field."""
    names: list[list[str]] = []
    for rows in emitted_tables(path):
        cells = [re.search(r"<td>(.*?)</td>", row, re.S) for row in rows]
        names.append([text_of(cell.group(1)) for cell in cells if cell])
    return names


def test_the_calibration_row_stands_beside_power():
    """#68: a Power row without a Calibration row prices a readout nobody has checked.

    Read off the rendered table rather than the source, so a row that exists in
    the file and is swallowed before it renders fails here as well as above.
    Adjacent, not merely present: the rule is that the two are read together.
    """
    with_power = [fields for fields in field_names(MEASUREMENT) if "Power" in fields]
    assert with_power, f"{MEASUREMENT.name} renders no table with a Power row"
    fields = with_power[0]
    after_power = fields[fields.index("Power") + 1]
    assert after_power == "Calibration", (
        f"{MEASUREMENT.name}: the row after Power renders as {after_power!r}, not Calibration"
    )


def test_what_it_records_carries_the_placement():
    """#68: every stakes readout records two numbers, and the template asks for them.

    Matched in the rendered list under the heading that owns it, so moving the
    line into another section is a failure and not a relocation.
    """
    html = RENDERER.render(MEASUREMENT.read_text())
    section = re.search(r"<h2>What it records</h2>(.*?)<h2>", html, re.S)
    assert section, f"{MEASUREMENT.name} renders no 'What it records' section"
    items = [text_of(li) for li in re.findall(r"<li>.*?</li>", section.group(1), re.S)]
    placement = [item for item in items if item.startswith("[ ] The placement under the ladder")]
    assert len(placement) == 1, f"{MEASUREMENT.name}: one placement item expected, got {placement}"
    assert "preregistration section 9" in placement[0]
    assert "two numbers, never one" in placement[0]
