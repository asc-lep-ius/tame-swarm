"""The issue templates are markdown a renderer reads, so the markdown is a test (#56).

`.gitlab/issue_templates/measurement.md` was broken by hand in two of the three
commits that touched it, both times by a defect no reader of the diff would see:
a dropped `-->` turned two rows into one comment that swallowed the markup
between them, and an unescaped `|t|` truncated a row's guidance at the pipe.
Both render as *missing text*, not as an error, so the only way to notice is to
render the file -- or to assert the properties every row of these tables has.

Run against the file as `b3ecfb3` left it, these flag line 45 and line 46, which
are exactly the two defects.
"""

from pathlib import Path

import pytest

TEMPLATES = sorted((Path(__file__).parent.parent / ".gitlab" / "issue_templates").glob("*.md"))
SEPARATOR = set("| -:")


def unescaped_pipes(line: str) -> int:
    """Delimiters GFM will read, counting `\\|` as a literal and `\\\\` as a backslash.

    Not a lookbehind: ``(?<!\\\\)\\|`` reads the pipe in ``\\\\|`` as escaped, when
    that is an escaped *backslash* followed by a real delimiter. A row the
    renderer truncates would have counted correct and passed, which is the exact
    defect class this file exists to catch.
    """
    return line.replace("\\\\", "").replace("\\|", "").count("|")


def tables(path: Path) -> list[tuple[int, list[tuple[int, str]]]]:
    """Each table in the file as (delimiters its header declares, its rows).

    The width comes off the header rather than a constant, so a three-column
    table added later is checked against its own shape instead of failing with
    advice to escape a pipe that is a real delimiter.
    """
    found: list[tuple[int, list[tuple[int, str]]]] = []
    rows: list[tuple[int, str]] = []
    for number, raw in enumerate(path.read_text().split("\n"), 1):
        line = raw.strip()
        if not line.startswith("|"):
            if rows:
                found.append((unescaped_pipes(rows[0][1]), rows))
                rows = []
            continue
        rows.append((number, line))
    if rows:
        found.append((unescaped_pipes(rows[0][1]), rows))
    return found


def test_there_are_templates_to_check():
    """A glob that matches nothing passes every parametrised case below."""
    assert TEMPLATES
    assert any(path.name == "measurement.md" for path in TEMPLATES)
    assert all(tables(path) for path in TEMPLATES)


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_every_comment_in_a_table_row_opens_and_closes_on_its_own_row(path: Path):
    """An unterminated comment runs on to the next row's `-->` and eats a whole row."""
    for _, rows in tables(path):
        for number, line in rows:
            assert line.count("<!--") == line.count("-->"), (
                f"{path.name}:{number} leaves an HTML comment open, so it swallows the rows "
                f"between here and the next `-->`:\n{line[:120]}"
            )
            # Counting alone would pass `--> stray <!--`, which is balanced and
            # still ends the row inside a comment.
            assert "-->" not in line.split("<!--")[0], (
                f"{path.name}:{number} closes a comment it never opened, so the row ends "
                f"inside one:\n{line[:120]}"
            )


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_every_row_carries_the_delimiters_its_header_declares(path: Path):
    """GFM drops every cell past the header's width, silently and without an error."""
    for width, rows in tables(path):
        for number, line in rows:
            if not set(line) - SEPARATOR:
                continue
            found = unescaped_pipes(line)
            assert found == width, (
                f"{path.name}:{number} has {found} unescaped pipes where this table's header "
                f"declares {width}; a cell is truncated at the extra one, or a cell is "
                f"missing. Write `\\|` for a pipe inside a cell:\n{line[:120]}"
            )
