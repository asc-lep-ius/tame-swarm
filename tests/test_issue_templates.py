"""The issue templates are markdown a renderer reads, so the markdown is a test (#56).

`.gitlab/issue_templates/measurement.md` was broken by hand in two of the three
commits that touched it, both times by a defect no reader of the diff would see:
a dropped `-->` turned two rows into one comment that swallowed the markup
between them, and an unescaped `|t|` truncated a row's guidance at the pipe.
Both render as *missing text*, not as an error, so the only way to notice is to
render the file -- or to assert the two properties every row of these tables has.
"""

import re
from pathlib import Path

import pytest

TEMPLATES = sorted((Path(__file__).parent.parent / ".gitlab" / "issue_templates").glob("*.md"))
# A table row here is `| Field | <!-- guidance --> |`: three delimiters, and any
# pipe inside the guidance escaped so the renderer does not read it as a fourth.
DELIMITERS = 3
UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def table_rows(path: Path) -> list[tuple[int, str]]:
    return [
        (number, line)
        for number, line in enumerate(path.read_text().split("\n"), 1)
        if line.startswith("|")
    ]


def test_there_are_templates_to_check():
    """A glob that matches nothing passes every parametrised case below."""
    assert TEMPLATES
    assert any(path.name == "measurement.md" for path in TEMPLATES)


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_every_comment_in_a_table_row_is_closed_on_its_own_row(path: Path):
    """An unterminated comment runs on to the next row's `-->` and eats a whole row."""
    for number, line in table_rows(path):
        assert line.count("<!--") == line.count("-->"), (
            f"{path.name}:{number} leaves an HTML comment open, so it swallows the rows "
            f"between here and the next `-->`:\n{line[:120]}"
        )


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_a_two_column_row_carries_exactly_three_unescaped_pipes(path: Path):
    """GFM drops every cell past the header's width, silently and without an error."""
    for number, line in table_rows(path):
        if not line.strip("| -:"):  # the `|---|---|` separator
            continue
        found = len(UNESCAPED_PIPE.findall(line))
        assert found == DELIMITERS, (
            f"{path.name}:{number} has {found} unescaped pipes where a two-column row has "
            f"{DELIMITERS}; the cell is truncated at the extra one. Write `\\|`:\n{line[:120]}"
        )
