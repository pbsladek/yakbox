from __future__ import annotations

import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from yakbox.cli import main


@pytest.mark.parametrize(
    "arguments",
    [
        ("--help",),
        ("init", "--help"),
        ("plan", "--help"),
        ("audition", "--help"),
        ("build", "--help"),
        ("release", "check", "--help"),
        ("artifacts", "clean", "--help"),
        ("artifacts", "trash", "restore", "--help"),
        ("cloud", "batch", "--help"),
        ("cloud", "voices", "recordings", "create", "--help"),
        ("doctor", "--help"),
    ],
)
def test_documented_command_surfaces_have_configuration_free_help(
    arguments: tuple[str, ...],
) -> None:
    result = CliRunner().invoke(main, list(arguments))
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_documentation_index_links_resolve() -> None:
    root = Path(__file__).parents[2]
    index = root / "docs" / "README.md"
    links = re.findall(r"\[[^]]+]\(([^)]+\.md)\)", index.read_text(encoding="utf-8"))

    assert links
    assert all((index.parent / link).is_file() for link in links)
