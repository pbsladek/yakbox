from __future__ import annotations

from pathlib import Path

import pytest

from yakbox.audiobook.manifest import load_manifest
from yakbox.errors import ValidationError


@pytest.fixture
def named_workspace(book_workspace: Path) -> Path:
    with (book_workspace / "yakbox.toml").open("a") as stream:
        stream.write('\n[characters.narrator]\nprofile = "default"\n')
    return book_workspace


@pytest.mark.parametrize(
    ("method", "misspelled", "expected"),
    [
        ("profile", "defalt", "default"),
        ("target", "defalt", "default"),
        ("voice", "narator", "narrator"),
        ("character", "narator", "narrator"),
    ],
)
def test_manifest_lookup_suggests_close_names(
    named_workspace: Path, method: str, misspelled: str, expected: str
) -> None:
    manifest = load_manifest(named_workspace / "yakbox.toml")
    lookup = getattr(manifest, method)
    with pytest.raises(ValidationError, match=f"Did you mean '{expected}'"):
        lookup(misspelled)
    assert lookup(expected).name == expected


@pytest.mark.parametrize("method", ["profile", "target", "voice", "character"])
def test_manifest_lookup_omits_unrelated_suggestions(
    book_workspace: Path, method: str
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    with pytest.raises(ValidationError) as caught:
        getattr(manifest, method)("xyz123")
    assert "Unknown" in str(caught.value)
    assert "Did you mean" not in str(caught.value)
