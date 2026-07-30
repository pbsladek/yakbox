from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from yakbox.audiobook.manifest import load_manifest
from yakbox.audiobook.planner import plan_audiobook, shard_plan
from yakbox.audiobook.sources import (
    Pause,
    SpeechSegment,
    chunk_text,
    normalize_sources,
)
from yakbox.errors import ValidationError


def test_normalizes_directives_and_pronunciations(book_workspace: Path) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
    )

    assert len(document.chapters) == 1
    assert any(isinstance(item, Pause) for item in document.chapters[0].segments)
    spoken = " ".join(
        item.text
        for item in document.chapters[0].segments
        if isinstance(item, SpeechSegment)
    )
    assert "brief opening" in spoken


def test_exclude_only_and_omitted_markdown(tmp_path: Path) -> None:
    source = tmp_path / "chapter.md"
    source.write_text(
        "# Test\n\n"
        "Speak [the label](https://example.com), but not `code`.\n\n"
        "<!-- yakbox:speech:exclude:start -->hidden"
        "<!-- yakbox:speech:exclude:end -->"
        "<!-- yakbox:speech:only:start -->spoken alternate"
        "<!-- yakbox:speech:only:end -->\n",
        encoding="utf-8",
    )
    document = normalize_sources((source,))
    spoken = " ".join(
        item.text
        for item in document.chapters[0].segments
        if isinstance(item, SpeechSegment)
    )
    assert "the label" in spoken
    assert "https" not in spoken
    assert "code" not in spoken
    assert "hidden" not in spoken
    assert "spoken alternate" in spoken


def test_rejects_unbalanced_directive(tmp_path: Path) -> None:
    source = tmp_path / "bad.md"
    source.write_text(
        "<!-- yakbox:speech:exclude:start -->never closes", encoding="utf-8"
    )
    with pytest.raises(ValidationError, match="unclosed"):
        normalize_sources((source,))


def test_pause_is_standalone_and_source_lines_survive_exclusion(
    tmp_path: Path,
) -> None:
    source = tmp_path / "chapter.md"
    source.write_text(
        "# Test\n\n"
        "<!-- yakbox:speech:exclude:start -->\n"
        "hidden\n"
        "over lines\n"
        "<!-- yakbox:speech:exclude:end -->\n\n"
        "<!-- yakbox:speech:pause ms=250 -->\n\n"
        "Spoken on line nine.\n",
        encoding="utf-8",
    )
    chapter = normalize_sources((source,)).chapters[0]

    pause = next(item for item in chapter.segments if isinstance(item, Pause))
    spoken = next(item for item in chapter.segments if isinstance(item, SpeechSegment))
    assert pause.milliseconds == 250
    assert pause.source.start_line == 8
    assert spoken.source.start_line == 10


def test_rejects_inline_or_malformed_pause_directives(tmp_path: Path) -> None:
    source = tmp_path / "bad.md"
    source.write_text(
        "# Bad\n\nText <!-- yakbox:speech:pause ms=10 --> inline.",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="own line"):
        normalize_sources((source,))

    source.write_text(
        "# Bad\n\n<!-- yakbox:speech:pause milliseconds=10 -->",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="malformed"):
        normalize_sources((source,))


@given(
    text=st.text(
        alphabet=st.sampled_from(tuple("abc ABC.?!,;")),
        min_size=1,
        max_size=500,
    ),
    maximum=st.integers(min_value=1, max_value=50),
)
def test_chunking_is_deterministic_and_never_exceeds_provider_limit(
    text: str,
    maximum: int,
) -> None:
    first = chunk_text(text, maximum)
    second = chunk_text(text, maximum)

    assert first == second
    assert all(chunk and len(chunk) <= maximum for chunk in first)


def test_chunking_does_not_split_combining_or_joiner_sequences() -> None:
    assert chunk_text("a\u0301b", 2) == ("a\u0301", "b")
    with pytest.raises(ValidationError, match="grapheme"):
        chunk_text("a\u0301", 1)
    with pytest.raises(ValidationError, match="grapheme"):
        chunk_text("👩\u200d💻", 2)


@pytest.mark.parametrize(
    ("term", "match"),
    [
        (
            'written = "word"\nspoken = "say"\nstatus = "unknown"\n',
            "invalid status",
        ),
        (
            'written = "word"\nspoken = "say"\nstatus = "approved"\nmatch = "regex"\n',
            "match must be",
        ),
        (
            'written = "word"\nspoken = "say"\nstatus = "approved"\nenabled = "yes"\n',
            "enabled must be boolean",
        ),
    ],
)
def test_pronunciation_contract_rejects_ambiguous_values(
    tmp_path: Path,
    term: str,
    match: str,
) -> None:
    source = tmp_path / "book.md"
    source.write_text("# One\n\nA word.", encoding="utf-8")
    pronunciations = tmp_path / "pronunciations.toml"
    pronunciations.write_text(
        "schema_version = 1\n[[terms]]\n" + term,
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match=match):
        normalize_sources((source,), pronunciations=pronunciations)


def test_twenty_chapter_plan_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "book.md"
    source.write_text(
        "\n".join(f"# Chapter {index}\n\nText {index}." for index in range(1, 22)),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "yakbox.toml"
    manifest_path.write_text(
        '"$schema" = '
        '"https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"\n'
        'schema_version = 1\nsources = ["book.md"]\n'
        '[book]\ntitle = "Twenty One"\n'
        '[profiles.default]\nbackend = "fake"\nvoice = "narrator"\n'
        '[targets.default]\nprofile = "default"\n',
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    document = normalize_sources(manifest.sources)
    first = plan_audiobook(manifest, document)
    second = plan_audiobook(manifest, document)

    assert len(document.chapters) == 21
    assert first == second
    shards = shard_plan(first, 4)
    chapter_ids = [node.chapter_id for shard in shards for node in shard]
    assert set(chapter_ids) == {chapter.id for chapter in document.chapters}
