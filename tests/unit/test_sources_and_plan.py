from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from yakbox.audiobook.manifest import load_manifest
from yakbox.audiobook.planner import plan_audiobook, shard_plan
from yakbox.audiobook.sources import (
    ChunkBoundary,
    Pause,
    SpeechSegment,
    audit_pronunciations,
    chunk_text,
    normalize_sources,
    plan_text_chunks,
)
from yakbox.errors import ValidationError


def _write_routed_workspace(
    tmp_path: Path,
    source_text: str,
    *,
    assistance: str = "warn",
    mara_cfg_weight: float = 0.3,
) -> Path:
    (tmp_path / "book.md").write_text(source_text, encoding="utf-8")
    for name in ("narrator", "mara", "wren"):
        (tmp_path / f"{name}.wav").write_bytes(name.encode())
    manifest_path = tmp_path / "yakbox.toml"
    manifest_path.write_text(
        '"$schema" = "https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"\n'
        'schema_version = 1\nsources = ["book.md"]\n'
        '[book]\ntitle = "Routed Book"\n'
        '[voices.narrator]\ndisplay_name = "Narrator"\n'
        'reference_audio = "narrator.wav"\n'
        '[voices.mara]\ndisplay_name = "Mara"\n'
        'reference_audio = "mara.wav"\n'
        '[voices.wren]\ndisplay_name = "Wren (male)"\n'
        'reference_audio = "wren.wav"\n'
        '[profiles.narrator]\nbackend = "chatterbox-local"\n'
        'voice = "narrator"\ndevice = "cpu"\n'
        '[profiles.mara]\nbackend = "chatterbox-local"\n'
        'voice = "mara"\ndevice = "cpu"\n'
        '[profiles.wren]\nbackend = "chatterbox-local"\n'
        'voice = "wren"\ndevice = "cpu"\n'
        '[characters.narrator]\nprofile = "narrator"\n'
        '[characters.mara]\nprofile = "mara"\n'
        f"cfg_weight = {mara_cfg_weight}\nexaggeration = 0.7\nseed = 17\n"
        '[characters.wren]\ndisplay_name = "Wren"\nprofile = "wren"\n'
        "[dialogue]\n"
        f'attribution_assistance = "{assistance}"\n'
        "short_utterance_words = 3\n"
        '[targets.default]\nprofile = "narrator"\nchunk_chars = 500\n',
        encoding="utf-8",
    )
    return manifest_path


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


def test_pronunciation_audit_reports_usage_locations_and_shadowing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.md"
    source.write_text("# One\n\nNew York and unused text.\n", encoding="utf-8")
    pronunciations = tmp_path / "pronunciations.toml"
    pronunciations.write_text(
        "schema_version = 1\n\n"
        '[[terms]]\nwritten = "New York"\nspoken = "New York City"\n'
        'status = "approved"\nenabled = true\npriority = 10\n\n'
        '[[terms]]\nwritten = "York"\nspoken = "Yorkshire"\n'
        'status = "approved"\nenabled = true\n\n'
        '[[terms]]\nwritten = "missing"\nspoken = "present"\n'
        'status = "approved"\nenabled = true\n',
        encoding="utf-8",
    )

    audit = audit_pronunciations((source,), pronunciations)

    by_written = {rule.written: rule for rule in audit.rules}
    assert by_written["New York"].applied == 1
    assert by_written["New York"].locations[0].start_line == 3
    assert by_written["York"].matches == 1
    assert by_written["York"].shadowed == 1
    assert by_written["missing"].unused
    assert audit.unused_rules == 1
    assert audit.shadowed_matches == 1


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


def test_speaker_directive_routes_dialogue_and_narrates_attribution(
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.md"
    source.write_text(
        "# One\n\nNarration before the argument.\n\n"
        "<!-- yakbox:speech:speaker name=wren -->\n\n"
        '"Mara, step away from the console," Wren said.\n\n'
        "Narration resumes after Wren speaks.\n",
        encoding="utf-8",
    )

    document = normalize_sources((source,))
    speech = tuple(
        item
        for item in document.chapters[0].segments
        if isinstance(item, SpeechSegment)
    )

    assert [item.speaker for item in speech] == [
        "narrator",
        "wren",
        "narrator",
        "narrator",
    ]
    assert [item.speaker_explicit for item in speech] == [
        False,
        True,
        False,
        False,
    ]
    assert [item.text for item in speech[1:3]] == [
        "Mara, step away from the console,",
        "Wren said.",
    ]
    assert speech[1].source.start_line == 7


def test_narrator_directive_can_override_profile_without_splitting_dialogue(
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.md"
    source.write_text(
        "# One\n\n"
        "<!-- yakbox:speech:speaker name=narrator profile=narrator-retry -->\n\n"
        '"No cameras," the clerk said.\n\nNarration resumes.\n',
        encoding="utf-8",
    )

    document = normalize_sources((source,))
    speech = tuple(
        item
        for item in document.chapters[0].segments
        if isinstance(item, SpeechSegment)
    )

    assert [item.text for item in speech] == [
        '"No cameras," the clerk said.',
        "Narration resumes.",
    ]
    assert [item.speaker for item in speech] == ["narrator", "narrator"]
    assert [item.profile_override for item in speech] == ["narrator-retry", None]


def test_character_directive_can_override_attribution_profile(tmp_path: Path) -> None:
    source = tmp_path / "book.md"
    source.write_text(
        "# One\n\n"
        "<!-- yakbox:speech:speaker name=liora profile=liora-retry "
        "narrator_profile=narrator-retry -->\n\n"
        '"Quill," Liora said.\n',
        encoding="utf-8",
    )

    speech = tuple(
        item
        for item in normalize_sources((source,)).chapters[0].segments
        if isinstance(item, SpeechSegment)
    )

    assert [item.text for item in speech] == ["Quill,", "Liora said."]
    assert [item.speaker for item in speech] == ["liora", "narrator"]
    assert [item.profile_override for item in speech] == [
        "liora-retry",
        "narrator-retry",
    ]


def test_routed_dialogue_splits_action_tags_and_multiple_quotes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.md"
    source.write_text(
        "# One\n\n"
        "<!-- yakbox:speech:speaker name=mara -->\n\n"
        "Mara faced him. “No,” Mara said. “I will not leave.”\n\n"
        "<!-- yakbox:speech:speaker name=wren -->\n\n"
        "This entire unquoted turn belongs to Wren.\n\n"
        "<!-- yakbox:speech:speaker name=mara -->\n\n"
        "“An unmatched quote remains one character turn.\n",
        encoding="utf-8",
    )

    speech = tuple(
        item
        for item in normalize_sources((source,)).chapters[0].segments
        if isinstance(item, SpeechSegment)
    )

    assert [item.speaker for item in speech] == [
        "narrator",
        "mara",
        "narrator",
        "mara",
        "wren",
        "mara",
    ]
    assert [item.text for item in speech[:4]] == [
        "Mara faced him.",
        "No,",
        "Mara said.",
        "I will not leave.",
    ]
    assert speech[4].speaker_explicit
    assert speech[5].speaker_explicit


def test_routed_dialogue_preserves_internal_clause_and_sentence_boundaries(
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.md"
    source.write_text(
        "# One\n\n"
        "<!-- yakbox:speech:speaker name=character-1 -->\n\n"
        '"Third," she said. "A maid found him. Rask was first uniform up."\n',
        encoding="utf-8",
    )

    speech = tuple(
        item
        for item in normalize_sources((source,)).chapters[0].segments
        if isinstance(item, SpeechSegment)
    )

    assert [item.text for item in speech] == [
        "Third,",
        "she said.",
        "A maid found him. Rask was first uniform up.",
    ]
    assert [item.boundary_after for item in speech] == [
        ChunkBoundary.CLAUSE,
        ChunkBoundary.SENTENCE,
        ChunkBoundary.PARAGRAPH,
    ]


def test_routed_dialogue_strips_only_paired_character_quote_delimiters(
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.md"
    source.write_text(
        "# One\n\n"
        "<!-- yakbox:speech:speaker name=character-1 -->\n\n"
        "«Stay here.»\n\n"
        '"Quoted narration keeps its delimiters."\n',
        encoding="utf-8",
    )

    speech = tuple(
        item
        for item in normalize_sources((source,)).chapters[0].segments
        if isinstance(item, SpeechSegment)
    )

    assert [item.speaker for item in speech] == ["character-1", "narrator"]
    assert [item.text for item in speech] == [
        "Stay here.",
        '"Quoted narration keeps its delimiters."',
    ]


def test_speaker_directive_changes_document_identity_and_must_be_used(
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.md"
    source.write_text("# One\n\nSame words.\n", encoding="utf-8")
    narrator_hash = normalize_sources((source,)).sha256
    source.write_text(
        "# One\n\n<!-- yakbox:speech:speaker name=wren -->\n\nSame words.\n",
        encoding="utf-8",
    )
    assert normalize_sources((source,)).sha256 != narrator_hash

    source.write_text(
        "# One\n\n<!-- yakbox:speech:speaker name=wren -->\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match=r"book\.md:3:.*precede speech"):
        normalize_sources((source,))

    source.write_text(
        "# One\n\nText <!-- yakbox:speech:speaker name=wren --> inline.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="own line"):
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
    with pytest.raises(ValidationError, match="grapheme"):
        chunk_text("🇺🇸X", 1)


def test_chunking_prioritizes_sentence_boundaries_over_later_words() -> None:
    chunks = plan_text_chunks("Alpha sentence. beta gamma delta epsilon", 28)

    assert chunks[0].text == "Alpha sentence."
    assert chunks[0].boundary is ChunkBoundary.SENTENCE


def test_plan_marks_short_narration_with_policy_fingerprint(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
    )

    plan = plan_audiobook(manifest, document)
    synthesis = next(node for node in plan.nodes if node.stage == "synthesize")
    serialized = plan.to_dict(root=manifest.root)
    nodes = cast(list[dict[str, object]], serialized["nodes"])
    chunks = cast(list[dict[str, object]], nodes[0]["chunks"])
    short_marker = cast(dict[str, object], chunks[2]["short_utterance"])

    assert synthesis.chunk_short_utterances[0] is None
    assert synthesis.chunk_short_utterances[1] is None
    assert synthesis.chunk_short_utterances[2] is not None
    assert synthesis.chunk_short_utterances[2].word_count == 3
    assert synthesis.chunk_short_utterances[2].policy_fingerprint == (
        manifest.short_utterances.fingerprint
    )
    assert short_marker["reason"] == "word_count"


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
        (
            'written = "word"\nspoken = "say"\nstatus = "approved"\nlanguage = 42\n',
            "language must be a non-empty string",
        ),
        (
            'written = "word"\nspoken = "say"\nstatus = "approved"\nnotes = 42\n',
            "notes must be a string",
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
    synthesis = next(node for node in first.nodes if node.stage.value == "synthesize")
    assert len(synthesis.chunk_sources) == len(synthesis.chunks)
    serialized = first.to_dict(root=tmp_path)
    nodes = cast(list[dict[str, object]], serialized["nodes"])
    chunks = cast(list[dict[str, object]], nodes[0]["chunks"])
    source_value = cast(dict[str, object], chunks[0]["source"])
    assert source_value["path"] == "book.md"
    assert cast(int, source_value["start_line"]) > 0
    shards = shard_plan(first, 4)
    chapter_ids = [node.chapter_id for shard in shards for node in shard]
    assert set(chapter_ids) == {chapter.id for chapter in document.chapters}

    selected = plan_audiobook(manifest, document, chapter_selector="2-4,7")
    assert {node.chapter_id for node in selected.nodes} == {
        document.chapters[index - 1].id for index in (2, 3, 4, 7)
    }
    with pytest.raises(ValueError, match="starts after"):
        plan_audiobook(manifest, document, chapter_selector="4-2")


def test_local_plan_applies_backend_limit_and_records_boundaries(
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.md"
    source.write_text("# One\n\n" + "word " * 300, encoding="utf-8")
    manifest_path = tmp_path / "yakbox.toml"
    manifest_path.write_text(
        '"$schema" = "https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"\n'
        'schema_version = 1\nsources = ["book.md"]\n'
        '[book]\ntitle = "Book"\n'
        '[voices.narrator]\ndisplay_name = "Narrator"\n'
        '[profiles.default]\nbackend = "chatterbox-local"\nvoice = "narrator"\n'
        '[targets.default]\nprofile = "default"\nchunk_chars = 2800\n',
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)

    plan = plan_audiobook(manifest, normalize_sources(manifest.sources))
    synthesis = next(node for node in plan.nodes if node.stage.value == "synthesize")

    assert max(map(len, synthesis.chunks)) <= 500
    assert len(synthesis.chunk_boundaries) == len(synthesis.chunks)
    assert synthesis.chunk_boundaries[-1] == "paragraph"


def test_plan_records_character_routes_settings_and_attribution_findings(
    tmp_path: Path,
) -> None:
    manifest_path = _write_routed_workspace(
        tmp_path,
        "# One\n\nThe observatory lights dimmed while Mara watched the array.\n\n"
        "<!-- yakbox:speech:speaker name=wren -->\n\n"
        '"Mara, step away from the console before it locks onto us," Wren said.\n\n'
        "<!-- yakbox:speech:speaker name=mara -->\n\n"
        '"I have waited twelve years for this answer, and I am staying," Mara said.\n\n'
        "<!-- yakbox:speech:speaker name=mara -->\n\n"
        '"No."\n\n'
        '"It already knows both of our names."\n',
    )
    manifest = load_manifest(manifest_path)
    plan = plan_audiobook(manifest, normalize_sources(manifest.sources))
    synthesis = next(node for node in plan.nodes if node.stage.value == "synthesize")

    speech_routes = tuple(route for route in synthesis.chunk_routes if route.speaker)
    assert [route.speaker for route in speech_routes] == [
        "narrator",
        "wren",
        "narrator",
        "mara",
        "narrator",
        "mara",
        "narrator",
    ]
    mara_routes = tuple(route for route in speech_routes if route.speaker == "mara")
    assert all(route.profile == "mara" for route in mara_routes)
    assert all(route.cfg_weight == 0.3 for route in mara_routes)
    assert all(route.exaggeration == 0.7 for route in mara_routes)
    assert all(route.seed == 17 for route in mara_routes)
    assert {finding.code for finding in plan.attribution_findings} == {
        "short-dialogue",
        "unrouted-dialogue",
    }

    serialized = plan.to_dict(root=tmp_path)
    nodes = cast(list[dict[str, object]], serialized["nodes"])
    chunks = cast(list[dict[str, object]], nodes[0]["chunks"])
    assert chunks[0]["speaker"] == "narrator"
    assert chunks[1]["speaker"] == "wren"
    assert chunks[3]["performance"] == {
        "cfg_weight": 0.3,
        "exaggeration": 0.7,
        "seed": 17,
    }


def test_routing_policy_unknown_speakers_and_performance_change_fingerprints(
    tmp_path: Path,
) -> None:
    source_text = (
        "# One\n\n<!-- yakbox:speech:speaker name=mara -->\n\n"
        '"This turn is long enough to synthesize naturally," Mara said.\n'
    )
    manifest_path = _write_routed_workspace(tmp_path, source_text)
    first_manifest = load_manifest(manifest_path)
    document = normalize_sources(first_manifest.sources)
    first = plan_audiobook(first_manifest, document)

    changed_path = _write_routed_workspace(
        tmp_path,
        source_text,
        mara_cfg_weight=0.4,
    )
    changed_manifest = load_manifest(changed_path)
    changed = plan_audiobook(
        changed_manifest,
        normalize_sources(changed_manifest.sources),
    )
    assert changed.fingerprint != first.fingerprint

    error_path = _write_routed_workspace(
        tmp_path,
        '# One\n\n<!-- yakbox:speech:speaker name=unknown -->\n\n"Who am I?"\n',
    )
    error_manifest = load_manifest(error_path)
    with pytest.raises(ValidationError, match="Unknown character: unknown"):
        plan_audiobook(error_manifest, normalize_sources(error_manifest.sources))


def test_attribution_assistance_can_be_disabled_or_enforced(tmp_path: Path) -> None:
    source_text = '# One\n\n"Unrouted dialogue is narrated."\n'
    off_path = _write_routed_workspace(tmp_path, source_text, assistance="off")
    off_manifest = load_manifest(off_path)
    assert not plan_audiobook(
        off_manifest, normalize_sources(off_manifest.sources)
    ).attribution_findings

    error_path = _write_routed_workspace(tmp_path, source_text, assistance="error")
    error_manifest = load_manifest(error_path)
    with pytest.raises(ValidationError, match="Attribution assistance found 1"):
        plan_audiobook(error_manifest, normalize_sources(error_manifest.sources))
