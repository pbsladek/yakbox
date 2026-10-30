from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

import pytest

import yakbox.audiobook.manifest as manifests
from yakbox.errors import ValidationError


@pytest.mark.parametrize(
    "field,value",
    [
        ("$schema", "https://invalid/schema"),
        ("schema_version", 99),
        ("book.cover", 42),
        ("book.title", " "),
        ("book.series_position", False),
        ("book.series_position", " "),
        ("pronunciations", 42),
        ("sources", []),
        ("sources", [42]),
        ("source", []),
        ("voices", []),
        ("voices.narrator", "not a table"),
        ("voices.narrator.rights_basis", "unknown-license"),
        ("voices.narrator.reference_audio", 42),
        ("profiles", []),
        ("profiles.default", "not a table"),
        ("profiles.default.backend", "unknown"),
        ("profiles.default.voice", "unknown"),
        (
            "profiles.default",
            {"backend": "chatterbox-local", "voice": "narrator", "max_processes": 2},
        ),
        (
            "profiles.default",
            {"backend": "chatterbox-local", "voice": "narrator", "device": "tpu"},
        ),
        ("characters", []),
        ("characters.UPPERCASE", {"profile": "default"}),
        ("characters.narrator", {"profile": "default", "cfg_weight": 0.5}),
        ("dialogue", []),
        ("dialogue.expressive_tag_handling", "invalid"),
        ("dialogue.routes", 42),
        ("whisper_qa", []),
        ("whisper_qa.cache_directory", " "),
        ("whisper_qa.phoneme_backend", "hosted"),
        ("whisper_qa.phoneme_timeout_seconds", 0),
        ("whisper_qa.minimum_phoneme_confidence", 1.1),
        ("whisper_qa", {"phoneme_alignment": True, "phoneme_model": "custom/local"}),
        ("short_utterances", []),
        (
            "short_utterances",
            {"strategy": "context_extract", "alignment_model": "custom/local"},
        ),
        ("short_utterances.alignment_aliases", []),
        ("short_utterances.alignment_aliases", {"two words": ["word"]}),
        ("short_utterances.alignment_aliases", {"word": []}),
        ("short_utterances.alignment_aliases", {"word": [42]}),
        ("short_utterances.alignment_aliases", {"word": ["two words"]}),
        ("short_utterances.carrier_positions", []),
        ("short_utterances.carrier_positions", ["middle", "middle"]),
        ("targets", []),
        ("targets.default", "not a table"),
        ("targets.default.profile", "unknown"),
        ("targets.default.output_root", " "),
        ("targets.default.output_root", "/outside"),
        (
            "targets.default",
            {
                "profile": "default",
                "from_stage": "inspect",
                "through_stage": "synthesize",
            },
        ),
        (
            "targets.default",
            {"profile": "default", "quality_min_lufs": -10, "quality_max_lufs": -20},
        ),
        ("targets.default.extends", "default"),
        ("targets.default.extends", "missing"),
        ("targets.default.extends", 42),
        ("targets.default.max_estimated_spend", 1),
        ("targets.default.max_estimated_spend", "invalid"),
        ("targets.default.max_estimated_spend", "NaN"),
        ("retention", []),
        ("repairs", []),
        ("repairs.mode", "unsupported"),
        ("repairs.takes", 21),
        ("repairs", {"takes": 1, "minimum_passing_takes": 2}),
        ("runtime", []),
    ],
)
def test_manifest_rejects_invalid_sections_resources_and_paths_before_building(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    path = book_workspace / "yakbox.toml"
    raw = tomllib.loads(path.read_text())
    table = raw
    parts = field.split(".")
    for part in parts[:-1]:
        current = table.setdefault(part, {})
        assert isinstance(current, dict)
        table = cast(dict[str, object], current)
    table[parts[-1]] = value
    monkeypatch.setattr(manifests, "_read_manifest", lambda _path: raw)
    with pytest.raises(ValidationError):
        manifests.load_manifest(path)
    assert not (book_workspace / ".yakbox").exists()
    assert not (book_workspace / "build").exists()


@pytest.mark.parametrize("content", [None, b"[invalid", b"\xff"])
def test_manifest_missing_malformed_or_non_utf8_files_have_domain_errors(
    tmp_path: Path, content: bytes | None
) -> None:
    path = tmp_path / "yakbox.toml"
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(
        ValidationError, match=r"Manifest does not exist|Cannot read manifest"
    ):
        manifests.load_manifest(path)


def test_manifest_lookup_reports_unknown_profiles_and_voices(
    book_workspace: Path,
) -> None:
    manifest = manifests.load_manifest(book_workspace / "yakbox.toml")
    with pytest.raises(ValidationError, match="Unknown profile"):
        manifest.profile("missing")
    with pytest.raises(ValidationError, match="Unknown logical voice"):
        manifest.voice("missing")
