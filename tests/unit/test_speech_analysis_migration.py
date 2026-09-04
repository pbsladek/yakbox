from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import yakbox.speech as public_speech
import yakbox.speech.analysis_migration as migration_module
from yakbox.audiobook.sources import normalize_sources
from yakbox.errors import ValidationError
from yakbox.schemas import load_schema
from yakbox.speech.analysis_manifest import parse_draft_manifest_speech_analysis
from yakbox.speech.analysis_migration import (
    DRAFT_COMMAND_MAP,
    DRAFT_PYTHON_EXPORTS,
    SPEECH_ANALYSIS_CACHE_VERSION,
    plan_migrated_manifest,
    preview_manifest_migration,
    render_manifest_toml,
    write_manifest_migration,
)

ROOT = Path(__file__).parents[2]
SCHEMA_PATH = ROOT / "src/yakbox/schemas/audiobook-manifest-v2.schema.json"


def _workspace_snapshot(root: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (path.relative_to(root).as_posix(), path.read_bytes().hex())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def test_migration_preview_is_deterministic_read_only_and_schema_valid(
    book_workspace: Path,
) -> None:
    before = _workspace_snapshot(book_workspace)

    first = preview_manifest_migration(book_workspace / "yakbox.toml")
    second = preview_manifest_migration(book_workspace / "yakbox.toml")

    assert first.to_dict() == second.to_dict()
    assert first.fingerprint == second.fingerprint
    assert _workspace_snapshot(book_workspace) == before
    assert first.speech_analysis.cache.version == SPEECH_ANALYSIS_CACHE_VERSION
    assert first.speech_analysis.cache.legacy_evidence == "historical_only"
    assert first.draft_manifest["schema_version"] == 2
    assert "whisper_qa" not in first.draft_manifest
    parsed = parse_draft_manifest_speech_analysis(first.draft_manifest)
    assert parsed == first.speech_analysis
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(first.draft_manifest)


def test_migration_separates_pronunciation_hint_from_lexical_truth(
    book_workspace: Path,
) -> None:
    preview = preview_manifest_migration(book_workspace / "yakbox.toml")

    assert len(preview.pronunciations) == 1
    term = preview.pronunciations[0]
    assert term.written == "short"
    assert term.synthesis_hint == "brief"
    assert term.expected_lexical == ("short",)
    assert term.phonemes == ()
    assert term.review_required
    assert any(
        finding.code == "pronunciation_tokenization_changed" and finding.review_required
        for finding in preview.findings
    )


def test_migrated_fixture_uses_internal_v2_validation_and_real_planner(
    book_workspace: Path,
) -> None:
    preview = preview_manifest_migration(book_workspace / "yakbox.toml")
    manifest = preview._legacy_manifest
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
    )

    plan = plan_migrated_manifest(
        preview,
        document,
        target_name=manifest.targets[0].name,
    )

    assert plan.nodes
    assert plan.target == "default"
    schema_uri = preview.draft_manifest["$schema"]
    assert isinstance(schema_uri, str)
    assert schema_uri.endswith("/audiobook-manifest-v2.schema.json")


def test_cutover_surface_is_declared_but_not_public() -> None:
    assert ("yakbox whisper inspect", "yakbox speech inspect") in DRAFT_COMMAND_MAP
    assert "SpeechRecognizer" in DRAFT_PYTHON_EXPORTS
    assert not hasattr(public_speech, "SpeechRecognizer")


def test_migration_write_requires_resolutions_and_clean_destination(
    book_workspace: Path,
) -> None:
    preview = preview_manifest_migration(book_workspace / "yakbox.toml")
    destination = book_workspace / "migrated" / "yakbox.toml"

    with pytest.raises(ValidationError, match="explicit resolution"):
        write_manifest_migration(preview, destination=destination)
    assert not destination.exists()

    resolutions = tuple(
        finding.code for finding in preview.findings if finding.review_required
    )

    result = write_manifest_migration(
        preview,
        destination=destination,
        resolved_finding_codes=resolutions,
    )

    assert result.manifest_path == destination.resolve()
    assert result.backup_path is None
    parsed = tomllib.loads(destination.read_text(encoding="utf-8"))
    assert parsed == preview.draft_manifest
    assert parse_draft_manifest_speech_analysis(parsed) == preview.speech_analysis
    assert (
        result.pronunciation_path
        == (destination.parent / "pronunciations.toml").resolve()
    )
    assert result.pronunciation_path.is_file()
    Draft202012Validator(load_schema("audiobook-manifest-migration")).validate(
        result.to_dict()
    )
    Draft202012Validator(load_schema("pronunciations", version=2)).validate(
        tomllib.loads(result.pronunciation_path.read_text(encoding="utf-8"))
    )

    with pytest.raises(ValidationError, match="must not already exist"):
        write_manifest_migration(
            preview,
            destination=destination,
            resolved_finding_codes=resolutions,
        )


def test_in_place_migration_preserves_exact_backup(book_workspace: Path) -> None:
    source = book_workspace / "yakbox.toml"
    original = source.read_bytes()
    preview = preview_manifest_migration(source)
    resolutions = tuple(
        finding.code for finding in preview.findings if finding.review_required
    )

    result = write_manifest_migration(
        preview,
        resolved_finding_codes=resolutions,
    )

    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original
    assert tomllib.loads(source.read_text(encoding="utf-8"))["schema_version"] == 2
    pronunciation_backup = book_workspace / "pronunciations.toml.v1.bak"
    assert pronunciation_backup.is_file()
    assert (
        tomllib.loads(
            (book_workspace / "pronunciations.toml").read_text(encoding="utf-8")
        )["schema_version"]
        == 2
    )


def test_migration_rolls_back_every_changed_file_after_partial_commit(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = book_workspace / "yakbox.toml"
    pronunciation = book_workspace / "pronunciations.toml"
    original_manifest = source.read_bytes()
    original_pronunciation = pronunciation.read_bytes()
    preview = preview_manifest_migration(source)
    resolutions = tuple(
        finding.code for finding in preview.findings if finding.review_required
    )
    original_write = migration_module.atomic_write_bytes
    failed = False

    def fail_manifest_commit_once(
        path: Path,
        data: bytes,
        *,
        overwrite: bool = False,
    ) -> None:
        nonlocal failed
        if (
            path.resolve() == source.resolve()
            and data != original_manifest
            and not failed
        ):
            failed = True
            raise OSError("simulated manifest commit failure")
        original_write(path, data, overwrite=overwrite)

    monkeypatch.setattr(
        migration_module,
        "atomic_write_bytes",
        fail_manifest_commit_once,
    )

    with pytest.raises(OSError, match="simulated manifest commit failure"):
        write_manifest_migration(
            preview,
            resolved_finding_codes=resolutions,
        )

    assert source.read_bytes() == original_manifest
    assert pronunciation.read_bytes() == original_pronunciation
    assert (book_workspace / "yakbox.toml.v1.bak").read_bytes() == original_manifest
    assert (
        book_workspace / "pronunciations.toml.v1.bak"
    ).read_bytes() == original_pronunciation


def test_migration_refuses_stale_preview(book_workspace: Path) -> None:
    source = book_workspace / "yakbox.toml"
    preview = preview_manifest_migration(source)
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="changed after"):
        write_manifest_migration(preview)


def test_toml_renderer_is_deterministic_and_round_trips_preview(
    book_workspace: Path,
) -> None:
    preview = preview_manifest_migration(book_workspace / "yakbox.toml")
    first = render_manifest_toml(preview.draft_manifest)
    second = render_manifest_toml(preview.draft_manifest)

    assert first == second
    assert tomllib.loads(first.decode()) == preview.draft_manifest


@pytest.mark.parametrize(
    "example",
    (
        "tiny-book",
        "selective-rebuild",
        "m4b-release",
        "pronunciation-heavy",
        "multiple-voices",
    ),
)
def test_real_fake_backend_examples_plan_through_internal_v2(example: str) -> None:
    preview = preview_manifest_migration(ROOT / "examples" / example / "yakbox.toml")
    manifest = preview._legacy_manifest
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
        strip_attribution_tags=manifest.dialogue.strip_attribution_tags,
        dialogue_routes=manifest.dialogue.routes,
        expressive_tag_handling=manifest.dialogue.expressive_tag_handling,
        retain_first_attribution_per_scene=(
            manifest.dialogue.retain_first_attribution_per_scene
        ),
    )

    plan = plan_migrated_manifest(
        preview,
        document,
        target_name=manifest.targets[0].name,
    )

    assert plan.nodes
    assert all(profile.backend == "fake" for profile in manifest.profiles)
