from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from yakbox.audiobook.manifest import load_manifest
from yakbox.errors import ValidationError


def _write_manifest(tmp_path: Path, extra: str = "") -> Path:
    (tmp_path / "book.md").write_text("# One\n\nText.", encoding="utf-8")
    path = tmp_path / "yakbox.toml"
    path.write_text(
        '"$schema" = '
        '"https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"\n'
        "schema_version = 1\n"
        'sources = ["book.md"]\n'
        "[book]\n"
        'title = "Book"\n'
        "[profiles.default]\n"
        'backend = "fake"\n'
        'voice = "narrator"\n'
        "[targets.default]\n"
        'profile = "default"\n'
        f"{extra}",
        encoding="utf-8",
    )
    return path


def test_manifest_parses_hosted_confirmation_and_storage_limits(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(
        _write_manifest(
            tmp_path,
            "confirm_above_characters = 1000\n"
            "confirm_above_requests = 10\n"
            "storage_budget_bytes = 123456\n"
            'max_estimated_spend = "5.00"\n'
            'currency = "USD"\n'
            'pricing_source = "resemble-2026-07"\n'
            'price_per_character = "0.001"\n'
            "[retention]\n"
            "keep_successful_runs = 2\n"
            "audition_days = 14\n"
            "preview_days = 3\n"
            "raw_until_release = true\n",
        )
    )

    target = manifest.target("default")
    assert target.confirm_above_characters == 1_000
    assert target.confirm_above_requests == 10
    assert target.storage_budget_bytes == 123_456
    assert target.max_estimated_spend == Decimal("5.00")
    assert manifest.retention.keep_successful_runs == 2
    assert manifest.retention.audition_days == 14
    assert manifest.retention.preview_days == 3
    assert manifest.retention.raw_until_release


def test_target_inheritance_supports_draft_proof_and_release_modes(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(
        _write_manifest(
            tmp_path,
            "media_concurrency = 3\n"
            "[targets.draft]\n"
            'extends = "default"\n'
            'output_root = "build/draft"\n'
            "mastering = false\n"
            'through_stage = "synthesize"\n'
            "[targets.release]\n"
            'extends = "default"\n'
            'output_root = "build/release"\n'
            "m4b = true\n",
        )
    )

    draft = manifest.target("draft")
    release = manifest.target("release")
    assert draft.profile == "default"
    assert draft.media_concurrency == 3
    assert draft.mastering is False
    assert draft.through_stage == "synthesize"
    assert release.m4b is True
    assert release.through_stage == "inspect"


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ('output_root = "."\n', "workspace root"),
        (
            'max_estimated_spend = "1.00"\n',
            "monetary budget requires",
        ),
        ("provider_concurrency = 101\n", "at most 100"),
        ("media_concurrency = 33\n", "at most 32"),
        ("storage_budget_bytes = -1\n", "non-negative"),
        ('mastering = "yes"\n', "mastering must be boolean"),
        ('mp3_bitrate = "fast"\n', "must look like"),
        ("chunk_chars = true\n", "positive integer"),
    ],
)
def test_manifest_rejects_unsafe_or_incomplete_target_settings(
    tmp_path: Path,
    extra: str,
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        load_manifest(_write_manifest(tmp_path, extra))


def test_manifest_rejects_source_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.md"
    outside.write_text("# Outside", encoding="utf-8")
    path = _write_manifest(tmp_path)
    content = path.read_text(encoding="utf-8").replace(
        'sources = ["book.md"]',
        'sources = ["../outside.md"]',
    )
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValidationError, match="escapes"):
        load_manifest(path)


def test_manifest_rejects_output_root_containing_source(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, 'output_root = "generated"\n')
    generated = tmp_path / "generated"
    generated.mkdir()
    (tmp_path / "book.md").replace(generated / "book.md")
    content = path.read_text(encoding="utf-8").replace(
        'sources = ["book.md"]',
        'sources = ["generated/book.md"]',
    )
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValidationError, match="contains source"):
        load_manifest(path)


def test_manifest_rejects_wrong_profile_and_book_value_types(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    content = path.read_text(encoding="utf-8").replace(
        'title = "Book"',
        'title = "Book"\nlanguage = 42',
    )
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValidationError, match=r"book\.language"):
        load_manifest(path)

    path = _write_manifest(tmp_path)
    content = path.read_text(encoding="utf-8").replace(
        'backend = "fake"',
        'backend = "fake"\nexecutor = "local-process"',
    )
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValidationError, match="incompatible"):
        load_manifest(path)
