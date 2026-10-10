from __future__ import annotations

import json
from pathlib import Path

import pytest

from yakbox._files import sha256_bytes, sha256_file
from yakbox.audiobook.cache import apply_cache_cleanup, plan_cache_cleanup
from yakbox.errors import ArtifactError


def _cache_entry(workspace: Path, fingerprint: str, data: bytes) -> Path:
    audio = (
        workspace / ".yakbox/cache/synthesis" / fingerprint[:2] / f"{fingerprint}.wav"
    )
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(data)
    audio.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fingerprint": fingerprint,
                "size": len(data),
                "sha256": sha256_bytes(data),
            }
        ),
        encoding="utf-8",
    )
    return audio


def test_cache_cleanup_rejects_a_root_symlink_outside_the_workspace(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    audio = _cache_entry(outside, "a" * 64, b"keep this audio")
    workspace = tmp_path / "workspace"
    linked = workspace / ".yakbox/cache/synthesis"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(outside / ".yakbox/cache/synthesis", target_is_directory=True)

    with pytest.raises(ArtifactError, match="escapes managed root"):
        plan_cache_cleanup(workspace, max_bytes=0)

    assert audio.read_bytes() == b"keep this audio"


def test_cache_cleanup_revalidates_all_candidates_before_deleting_any(
    tmp_path: Path,
) -> None:
    first = _cache_entry(tmp_path, "a" * 64, b"first cached chunk")
    second = _cache_entry(tmp_path, "b" * 64, b"second cached chunk")
    plan = plan_cache_cleanup(tmp_path, max_bytes=0)
    _cache_entry(tmp_path, "b" * 64, b"newly regenerated second chunk")

    with pytest.raises(ArtifactError, match="changed after planning"):
        apply_cache_cleanup(plan)

    assert first.is_file()
    assert second.read_bytes() == b"newly regenerated second chunk"


def _pin_entry(workspace: Path, fingerprint: str) -> None:
    raw = workspace / "chapter.wav"
    raw.write_bytes(b"current assembled chapter")
    assembly = workspace / ".yakbox/assemblies/chapter.json"
    assembly.parent.mkdir(parents=True)
    assembly.write_text(
        json.dumps(
            {
                "raw_audio": "chapter.wav",
                "raw_sha256": sha256_file(raw),
                "chunks": [{"cache_fingerprint": fingerprint}],
            }
        ),
        encoding="utf-8",
    )


def test_cache_cleanup_preserves_chunks_pinned_after_planning(tmp_path: Path) -> None:
    fingerprint = "a" * 64
    audio = _cache_entry(tmp_path, fingerprint, b"cached chunk")
    plan = plan_cache_cleanup(tmp_path, max_bytes=0)
    _pin_entry(tmp_path, fingerprint)

    with pytest.raises(ArtifactError, match="changed after planning"):
        apply_cache_cleanup(plan)

    assert audio.is_file()


def test_cache_cleanup_preserves_pinned_chunks_with_invalid_metadata(
    tmp_path: Path,
) -> None:
    fingerprint = "a" * 64
    audio = _cache_entry(tmp_path, fingerprint, b"cached chunk")
    audio.with_suffix(".json").write_bytes(b"broken JSON")
    _pin_entry(tmp_path, fingerprint)

    assert plan_cache_cleanup(tmp_path, max_bytes=0).candidates == ()


@pytest.mark.parametrize("metadata", [b"broken JSON", b"\xff\xfe"])
def test_unpinned_corrupt_metadata_is_a_cleanup_candidate(
    tmp_path: Path, metadata: bytes
) -> None:
    audio = _cache_entry(tmp_path, "a" * 64, b"cached chunk")
    audio.with_suffix(".json").write_bytes(metadata)

    plan = plan_cache_cleanup(tmp_path)

    assert len(plan.candidates) == 1
    assert not plan.candidates[0].valid
    assert apply_cache_cleanup(plan) == 1
    assert not audio.exists()
