from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from yakbox._files import sha256_file
from yakbox.audiobook.artifacts import (
    ArtifactKind,
    ArtifactRecord,
    write_artifact_record,
)
from yakbox.audiobook.cleanup import (
    apply_cleanup,
    plan_cleanup,
    purge_trash,
    restore_trash,
)
from yakbox.errors import ArtifactError, BuildError


def _raw_artifact(workspace: Path, *, run_id: str = "run-complete") -> Path:
    root = workspace / "build"
    path = root / "raw" / "chapter.wav"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"tiny-wave-fixture")
    write_artifact_record(
        ArtifactRecord(
            schema_version=1,
            id="chapter:synthesize",
            kind=ArtifactKind.RAW,
            path=path.resolve(),
            sha256=sha256_file(path),
            size=path.stat().st_size,
            fingerprint="fixture",
            target="default",
            run_id=run_id,
            protected=False,
            media_type="audio/wav",
        ),
        root=root,
    )
    return path


def _preview_artifact(workspace: Path, *, run_id: str = "preview-run") -> Path:
    root = workspace / "build"
    path = root / "previews" / run_id / "sample.wav"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"tiny-preview")
    write_artifact_record(
        ArtifactRecord(
            schema_version=1,
            id=f"preview:{run_id}",
            kind=ArtifactKind.PREVIEW,
            path=path.resolve(),
            sha256=sha256_file(path),
            size=path.stat().st_size,
            fingerprint="fixture",
            target="default",
            run_id=run_id,
            protected=False,
            media_type="audio/wav",
        ),
        root=root,
    )
    return path


def _complete_run(workspace: Path, run_id: str) -> None:
    run = workspace / ".yakbox" / "runs" / run_id
    run.mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps({"run_id": run_id, "status": "complete"}),
        encoding="utf-8",
    )


def test_cleanup_revalidates_metadata_and_leaves_no_partial_trash(
    tmp_path: Path,
) -> None:
    artifact = _raw_artifact(tmp_path)
    plan = plan_cleanup(tmp_path, tmp_path / "build", raw_until_release=False)
    metadata = artifact.with_suffix(".wav.artifact.json")
    metadata.write_text(
        metadata.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="changed after planning"):
        apply_cleanup(plan)

    assert artifact.is_file()
    assert metadata.is_file()
    assert not (tmp_path / ".yakbox" / "trash" / plan.cleanup_id).exists()


def test_cleanup_obeys_target_lock(tmp_path: Path) -> None:
    artifact = _raw_artifact(tmp_path)
    plan = plan_cleanup(tmp_path, tmp_path / "build", raw_until_release=False)
    lock = tmp_path / ".yakbox" / "locks" / "default.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("held", encoding="utf-8")

    with pytest.raises(BuildError, match="already locked"):
        apply_cleanup(plan)

    assert artifact.is_file()


def test_restore_prevalidates_collisions_and_rejects_manifest_path_escape(
    tmp_path: Path,
) -> None:
    artifact = _raw_artifact(tmp_path)
    plan = plan_cleanup(tmp_path, tmp_path / "build", raw_until_release=False)
    trash = apply_cleanup(plan)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"collision")

    with pytest.raises(ArtifactError, match="already exists"):
        restore_trash(tmp_path, plan.cleanup_id)

    artifact.unlink()
    manifest_path = trash / "cleanup.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["items"][0]["original"] = str(tmp_path.parent / "escaped.wav")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="escapes managed root"):
        restore_trash(tmp_path, plan.cleanup_id)
    assert not (tmp_path.parent / "escaped.wav").exists()


def test_incomplete_run_artifacts_are_not_cleanup_candidates(tmp_path: Path) -> None:
    _raw_artifact(tmp_path, run_id="incomplete")
    run = tmp_path / ".yakbox" / "runs" / "incomplete"
    run.mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps({"run_id": "incomplete", "status": "failed"}),
        encoding="utf-8",
    )

    assert (
        plan_cleanup(
            tmp_path,
            tmp_path / "build",
            raw_until_release=False,
        ).candidates
        == ()
    )


def test_interrupted_run_without_summary_is_protected(tmp_path: Path) -> None:
    _raw_artifact(tmp_path, run_id="interrupted")
    run = tmp_path / ".yakbox" / "runs" / "interrupted"
    run.mkdir(parents=True)
    (run / "journal.ndjson").write_text("interrupted\n", encoding="utf-8")

    assert (
        plan_cleanup(
            tmp_path,
            tmp_path / "build",
            raw_until_release=False,
        ).candidates
        == ()
    )


def test_retention_keeps_only_artifacts_owned_by_recent_successful_runs(
    tmp_path: Path,
) -> None:
    artifact = _raw_artifact(tmp_path, run_id="001-old")
    _complete_run(tmp_path, "001-old")
    _complete_run(tmp_path, "002-new")

    retained = plan_cleanup(
        tmp_path,
        tmp_path / "build",
        raw_until_release=False,
        keep_successful_runs=2,
    )
    expired = plan_cleanup(
        tmp_path,
        tmp_path / "build",
        raw_until_release=False,
        keep_successful_runs=1,
    )

    assert retained.candidates == ()
    assert [item.path for item in expired.candidates] == [artifact.resolve()]


def test_raw_waits_for_release_and_preview_uses_age_policy(tmp_path: Path) -> None:
    raw = _raw_artifact(tmp_path)
    preview = _preview_artifact(tmp_path)

    before_release = plan_cleanup(
        tmp_path,
        tmp_path / "build",
        preview_days=7,
    )
    assert before_release.candidates == ()

    release = tmp_path / "build" / "release" / "release-1"
    release.mkdir(parents=True)
    (release / "release.json").write_text(
        json.dumps({"master_wavs": [], "delivery_mp3s": []}),
        encoding="utf-8",
    )
    old = time.time() - 10 * 24 * 60 * 60
    os.utime(preview, (old, old))

    after_release = plan_cleanup(
        tmp_path,
        tmp_path / "build",
        preview_days=7,
    )
    assert {item.path for item in after_release.candidates} == {
        raw.resolve(),
        preview.resolve(),
    }


def test_purge_is_explicit_and_scoped_to_quarantine(tmp_path: Path) -> None:
    _raw_artifact(tmp_path)
    plan = plan_cleanup(tmp_path, tmp_path / "build", raw_until_release=False)
    trash = apply_cleanup(plan)

    assert purge_trash(tmp_path, plan.cleanup_id) == 1
    assert not trash.exists()
    assert purge_trash(tmp_path, plan.cleanup_id) == 0


def test_restore_can_select_one_relative_artifact_then_restore_remainder(
    tmp_path: Path,
) -> None:
    raw = _raw_artifact(tmp_path)
    preview = _preview_artifact(tmp_path)
    plan = plan_cleanup(tmp_path, tmp_path / "build", raw_until_release=False)
    apply_cleanup(plan)

    assert (
        restore_trash(
            tmp_path,
            plan.cleanup_id,
            relative_path=Path("raw/chapter.wav"),
        )
        == 1
    )
    assert raw.read_bytes() == b"tiny-wave-fixture"
    assert not preview.exists()

    assert restore_trash(tmp_path, plan.cleanup_id) == 1
    assert preview.read_bytes() == b"tiny-preview"

    with pytest.raises(ArtifactError, match="must be relative"):
        restore_trash(
            tmp_path,
            plan.cleanup_id,
            relative_path=Path("../source.md"),
        )
