from __future__ import annotations

import json
from pathlib import Path

import pytest

from yakbox._files import sha256_file
from yakbox.audiobook.artifacts import (
    ArtifactKind,
    ArtifactRecord,
    inventory_artifacts,
    repair_artifact_metadata,
    verify_artifact,
    write_artifact_record,
)
from yakbox.errors import ArtifactError


def _record(root: Path, name: str, *, artifact_id: str) -> ArtifactRecord:
    path = root / "raw" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode())
    record = ArtifactRecord(
        schema_version=1,
        id=artifact_id,
        kind=ArtifactKind.RAW,
        path=path.resolve(),
        sha256=sha256_file(path),
        size=path.stat().st_size,
        fingerprint="fixture",
        target="default",
        run_id="run",
        protected=False,
        media_type="audio/wav",
    )
    write_artifact_record(record, root=root)
    return record


def test_inventory_rejects_absolute_or_escaping_manifest_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "build"
    record = _record(root, "one.wav", artifact_id="one")
    metadata = record.path.with_suffix(".wav.artifact.json")
    raw = json.loads(metadata.read_text(encoding="utf-8"))
    raw["path"] = str(tmp_path.parent / "outside.wav")
    metadata.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ArtifactError, match="must be relative"):
        inventory_artifacts(root)


def test_inventory_rejects_duplicate_ids_and_media_kind_conflicts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "build"
    _record(root, "one.wav", artifact_id="same")
    second = _record(root, "two.wav", artifact_id="same")
    with pytest.raises(ArtifactError, match="Duplicate artifact"):
        inventory_artifacts(root)

    second_metadata = second.path.with_suffix(".wav.artifact.json")
    raw = json.loads(second_metadata.read_text(encoding="utf-8"))
    raw["id"] = "two"
    raw["media_type"] = "audio/mpeg"
    second_metadata.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ArtifactError, match="media type conflicts"):
        inventory_artifacts(root)


def test_verify_artifact_checks_size_and_digest(tmp_path: Path) -> None:
    record = _record(tmp_path / "build", "one.wav", artifact_id="one")
    assert verify_artifact(record) == (True, None)

    record.path.write_bytes(b"changed")
    valid, error = verify_artifact(record)
    assert not valid
    assert error in {"size differs from manifest", "digest differs from manifest"}


def test_repair_metadata_explicitly_accepts_valid_current_json(
    tmp_path: Path,
) -> None:
    root = tmp_path / "build"
    path = root / "reports" / "report.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"version": 1}', encoding="utf-8")
    record = ArtifactRecord(
        schema_version=1,
        id="report",
        kind=ArtifactKind.REPORT,
        path=path.resolve(),
        sha256=sha256_file(path),
        size=path.stat().st_size,
        fingerprint="fixture",
        target="default",
        run_id="run",
        protected=False,
        media_type="application/json",
    )
    write_artifact_record(record, root=root)
    path.write_text('{"version": 2}', encoding="utf-8")

    repaired = repair_artifact_metadata(record, root=root)

    assert repaired.sha256 == sha256_file(path)
    assert repaired.size == path.stat().st_size
    assert verify_artifact(repaired) == (True, None)

    path.write_text("{", encoding="utf-8")
    with pytest.raises(ArtifactError, match="malformed JSON"):
        repair_artifact_metadata(repaired, root=root)
