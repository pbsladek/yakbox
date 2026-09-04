from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from yakbox.errors import ModelIntegrityError
from yakbox.speech.analysis_worker_artifact import (
    build_worker_artifact,
    packaged_worker_artifact_path,
    source_worker_artifact_bytes,
    verify_packaged_worker_artifact,
    verify_worker_artifact,
    worker_artifact_bytes,
)


def test_worker_artifact_build_is_byte_reproducible(tmp_path: Path) -> None:
    first = build_worker_artifact(tmp_path / "first.pyz")
    second = build_worker_artifact(tmp_path / "second.pyz")

    assert first.sha256 == second.sha256
    assert first.size_bytes == second.size_bytes
    assert first.entry_count == second.entry_count
    assert first.manifest_fingerprint == second.manifest_fingerprint
    assert first.path.read_bytes() == second.path.read_bytes()


def test_packaged_worker_is_current_with_reviewed_source() -> None:
    verified = verify_packaged_worker_artifact()

    assert verified.path == packaged_worker_artifact_path().resolve()
    assert worker_artifact_bytes() == source_worker_artifact_bytes()


def test_worker_artifact_manifest_covers_every_safe_entry(tmp_path: Path) -> None:
    artifact = build_worker_artifact(tmp_path / "worker.pyz")

    verified = verify_worker_artifact(artifact.path)

    assert verified == artifact
    with zipfile.ZipFile(artifact.path) as archive:
        names = tuple(item.filename for item in archive.infolist())
    assert names == tuple(sorted(names))
    assert "__main__.py" in names
    assert "yakbox/speech/analysis_worker.py" in names
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_worker_artifact_rejects_modified_entry(tmp_path: Path) -> None:
    artifact = build_worker_artifact(tmp_path / "worker.pyz")
    with zipfile.ZipFile(artifact.path, "a") as archive:
        archive.writestr("unexpected.py", "pass\n")

    with pytest.raises(ModelIntegrityError, match="differ"):
        verify_worker_artifact(artifact.path)
