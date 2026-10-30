from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.unit.test_speech_model_registry import (
    _model_path,
    _package,
    _record,
    _registry,
)

import yakbox.phoneme_models as phonemes
from yakbox.errors import BackendUnavailableError, ModelIntegrityError, ValidationError
from yakbox.speech.model_registry import (
    ModelFileRecord,
    ModelRegistry,
    ModelRegistryData,
)


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("package", "name", ""),
        ("package", "license", "unknown"),
        ("package", "source_revision", ""),
        ("package", "source_revision", "main"),
        ("package", "sdist_sha256", "invalid"),
        ("file", "path", "../outside"),
        ("file", "path", "/outside"),
        ("file", "size_bytes", 0),
        ("file", "digest_kind", "unknown"),
        ("model", "files", ()),
        ("model", "license", "unknown"),
        ("model", "converted_revision", "main"),
        ("model", "total_size_bytes", 999),
        ("registry", "schema_version", 2),
        ("registry", "qualification_language", "fr"),
        ("registry", "packages", ()),
    ],
)
def test_model_provenance_requires_licensed_pinned_allowlisted_artifacts(
    kind: str, field: str, value: object
) -> None:
    model = _record("config.json", b"{}")
    values = {
        "package": _package(),
        "file": model.files[0],
        "model": model,
        "registry": ModelRegistryData(1, "en", "2026-10-15", (_package(),), (model,)),
    }
    with pytest.raises(ValidationError):
        replace(values[kind], **{field: value})


def test_model_registry_rejects_duplicate_file_and_engine_identities() -> None:
    model = _record("config.json", b"{}")
    with pytest.raises(ValidationError, match="file paths must be unique"):
        replace(model, files=(model.files[0], model.files[0]), total_size_bytes=4)
    with pytest.raises(ValidationError, match="engine names must be unique"):
        ModelRegistryData(1, "en", "2026-10-15", (_package(),), (model, model))


def test_model_registry_verifies_git_blob_digests_without_trusting_filenames(
    tmp_path: Path,
) -> None:
    payload = b'{"model_type":"fixture"}\n'
    model = _record("config.json", payload)
    digest = hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload, usedforsecurity=False
    ).hexdigest()
    model = replace(
        model, files=(ModelFileRecord("config.json", len(payload), "git-sha1", digest),)
    )
    registry = ModelRegistry(
        tmp_path, data=ModelRegistryData(1, "en", "2026-10-15", (_package(),), (model,))
    )
    root = _model_path(tmp_path)
    root.mkdir(parents=True)
    (root / "config.json").write_bytes(payload)
    assert registry.status("whisper").verified
    (root / "config.json").write_bytes(b"x" * len(payload))
    assert not registry.status("whisper").verified


def test_model_registry_refuses_non_directory_snapshots_and_unknown_engines(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path, "config.json", b"{}")
    destination = _model_path(tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"not-a-directory")
    assert "model_path_not_directory" in registry.status("whisper").issues
    assert registry.verify_snapshot_directory("whisper", destination) == (
        "model_path_not_directory",
    )
    with pytest.raises(ModelIntegrityError, match="Refusing to replace"):
        registry.install("whisper")
    with pytest.raises(ValidationError, match="Unknown registered"):
        registry.record("missing")


def _local_phoneme_model(root: Path) -> None:
    (root / "config.json").write_text("{}")
    (root / "vocab.json").write_text('{"<pad>":0,"a":1}')
    (root / "model.safetensors").write_bytes(b"fixture")


def test_phoneme_lifecycle_resolves_only_verified_files_and_explicitly_pinned_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _local_phoneme_model(tmp_path)
    status = phonemes.phoneme_model_status(str(tmp_path), None)
    assert status.verified
    assert phonemes.require_phoneme_model_path(str(tmp_path), None) == tmp_path
    hub = SimpleNamespace(snapshot_download=Mock(return_value=str(tmp_path)))
    monkeypatch.setattr(phonemes, "_hub", lambda: hub)
    monkeypatch.setattr(phonemes, "phoneme_model_status", lambda *_args: status)
    assert phonemes.install_phoneme_model("example/phoneme", "pinned") == status
    hub.snapshot_download.assert_called_once_with(
        repo_id="example/phoneme", revision="pinned"
    )
    with pytest.raises(BackendUnavailableError, match="pinned revision"):
        phonemes.install_phoneme_model("example/phoneme", None)
    hub.snapshot_download.side_effect = OSError("offline")
    with pytest.raises(BackendUnavailableError, match="Could not install"):
        phonemes.install_phoneme_model("example/phoneme", "pinned")
    monkeypatch.setattr(
        phonemes,
        "phoneme_model_status",
        lambda *_args: replace(status, local_path=None, verified=False),
    )
    with pytest.raises(BackendUnavailableError, match="not installed"):
        phonemes.require_phoneme_model_path("missing", None)
    monkeypatch.setattr(
        phonemes,
        "phoneme_model_status",
        lambda *_args: replace(status, verified=False, issues=("vocabulary_missing",)),
    )
    with pytest.raises(BackendUnavailableError, match="vocabulary_missing"):
        phonemes.require_phoneme_model_path("broken", None)
