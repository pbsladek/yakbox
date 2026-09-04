from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from yakbox import __version__
from yakbox._files import sha256_file
from yakbox.errors import ModelIntegrityError, ValidationError
from yakbox.speech.model_registry import (
    ModelFileRecord,
    ModelRecord,
    ModelRegistry,
    ModelRegistryData,
    PackageRecord,
    load_model_registry,
    load_qualification_model_registry,
)

REVISION_A = "a" * 40
REVISION_B = "b" * 40
SHA_A = "a" * 64


def _package() -> PackageRecord:
    return PackageRecord(
        name="yakbox",
        version=__version__,
        license="MIT",
        source_url="https://example.invalid/yakbox",
        source_revision=REVISION_A,
        wheel_sha256=SHA_A,
        sdist_sha256="",
        source_verified=True,
        has_ci=True,
        has_tests=True,
        has_security_policy=True,
        signed_releases=True,
        trusted_publishing=True,
    )


def _record(path: str, data: bytes) -> ModelRecord:
    return ModelRecord(
        engine="whisper",
        backend_package="yakbox",
        converted_repository="example/converted",
        converted_revision=REVISION_A,
        upstream_repository="example/upstream",
        upstream_revision=REVISION_B,
        license="MIT",
        source_url="https://example.invalid/upstream",
        converted_url="https://example.invalid/converted",
        conversion_source="https://example.invalid/converter",
        conversion_tool="test-converter",
        conversion_tool_version="1",
        conversion_recipe_fingerprint=SHA_A,
        precision="bf16",
        conversion_verified=True,
        total_size_bytes=len(data),
        files=(
            ModelFileRecord(
                path=path,
                size_bytes=len(data),
                digest_kind="sha256",
                digest=hashlib.sha256(data).hexdigest(),
            ),
        ),
    )


def _registry(root: Path, path: str, data: bytes) -> ModelRegistry:
    return ModelRegistry(
        root,
        data=ModelRegistryData(
            1, "en", "2026-08-13", (_package(),), (_record(path, data),)
        ),
    )


def _model_path(root: Path) -> Path:
    return root / "whisper" / REVISION_A


def test_packaged_registry_is_pinned_licensed_and_sized() -> None:
    registry = load_model_registry()

    assert {model.engine for model in registry.models} == {
        "whisper",
        "parakeet",
        "qwen",
        "qwen-forced",
    }
    assert all(len(model.converted_revision) == 40 for model in registry.models)
    assert all(len(model.upstream_revision) == 40 for model in registry.models)
    assert all(
        model.total_size_bytes == sum(item.size_bytes for item in model.files)
        for model in registry.models
    )
    assert all(package.source_verified for package in registry.packages)


def test_qualification_candidates_are_pinned_but_not_default_models() -> None:
    default = load_model_registry()
    candidates = load_qualification_model_registry()

    assert {model.engine for model in candidates.models} == {
        "qwen-8bit",
        "qwen-forced-8bit",
    }
    assert not {model.engine for model in candidates.models} & {
        model.engine for model in default.models
    }
    assert {model.precision for model in candidates.models} == {"8bit-affine-group64"}
    assert {package.name for package in candidates.packages} == {"mlx-audio"}
    assert all(model.license == "Apache-2.0" for model in candidates.models)
    assert all(
        model.total_size_bytes == sum(item.size_bytes for item in model.files)
        for model in candidates.models
    )


def test_registry_verifies_exact_local_bytes_without_network(tmp_path: Path) -> None:
    data = b'{"model_type":"test"}\n'
    registry = _registry(tmp_path, "config.json", data)
    destination = _model_path(tmp_path)
    destination.mkdir(parents=True)
    config = destination / "config.json"
    config.write_bytes(data)

    status = registry.status("whisper")

    assert status.verified is True
    assert status.size_bytes == len(data)
    assert status.directory_fingerprint is not None
    assert registry.require_path("whisper") == destination
    assert registry.verify_snapshot_directory("whisper", destination) == ()
    assert sha256_file(config) == hashlib.sha256(data).hexdigest()

    config.write_bytes(b"changed")
    assert registry.verify_snapshot_directory("whisper", destination) == (
        "model_file_size:config.json",
        "model_json_invalid:config.json",
    )


@pytest.mark.parametrize(
    ("relative_path", "data", "issue"),
    [
        ("model.py", b"raise SystemExit\n", "model_code_not_allowed"),
        (
            "config.json",
            b'{"auto_map":{"AutoModel":"model.Custom"}}\n',
            "model_remote_code_mapping:config.json",
        ),
    ],
)
def test_registry_rejects_model_code_and_remote_code_mappings(
    tmp_path: Path,
    relative_path: str,
    data: bytes,
    issue: str,
) -> None:
    registry = _registry(tmp_path, relative_path, data)
    destination = _model_path(tmp_path)
    destination.mkdir(parents=True)
    (destination / relative_path).write_bytes(data)

    status = registry.status("whisper")

    assert status.verified is False
    assert issue in status.issues
    with pytest.raises(ModelIntegrityError):
        registry.require_path("whisper")


def test_registry_rejects_unexpected_and_changed_files(tmp_path: Path) -> None:
    data = b'{"model_type":"test"}\n'
    registry = _registry(tmp_path, "config.json", data)
    destination = _model_path(tmp_path)
    destination.mkdir(parents=True)
    (destination / "config.json").write_bytes(data + b"changed")
    (destination / "extra.txt").write_text("unexpected", encoding="utf-8")

    status = registry.status("whisper")

    assert status.verified is False
    assert "model_file_size:config.json" in status.issues
    assert "unexpected_model_file:extra.txt" in status.issues


class _FakeHub:
    def __init__(self, snapshot: Path) -> None:
        self.snapshot = snapshot
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def snapshot_download(
        self,
        *,
        repo_id: str,
        revision: str,
        allow_patterns: list[str],
        local_dir: str,
    ) -> str:
        self.calls.append((repo_id, revision, tuple(allow_patterns)))
        destination = Path(local_dir)
        for relative in allow_patterns:
            source = self.snapshot / relative
            if source.is_file():
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        return str(destination)


def test_install_is_explicit_allowlisted_and_then_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b'{"model_type":"test"}\n'
    root = tmp_path / "models"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_bytes(data)
    (snapshot / "not-allowlisted.txt").write_text("ignored", encoding="utf-8")
    hub = _FakeHub(snapshot)
    monkeypatch.setattr("yakbox.speech.model_registry._hub", lambda: hub)
    registry = _registry(root, "config.json", data)

    installed = registry.install("whisper")

    assert installed.verified is True
    assert hub.calls == [("example/converted", REVISION_A, ("config.json",))]
    assert not (_model_path(root) / "not-allowlisted.txt").exists()

    monkeypatch.setattr(
        "yakbox.speech.model_registry._hub",
        lambda: (_ for _ in ()).throw(AssertionError("network access")),
    )
    assert registry.require_path("whisper") == _model_path(root)


def test_install_refuses_unverified_conversion_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b'{"model_type":"test"}\n'
    record = replace(_record("config.json", data), conversion_verified=False)
    registry = ModelRegistry(
        tmp_path / "models",
        data=ModelRegistryData(1, "en", "2026-08-13", (_package(),), (record,)),
    )
    monkeypatch.setattr(
        "yakbox.speech.model_registry._hub",
        lambda: (_ for _ in ()).throw(AssertionError("network access")),
    )

    with pytest.raises(ModelIntegrityError, match="conversion provenance"):
        registry.install("whisper")


def test_registry_rejects_unknown_licenses_and_non_hex_digests() -> None:
    with pytest.raises(ValidationError, match="license"):
        PackageRecord(
            name="unsafe",
            version="1",
            license="Proprietary",
            source_url="https://example.invalid",
            source_revision=REVISION_A,
            wheel_sha256=SHA_A,
            sdist_sha256="",
            source_verified=True,
            has_ci=False,
            has_tests=False,
            has_security_policy=False,
            signed_releases=False,
            trusted_publishing=False,
        )
    with pytest.raises(ValidationError, match="digest"):
        ModelFileRecord("model.bin", 1, "sha256", "z" * 64)
