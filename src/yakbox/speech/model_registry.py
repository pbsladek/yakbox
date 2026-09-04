"""Pinned, local-only model registry for speech-analysis engines."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Protocol, cast

from yakbox._files import safe_child, sha256_file
from yakbox.errors import (
    BackendUnavailableError,
    ModelIntegrityError,
    ValidationError,
)
from yakbox.speech.analysis_fingerprints import semantic_fingerprint

MODEL_REGISTRY_VERSION = 1
_MODEL_REGISTRY_RESOURCE = "data/speech-model-registry-v1.toml"
_MODEL_CANDIDATES_RESOURCE = "data/speech-model-candidates-v1.toml"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SOFTWARE_LICENSES = frozenset(
    {"Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MIT"}
)
_MODEL_LICENSES = _SOFTWARE_LICENSES | {"CC-BY-4.0"}


class _HuggingFaceHub(Protocol):
    def snapshot_download(
        self,
        *,
        repo_id: str,
        revision: str,
        allow_patterns: list[str],
        local_dir: str,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class PackageRecord:
    """Reviewed Python distribution provenance."""

    name: str
    version: str
    license: str
    source_url: str
    source_revision: str
    wheel_sha256: str
    sdist_sha256: str
    source_verified: bool
    has_ci: bool
    has_tests: bool
    has_security_policy: bool
    signed_releases: bool
    trusted_publishing: bool

    def __post_init__(self) -> None:
        if not self.name or not self.version or not self.source_url:
            raise ValidationError("Speech package provenance is incomplete")
        if self.license not in _SOFTWARE_LICENSES:
            raise ValidationError("Speech package license is not approved")
        _require_sha256(self.wheel_sha256, "package wheel digest")
        if self.sdist_sha256:
            _require_sha256(self.sdist_sha256, "package source digest")
        if self.source_verified and not self.source_revision:
            raise ValidationError(
                "Source-verified speech packages require an immutable revision"
            )
        if self.source_revision and _REVISION.fullmatch(self.source_revision) is None:
            raise ValidationError("Speech package source revision is not immutable")


@dataclass(frozen=True, slots=True)
class ModelFileRecord:
    """One required regular file and its immutable content digest."""

    path: str
    size_bytes: int
    digest_kind: str
    digest: str

    def __post_init__(self) -> None:
        candidate = Path(self.path)
        if candidate.is_absolute() or ".." in candidate.parts or not self.path:
            raise ValidationError("Model registry file path must be relative and safe")
        if self.size_bytes <= 0:
            raise ValidationError("Model registry file size must be positive")
        if self.digest_kind not in {"sha256", "git-sha1"}:
            raise ValidationError("Unsupported model registry digest kind")
        pattern = _SHA256 if self.digest_kind == "sha256" else _REVISION
        if pattern.fullmatch(self.digest) is None:
            raise ValidationError("Model registry digest is invalid")


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """Complete immutable model and conversion provenance."""

    engine: str
    backend_package: str
    converted_repository: str
    converted_revision: str
    upstream_repository: str
    upstream_revision: str
    license: str
    source_url: str
    converted_url: str
    conversion_source: str
    conversion_tool: str
    conversion_tool_version: str
    conversion_recipe_fingerprint: str
    precision: str
    conversion_verified: bool
    total_size_bytes: int
    files: tuple[ModelFileRecord, ...]

    def __post_init__(self) -> None:
        if not self.files:
            raise ValidationError(
                "Registered model requires an explicit file allowlist"
            )
        paths = tuple(item.path for item in self.files)
        if len(paths) != len(set(paths)):
            raise ValidationError("Registered model file paths must be unique")
        if self.license not in _MODEL_LICENSES:
            raise ValidationError("Speech model license is not approved")
        for revision in (self.converted_revision, self.upstream_revision):
            if _REVISION.fullmatch(revision) is None:
                raise ValidationError("Speech model revision is not immutable")
        _require_sha256(
            self.conversion_recipe_fingerprint,
            "model conversion recipe fingerprint",
        )
        if self.total_size_bytes != sum(item.size_bytes for item in self.files):
            raise ValidationError("Speech model total size does not match its files")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-model-record-v1", self)


@dataclass(frozen=True, slots=True)
class ModelStatus:
    """Offline installation and integrity state for one registered model."""

    engine: str
    installed: bool
    verified: bool
    local_path: Path | None
    size_bytes: int
    file_count: int
    directory_fingerprint: str | None
    package_available: bool
    package_version: str | None
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "local_path": str(self.local_path) if self.local_path else None,
            "issues": list(self.issues),
        }


@dataclass(frozen=True, slots=True)
class ModelRegistryData:
    """Validated packaged registry data."""

    schema_version: int
    qualification_language: str
    reviewed_at: str
    packages: tuple[PackageRecord, ...]
    models: tuple[ModelRecord, ...]

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_REGISTRY_VERSION:
            raise ValidationError("Unsupported speech model registry version")
        if self.qualification_language != "en":
            raise ValidationError("Initial speech model registry must qualify English")
        engines = tuple(model.engine for model in self.models)
        if len(engines) != len(set(engines)):
            raise ValidationError("Speech model registry engine names must be unique")
        package_names = {package.name for package in self.packages}
        missing_packages = tuple(
            model.backend_package
            for model in self.models
            if model.backend_package not in package_names
        )
        if missing_packages:
            raise ValidationError(
                "Speech model references an unreviewed backend package"
            )

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-model-registry-v1", self)


class ModelRegistry:
    """Install explicitly and resolve only verified local model snapshots."""

    def __init__(
        self,
        root: Path,
        *,
        data: ModelRegistryData | None = None,
    ) -> None:
        self.root = root.resolve()
        self.data = data or load_model_registry()

    def engines(self) -> tuple[str, ...]:
        return tuple(sorted(model.engine for model in self.data.models))

    def record(self, engine: str) -> ModelRecord:
        matches = tuple(model for model in self.data.models if model.engine == engine)
        if len(matches) != 1:
            raise ValidationError(f"Unknown registered speech model: {engine}")
        return matches[0]

    def status(self, engine: str) -> ModelStatus:
        record = self.record(engine)
        destination = self._destination(record)
        issues: list[str] = []
        package_version = _package_version(record.backend_package)
        package = next(
            (
                item
                for item in self.data.packages
                if item.name == record.backend_package
            ),
            None,
        )
        if package_version is None:
            issues.append("backend_package_unavailable")
        elif package is None or package_version != package.version:
            issues.append("backend_package_unqualified")
        if not record.conversion_verified:
            issues.append("conversion_provenance_unverified")
        if destination.is_dir():
            issues.extend(_verify_model_directory(destination, record))
        elif destination.exists():
            issues.append("model_path_not_directory")
        else:
            issues.append("model_not_installed")
        verified = not issues
        return ModelStatus(
            engine=engine,
            installed=destination.is_dir(),
            verified=verified,
            local_path=destination if destination.is_dir() else None,
            size_bytes=(record.total_size_bytes if destination.is_dir() else 0),
            file_count=len(record.files) if destination.is_dir() else 0,
            directory_fingerprint=(
                _directory_fingerprint(record)
                if destination.is_dir() and verified
                else None
            ),
            package_available=package_version is not None,
            package_version=package_version,
            issues=tuple(dict.fromkeys(issues)),
        )

    def verify_snapshot_directory(
        self,
        engine: str,
        path: Path,
    ) -> tuple[str, ...]:
        """Verify an external conversion output against one pinned model record."""
        record = self.record(engine)
        if path.is_symlink():
            return ("model_path_symlink",)
        if not path.is_dir():
            return ("model_path_not_directory",)
        return _verify_model_directory(path, record)

    def require_path(self, engine: str) -> Path:
        status = self.status(engine)
        if not status.installed:
            raise BackendUnavailableError(
                f"Speech-analysis model {engine!r} is not installed; install it "
                "with the explicit model lifecycle service"
            )
        if not status.verified or status.local_path is None:
            raise ModelIntegrityError(
                f"Speech-analysis model {engine!r} failed verification: "
                + ", ".join(status.issues)
            )
        return status.local_path

    def install(self, engine: str) -> ModelStatus:
        """Download one pinned model, materialize its allowlist, and verify it."""
        record = self.record(engine)
        if not record.conversion_verified:
            raise ModelIntegrityError(
                f"Refusing to install {engine!r} before conversion provenance "
                "is verified"
            )
        existing = self.status(engine)
        if existing.verified:
            return existing
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self._destination(record)
        if destination.exists():
            raise ModelIntegrityError(
                f"Refusing to replace unverified model directory: {destination}"
            )
        acquisition = Path(
            tempfile.mkdtemp(prefix=f".{engine}.download.", dir=self.root)
        )
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{engine}.", suffix=".part", dir=self.root)
        )
        try:
            _acquire_model_files(record, acquisition, temporary)
            issues = _verify_model_directory(temporary, record)
            if issues:
                raise ModelIntegrityError(
                    "Downloaded model failed verification: " + ", ".join(issues)
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(destination)
        finally:
            if acquisition.exists():
                shutil.rmtree(acquisition)
            if temporary.exists():
                shutil.rmtree(temporary)
        return self.status(engine)

    def _destination(self, record: ModelRecord) -> Path:
        return safe_child(
            self.root,
            self.root / record.engine / record.converted_revision,
        )


def default_model_root() -> Path:
    """Return Yakbox's platform-owned model directory."""
    home = Path.home()
    if platform.system() == "Darwin":
        return home / "Library" / "Caches" / "yakbox" / "models"
    if os.name == "nt":
        return home / "AppData" / "Local" / "yakbox" / "models"
    return home / ".cache" / "yakbox" / "models"


def default_qualification_model_root() -> Path:
    """Return the separate cache for non-default qualification candidates."""
    return default_model_root().parent / "qualification-models"


def load_model_registry() -> ModelRegistryData:
    """Load and validate the packaged immutable model registry."""
    raw = tomllib.loads(
        files("yakbox").joinpath(_MODEL_REGISTRY_RESOURCE).read_text(encoding="utf-8")
    )
    packages_raw = _list_of_mappings(raw, "packages")
    models_raw = _list_of_mappings(raw, "models")
    packages = tuple(_package_record(item) for item in packages_raw)
    models = tuple(_model_record(item) for item in models_raw)
    return ModelRegistryData(
        schema_version=_integer(raw, "schema_version"),
        qualification_language=_string(raw, "qualification_language"),
        reviewed_at=_string(raw, "reviewed_at"),
        packages=packages,
        models=models,
    )


def load_qualification_model_registry() -> ModelRegistryData:
    """Load pinned candidates that may be measured but never selected by builds."""
    raw = tomllib.loads(
        files("yakbox").joinpath(_MODEL_CANDIDATES_RESOURCE).read_text(encoding="utf-8")
    )
    models = tuple(_model_record(item) for item in _list_of_mappings(raw, "models"))
    required_packages = {model.backend_package for model in models}
    packages = tuple(
        package
        for package in load_model_registry().packages
        if package.name in required_packages
    )
    if {package.name for package in packages} != required_packages:
        raise ValidationError("Qualification model uses an unreviewed package")
    return ModelRegistryData(
        schema_version=_integer(raw, "schema_version"),
        qualification_language=_string(raw, "qualification_language"),
        reviewed_at=_string(raw, "reviewed_at"),
        packages=packages,
        models=models,
    )


def _verify_model_directory(path: Path, record: ModelRecord) -> tuple[str, ...]:
    issues: list[str] = []
    expected = {item.path: item for item in record.files}
    actual: set[str] = set()
    for candidate in path.rglob("*"):
        relative = candidate.relative_to(path).as_posix()
        if candidate.is_symlink():
            issues.append(f"model_symlink:{relative}")
            continue
        if candidate.is_file():
            actual.add(relative)
    issues.extend(
        f"unexpected_model_file:{unexpected}"
        for unexpected in sorted(actual - expected.keys())
    )
    issues.extend(
        f"missing_model_file:{missing}" for missing in sorted(expected.keys() - actual)
    )
    for relative in sorted(actual & expected.keys()):
        file_record = expected[relative]
        candidate = path / relative
        if candidate.stat().st_size != file_record.size_bytes:
            issues.append(f"model_file_size:{relative}")
            continue
        if _file_digest(candidate, file_record.digest_kind) != file_record.digest:
            issues.append(f"model_file_digest:{relative}")
    issues.extend(_unsafe_model_configuration_issues(path, actual))
    return tuple(issues)


def _acquire_model_files(
    record: ModelRecord,
    acquisition: Path,
    destination: Path,
) -> None:
    downloaded = Path(
        _hub().snapshot_download(
            repo_id=record.converted_repository,
            revision=record.converted_revision,
            allow_patterns=[item.path for item in record.files],
            local_dir=str(acquisition),
        )
    ).absolute()
    if downloaded != acquisition.absolute():
        raise ModelIntegrityError(
            "Model installer returned an unexpected acquisition directory"
        )
    for file_record in record.files:
        untrusted_source = acquisition / file_record.path
        if untrusted_source.is_symlink():
            raise ModelIntegrityError(
                "Downloaded model file is missing or unsafe: " + file_record.path
            )
        source = safe_child(acquisition, untrusted_source)
        target = safe_child(destination, destination / file_record.path)
        if not source.is_file():
            raise ModelIntegrityError(
                "Downloaded model file is missing or unsafe: " + file_record.path
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _unsafe_model_configuration_issues(
    root: Path, relative_paths: set[str]
) -> tuple[str, ...]:
    issues: list[str] = []
    if any(Path(relative).suffix == ".py" for relative in relative_paths):
        issues.append("model_code_not_allowed")
    for relative in sorted(relative_paths):
        if Path(relative).suffix != ".json":
            continue
        try:
            value = json.loads((root / relative).read_text(encoding="utf-8"))
        except OSError, UnicodeDecodeError, json.JSONDecodeError:
            issues.append(f"model_json_invalid:{relative}")
            continue
        if _contains_remote_code_mapping(value):
            issues.append(f"model_remote_code_mapping:{relative}")
    return tuple(issues)


def _contains_remote_code_mapping(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            (key == "auto_map" and bool(item)) or _contains_remote_code_mapping(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_remote_code_mapping(item) for item in value)
    return False


def _file_digest(path: Path, kind: str) -> str:
    if kind == "sha256":
        return sha256_file(path)
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_fingerprint(record: ModelRecord) -> str:
    return semantic_fingerprint(
        "speech-model-directory-v1",
        tuple(
            (item.path, item.size_bytes, item.digest_kind, item.digest)
            for item in record.files
        ),
    )


def _hub() -> _HuggingFaceHub:
    try:
        return cast(_HuggingFaceHub, importlib.import_module("huggingface_hub"))
    except ImportError as error:
        raise BackendUnavailableError(
            "Model installation requires huggingface-hub; install one of the "
            "speech-analysis extras"
        ) from error


def _package_version(name: str) -> str | None:
    if importlib.util.find_spec(name.replace("-", "_")) is None:
        return None
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _package_record(raw: Mapping[str, object]) -> PackageRecord:
    return PackageRecord(
        name=_string(raw, "name"),
        version=_string(raw, "version"),
        license=_string(raw, "license"),
        source_url=_string(raw, "source_url"),
        source_revision=_string(raw, "source_revision", required=False),
        wheel_sha256=_string(raw, "wheel_sha256"),
        sdist_sha256=_string(raw, "sdist_sha256", required=False),
        source_verified=_boolean(raw, "source_verified"),
        has_ci=_boolean(raw, "has_ci"),
        has_tests=_boolean(raw, "has_tests"),
        has_security_policy=_boolean(raw, "has_security_policy"),
        signed_releases=_boolean(raw, "signed_releases"),
        trusted_publishing=_boolean(raw, "trusted_publishing"),
    )


def _model_record(raw: Mapping[str, object]) -> ModelRecord:
    files_raw = _list_of_mappings(raw, "files")
    return ModelRecord(
        engine=_string(raw, "engine"),
        backend_package=_string(raw, "backend_package"),
        converted_repository=_string(raw, "converted_repository"),
        converted_revision=_string(raw, "converted_revision"),
        upstream_repository=_string(raw, "upstream_repository"),
        upstream_revision=_string(raw, "upstream_revision"),
        license=_string(raw, "license"),
        source_url=_string(raw, "source_url"),
        converted_url=_string(raw, "converted_url"),
        conversion_source=_string(raw, "conversion_source"),
        conversion_tool=_string(raw, "conversion_tool"),
        conversion_tool_version=_string(raw, "conversion_tool_version"),
        conversion_recipe_fingerprint=_string(raw, "conversion_recipe_fingerprint"),
        precision=_string(raw, "precision"),
        conversion_verified=_boolean(raw, "conversion_verified"),
        total_size_bytes=_integer(raw, "total_size_bytes"),
        files=tuple(
            ModelFileRecord(
                path=_string(item, "path"),
                size_bytes=_integer(item, "size_bytes"),
                digest_kind=_string(item, "digest_kind"),
                digest=_string(item, "digest"),
            )
            for item in files_raw
        ),
    )


def _list_of_mappings(
    raw: Mapping[str, object], key: str
) -> tuple[Mapping[str, object], ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise ValidationError(f"Speech model registry {key!r} must be a table array")
    return tuple(cast(Mapping[str, object], item) for item in value)


def _string(raw: Mapping[str, object], key: str, *, required: bool = True) -> str:
    value = raw.get(key)
    if value is None and not required:
        return ""
    if isinstance(value, str) and (value or not required):
        return value
    raise ValidationError(f"Speech model registry {key!r} must be a string")


def _integer(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, bool) and isinstance(value, int):
        return value
    raise ValidationError(f"Speech model registry {key!r} must be an integer")


def _boolean(raw: Mapping[str, object], key: str) -> bool:
    value = raw.get(key)
    if isinstance(value, bool):
        return value
    raise ValidationError(f"Speech model registry {key!r} must be a boolean")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label.capitalize()} must be a SHA-256")


__all__ = [
    "MODEL_REGISTRY_VERSION",
    "ModelFileRecord",
    "ModelRecord",
    "ModelRegistry",
    "ModelRegistryData",
    "ModelStatus",
    "PackageRecord",
    "default_model_root",
    "default_qualification_model_root",
    "load_model_registry",
    "load_qualification_model_registry",
]
