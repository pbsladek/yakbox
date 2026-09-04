"""Reproducible, hash-verifiable zip application for analysis workers."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from yakbox._files import atomic_write_bytes, sha256_bytes, sha256_file
from yakbox.errors import ModelIntegrityError

WORKER_ARTIFACT_FORMAT_VERSION = 1
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_WORKER_MODULES = (
    "_files.py",
    "contracts.py",
    "errors.py",
    "speech/accelerator.py",
    "speech/analysis_adapters.py",
    "speech/analysis_fingerprints.py",
    "speech/analysis_models.py",
    "speech/analysis_protocol.py",
    "speech/analysis_runtime_identity.py",
    "speech/analysis_scheduler.py",
    "speech/analysis_serialization.py",
    "speech/analysis_services.py",
    "speech/analysis_worker.py",
    "speech/model_registry.py",
    "speech/normalization.py",
    "data/speech-model-registry-v1.toml",
)
_MAIN = b"from yakbox.speech.analysis_worker import main\nraise SystemExit(main())\n"
_PACKAGED_ARTIFACT = "runtimes/analysis-worker.pyz"


@dataclass(frozen=True, slots=True)
class WorkerArtifact:
    path: Path
    sha256: str
    size_bytes: int
    entry_count: int
    manifest_fingerprint: str


def build_worker_artifact(destination: Path) -> WorkerArtifact:
    """Build identical bytes from identical reviewed source files."""
    entries = _source_entries()
    payload = _artifact_bytes(entries)
    atomic_write_bytes(destination, payload, overwrite=True)
    destination.chmod(0o644)
    return verify_worker_artifact(destination)


def verify_worker_artifact(path: Path) -> WorkerArtifact:
    """Reject missing, extra, duplicated, unsafe, or modified zip entries."""
    resolved = path.resolve()
    try:
        with zipfile.ZipFile(resolved) as archive:
            names = tuple(item.filename for item in archive.infolist())
            if len(names) != len(set(names)):
                raise ModelIntegrityError(
                    "Analysis worker artifact has duplicate entries"
                )
            if any(not _safe_archive_name(name) for name in names):
                raise ModelIntegrityError("Analysis worker artifact has an unsafe path")
            manifest_raw = archive.read("worker-manifest.json")
            manifest = json.loads(manifest_raw)
            if not isinstance(manifest, dict):
                raise ModelIntegrityError("Analysis worker manifest is invalid")
            declared = manifest.get("entries")
            if not isinstance(declared, dict):
                raise ModelIntegrityError("Analysis worker entries are invalid")
            expected_names = tuple(sorted((*declared, "worker-manifest.json")))
            if names != expected_names:
                raise ModelIntegrityError(
                    "Analysis worker entries differ from manifest"
                )
            for name, digest in declared.items():
                if not isinstance(name, str) or not isinstance(digest, str):
                    raise ModelIntegrityError(
                        "Analysis worker entry identity is invalid"
                    )
                if sha256_bytes(archive.read(name)) != digest:
                    raise ModelIntegrityError("Analysis worker entry digest differs")
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        raise ModelIntegrityError(
            "Analysis worker artifact cannot be verified"
        ) from error
    return WorkerArtifact(
        resolved,
        sha256_file(resolved),
        resolved.stat().st_size,
        len(names),
        sha256_bytes(manifest_raw),
    )


def worker_artifact_bytes() -> bytes:
    """Read the immutable artifact shipped in the Yakbox distribution."""
    path = packaged_worker_artifact_path()
    try:
        return path.read_bytes()
    except OSError as error:
        raise ModelIntegrityError(
            "The packaged analysis worker artifact is unavailable"
        ) from error


def source_worker_artifact_bytes() -> bytes:
    """Build candidate bytes from reviewed source for an explicit dev check."""
    return _artifact_bytes(_source_entries())


def packaged_worker_artifact_path() -> Path:
    return Path(__file__).parents[1] / _PACKAGED_ARTIFACT


def verify_packaged_worker_artifact() -> WorkerArtifact:
    """Prove the shipped artifact is valid and current with reviewed source."""
    path = packaged_worker_artifact_path()
    verified = verify_worker_artifact(path)
    if path.read_bytes() != source_worker_artifact_bytes():
        raise ModelIntegrityError(
            "The packaged analysis worker differs from reviewed worker source"
        )
    return verified


def _source_entries() -> dict[str, bytes]:
    package_root = Path(__file__).parents[1]
    entries = {
        f"yakbox/{name}": (package_root / name).read_bytes() for name in _WORKER_MODULES
    }
    entries["yakbox/__init__.py"] = (package_root / "__init__.py").read_bytes()
    # Keep package initializers deliberately inert in the isolated artifact.
    entries["yakbox/speech/__init__.py"] = b'"""Isolated analysis worker."""\n'
    entries["__main__.py"] = _MAIN
    return entries


def _artifact_bytes(entries: Mapping[str, bytes]) -> bytes:
    declared = {name: sha256_bytes(value) for name, value in sorted(entries.items())}
    manifest = json.dumps(
        {
            "format_version": WORKER_ARTIFACT_FORMAT_VERSION,
            "entries": declared,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    values = {**entries, "worker-manifest.json": manifest}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in sorted(values.items()):
            info = zipfile.ZipInfo(name, _ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
    return stream.getvalue()


def _safe_archive_name(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def main(arguments: list[str] | None = None) -> int:
    """Explicitly rebuild the checked-in package resource for development."""
    values = sys.argv[1:] if arguments is None else arguments
    if values != ["--write-packaged"]:
        raise SystemExit("usage: analysis_worker_artifact.py --write-packaged")
    artifact = build_worker_artifact(packaged_worker_artifact_path())
    sys.stdout.write(f"{artifact.sha256}  {_PACKAGED_ARTIFACT}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "WORKER_ARTIFACT_FORMAT_VERSION",
    "WorkerArtifact",
    "build_worker_artifact",
    "main",
    "packaged_worker_artifact_path",
    "source_worker_artifact_bytes",
    "verify_packaged_worker_artifact",
    "verify_worker_artifact",
    "worker_artifact_bytes",
]
