from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path

from yakbox._files import atomic_write_json, safe_child, sha256_file
from yakbox.audio import inspect_audio
from yakbox.contracts import runtime_metadata, schema_uri
from yakbox.errors import ArtifactError

SHA256_HEX_LENGTH = 64


class ArtifactKind(StrEnum):
    RAW = "raw"
    MASTER = "master"
    DELIVERY = "delivery"
    REPORT = "report"
    AUDITION = "audition"
    PREVIEW = "preview"
    RELEASE = "release"


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    schema_version: int
    id: str
    kind: ArtifactKind
    path: Path
    sha256: str
    size: int
    fingerprint: str
    target: str
    run_id: str
    protected: bool
    dependencies: tuple[str, ...] = ()
    media_type: str | None = None
    logical_voice: str | None = None
    reference_audio_sha256: str | None = None
    reference_rights_basis: str | None = None
    watermark_disclosure: str | None = None

    def to_dict(self, *, root: Path) -> dict[str, object]:
        value = asdict(self)
        value.update(runtime_metadata("audiobook-artifact"))
        value["kind"] = self.kind.value
        value["path"] = self.path.resolve().relative_to(root.resolve()).as_posix()
        value["dependencies"] = list(self.dependencies)
        return value


@dataclass(frozen=True, slots=True)
class InventoryReport:
    root: Path
    records: tuple[ArtifactRecord, ...]
    unknown_files: tuple[Path, ...]
    total_bytes: int
    managed_bytes: int

    def to_dict(self, *, workspace: Path) -> dict[str, object]:
        return {
            **runtime_metadata("audiobook-inventory"),
            "root": str(self.root),
            "artifacts": [record.to_dict(root=workspace) for record in self.records],
            "unknown_files": [str(path) for path in self.unknown_files],
            "total_bytes": self.total_bytes,
            "managed_bytes": self.managed_bytes,
        }


def write_artifact_record(record: ArtifactRecord, *, root: Path) -> Path:
    metadata_path = record.path.with_suffix(f"{record.path.suffix}.artifact.json")
    atomic_write_json(metadata_path, record.to_dict(root=root))
    return metadata_path


def load_artifact_record(path: Path, *, root: Path) -> ArtifactRecord:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or raw.get("$schema") != schema_uri("audiobook-artifact")
            or raw.get("schema_version") != 1
        ):
            raise ArtifactError(f"Unsupported artifact manifest {path}")
        relative = Path(str(raw["path"]))
        if relative.is_absolute():
            raise ArtifactError(f"Artifact path must be relative: {path}")
        artifact_path = safe_child(root, root / relative)
        record = ArtifactRecord(
            schema_version=int(raw["schema_version"]),
            id=str(raw["id"]),
            kind=ArtifactKind(raw["kind"]),
            path=artifact_path,
            sha256=str(raw["sha256"]),
            size=int(raw["size"]),
            fingerprint=str(raw["fingerprint"]),
            target=str(raw["target"]),
            run_id=str(raw["run_id"]),
            protected=bool(raw["protected"]),
            dependencies=tuple(str(item) for item in raw.get("dependencies", [])),
            media_type=raw.get("media_type"),
            logical_voice=raw.get("logical_voice"),
            reference_audio_sha256=raw.get("reference_audio_sha256"),
            reference_rights_basis=raw.get("reference_rights_basis"),
            watermark_disclosure=raw.get("watermark_disclosure"),
        )
        _validate_record_fields(record, path)
        return record
    except ArtifactError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Invalid artifact manifest {path}: {error}") from error


def verify_artifact(record: ArtifactRecord) -> tuple[bool, str | None]:
    if not record.path.is_file():
        return False, "file is missing"
    if record.path.stat().st_size != record.size:
        return False, "size differs from manifest"
    if sha256_file(record.path) != record.sha256:
        return False, "digest differs from manifest"
    return True, None


def repair_artifact_metadata(
    record: ArtifactRecord,
    *,
    root: Path,
) -> ArtifactRecord:
    """Explicitly accept current artifact bytes and refresh digest metadata."""
    if not record.path.is_file() or record.path.stat().st_size == 0:
        raise ArtifactError(f"Cannot repair missing or empty artifact: {record.path}")
    if record.media_type == "application/json":
        try:
            json.loads(record.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactError(
                f"Cannot repair malformed JSON artifact: {record.path}"
            ) from error
    elif record.media_type and record.media_type.startswith("audio/"):
        inspection = inspect_audio(record.path)
        if not inspection.valid:
            raise ArtifactError(
                f"Cannot repair invalid audio artifact {record.path}: "
                + "; ".join(inspection.issues)
            )
    repaired = replace(
        record,
        sha256=sha256_file(record.path),
        size=record.path.stat().st_size,
    )
    write_artifact_record(repaired, root=root)
    return repaired


def inventory_artifacts(root: Path) -> InventoryReport:
    root = root.resolve()
    records: list[ArtifactRecord] = []
    artifact_ids: set[str] = set()
    metadata_paths = set(root.rglob("*.artifact.json")) if root.exists() else set()
    managed: set[Path] = set(metadata_paths)
    for metadata_path in sorted(metadata_paths):
        record = load_artifact_record(metadata_path, root=root)
        if record.id in artifact_ids:
            raise ArtifactError(f"Duplicate artifact identity: {record.id}")
        records.append(record)
        artifact_ids.add(record.id)
        managed.add(record.path)
    files = (
        {path for path in root.rglob("*") if path.is_file()} if root.exists() else set()
    )
    unknown = tuple(sorted(files - managed))
    return InventoryReport(
        root=root,
        records=tuple(records),
        unknown_files=unknown,
        total_bytes=sum(path.stat().st_size for path in files),
        managed_bytes=sum(record.size for record in records if record.path.exists()),
    )


def _validate_record_fields(record: ArtifactRecord, metadata_path: Path) -> None:
    if not record.id or not record.target or not record.run_id:
        raise ArtifactError(f"Artifact identity is incomplete: {metadata_path}")
    if (
        record.size < 0
        or len(record.sha256) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in record.sha256)
    ):
        raise ArtifactError(f"Artifact size or digest is invalid: {metadata_path}")
    expected_media = {
        ArtifactKind.RAW: "audio/wav",
        ArtifactKind.MASTER: "audio/wav",
        ArtifactKind.AUDITION: "audio/wav",
        ArtifactKind.PREVIEW: "audio/wav",
        ArtifactKind.DELIVERY: "audio/mpeg",
        ArtifactKind.REPORT: "application/json",
    }.get(record.kind)
    if expected_media is not None and record.media_type != expected_media:
        raise ArtifactError(
            f"Artifact media type conflicts with its managed kind: {metadata_path}"
        )
