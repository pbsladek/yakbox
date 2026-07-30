from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import uuid4

from yakbox._files import atomic_write_json, safe_child, sha256_file
from yakbox.audiobook.artifacts import (
    ArtifactKind,
    ArtifactRecord,
    inventory_artifacts,
    verify_artifact,
)
from yakbox.audiobook.journal import target_lock
from yakbox.contracts import runtime_metadata, schema_uri
from yakbox.errors import ArtifactError

type RestoreItem = tuple[Path, Path, Path, Path]


@dataclass(frozen=True, slots=True)
class CleanupCandidate:
    id: str
    path: Path
    metadata_path: Path
    size: int
    sha256: str
    metadata_sha256: str
    reason: str


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    schema_version: int
    cleanup_id: str
    workspace: Path
    artifact_root: Path
    candidates: tuple[CleanupCandidate, ...]
    target: str = "default"
    current_paths: tuple[Path, ...] = ()

    @property
    def bytes_reclaimable(self) -> int:
        return sum(item.size for item in self.candidates)

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("audiobook-cleanup-plan"),
            "cleanup_id": self.cleanup_id,
            "workspace": str(self.workspace),
            "artifact_root": str(self.artifact_root),
            "target": self.target,
            "current_paths": [str(path) for path in self.current_paths],
            "bytes_to_quarantine": self.bytes_reclaimable,
            "bytes_reclaimable_after_purge": self.bytes_reclaimable,
            "bytes_reclaimable": self.bytes_reclaimable,
            "candidates": [
                {
                    "id": item.id,
                    "path": str(item.path),
                    "metadata_path": str(item.metadata_path),
                    "size": item.size,
                    "sha256": item.sha256,
                    "metadata_sha256": item.metadata_sha256,
                    "reason": item.reason,
                }
                for item in self.candidates
            ],
        }


def plan_cleanup(
    workspace: Path,
    artifact_root: Path,
    *,
    kind: ArtifactKind | None = None,
    older_than_days: int | None = None,
    target: str = "default",
    keep_successful_runs: int = 0,
    audition_days: int | None = None,
    preview_days: int | None = None,
    raw_until_release: bool = True,
    current_paths: tuple[Path, ...] = (),
) -> CleanupPlan:
    inventory = inventory_artifacts(artifact_root)
    incomplete_runs = _incomplete_run_ids(workspace)
    release_paths = _release_references(artifact_root)
    has_release = _has_release_manifest(artifact_root)
    retained_paths = _retained_artifact_paths(
        workspace,
        artifact_root,
        keep_successful_runs,
    )
    resolved_current_paths = tuple(
        safe_child(artifact_root, path) for path in current_paths
    )
    candidates: list[CleanupCandidate] = []
    for record in inventory.records:
        if not _cleanup_eligible(
            record,
            incomplete_runs=incomplete_runs,
            release_paths=release_paths,
            current_paths=resolved_current_paths,
            retained_paths=retained_paths,
            raw_until_release=raw_until_release,
            has_release=has_release,
            kind=kind,
        ):
            continue
        retention_days = (
            older_than_days
            if older_than_days is not None
            else audition_days
            if record.kind is ArtifactKind.AUDITION
            else preview_days
            if record.kind is ArtifactKind.PREVIEW
            else None
        )
        cutoff = (
            datetime.now(UTC) - timedelta(days=retention_days)
            if retention_days is not None
            else None
        )
        modified = datetime.fromtimestamp(record.path.stat().st_mtime, tz=UTC)
        if cutoff is not None and modified >= cutoff:
            continue
        valid, error = verify_artifact(record)
        if not valid:
            raise ArtifactError(
                f"Refusing stale cleanup plan for {record.path}: {error}"
            )
        candidates.append(
            CleanupCandidate(
                id=record.id,
                path=record.path,
                metadata_path=record.path.with_suffix(
                    f"{record.path.suffix}.artifact.json"
                ),
                size=record.size,
                sha256=record.sha256,
                metadata_sha256=sha256_file(
                    record.path.with_suffix(f"{record.path.suffix}.artifact.json")
                ),
                reason="managed artifact outside the protected current graph",
            )
        )
    return CleanupPlan(
        schema_version=1,
        cleanup_id=f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}",
        workspace=workspace.resolve(),
        artifact_root=artifact_root.resolve(),
        candidates=tuple(candidates),
        target=target,
        current_paths=resolved_current_paths,
    )


def _cleanup_eligible(
    record: ArtifactRecord,
    *,
    incomplete_runs: set[str],
    release_paths: set[Path],
    current_paths: tuple[Path, ...],
    retained_paths: set[Path],
    raw_until_release: bool,
    has_release: bool,
    kind: ArtifactKind | None,
) -> bool:
    path = record.path.resolve()
    if record.protected or record.kind is ArtifactKind.RELEASE:
        return False
    if record.run_id in incomplete_runs or path in release_paths:
        return False
    if path in current_paths or path in retained_paths:
        return False
    if raw_until_release and record.kind is ArtifactKind.RAW and not has_release:
        return False
    if kind is not None and record.kind is not kind:
        return False
    return record.path.is_file()


def apply_cleanup(plan: CleanupPlan) -> Path:
    with target_lock(plan.workspace / ".yakbox", plan.target):
        trash = plan.workspace / ".yakbox" / "trash" / plan.cleanup_id
        if trash.exists():
            raise ArtifactError(f"Cleanup already exists: {plan.cleanup_id}")
        _validate_cleanup_plan(plan)
        trash.mkdir(parents=True)
        moved: list[dict[str, object]] = []
        try:
            for candidate in plan.candidates:
                relative = candidate.path.relative_to(plan.artifact_root)
                destination = trash / "files" / relative
                metadata_destination = (
                    trash
                    / "metadata"
                    / relative.with_suffix(f"{relative.suffix}.artifact.json")
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                metadata_destination.parent.mkdir(parents=True, exist_ok=True)
                candidate.path.replace(destination)
                candidate.metadata_path.replace(metadata_destination)
                moved.append(
                    {
                        "id": candidate.id,
                        "original": str(candidate.path),
                        "quarantined": str(destination.relative_to(trash)),
                        "metadata_original": str(candidate.metadata_path),
                        "metadata_quarantined": str(
                            metadata_destination.relative_to(trash)
                        ),
                        "sha256": candidate.sha256,
                        "metadata_sha256": candidate.metadata_sha256,
                        "size": candidate.size,
                    }
                )
            atomic_write_json(
                trash / "cleanup.json",
                {
                    **runtime_metadata("audiobook-quarantine"),
                    "cleanup_id": plan.cleanup_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "artifact_root": str(plan.artifact_root),
                    "target": plan.target,
                    "items": moved,
                },
            )
        except Exception:
            _rollback_cleanup_moves(trash, moved)
            if trash.exists():
                shutil.rmtree(trash)
            raise
    return trash


def restore_trash(
    workspace: Path,
    cleanup_id: str,
    *,
    relative_path: Path | None = None,
) -> int:
    trash_root = workspace.resolve() / ".yakbox" / "trash"
    trash = safe_child(trash_root, trash_root / cleanup_id)
    manifest_path = trash / "cleanup.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Cannot read cleanup {cleanup_id}: {error}") from error
    target, prepared = _prepare_restore(
        trash,
        cleanup_id,
        raw,
        relative_path=relative_path,
    )
    restored: list[RestoreItem] = []
    try:
        with target_lock(workspace.resolve() / ".yakbox", target):
            for source, destination, metadata_source, metadata_destination in prepared:
                destination.parent.mkdir(parents=True, exist_ok=True)
                metadata_destination.parent.mkdir(parents=True, exist_ok=True)
                source.replace(destination)
                metadata_source.replace(metadata_destination)
                restored.append(
                    (source, destination, metadata_source, metadata_destination)
                )
    except Exception:
        for source, destination, metadata_source, metadata_destination in reversed(
            restored
        ):
            if destination.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                destination.replace(source)
            if metadata_destination.exists():
                metadata_source.parent.mkdir(parents=True, exist_ok=True)
                metadata_destination.replace(metadata_source)
        raise
    atomic_write_json(
        trash / "restored.json",
        {
            **runtime_metadata("audiobook-cleanup-report"),
            "cleanup_id": cleanup_id,
            "restored_at": datetime.now(UTC).isoformat(),
            "count": len(restored),
        },
    )
    return len(restored)


def _prepare_restore(
    trash: Path,
    cleanup_id: str,
    raw: dict[str, object],
    *,
    relative_path: Path | None,
) -> tuple[str, list[RestoreItem]]:
    if (
        raw.get("$schema") != schema_uri("audiobook-quarantine")
        or raw.get("schema_version") != 1
        or raw.get("cleanup_id") != cleanup_id
    ):
        raise ArtifactError("Cleanup manifest identity or schema is invalid")
    artifact_root = Path(str(raw.get("artifact_root", ""))).resolve()
    target = str(raw.get("target", "default"))
    items = raw.get("items", [])
    if not isinstance(items, list):
        raise ArtifactError("Cleanup manifest items are invalid")
    selected = _restore_selection(artifact_root, relative_path)
    prepared: list[RestoreItem] = []
    matched = False
    for item in items:
        if not isinstance(item, dict):
            raise ArtifactError("Cleanup manifest contains an invalid item")
        item = cast(dict[str, object], item)
        source = safe_child(trash, trash / str(item["quarantined"]))
        destination = safe_child(artifact_root, Path(str(item["original"])))
        metadata_source = safe_child(trash, trash / str(item["metadata_quarantined"]))
        metadata_destination = safe_child(
            artifact_root, Path(str(item["metadata_original"]))
        )
        item_relative = destination.relative_to(artifact_root)
        if selected is not None and item_relative != selected:
            continue
        matched = True
        if (
            not source.exists()
            and not metadata_source.exists()
            and destination.is_file()
            and metadata_destination.is_file()
        ):
            if sha256_file(destination) != item["sha256"] or (
                item.get("metadata_sha256") is not None
                and sha256_file(metadata_destination) != item["metadata_sha256"]
            ):
                raise ArtifactError(
                    f"Previously restored artifact changed: {destination}"
                )
            continue
        _validate_restore_item(
            source,
            destination,
            metadata_source,
            metadata_destination,
            item,
        )
        prepared.append((source, destination, metadata_source, metadata_destination))
    if selected is not None and not matched:
        raise ArtifactError(f"Cleanup does not contain artifact path: {selected}")
    return target, prepared


def _restore_selection(artifact_root: Path, relative_path: Path | None) -> Path | None:
    if relative_path is None:
        return None
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ArtifactError("Restore --path must be relative to the artifact root")
    selected = safe_child(artifact_root, artifact_root / relative_path)
    return selected.relative_to(artifact_root)


def _validate_restore_item(
    source: Path,
    destination: Path,
    metadata_source: Path,
    metadata_destination: Path,
    item: dict[str, object],
) -> None:
    if destination.exists() or metadata_destination.exists():
        raise ArtifactError(f"Restore destination already exists: {destination}")
    if not source.is_file() or not metadata_source.is_file():
        raise ArtifactError(f"Quarantined files are incomplete: {source}")
    if sha256_file(source) != item["sha256"]:
        raise ArtifactError(f"Quarantined artifact digest mismatch: {source}")
    metadata_digest = item.get("metadata_sha256")
    if metadata_digest is not None and sha256_file(metadata_source) != metadata_digest:
        raise ArtifactError(f"Quarantined metadata digest mismatch: {metadata_source}")


def _validate_cleanup_plan(plan: CleanupPlan) -> None:
    inventory = inventory_artifacts(plan.artifact_root)
    current = {record.id: record for record in inventory.records}
    incomplete_runs = _incomplete_run_ids(plan.workspace)
    release_paths = _release_references(plan.artifact_root)
    current_paths = set(plan.current_paths)
    seen: set[str] = set()
    for candidate in plan.candidates:
        if candidate.id in seen:
            raise ArtifactError(f"Cleanup plan repeats artifact {candidate.id}")
        seen.add(candidate.id)
        safe_child(plan.artifact_root, candidate.path)
        safe_child(plan.artifact_root, candidate.metadata_path)
        record = current.get(candidate.id)
        if (
            record is None
            or record.path != candidate.path.resolve()
            or record.protected
            or record.kind is ArtifactKind.RELEASE
            or record.run_id in incomplete_runs
            or record.path.resolve() in release_paths
            or record.path.resolve() in current_paths
        ):
            raise ArtifactError(
                f"Artifact is no longer eligible for cleanup: {candidate.path}"
            )
        if (
            not candidate.path.is_file()
            or not candidate.metadata_path.is_file()
            or candidate.path.stat().st_size != candidate.size
            or sha256_file(candidate.path) != candidate.sha256
            or sha256_file(candidate.metadata_path) != candidate.metadata_sha256
        ):
            raise ArtifactError(f"Artifact changed after planning: {candidate.path}")


def _rollback_cleanup_moves(
    trash: Path,
    moved: list[dict[str, object]],
) -> None:
    for item in reversed(moved):
        quarantined = trash / str(item["quarantined"])
        original = Path(str(item["original"]))
        if quarantined.exists():
            original.parent.mkdir(parents=True, exist_ok=True)
            quarantined.replace(original)
        metadata_quarantined = trash / str(item["metadata_quarantined"])
        metadata_original = Path(str(item["metadata_original"]))
        if metadata_quarantined.exists():
            metadata_original.parent.mkdir(parents=True, exist_ok=True)
            metadata_quarantined.replace(metadata_original)


def _incomplete_run_ids(workspace: Path) -> set[str]:
    runs = workspace.resolve() / ".yakbox" / "runs"
    if not runs.exists():
        return set()
    result: set[str] = set()
    for directory in (path for path in runs.iterdir() if path.is_dir()):
        path = directory / "run.json"
        if not path.is_file():
            result.add(directory.name)
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactError(f"Cannot classify build run {path}: {error}") from error
        run_id = raw.get("run_id")
        status = raw.get("status")
        if not isinstance(run_id, str) or not isinstance(status, str):
            raise ArtifactError(f"Cannot classify build run {path}")
        if run_id != directory.name:
            raise ArtifactError(f"Build run identity mismatch: {path}")
        if status != "complete":
            result.add(run_id)
    return result


def _release_references(artifact_root: Path) -> set[Path]:
    references: set[Path] = set()
    release_root = artifact_root.resolve() / "release"
    if not release_root.exists():
        return references
    for path in release_root.glob("*/release.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactError(
                f"Cannot classify immutable release {path}: {error}"
            ) from error
        for key in ("master_wavs", "delivery_mp3s"):
            values = raw.get(key, [])
            if not isinstance(values, list):
                raise ArtifactError(f"Invalid immutable release manifest: {path}")
            for item in values:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    raise ArtifactError(f"Invalid immutable release manifest: {path}")
                candidate = Path(str(item["path"]))
                if not candidate.is_absolute():
                    candidate = artifact_root / candidate
                references.add(candidate.resolve())
    return references


def _has_release_manifest(artifact_root: Path) -> bool:
    release_root = artifact_root.resolve() / "release"
    return release_root.exists() and any(release_root.glob("*/release.json"))


def _retained_artifact_paths(
    workspace: Path,
    artifact_root: Path,
    keep_successful_runs: int,
) -> set[Path]:
    if keep_successful_runs < 0:
        raise ArtifactError("keep_successful_runs cannot be negative")
    if keep_successful_runs == 0:
        return set()
    retained_runs = _retained_run_ids(workspace, keep_successful_runs)
    return {
        record.path.resolve()
        for record in inventory_artifacts(artifact_root).records
        if record.run_id in retained_runs
    }


def _retained_run_ids(
    workspace: Path,
    keep_successful_runs: int,
) -> set[str]:
    runs = workspace.resolve() / ".yakbox" / "runs"
    if not runs.exists():
        return set()
    successful: set[str] = set()
    for path in sorted(runs.glob("*/run.json"), reverse=True):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactError(f"Cannot classify build run {path}: {error}") from error
        if not isinstance(raw, dict):
            raise ArtifactError(f"Cannot classify build run {path}")
        run = cast(dict[str, object], raw)
        run_id = run.get("run_id")
        if run.get("status") == "complete":
            if not isinstance(run_id, str) or run_id != path.parent.name:
                raise ArtifactError(f"Build run identity mismatch: {path}")
            successful.add(run_id)
        if len(successful) == keep_successful_runs:
            break
    return successful


def purge_trash(workspace: Path, cleanup_id: str | None = None) -> int:
    trash_root = workspace.resolve() / ".yakbox" / "trash"
    if not trash_root.exists():
        return 0
    targets = (
        [safe_child(trash_root, trash_root / cleanup_id)]
        if cleanup_id is not None
        else [path for path in trash_root.iterdir() if path.is_dir()]
    )
    count = 0
    for target in targets:
        safe_child(trash_root, target)
        if target.exists():
            shutil.rmtree(target)
            count += 1
    return count
