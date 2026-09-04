"""Approved, source-checked audio overrides for localized chapter repair."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_json, safe_child, sha256_file
from yakbox.contracts import runtime_metadata, schema_uri
from yakbox.errors import ArtifactError, ValidationError


@dataclass(frozen=True, slots=True)
class ApprovedRepair:
    """One reviewed replacement bound to exact source and routing inputs."""

    chunk_id: str
    text_sha256: str
    profile: str
    audio_path: Path
    audio_sha256: str
    source_path: str
    source_start_line: int
    source_end_line: int
    repair_id: str
    take: int

    @property
    def fingerprint(self) -> str:
        """Return the build-invalidating identity of this reviewed decision."""
        payload = json.dumps(
            {
                "version": 1,
                "chunk_id": self.chunk_id,
                "text_sha256": self.text_sha256,
                "profile": self.profile,
                "audio_sha256": self.audio_sha256,
                "repair_id": self.repair_id,
                "take": self.take,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self, *, workspace: Path) -> dict[str, object]:
        """Serialize a decision using workspace-relative managed paths."""
        value = asdict(self)
        value["audio_path"] = self.audio_path.relative_to(workspace).as_posix()
        value["source"] = {
            "path": value.pop("source_path"),
            "start_line": value.pop("source_start_line"),
            "end_line": value.pop("source_end_line"),
        }
        value["fingerprint"] = self.fingerprint
        return value


@dataclass(frozen=True, slots=True)
class RepairInstall:
    """Validated candidate inputs ready for one atomic approval commit."""

    chunk_id: str
    text_sha256: str
    profile: str
    source_path: str
    source_start_line: int
    source_end_line: int
    repair_id: str
    take: int
    candidate_audio: Path


def approved_repairs_path(workspace: Path, target: str) -> Path:
    """Return the reviewed replacement store for one target."""
    return workspace.resolve() / ".yakbox" / "repairs" / target / "approved.json"


def load_approved_repairs(workspace: Path, target: str) -> tuple[ApprovedRepair, ...]:
    """Load and validate every approved replacement for a target."""
    path = approved_repairs_path(workspace, target)
    if not path.exists():
        return ()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(
            f"Cannot read approved repairs {path}: {error}"
        ) from error
    if not isinstance(raw, dict) or raw.get("$schema") != schema_uri(
        "audiobook-repairs"
    ):
        raise ValidationError(f"Unsupported approved repair document: {path}")
    values = raw.get("repairs")
    if not isinstance(values, list):
        raise ValidationError(f"Approved repair document has no repairs list: {path}")
    repairs = tuple(_repair_from_dict(value, workspace=workspace) for value in values)
    ids = [repair.chunk_id for repair in repairs]
    if len(ids) != len(set(ids)):
        raise ValidationError(f"Approved repairs contain duplicate chunk IDs: {path}")
    return repairs


def find_approved_repair(
    workspace: Path,
    target: str,
    *,
    chunk_id: str,
    text_sha256: str,
    profile: str,
) -> ApprovedRepair | None:
    """Return a current replacement; stale decisions fail closed by not matching."""
    return next(
        (
            repair
            for repair in load_approved_repairs(workspace, target)
            if repair.chunk_id == chunk_id
            and repair.text_sha256 == text_sha256
            and repair.profile == profile
        ),
        None,
    )


def repair_fingerprint(
    workspace: Path,
    target: str,
    *,
    chunk_id: str,
    text_sha256: str,
    profile: str | None,
) -> str | None:
    """Return a matching reviewed replacement fingerprint for planning."""
    if profile is None:
        return None
    repair = find_approved_repair(
        workspace,
        target,
        chunk_id=chunk_id,
        text_sha256=text_sha256,
        profile=profile,
    )
    return repair.fingerprint if repair is not None else None


def install_approved_repair(
    workspace: Path,
    target: str,
    *,
    chunk_id: str,
    text_sha256: str,
    profile: str,
    source_path: str,
    source_start_line: int,
    source_end_line: int,
    repair_id: str,
    take: int,
    candidate_audio: Path,
) -> ApprovedRepair:
    """Copy an auditioned take into managed storage and atomically approve it."""
    return install_approved_repairs(
        workspace,
        target,
        installs=(
            RepairInstall(
                chunk_id=chunk_id,
                text_sha256=text_sha256,
                profile=profile,
                source_path=source_path,
                source_start_line=source_start_line,
                source_end_line=source_end_line,
                repair_id=repair_id,
                take=take,
                candidate_audio=candidate_audio,
            ),
        ),
    )[0]


def install_approved_repairs(
    workspace: Path,
    target: str,
    *,
    installs: tuple[RepairInstall, ...],
) -> tuple[ApprovedRepair, ...]:
    """Validate, cache, and atomically commit several reviewed replacements."""
    if not installs:
        raise ValidationError("Repair approval batch must not be empty")
    chunk_ids = tuple(item.chunk_id for item in installs)
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValidationError("Repair approval batch contains duplicate chunk IDs")
    prepared = tuple(_prepare_approved_repair(workspace, item) for item in installs)
    replaced = set(chunk_ids)
    existing = [
        item
        for item in load_approved_repairs(workspace, target)
        if item.chunk_id not in replaced
    ]
    existing.extend(prepared)
    path = approved_repairs_path(workspace, target)
    atomic_write_json(
        path,
        {
            **runtime_metadata("audiobook-repairs"),
            "target": target,
            "repairs": [
                item.to_dict(workspace=workspace)
                for item in sorted(existing, key=lambda value: value.chunk_id)
            ],
        },
    )
    return prepared


def _prepare_approved_repair(
    workspace: Path,
    install: RepairInstall,
) -> ApprovedRepair:
    take = install.take
    if take < 1:
        raise ValidationError("Approved repair take must be positive")
    candidate_audio = install.candidate_audio
    if not candidate_audio.is_file():
        raise ArtifactError(f"Repair candidate is missing: {candidate_audio}")
    audio_sha256 = sha256_file(candidate_audio)
    repair_root = workspace.resolve() / ".yakbox" / "cache" / "repairs"
    destination = repair_root / audio_sha256[:2] / f"{audio_sha256}.wav"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(candidate_audio, destination)
    if sha256_file(destination) != audio_sha256:
        raise ArtifactError("Approved repair cache copy failed digest verification")
    return ApprovedRepair(
        chunk_id=install.chunk_id,
        text_sha256=install.text_sha256,
        profile=install.profile,
        audio_path=destination.resolve(),
        audio_sha256=audio_sha256,
        source_path=install.source_path,
        source_start_line=install.source_start_line,
        source_end_line=install.source_end_line,
        repair_id=install.repair_id,
        take=take,
    )


def _repair_from_dict(value: object, *, workspace: Path) -> ApprovedRepair:
    if not isinstance(value, dict):
        raise ValidationError("Approved repair entry must be an object")
    repair_value = cast(dict[str, object], value)
    source = repair_value.get("source")
    if not isinstance(source, dict):
        raise ValidationError("Approved repair source must be an object")
    source_value = cast(dict[str, object], source)
    try:
        audio = safe_child(workspace, workspace / str(repair_value["audio_path"]))
        repair = ApprovedRepair(
            chunk_id=str(repair_value["chunk_id"]),
            text_sha256=str(repair_value["text_sha256"]),
            profile=str(repair_value["profile"]),
            audio_path=audio,
            audio_sha256=str(repair_value["audio_sha256"]),
            source_path=str(source_value["path"]),
            source_start_line=int(cast(int, source_value["start_line"])),
            source_end_line=int(cast(int, source_value["end_line"])),
            repair_id=str(repair_value["repair_id"]),
            take=int(cast(int, repair_value["take"])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValidationError("Approved repair entry has invalid fields") from error
    if not audio.is_file() or sha256_file(audio) != repair.audio_sha256:
        raise ValidationError(f"Approved repair audio is missing or invalid: {audio}")
    fingerprint = repair_value.get("fingerprint")
    if fingerprint is not None and fingerprint != repair.fingerprint:
        raise ValidationError("Approved repair fingerprint is invalid")
    return repair
