from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from yakbox._files import atomic_write_json, sha256_file
from yakbox.audiobook.planner import BuildPlan, shard_plan
from yakbox.contracts import runtime_metadata
from yakbox.errors import BuildError


@dataclass(frozen=True, slots=True)
class ShardManifest:
    """Portable assignment of build nodes and artifacts to one worker shard."""

    schema_version: int
    plan_fingerprint: str
    target: str
    index: int
    count: int
    node_ids: tuple[str, ...]
    artifact_paths: tuple[Path, ...]

    def to_dict(self, *, root: Path) -> dict[str, object]:
        """Serialize shard paths relative to the build root."""
        return {
            **runtime_metadata("audiobook-shard"),
            "plan_fingerprint": self.plan_fingerprint,
            "target": self.target,
            "index": self.index,
            "count": self.count,
            "node_ids": list(self.node_ids),
            "artifact_paths": [
                path.relative_to(root).as_posix()
                if path.is_relative_to(root)
                else str(path)
                for path in self.artifact_paths
            ],
        }


def export_shard_manifests(
    plan: BuildPlan,
    directory: Path,
    *,
    count: int,
    root: Path,
) -> tuple[Path, ...]:
    """Write deterministic non-overlapping shard manifests for a build plan."""
    shards = shard_plan(plan, count)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, nodes in enumerate(shards, 1):
        manifest = ShardManifest(
            schema_version=1,
            plan_fingerprint=plan.fingerprint,
            target=plan.target,
            index=index,
            count=len(shards),
            node_ids=tuple(node.id for node in nodes),
            artifact_paths=tuple(node.output for node in nodes),
        )
        path = directory / f"shard-{index:04d}-of-{len(shards):04d}.json"
        atomic_write_json(path, manifest.to_dict(root=root), overwrite=False)
        paths.append(path)
    return tuple(paths)


def verify_shard_manifests(
    paths: tuple[Path, ...], *, root: Path
) -> tuple[ShardManifest, ...]:
    """Verify shard identity, completeness, ordering, and artifact containment."""
    if not paths:
        raise BuildError("No shard manifests were supplied")
    shards = tuple(_load_shard(path, root=root) for path in paths)
    fingerprints = {shard.plan_fingerprint for shard in shards}
    targets = {shard.target for shard in shards}
    counts = {shard.count for shard in shards}
    if len(fingerprints) != 1 or len(targets) != 1 or len(counts) != 1:
        raise BuildError("Shard manifests are not compatible")
    expected_count = shards[0].count
    indexes = {shard.index for shard in shards}
    if indexes != set(range(1, expected_count + 1)):
        raise BuildError("Shard manifest set is incomplete")
    node_ids = [node_id for shard in shards for node_id in shard.node_ids]
    if len(node_ids) != len(set(node_ids)):
        raise BuildError("Shard manifests contain duplicate nodes")
    for shard in shards:
        for artifact in shard.artifact_paths:
            metadata = artifact.with_suffix(f"{artifact.suffix}.artifact.json")
            if not artifact.is_file() or not metadata.is_file():
                raise BuildError(f"Shard artifact is incomplete: {artifact}")
            try:
                raw = json.loads(metadata.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise BuildError(f"Invalid artifact manifest {metadata}") from error
            if raw.get("sha256") != sha256_file(artifact):
                raise BuildError(f"Shard artifact digest mismatch: {artifact}")
    return tuple(sorted(shards, key=lambda shard: shard.index))


def _load_shard(path: Path, *, root: Path) -> ShardManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BuildError(f"Cannot read shard manifest {path}: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise BuildError(f"Unsupported shard manifest {path}")
    node_ids = raw.get("node_ids")
    artifact_paths = raw.get("artifact_paths")
    if not isinstance(node_ids, list) or not isinstance(artifact_paths, list):
        raise BuildError(f"Invalid shard manifest {path}")
    return ShardManifest(
        schema_version=1,
        plan_fingerprint=str(raw.get("plan_fingerprint", "")),
        target=str(raw.get("target", "")),
        index=_required_int(raw, "index"),
        count=_required_int(raw, "count"),
        node_ids=tuple(str(item) for item in node_ids),
        artifact_paths=tuple((root / str(item)).resolve() for item in artifact_paths),
    )


def _required_int(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise BuildError(f"Shard field {key!r} must be an integer")
    return value
