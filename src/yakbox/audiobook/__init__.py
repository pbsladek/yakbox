"""Public audiobook planning, build, and artifact APIs."""

from yakbox.audiobook.artifacts import (
    ArtifactKind,
    ArtifactRecord,
    InventoryReport,
    inventory_artifacts,
    repair_artifact_metadata,
)
from yakbox.audiobook.build import (
    BuildChangeSummary,
    BuildPreflight,
    BuildResult,
    ReleaseCheck,
    assemble_release,
    audition_audiobook,
    build_audiobook,
    check_release,
    preflight_audiobook_build,
    preview_audiobook,
)
from yakbox.audiobook.cleanup import (
    CleanupPlan,
    apply_cleanup,
    plan_cleanup,
    purge_trash,
    restore_trash,
)
from yakbox.audiobook.manifest import AudiobookManifest, load_manifest
from yakbox.audiobook.planner import BuildPlan, plan_audiobook
from yakbox.audiobook.shards import (
    ShardManifest,
    export_shard_manifests,
    verify_shard_manifests,
)
from yakbox.audiobook.sources import NormalizedDocument, normalize_sources

__all__ = [
    "ArtifactKind",
    "ArtifactRecord",
    "AudiobookManifest",
    "BuildChangeSummary",
    "BuildPlan",
    "BuildPreflight",
    "BuildResult",
    "CleanupPlan",
    "InventoryReport",
    "NormalizedDocument",
    "ReleaseCheck",
    "ShardManifest",
    "apply_cleanup",
    "assemble_release",
    "audition_audiobook",
    "build_audiobook",
    "check_release",
    "export_shard_manifests",
    "inventory_artifacts",
    "load_manifest",
    "normalize_sources",
    "plan_audiobook",
    "plan_cleanup",
    "preflight_audiobook_build",
    "preview_audiobook",
    "purge_trash",
    "repair_artifact_metadata",
    "restore_trash",
    "verify_shard_manifests",
]
