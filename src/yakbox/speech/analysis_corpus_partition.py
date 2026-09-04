"""Freeze voice-disjoint calibration and held-out corpus partitions."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_json
from yakbox.contracts import runtime_metadata, schema_uri
from yakbox.errors import ValidationError
from yakbox.speech.analysis_corpus_sources import CorpusSourceInventory
from yakbox.speech.analysis_corpus_truth import ApprovedCorpusTranscripts
from yakbox.speech.analysis_fingerprints import semantic_fingerprint

_DEFAULT_CALIBRATION_VOICE_COUNT = 6
_MINIMUM_HELD_OUT_CLUSTERS = 52
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOP_FIELDS = {
    "$schema",
    "schema_version",
    "yakbox_version",
    "timestamp",
    "fingerprint",
    "source_inventory_fingerprint",
    "transcript_truth_fingerprint",
    "partition_seed_fingerprint",
    "minimum_held_out_clusters",
    "calibration_voice_count",
    "held_out_voice_count",
    "calibration_cluster_count",
    "held_out_cluster_count",
    "assignments",
}
_ASSIGNMENT_FIELDS = {
    "source_window_id",
    "source_passage_group",
    "voice",
    "partition",
    "fingerprint",
}


class QualificationPartition(StrEnum):
    CALIBRATION = "calibration"
    HELD_OUT = "held_out"


@dataclass(frozen=True, slots=True)
class CorpusPartitionAssignment:
    source_window_id: str
    source_passage_group: str
    voice: str
    partition: QualificationPartition

    def __post_init__(self) -> None:
        if not self.source_window_id or not self.source_passage_group or not self.voice:
            raise ValidationError("Corpus partition assignment is incomplete")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-partition-assignment-v1", self)


@dataclass(frozen=True, slots=True)
class FrozenCorpusPartition:
    source_inventory_fingerprint: str
    transcript_truth_fingerprint: str
    partition_seed_fingerprint: str
    minimum_held_out_clusters: int
    assignments: tuple[CorpusPartitionAssignment, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.source_inventory_fingerprint, "source inventory fingerprint"),
            (self.transcript_truth_fingerprint, "transcript truth fingerprint"),
            (self.partition_seed_fingerprint, "partition seed fingerprint"),
        ):
            _require_sha256(value, label)
        identifiers = tuple(item.source_window_id for item in self.assignments)
        passages = tuple(item.source_passage_group for item in self.assignments)
        if (
            self.minimum_held_out_clusters < 1
            or not self.assignments
            or identifiers != tuple(sorted(set(identifiers)))
            or len(passages) != len(set(passages))
        ):
            raise ValidationError("Frozen corpus partition is inconsistent")
        by_voice: dict[str, set[QualificationPartition]] = defaultdict(set)
        for item in self.assignments:
            by_voice[item.voice].add(item.partition)
        if any(len(partitions) != 1 for partitions in by_voice.values()):
            raise ValidationError(
                "A corpus voice cannot cross qualification partitions"
            )
        counts = Counter(item.partition for item in self.assignments)
        if (
            counts[QualificationPartition.CALIBRATION] < 1
            or counts[QualificationPartition.HELD_OUT] < self.minimum_held_out_clusters
        ):
            raise ValidationError("Corpus partition has insufficient clusters")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-partition-v1", self)

    @property
    def calibration_cluster_count(self) -> int:
        return sum(
            item.partition is QualificationPartition.CALIBRATION
            for item in self.assignments
        )

    @property
    def held_out_cluster_count(self) -> int:
        return sum(
            item.partition is QualificationPartition.HELD_OUT
            for item in self.assignments
        )

    @property
    def calibration_voice_count(self) -> int:
        return len(
            {
                item.voice
                for item in self.assignments
                if item.partition is QualificationPartition.CALIBRATION
            }
        )

    @property
    def held_out_voice_count(self) -> int:
        return len(
            {
                item.voice
                for item in self.assignments
                if item.partition is QualificationPartition.HELD_OUT
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("speech-corpus-partition"),
            "fingerprint": self.fingerprint,
            "source_inventory_fingerprint": self.source_inventory_fingerprint,
            "transcript_truth_fingerprint": self.transcript_truth_fingerprint,
            "partition_seed_fingerprint": self.partition_seed_fingerprint,
            "minimum_held_out_clusters": self.minimum_held_out_clusters,
            "calibration_voice_count": self.calibration_voice_count,
            "held_out_voice_count": self.held_out_voice_count,
            "calibration_cluster_count": self.calibration_cluster_count,
            "held_out_cluster_count": self.held_out_cluster_count,
            "assignments": [
                {
                    "source_window_id": item.source_window_id,
                    "source_passage_group": item.source_passage_group,
                    "voice": item.voice,
                    "partition": item.partition.value,
                    "fingerprint": item.fingerprint,
                }
                for item in self.assignments
            ],
        }


def freeze_corpus_partition(
    inventory: CorpusSourceInventory,
    transcripts: ApprovedCorpusTranscripts,
    *,
    partition_seed: str,
    calibration_voice_count: int = _DEFAULT_CALIBRATION_VOICE_COUNT,
    minimum_held_out_clusters: int = _MINIMUM_HELD_OUT_CLUSTERS,
) -> FrozenCorpusPartition:
    """Assign whole voices deterministically so held-out audio stays unseen."""
    seed = partition_seed.strip()
    if not seed:
        raise ValidationError("Corpus partition seed cannot be empty")
    if transcripts.source_inventory_fingerprint != inventory.fingerprint:
        raise ValidationError("Transcript truth does not match the source inventory")
    transcript_ids = tuple(item.source_window_id for item in transcripts.cases)
    inventory_ids = tuple(item.source_window_id for item in inventory.windows)
    if transcript_ids != inventory_ids:
        raise ValidationError("Transcript truth does not cover the source inventory")
    voices = tuple(sorted({item.voice for item in inventory.windows}))
    if calibration_voice_count < 1 or calibration_voice_count >= len(voices):
        raise ValidationError("Calibration voice count cannot create both partitions")
    ranked_voices = tuple(
        sorted(
            voices,
            key=lambda voice: semantic_fingerprint(
                "speech-corpus-partition-rank-v1",
                (seed, transcripts.fingerprint, voice),
            ),
        )
    )
    calibration_voices = frozenset(ranked_voices[:calibration_voice_count])
    assignments = tuple(
        CorpusPartitionAssignment(
            item.source_window_id,
            item.source_passage_group,
            item.voice,
            (
                QualificationPartition.CALIBRATION
                if item.voice in calibration_voices
                else QualificationPartition.HELD_OUT
            ),
        )
        for item in inventory.windows
    )
    return FrozenCorpusPartition(
        inventory.fingerprint,
        transcripts.fingerprint,
        semantic_fingerprint("speech-corpus-partition-seed-v1", seed),
        minimum_held_out_clusters,
        assignments,
    )


def write_frozen_corpus_partition(
    path: Path,
    partition: FrozenCorpusPartition,
) -> None:
    atomic_write_json(path, partition.to_dict())


def load_frozen_corpus_partition(
    path: Path,
    *,
    inventory: CorpusSourceInventory,
    transcripts: ApprovedCorpusTranscripts,
) -> FrozenCorpusPartition:
    """Load a frozen split and reject stale truth or changed assignments."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError("Cannot read frozen corpus partition") from error
    if not isinstance(raw, dict) or set(raw) != _TOP_FIELDS:
        raise ValidationError("Frozen corpus partition fields are invalid")
    if (
        raw.get("$schema") != schema_uri("speech-corpus-partition")
        or raw.get("schema_version") != 1
        or not isinstance(raw.get("yakbox_version"), str)
        or not _utc_timestamp(raw.get("timestamp"))
        or raw.get("source_inventory_fingerprint") != inventory.fingerprint
        or raw.get("transcript_truth_fingerprint") != transcripts.fingerprint
    ):
        raise ValidationError("Frozen corpus partition metadata is invalid")
    values = raw.get("assignments")
    if not isinstance(values, list):
        raise ValidationError("Frozen corpus assignments are invalid")
    assignments = tuple(_load_assignment(item) for item in values)
    partition = FrozenCorpusPartition(
        _text(raw, "source_inventory_fingerprint"),
        _text(raw, "transcript_truth_fingerprint"),
        _text(raw, "partition_seed_fingerprint"),
        _integer(raw, "minimum_held_out_clusters"),
        assignments,
    )
    expected = {
        item.source_window_id: (item.source_passage_group, item.voice)
        for item in inventory.windows
    }
    actual = {
        item.source_window_id: (item.source_passage_group, item.voice)
        for item in assignments
    }
    if (
        expected != actual
        or raw.get("fingerprint") != partition.fingerprint
        or raw.get("calibration_voice_count") != partition.calibration_voice_count
        or raw.get("held_out_voice_count") != partition.held_out_voice_count
        or raw.get("calibration_cluster_count") != partition.calibration_cluster_count
        or raw.get("held_out_cluster_count") != partition.held_out_cluster_count
    ):
        raise ValidationError("Frozen corpus partition identity differs")
    return partition


def _load_assignment(value: object) -> CorpusPartitionAssignment:
    if not isinstance(value, dict) or set(value) != _ASSIGNMENT_FIELDS:
        raise ValidationError("Frozen corpus assignment fields are invalid")
    raw = cast(dict[str, object], value)
    try:
        partition = QualificationPartition(_text(raw, "partition"))
    except ValueError as error:
        raise ValidationError(
            "Frozen corpus assignment partition is invalid"
        ) from error
    assignment = CorpusPartitionAssignment(
        _text(raw, "source_window_id"),
        _text(raw, "source_passage_group"),
        _text(raw, "voice"),
        partition,
    )
    if raw.get("fingerprint") != assignment.fingerprint:
        raise ValidationError("Frozen corpus assignment fingerprint differs")
    return assignment


def _text(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Frozen corpus partition {key} must be text")
    return value


def _integer(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"Frozen corpus partition {key} must be an integer")
    return value


def _utc_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.utcoffset() == timedelta(0)


def _require_sha256(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label.capitalize()} must be a SHA-256")


__all__ = [
    "CorpusPartitionAssignment",
    "FrozenCorpusPartition",
    "QualificationPartition",
    "freeze_corpus_partition",
    "load_frozen_corpus_partition",
    "write_frozen_corpus_partition",
]
