"""Draft and validate two-pass word-boundary truth for a corpus subset."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_json
from yakbox.contracts import runtime_metadata, schema_uri
from yakbox.errors import ValidationError
from yakbox.speech.analysis_corpus_partition import (
    CorpusPartitionAssignment,
    FrozenCorpusPartition,
    QualificationPartition,
)
from yakbox.speech.analysis_corpus_truth import (
    ApprovedCorpusTranscripts,
    ApprovedTranscriptCase,
)
from yakbox.speech.analysis_fingerprints import semantic_fingerprint, text_fingerprint
from yakbox.speech.analysis_models import AlignmentPurpose, ForcedAlignmentResult

_DEFAULT_REVIEW_CASE_COUNT = 12
_MINIMUM_REVIEW_CASE_COUNT = 2
_REVIEW_PASS_COUNT = 2
_SHA256 = re.compile(r"[0-9a-f]{64}")
_AUTHORING_FIELDS = {
    "$schema",
    "schema_version",
    "yakbox_version",
    "timestamp",
    "fingerprint",
    "transcript_truth_fingerprint",
    "partition_fingerprint",
    "review_status",
    "case_count",
    "cases",
}
_CASE_FIELDS = {
    "case_id",
    "source_passage_group",
    "voice",
    "partition",
    "audio_digest",
    "sample_rate",
    "frame_count",
    "alignment_fingerprint",
    "tokens",
    "passes",
    "accepted_boundaries",
    "adjudicator_fingerprint",
    "review_status",
    "fingerprint",
}
_TOKEN_FIELDS = {
    "token_index",
    "text",
    "text_hash",
    "proposed_start_frame",
    "proposed_end_frame",
}
_PASS_FIELDS = {"pass_index", "reviewer_fingerprint", "boundaries"}
_BOUNDARY_FIELDS = {"token_index", "start_frame", "end_frame"}


@dataclass(frozen=True, slots=True)
class ReviewedBoundary:
    token_index: int
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if (
            self.token_index < 0
            or self.start_frame < 0
            or self.end_frame <= self.start_frame
        ):
            raise ValidationError("Reviewed word boundary is invalid")


@dataclass(frozen=True, slots=True)
class BoundaryReviewPass:
    pass_index: int
    reviewer_fingerprint: str
    boundaries: tuple[ReviewedBoundary, ...]

    def __post_init__(self) -> None:
        if self.pass_index not in {1, 2}:
            raise ValidationError("Boundary review pass index must be one or two")
        _require_sha256(self.reviewer_fingerprint, "boundary reviewer fingerprint")
        _validate_boundary_sequence(self.boundaries)

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-boundary-review-pass-v1", self)


@dataclass(frozen=True, slots=True)
class ApprovedBoundaryCase:
    case_id: str
    source_passage_group: str
    voice: str
    partition: QualificationPartition
    audio_digest: str
    sample_rate: int
    frame_count: int
    token_hashes: tuple[str, ...]
    alignment_fingerprint: str
    draft_case_fingerprint: str
    passes: tuple[BoundaryReviewPass, BoundaryReviewPass]
    accepted_boundaries: tuple[ReviewedBoundary, ...]
    adjudicator_fingerprint: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.source_passage_group or not self.voice:
            raise ValidationError("Approved boundary case identity is incomplete")
        for value, label in (
            (self.audio_digest, "boundary audio digest"),
            (self.alignment_fingerprint, "boundary alignment fingerprint"),
            (self.draft_case_fingerprint, "boundary draft-case fingerprint"),
            (self.adjudicator_fingerprint, "boundary adjudicator fingerprint"),
            *((value, "boundary token hash") for value in self.token_hashes),
        ):
            _require_sha256(value, label)
        if self.sample_rate < 1 or self.frame_count < 1 or not self.token_hashes:
            raise ValidationError("Approved boundary case audio shape is invalid")
        if tuple(item.pass_index for item in self.passes) != (1, 2):
            raise ValidationError("Approved boundary case requires two ordered passes")
        for review_pass in self.passes:
            _validate_case_boundaries(
                review_pass.boundaries,
                token_count=len(self.token_hashes),
                frame_count=self.frame_count,
            )
        _validate_case_boundaries(
            self.accepted_boundaries,
            token_count=len(self.token_hashes),
            frame_count=self.frame_count,
        )

    @property
    def accepted_boundaries_fingerprint(self) -> str:
        return semantic_fingerprint(
            "speech-corpus-accepted-boundaries-v1",
            self.accepted_boundaries,
        )

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-approved-boundary-case-v1", self)


@dataclass(frozen=True, slots=True)
class ApprovedCorpusBoundaryTruth:
    transcript_truth_fingerprint: str
    partition_fingerprint: str
    draft_fingerprint: str
    cases: tuple[ApprovedBoundaryCase, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.transcript_truth_fingerprint, "transcript truth fingerprint"),
            (self.partition_fingerprint, "partition fingerprint"),
            (self.draft_fingerprint, "boundary draft fingerprint"),
        ):
            _require_sha256(value, label)
        identifiers = tuple(item.case_id for item in self.cases)
        if not self.cases or identifiers != tuple(sorted(set(identifiers))):
            raise ValidationError("Approved corpus boundary truth is inconsistent")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-boundary-truth-v1", self)

    def to_report_dict(self) -> dict[str, object]:
        """Serialize approved boundary truth without transcript plaintext."""
        return {
            **runtime_metadata("speech-corpus-boundary-truth"),
            "fingerprint": self.fingerprint,
            "transcript_truth_fingerprint": self.transcript_truth_fingerprint,
            "partition_fingerprint": self.partition_fingerprint,
            "draft_fingerprint": self.draft_fingerprint,
            "case_count": len(self.cases),
            "cases": [_boundary_case_report(item) for item in self.cases],
        }


def select_boundary_review_case_ids(
    partition: FrozenCorpusPartition,
    *,
    case_count: int = _DEFAULT_REVIEW_CASE_COUNT,
) -> tuple[str, ...]:
    """Select a deterministic subset balanced across partitions and voices."""
    voices: dict[QualificationPartition, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in partition.assignments:
        voices[item.partition][item.voice].append(item.source_window_id)
    if case_count < _MINIMUM_REVIEW_CASE_COUNT:
        raise ValidationError("Boundary review requires at least two cases")
    calibration_target = min(
        case_count // 2,
        len(voices[QualificationPartition.CALIBRATION]),
    )
    held_out_target = case_count - calibration_target
    if held_out_target > len(voices[QualificationPartition.HELD_OUT]):
        raise ValidationError("Boundary review subset lacks enough distinct voices")
    selected = (
        *_select_partition_cases(
            voices[QualificationPartition.CALIBRATION],
            count=calibration_target,
            partition=partition,
        ),
        *_select_partition_cases(
            voices[QualificationPartition.HELD_OUT],
            count=held_out_target,
            partition=partition,
        ),
    )
    return tuple(sorted(selected))


def build_boundary_review_authoring(
    transcripts: ApprovedCorpusTranscripts,
    partition: FrozenCorpusPartition,
    alignments: dict[str, ForcedAlignmentResult],
    *,
    case_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Build a private two-pass worksheet from authoritative Qwen timing."""
    if partition.transcript_truth_fingerprint != transcripts.fingerprint:
        raise ValidationError("Boundary partition and transcript truth differ")
    selected = case_ids or select_boundary_review_case_ids(partition)
    if not selected or selected != tuple(sorted(set(selected))):
        raise ValidationError("Boundary review case selection is inconsistent")
    transcripts_by_id = {item.source_window_id: item for item in transcripts.cases}
    assignments = {item.source_window_id: item for item in partition.assignments}
    cases: list[dict[str, object]] = []
    immutable: list[object] = []
    for case_id in selected:
        try:
            transcript = transcripts_by_id[case_id]
            assignment = assignments[case_id]
            alignment = alignments[case_id]
        except KeyError as error:
            raise ValidationError(
                "Boundary review case evidence is incomplete"
            ) from error
        case, identity = _draft_boundary_case(transcript, assignment, alignment)
        cases.append(case)
        immutable.append(identity)
    draft_fingerprint = _boundary_draft_fingerprint(
        transcripts.fingerprint,
        partition.fingerprint,
        tuple(immutable),
    )
    return {
        **runtime_metadata("speech-corpus-boundary-authoring"),
        "fingerprint": draft_fingerprint,
        "transcript_truth_fingerprint": transcripts.fingerprint,
        "partition_fingerprint": partition.fingerprint,
        "review_status": "pending",
        "case_count": len(cases),
        "cases": cases,
    }


def write_boundary_review_authoring(path: Path, authoring: dict[str, object]) -> None:
    atomic_write_json(path, authoring)


def load_approved_corpus_boundaries(
    path: Path,
    *,
    transcripts: ApprovedCorpusTranscripts,
    partition: FrozenCorpusPartition,
) -> ApprovedCorpusBoundaryTruth:
    """Load fully adjudicated two-pass boundary truth and record disagreement."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError("Cannot read corpus boundary authoring source") from error
    if not isinstance(raw, dict) or set(raw) != _AUTHORING_FIELDS:
        raise ValidationError("Corpus boundary authoring fields are invalid")
    if not _approved_authoring_metadata(
        cast(dict[str, object], raw),
        transcripts=transcripts,
        partition=partition,
    ):
        raise ValidationError("Corpus boundary approval metadata is invalid")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list):
        raise ValidationError("Corpus boundary approval cases are invalid")
    transcript_by_id = {item.source_window_id: item for item in transcripts.cases}
    assignment_by_id = {item.source_window_id: item for item in partition.assignments}
    approved: list[ApprovedBoundaryCase] = []
    immutable: list[object] = []
    for value in cases_raw:
        case, identity = _load_approved_boundary_case(
            value,
            transcripts=transcript_by_id,
            assignments=assignment_by_id,
        )
        approved.append(case)
        immutable.append(identity)
    draft_fingerprint = _boundary_draft_fingerprint(
        transcripts.fingerprint,
        partition.fingerprint,
        tuple(immutable),
    )
    approved_sorted = tuple(sorted(approved, key=lambda item: item.case_id))
    if (
        raw.get("case_count") != len(approved_sorted)
        or raw.get("fingerprint") != draft_fingerprint
    ):
        raise ValidationError("Corpus boundary approval identity differs")
    return ApprovedCorpusBoundaryTruth(
        transcripts.fingerprint,
        partition.fingerprint,
        draft_fingerprint,
        approved_sorted,
    )


def write_approved_boundary_report(
    path: Path,
    truth: ApprovedCorpusBoundaryTruth,
) -> None:
    atomic_write_json(path, truth.to_report_dict())


def _select_partition_cases(
    voices: dict[str, list[str]],
    *,
    count: int,
    partition: FrozenCorpusPartition,
) -> tuple[str, ...]:
    ranked_voices = sorted(
        voices,
        key=lambda voice: semantic_fingerprint(
            "speech-corpus-boundary-voice-rank-v1",
            (partition.fingerprint, voice),
        ),
    )
    selected = [
        min(
            voices[voice],
            key=lambda case_id: semantic_fingerprint(
                "speech-corpus-boundary-case-rank-v1",
                (partition.fingerprint, case_id),
            ),
        )
        for voice in ranked_voices[:count]
    ]
    return tuple(selected)


def _draft_boundary_case(
    transcript: ApprovedTranscriptCase,
    assignment: CorpusPartitionAssignment,
    alignment: ForcedAlignmentResult,
) -> tuple[dict[str, object], tuple[object, ...]]:
    if transcript.source_window_id != assignment.source_window_id:
        raise ValidationError("Boundary review source identities differ")
    token_hashes = tuple(
        text_fingerprint(token) for token in transcript.accepted_tokens
    )
    if (
        alignment.engine != "qwen-forced"
        or alignment.purpose is not AlignmentPurpose.VERIFIED_TARGET
        or alignment.span.audio_digest != transcript.audio_digest
        or alignment.expected_lexical_span_hash != transcript.accepted_tokens_hash
        or alignment.coverage_ratio != 1
        or alignment.issues
        or tuple(item.text_hash for item in alignment.units) != token_hashes
    ):
        raise ValidationError("Boundary review forced alignment is not authoritative")
    tokens = [
        {
            "token_index": index,
            "text": text,
            "text_hash": token_hash,
            "proposed_start_frame": unit.start_frame,
            "proposed_end_frame": unit.end_frame,
        }
        for index, (text, token_hash, unit) in enumerate(
            zip(transcript.accepted_tokens, token_hashes, alignment.units, strict=True)
        )
    ]
    identity = (
        transcript.source_window_id,
        transcript.source_passage_group,
        transcript.voice,
        assignment.partition.value,
        transcript.audio_digest,
        alignment.span.sample_rate,
        alignment.span.end_frame - alignment.span.start_frame,
        token_hashes,
        tuple((unit.start_frame, unit.end_frame) for unit in alignment.units),
        alignment.fingerprint,
    )
    fingerprint = semantic_fingerprint("speech-corpus-boundary-draft-case-v1", identity)
    return {
        "case_id": transcript.source_window_id,
        "source_passage_group": transcript.source_passage_group,
        "voice": transcript.voice,
        "partition": assignment.partition.value,
        "audio_digest": transcript.audio_digest,
        "sample_rate": alignment.span.sample_rate,
        "frame_count": alignment.span.end_frame - alignment.span.start_frame,
        "alignment_fingerprint": alignment.fingerprint,
        "tokens": tokens,
        "passes": [
            {"pass_index": 1, "reviewer_fingerprint": "", "boundaries": []},
            {"pass_index": 2, "reviewer_fingerprint": "", "boundaries": []},
        ],
        "accepted_boundaries": [],
        "adjudicator_fingerprint": "",
        "review_status": "pending",
        "fingerprint": fingerprint,
    }, identity


def _load_approved_boundary_case(
    value: object,
    *,
    transcripts: dict[str, ApprovedTranscriptCase],
    assignments: dict[str, CorpusPartitionAssignment],
) -> tuple[ApprovedBoundaryCase, tuple[object, ...]]:
    if not isinstance(value, dict) or set(value) != _CASE_FIELDS:
        raise ValidationError("Corpus boundary approval case fields are invalid")
    raw = cast(dict[str, object], value)
    case_id = _text(raw, "case_id")
    try:
        transcript = transcripts[case_id]
        assignment = assignments[case_id]
    except KeyError as error:
        raise ValidationError("Corpus boundary approval case is unknown") from error
    tokens, proposals = _load_tokens(raw.get("tokens"), transcript=transcript)
    sample_rate = _integer(raw, "sample_rate")
    frame_count = _integer(raw, "frame_count")
    identity = (
        case_id,
        _text(raw, "source_passage_group"),
        _text(raw, "voice"),
        _text(raw, "partition"),
        _text(raw, "audio_digest"),
        sample_rate,
        frame_count,
        tokens,
        proposals,
        _text(raw, "alignment_fingerprint"),
    )
    draft_case_fingerprint = semantic_fingerprint(
        "speech-corpus-boundary-draft-case-v1", identity
    )
    _validate_case_identity(
        raw,
        transcript=transcript,
        assignment=assignment,
        fingerprint=draft_case_fingerprint,
    )
    passes = _load_passes(
        raw.get("passes"), token_count=len(tokens), frame_count=frame_count
    )
    accepted = _load_boundaries(raw.get("accepted_boundaries"))
    approved = ApprovedBoundaryCase(
        case_id,
        _text(raw, "source_passage_group"),
        _text(raw, "voice"),
        QualificationPartition(_text(raw, "partition")),
        _text(raw, "audio_digest"),
        sample_rate,
        frame_count,
        tokens,
        _text(raw, "alignment_fingerprint"),
        draft_case_fingerprint,
        passes,
        accepted,
        _text(raw, "adjudicator_fingerprint"),
    )
    return approved, identity


def _load_tokens(
    value: object,
    *,
    transcript: ApprovedTranscriptCase,
) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    if not isinstance(value, list):
        raise ValidationError("Corpus boundary approval tokens are invalid")
    hashes: list[str] = []
    proposals: list[tuple[int, int]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != _TOKEN_FIELDS:
            raise ValidationError("Corpus boundary approval token fields are invalid")
        raw = cast(dict[str, object], item)
        text = _text(raw, "text")
        token_hash = _text(raw, "text_hash")
        if (
            raw.get("token_index") != index
            or index >= len(transcript.accepted_tokens)
            or text != transcript.accepted_tokens[index]
            or token_hash != text_fingerprint(text)
        ):
            raise ValidationError("Corpus boundary approval token identity differs")
        hashes.append(token_hash)
        proposals.append(
            (_integer(raw, "proposed_start_frame"), _integer(raw, "proposed_end_frame"))
        )
    if len(hashes) != len(transcript.accepted_tokens):
        raise ValidationError("Corpus boundary approval token count differs")
    return tuple(hashes), tuple(proposals)


def _load_passes(
    value: object,
    *,
    token_count: int,
    frame_count: int,
) -> tuple[BoundaryReviewPass, BoundaryReviewPass]:
    if not isinstance(value, list) or len(value) != _REVIEW_PASS_COUNT:
        raise ValidationError("Corpus boundary approval requires two review passes")
    passes: list[BoundaryReviewPass] = []
    for expected_index, item in enumerate(value, start=1):
        if not isinstance(item, dict) or set(item) != _PASS_FIELDS:
            raise ValidationError("Corpus boundary review pass fields are invalid")
        raw = cast(dict[str, object], item)
        review_pass = BoundaryReviewPass(
            _integer(raw, "pass_index"),
            _text(raw, "reviewer_fingerprint"),
            _load_boundaries(raw.get("boundaries")),
        )
        if review_pass.pass_index != expected_index:
            raise ValidationError("Corpus boundary review passes are out of order")
        _validate_case_boundaries(
            review_pass.boundaries,
            token_count=token_count,
            frame_count=frame_count,
        )
        passes.append(review_pass)
    return cast(tuple[BoundaryReviewPass, BoundaryReviewPass], tuple(passes))


def _load_boundaries(value: object) -> tuple[ReviewedBoundary, ...]:
    if not isinstance(value, list):
        raise ValidationError("Corpus boundary review boundaries are invalid")
    boundaries: list[ReviewedBoundary] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != _BOUNDARY_FIELDS:
            raise ValidationError("Corpus boundary review boundary fields are invalid")
        raw = cast(dict[str, object], item)
        boundaries.append(
            ReviewedBoundary(
                _integer(raw, "token_index"),
                _integer(raw, "start_frame"),
                _integer(raw, "end_frame"),
            )
        )
    return tuple(boundaries)


def _validate_case_identity(
    raw: dict[str, object],
    *,
    transcript: ApprovedTranscriptCase,
    assignment: CorpusPartitionAssignment,
    fingerprint: str,
) -> None:
    if (
        raw.get("review_status") != "approved"
        or raw.get("fingerprint") != fingerprint
        or raw.get("source_passage_group") != transcript.source_passage_group
        or raw.get("voice") != transcript.voice
        or raw.get("partition") != assignment.partition.value
        or raw.get("audio_digest") != transcript.audio_digest
    ):
        raise ValidationError("Corpus boundary approval case identity differs")


def _approved_authoring_metadata(
    raw: dict[str, object],
    *,
    transcripts: ApprovedCorpusTranscripts,
    partition: FrozenCorpusPartition,
) -> bool:
    return (
        raw.get("$schema") == schema_uri("speech-corpus-boundary-authoring")
        and raw.get("schema_version") == 1
        and isinstance(raw.get("yakbox_version"), str)
        and _utc_timestamp(raw.get("timestamp"))
        and raw.get("review_status") == "approved"
        and raw.get("transcript_truth_fingerprint") == transcripts.fingerprint
        and raw.get("partition_fingerprint") == partition.fingerprint
    )


def _boundary_draft_fingerprint(
    transcript_truth_fingerprint: str,
    partition_fingerprint: str,
    cases: tuple[object, ...],
) -> str:
    return semantic_fingerprint(
        "speech-corpus-boundary-draft-v1",
        (transcript_truth_fingerprint, partition_fingerprint, cases),
    )


def _validate_boundary_sequence(boundaries: tuple[ReviewedBoundary, ...]) -> None:
    if not boundaries:
        raise ValidationError("Boundary review pass cannot be empty")
    previous_end = 0
    for expected_index, item in enumerate(boundaries):
        if item.token_index != expected_index or item.start_frame < previous_end:
            raise ValidationError("Reviewed boundaries must be ordered and monotonic")
        previous_end = item.end_frame


def _validate_case_boundaries(
    boundaries: tuple[ReviewedBoundary, ...],
    *,
    token_count: int,
    frame_count: int,
) -> None:
    _validate_boundary_sequence(boundaries)
    if len(boundaries) != token_count or boundaries[-1].end_frame > frame_count:
        raise ValidationError("Reviewed boundaries do not cover the approved tokens")


def _boundary_case_report(case: ApprovedBoundaryCase) -> dict[str, object]:
    first, second = case.passes
    disagreements = [
        {
            "token_index": index,
            "start_difference_frames": abs(one.start_frame - two.start_frame),
            "end_difference_frames": abs(one.end_frame - two.end_frame),
        }
        for index, (one, two) in enumerate(
            zip(first.boundaries, second.boundaries, strict=True)
        )
    ]
    return {
        "case_id": case.case_id,
        "source_passage_group": case.source_passage_group,
        "voice": case.voice,
        "partition": case.partition.value,
        "audio_digest": case.audio_digest,
        "sample_rate": case.sample_rate,
        "frame_count": case.frame_count,
        "token_hashes": list(case.token_hashes),
        "alignment_fingerprint": case.alignment_fingerprint,
        "draft_case_fingerprint": case.draft_case_fingerprint,
        "review_passes": [
            {
                "pass_index": item.pass_index,
                "reviewer_fingerprint": item.reviewer_fingerprint,
                "fingerprint": item.fingerprint,
            }
            for item in case.passes
        ],
        "boundary_disagreement": disagreements,
        "accepted_boundaries_fingerprint": case.accepted_boundaries_fingerprint,
        "adjudicator_fingerprint": case.adjudicator_fingerprint,
        "fingerprint": case.fingerprint,
    }


def _text(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Corpus boundary {key} must be text")
    return value


def _integer(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"Corpus boundary {key} must be an integer")
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
    "ApprovedBoundaryCase",
    "ApprovedCorpusBoundaryTruth",
    "BoundaryReviewPass",
    "ReviewedBoundary",
    "build_boundary_review_authoring",
    "load_approved_corpus_boundaries",
    "select_boundary_review_case_ids",
    "write_approved_boundary_report",
    "write_boundary_review_authoring",
]
