"""Durable, evaluation-bound review artifacts for speech qualification."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_bytes
from yakbox.contracts import runtime_metadata, schema_uri
from yakbox.errors import ValidationError
from yakbox.speech.analysis_evaluation import (
    ReviewDecision,
    ReviewerDisposition,
    ReviewKind,
    SpeechAnalysisEvaluation,
    approve_calibration,
)
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_models import ConsensusOutcome
from yakbox.speech.analysis_policy import CalibrationTable

_SHA256_LENGTH = 64


class QualificationReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"


@dataclass(frozen=True, slots=True)
class QualificationReviewEntry:
    case_id: str
    kind: ReviewKind
    pass_index: int
    reviewer_fingerprint: str | None
    decision: ReviewDecision | None

    def __post_init__(self) -> None:
        if not self.case_id or self.pass_index < 1:
            raise ValidationError("Qualification review entry identity is invalid")
        if (self.reviewer_fingerprint is None) != (self.decision is None):
            raise ValidationError(
                "Qualification review decision and reviewer must be completed together"
            )
        if self.reviewer_fingerprint is not None:
            _require_sha256(self.reviewer_fingerprint, "reviewer fingerprint")

    @property
    def identity(self) -> tuple[str, ReviewKind, int]:
        return self.case_id, self.kind, self.pass_index


@dataclass(frozen=True, slots=True)
class QualificationReview:
    schema_version: int
    yakbox_version: str
    timestamp: str
    status: QualificationReviewStatus
    evaluation_fingerprint: str
    corpus_fingerprint: str
    policy_fingerprint: str
    protocol_fingerprint: str
    order_seed_fingerprint: str
    entries: tuple[QualificationReviewEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.yakbox_version or not self.entries:
            raise ValidationError("Qualification review must be versioned and nonempty")
        try:
            created = datetime.fromisoformat(self.timestamp)
        except ValueError as error:
            raise ValidationError(
                "Qualification review timestamp is invalid"
            ) from error
        if created.utcoffset() != timedelta(0):
            raise ValidationError("Qualification review timestamp must be UTC")
        for value, label in (
            (self.evaluation_fingerprint, "evaluation fingerprint"),
            (self.corpus_fingerprint, "corpus fingerprint"),
            (self.policy_fingerprint, "policy fingerprint"),
            (self.protocol_fingerprint, "protocol fingerprint"),
            (self.order_seed_fingerprint, "review-order seed fingerprint"),
        ):
            _require_sha256(value, label)
        identities = tuple(entry.identity for entry in self.entries)
        if len(identities) != len(set(identities)):
            raise ValidationError("Qualification review entries must be unique")
        if self.status is QualificationReviewStatus.APPROVED and any(
            entry.decision is None for entry in self.entries
        ):
            raise ValidationError("Approved qualification review is incomplete")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "speech-qualification-review-v1",
            {
                "schema_version": self.schema_version,
                "status": self.status,
                "evaluation_fingerprint": self.evaluation_fingerprint,
                "corpus_fingerprint": self.corpus_fingerprint,
                "policy_fingerprint": self.policy_fingerprint,
                "protocol_fingerprint": self.protocol_fingerprint,
                "order_seed_fingerprint": self.order_seed_fingerprint,
                "entries": tuple(
                    sorted(self.entries, key=lambda entry: entry.identity)
                ),
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "$schema": schema_uri("speech-qualification-review"),
            "schema_version": self.schema_version,
            "yakbox_version": self.yakbox_version,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "protocol_fingerprint": self.protocol_fingerprint,
            "order_seed_fingerprint": self.order_seed_fingerprint,
            "reviews": [
                {
                    "case_id": entry.case_id,
                    "kind": entry.kind.value,
                    "pass_index": entry.pass_index,
                    "reviewer_fingerprint": entry.reviewer_fingerprint or "",
                    "decision": (
                        entry.decision.value
                        if entry.decision is not None
                        else "pending"
                    ),
                }
                for entry in self.entries
            ],
        }

    def dispositions(self) -> tuple[ReviewerDisposition, ...]:
        if self.status is not QualificationReviewStatus.APPROVED:
            raise ValidationError("Qualification review is not approved")
        dispositions: list[ReviewerDisposition] = []
        for entry in self.entries:
            if entry.reviewer_fingerprint is None or entry.decision is None:
                raise ValidationError("Qualification review is incomplete")
            dispositions.append(
                ReviewerDisposition(
                    entry.case_id,
                    self.policy_fingerprint,
                    entry.reviewer_fingerprint,
                    entry.pass_index,
                    entry.kind,
                    entry.decision,
                )
            )
        return tuple(dispositions)


def reviewer_fingerprint(label: str) -> str:
    """Create a stable pseudonymous reviewer identity from a private label."""
    normalized = label.strip()
    if not normalized:
        raise ValidationError("Reviewer label cannot be empty")
    return semantic_fingerprint("speech-qualification-reviewer-v1", normalized)


def qualification_review_template(
    evaluation: SpeechAnalysisEvaluation,
    *,
    randomization_seed: str,
    timestamp: str | None = None,
) -> str:
    """Create a pending TOML review with a stable, seed-randomized order."""
    seed = randomization_seed.strip()
    if not seed:
        raise ValidationError("Qualification review randomization seed is required")
    seed_fingerprint = semantic_fingerprint(
        "speech-qualification-review-order-seed-v1", seed
    )
    requirements = _required_entries(evaluation)
    ordered = tuple(
        sorted(
            requirements,
            key=lambda identity: semantic_fingerprint(
                "speech-qualification-review-order-v1",
                (seed, evaluation.fingerprint, identity),
            ),
        )
    )
    metadata = runtime_metadata("speech-qualification-review", timestamp=timestamp)
    lines = [
        f'"$schema" = {_toml_string(str(metadata["$schema"]))}',
        f"schema_version = {metadata['schema_version']}",
        f"yakbox_version = {_toml_string(str(metadata['yakbox_version']))}",
        f"timestamp = {_toml_string(str(metadata['timestamp']))}",
        'status = "pending"',
        f'evaluation_fingerprint = "{evaluation.fingerprint}"',
        f'corpus_fingerprint = "{evaluation.corpus_fingerprint}"',
        f'policy_fingerprint = "{evaluation.policy_fingerprint}"',
        f'protocol_fingerprint = "{evaluation.protocol_fingerprint}"',
        f'order_seed_fingerprint = "{seed_fingerprint}"',
    ]
    for case_id, kind, pass_index in ordered:
        lines.extend(
            (
                "",
                "[[reviews]]",
                f"case_id = {_toml_string(case_id)}",
                f"kind = {_toml_string(kind.value)}",
                f"pass_index = {pass_index}",
                'reviewer_fingerprint = ""',
                'decision = "pending"',
            )
        )
    return "\n".join(lines) + "\n"


def write_qualification_review_template(
    path: Path,
    evaluation: SpeechAnalysisEvaluation,
    *,
    randomization_seed: str,
    timestamp: str | None = None,
    overwrite: bool = False,
) -> None:
    """Atomically write a pending qualification review artifact."""
    content = qualification_review_template(
        evaluation,
        randomization_seed=randomization_seed,
        timestamp=timestamp,
    )
    atomic_write_bytes(path, content.encode(), overwrite=overwrite)


def load_qualification_review(
    path: Path,
    *,
    evaluation: SpeechAnalysisEvaluation,
) -> QualificationReview:
    """Load, bind, and structurally validate one qualification review."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValidationError("Cannot read speech qualification review") from error
    if not isinstance(raw, dict):
        raise ValidationError("Speech qualification review must be a TOML table")
    _require_keys(
        raw,
        {
            "$schema",
            "schema_version",
            "yakbox_version",
            "timestamp",
            "status",
            "evaluation_fingerprint",
            "corpus_fingerprint",
            "policy_fingerprint",
            "protocol_fingerprint",
            "order_seed_fingerprint",
            "reviews",
        },
    )
    entries_raw = raw["reviews"]
    if not isinstance(entries_raw, list) or not entries_raw:
        raise ValidationError("Speech qualification review requires review entries")
    entries = tuple(_load_entry(item) for item in entries_raw)
    try:
        status = QualificationReviewStatus(_string(raw, "status"))
    except ValueError as error:
        raise ValidationError("Qualification review status is invalid") from error
    review = QualificationReview(
        schema_version=_integer(raw, "schema_version"),
        yakbox_version=_string(raw, "yakbox_version"),
        timestamp=_string(raw, "timestamp"),
        status=status,
        evaluation_fingerprint=_string(raw, "evaluation_fingerprint"),
        corpus_fingerprint=_string(raw, "corpus_fingerprint"),
        policy_fingerprint=_string(raw, "policy_fingerprint"),
        protocol_fingerprint=_string(raw, "protocol_fingerprint"),
        order_seed_fingerprint=_string(raw, "order_seed_fingerprint"),
        entries=entries,
    )
    if _string(raw, "$schema") != schema_uri("speech-qualification-review"):
        raise ValidationError("Qualification review schema URI is invalid")
    _bind_review(review, evaluation)
    return review


def approve_calibration_review(
    table: CalibrationTable,
    evaluation: SpeechAnalysisEvaluation,
    review: QualificationReview,
) -> CalibrationTable:
    """Approve calibration from a complete, evaluation-bound review artifact."""
    _bind_review(review, evaluation)
    approved = approve_calibration(table, evaluation, review.dispositions())
    return replace(
        approved,
        reviewer_disposition_fingerprint=review.fingerprint,
    )


def _load_entry(raw: object) -> QualificationReviewEntry:
    if not isinstance(raw, dict):
        raise ValidationError("Qualification review entry must be a table")
    values = cast(dict[str, object], raw)
    _require_keys(
        values,
        {"case_id", "kind", "pass_index", "reviewer_fingerprint", "decision"},
    )
    try:
        kind = ReviewKind(_string(values, "kind"))
    except ValueError as error:
        raise ValidationError("Qualification review kind is invalid") from error
    decision_value = _string(values, "decision", allow_empty=False)
    reviewer_value = _string(values, "reviewer_fingerprint", allow_empty=True)
    if decision_value == "pending":
        decision = None
    else:
        try:
            decision = ReviewDecision(decision_value)
        except ValueError as error:
            raise ValidationError("Qualification review decision is invalid") from error
    return QualificationReviewEntry(
        case_id=_string(values, "case_id"),
        kind=kind,
        pass_index=_integer(values, "pass_index"),
        reviewer_fingerprint=reviewer_value or None,
        decision=decision,
    )


def _bind_review(
    review: QualificationReview,
    evaluation: SpeechAnalysisEvaluation,
) -> None:
    actual_binding = (
        review.evaluation_fingerprint,
        review.corpus_fingerprint,
        review.policy_fingerprint,
        review.protocol_fingerprint,
    )
    expected_binding = (
        evaluation.fingerprint,
        evaluation.corpus_fingerprint,
        evaluation.policy_fingerprint,
        evaluation.protocol_fingerprint,
    )
    if actual_binding != expected_binding:
        raise ValidationError("Qualification review does not match the evaluation")
    required = set(_required_entries(evaluation))
    actual = {entry.identity for entry in review.entries}
    if actual != required:
        raise ValidationError(
            "Qualification review entries do not match required cases"
        )


def _required_entries(
    evaluation: SpeechAnalysisEvaluation,
) -> tuple[tuple[str, ReviewKind, int], ...]:
    required: list[tuple[str, ReviewKind, int]] = []
    for item in evaluation.cases:
        if item.observed.consensus.outcome is ConsensusOutcome.ACCEPTED:
            required.append((item.case.case_id, ReviewKind.AUTOMATIC_ACCEPTANCE, 1))
        if "persistent_valid_dissent" in item.observed.consensus.reason_codes:
            required.append((item.case.case_id, ReviewKind.VALID_DISSENT, 1))
        if item.case.boundary_review_required:
            required.extend(
                (
                    (item.case.case_id, ReviewKind.BOUNDARY, 1),
                    (item.case.case_id, ReviewKind.BOUNDARY, 2),
                )
            )
    if not required:
        raise ValidationError("Evaluation has no required qualification reviews")
    return tuple(required)


def _require_keys(raw: dict[str, object], expected: set[str]) -> None:
    if set(raw) != expected:
        raise ValidationError("Qualification review fields are invalid")


def _string(raw: dict[str, object], key: str, *, allow_empty: bool = False) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ValidationError(f"Qualification review {key!r} must be text")
    return value


def _integer(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"Qualification review {key!r} must be an integer")
    return value


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _require_sha256(value: str, label: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValidationError(f"{label.capitalize()} must be a SHA-256")


__all__ = [
    "QualificationReview",
    "QualificationReviewEntry",
    "QualificationReviewStatus",
    "approve_calibration_review",
    "load_qualification_review",
    "qualification_review_template",
    "reviewer_fingerprint",
    "write_qualification_review_template",
]
