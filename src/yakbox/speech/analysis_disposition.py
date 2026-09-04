"""Durable, evidence-bound dispositions for review-eligible speech defects."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

from yakbox._files import atomic_write_json, safe_child, sha256_file
from yakbox.contracts import runtime_metadata, schema_uri, utc_timestamp
from yakbox.errors import ValidationError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint, text_fingerprint

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}")
_MAXIMUM_NOTES_BYTES = 4_096
_CANDIDATE_FIELDS = frozenset(
    {
        "$schema",
        "schema_version",
        "yakbox_version",
        "timestamp",
        "review_id",
        "fingerprint",
        "audio_digest",
        "spoken_text_plan_fingerprint",
        "expected_span_hash",
        "policy_fingerprint",
        "evidence_fingerprints",
        "reason_codes",
        "review_eligible",
        "hard_failure_codes",
        "bound_files",
    }
)
_DISPOSITION_FIELDS = frozenset(
    {
        "$schema",
        "schema_version",
        "yakbox_version",
        "timestamp",
        "review_id",
        "fingerprint",
        "candidate_fingerprint",
        "audio_digest",
        "spoken_text_plan_fingerprint",
        "expected_span_hash",
        "policy_fingerprint",
        "evidence_fingerprints",
        "decision",
        "reviewer_fingerprint",
        "resolved_at",
        "notes",
        "notes_hash",
    }
)
_BOUND_FILE_FIELDS = frozenset(
    {"kind", "relative_path", "sha256", "evidence_fingerprint"}
)


class HumanReviewDecision(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"


class HumanReviewState(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class BoundReviewFile:
    """One managed artifact whose bytes must still match at resolution time."""

    kind: str
    relative_path: str
    sha256: str
    evidence_fingerprint: str

    def __post_init__(self) -> None:
        if _IDENTIFIER.fullmatch(self.kind) is None:
            raise ValidationError("Bound review file kind is invalid")
        _safe_relative_path(self.relative_path)
        _require_sha256(self.sha256, "bound file digest")
        _require_sha256(self.evidence_fingerprint, "bound evidence fingerprint")


@dataclass(frozen=True, slots=True)
class HumanReviewCandidate:
    """A soft rejection bound to exact audio, text, policy, and evidence."""

    review_id: str
    created_at: str
    audio_digest: str
    spoken_text_plan_fingerprint: str
    expected_span_hash: str
    policy_fingerprint: str
    evidence_fingerprints: tuple[str, ...]
    reason_codes: tuple[str, ...]
    review_eligible: bool
    hard_failure_codes: tuple[str, ...]
    bound_files: tuple[BoundReviewFile, ...]

    def __post_init__(self) -> None:
        if _IDENTIFIER.fullmatch(self.review_id) is None:
            raise ValidationError("Human review identifier is invalid")
        _require_utc(self.created_at, "review candidate timestamp")
        for value, label in (
            (self.audio_digest, "review audio digest"),
            (self.spoken_text_plan_fingerprint, "spoken text plan fingerprint"),
            (self.expected_span_hash, "expected span hash"),
            (self.policy_fingerprint, "review policy fingerprint"),
        ):
            _require_sha256(value, label)
        if (
            not self.evidence_fingerprints
            or tuple(sorted(set(self.evidence_fingerprints)))
            != self.evidence_fingerprints
            or any(
                _SHA256.fullmatch(item) is None for item in self.evidence_fingerprints
            )
        ):
            raise ValidationError(
                "Review evidence fingerprints must be unique SHA-256s"
            )
        if not self.reason_codes:
            raise ValidationError("Review reason codes cannot be empty")
        for values, label in (
            (self.reason_codes, "review reason codes"),
            (self.hard_failure_codes, "hard failure codes"),
        ):
            if tuple(sorted(set(values))) != values or any(
                _IDENTIFIER.fullmatch(item) is None for item in values
            ):
                raise ValidationError(f"{label.capitalize()} are invalid")
        if not self.bound_files:
            raise ValidationError("Human review requires bound evidence files")
        identities = tuple((item.kind, item.relative_path) for item in self.bound_files)
        if len(identities) != len(set(identities)):
            raise ValidationError("Bound review files must be unique")
        _validate_binding_topology(self)

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "speech-human-review-candidate-v1",
            {
                "review_id": self.review_id,
                "audio_digest": self.audio_digest,
                "spoken_text_plan_fingerprint": (self.spoken_text_plan_fingerprint),
                "expected_span_hash": self.expected_span_hash,
                "policy_fingerprint": self.policy_fingerprint,
                "evidence_fingerprints": self.evidence_fingerprints,
                "reason_codes": self.reason_codes,
                "review_eligible": self.review_eligible,
                "hard_failure_codes": self.hard_failure_codes,
                "bound_files": tuple(
                    sorted(
                        self.bound_files,
                        key=lambda item: (item.kind, item.relative_path),
                    )
                ),
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata(
                "speech-human-review-candidate", timestamp=self.created_at
            ),
            "review_id": self.review_id,
            "fingerprint": self.fingerprint,
            "audio_digest": self.audio_digest,
            "spoken_text_plan_fingerprint": self.spoken_text_plan_fingerprint,
            "expected_span_hash": self.expected_span_hash,
            "policy_fingerprint": self.policy_fingerprint,
            "evidence_fingerprints": list(self.evidence_fingerprints),
            "reason_codes": list(self.reason_codes),
            "review_eligible": self.review_eligible,
            "hard_failure_codes": list(self.hard_failure_codes),
            "bound_files": [
                {
                    "kind": item.kind,
                    "relative_path": item.relative_path,
                    "sha256": item.sha256,
                    "evidence_fingerprint": item.evidence_fingerprint,
                }
                for item in self.bound_files
            ],
        }


@dataclass(frozen=True, slots=True)
class HumanDisposition:
    """Explicit reviewer decision transitively bound to one candidate."""

    review_id: str
    candidate_fingerprint: str
    audio_digest: str
    spoken_text_plan_fingerprint: str
    expected_span_hash: str
    policy_fingerprint: str
    evidence_fingerprints: tuple[str, ...]
    decision: HumanReviewDecision
    reviewer_fingerprint: str
    resolved_at: str
    notes: str
    notes_hash: str

    def __post_init__(self) -> None:
        if _IDENTIFIER.fullmatch(self.review_id) is None:
            raise ValidationError("Human disposition identifier is invalid")
        for value, label in (
            (self.candidate_fingerprint, "review candidate fingerprint"),
            (self.audio_digest, "review audio digest"),
            (self.spoken_text_plan_fingerprint, "spoken text plan fingerprint"),
            (self.expected_span_hash, "expected span hash"),
            (self.policy_fingerprint, "review policy fingerprint"),
            (self.reviewer_fingerprint, "reviewer fingerprint"),
            (self.notes_hash, "review notes hash"),
        ):
            _require_sha256(value, label)
        if (
            not self.evidence_fingerprints
            or tuple(sorted(set(self.evidence_fingerprints)))
            != self.evidence_fingerprints
            or any(
                _SHA256.fullmatch(item) is None for item in self.evidence_fingerprints
            )
        ):
            raise ValidationError("Disposition evidence fingerprints are invalid")
        _require_utc(self.resolved_at, "disposition timestamp")
        _validate_notes(self.notes)
        if self.notes_hash != text_fingerprint(self.notes):
            raise ValidationError("Human disposition notes hash differs")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "speech-human-disposition-v1",
            {
                "review_id": self.review_id,
                "candidate_fingerprint": self.candidate_fingerprint,
                "audio_digest": self.audio_digest,
                "spoken_text_plan_fingerprint": (self.spoken_text_plan_fingerprint),
                "expected_span_hash": self.expected_span_hash,
                "policy_fingerprint": self.policy_fingerprint,
                "evidence_fingerprints": self.evidence_fingerprints,
                "decision": self.decision,
                "reviewer_fingerprint": self.reviewer_fingerprint,
                "notes": self.notes,
                "notes_hash": self.notes_hash,
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("speech-human-disposition", timestamp=self.resolved_at),
            "review_id": self.review_id,
            "fingerprint": self.fingerprint,
            "candidate_fingerprint": self.candidate_fingerprint,
            "audio_digest": self.audio_digest,
            "spoken_text_plan_fingerprint": self.spoken_text_plan_fingerprint,
            "expected_span_hash": self.expected_span_hash,
            "policy_fingerprint": self.policy_fingerprint,
            "evidence_fingerprints": list(self.evidence_fingerprints),
            "decision": self.decision.value,
            "reviewer_fingerprint": self.reviewer_fingerprint,
            "resolved_at": self.resolved_at,
            "notes": self.notes,
            "notes_hash": self.notes_hash,
        }


@dataclass(frozen=True, slots=True)
class HumanReviewStatus:
    review_id: str
    state: HumanReviewState
    candidate_fingerprint: str
    disposition_fingerprint: str | None
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "review_id": self.review_id,
            "state": self.state.value,
            "candidate_fingerprint": self.candidate_fingerprint,
            "disposition_fingerprint": self.disposition_fingerprint,
            "issues": list(self.issues),
        }


class HumanReviewStore:
    """Atomically manage candidates and dispositions under one workspace root."""

    def __init__(self, root: Path, *, evidence_root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.evidence_root = evidence_root.expanduser().resolve()

    def register(
        self,
        candidate: HumanReviewCandidate,
        *,
        overwrite: bool = False,
    ) -> None:
        self._validate_bound_files(candidate)
        atomic_write_json(
            self._candidate_path(candidate.review_id),
            candidate.to_dict(),
            overwrite=overwrite,
        )

    def list(self) -> tuple[HumanReviewStatus, ...]:
        directory = safe_child(self.root, self.root / "candidates")
        if not directory.is_dir():
            return ()
        return tuple(
            self.show(path.stem.removesuffix(".candidate"))
            for path in sorted(directory.glob("*.candidate.json"))
            if not path.is_symlink()
        )

    def show(self, review_id: str) -> HumanReviewStatus:
        candidate = self._load_candidate(review_id)
        issues = list(self._bound_file_issues(candidate))
        disposition = self._load_disposition(review_id, required=False)
        if (
            disposition is not None
            and disposition.candidate_fingerprint != candidate.fingerprint
        ):
            issues.append("disposition_binding_stale")
        if issues:
            state = HumanReviewState.STALE
        elif disposition is None:
            state = HumanReviewState.PENDING
        elif disposition.decision is HumanReviewDecision.ACCEPT:
            state = HumanReviewState.ACCEPTED
        else:
            state = HumanReviewState.REJECTED
        return HumanReviewStatus(
            review_id,
            state,
            candidate.fingerprint,
            disposition.fingerprint if disposition is not None else None,
            tuple(sorted(set(issues))),
        )

    def resolve(
        self,
        review_id: str,
        *,
        expected_candidate_fingerprint: str,
        decision: HumanReviewDecision,
        reviewer_identifier: str,
        notes: str,
    ) -> HumanDisposition:
        _require_sha256(
            expected_candidate_fingerprint, "expected candidate fingerprint"
        )
        candidate = self._load_candidate(review_id)
        if candidate.fingerprint != expected_candidate_fingerprint:
            raise ValidationError("Review candidate changed after it was shown")
        if not candidate.review_eligible or candidate.hard_failure_codes:
            raise ValidationError(
                "Only a soft review-eligible rejection can be resolved"
            )
        issues = self._bound_file_issues(candidate)
        if issues:
            raise ValidationError("Review candidate evidence is stale: " + issues[0])
        reviewer = reviewer_identifier.strip()
        if not reviewer:
            raise ValidationError("Reviewer identifier is required")
        _validate_notes(notes)
        disposition = HumanDisposition(
            review_id=candidate.review_id,
            candidate_fingerprint=candidate.fingerprint,
            audio_digest=candidate.audio_digest,
            spoken_text_plan_fingerprint=candidate.spoken_text_plan_fingerprint,
            expected_span_hash=candidate.expected_span_hash,
            policy_fingerprint=candidate.policy_fingerprint,
            evidence_fingerprints=candidate.evidence_fingerprints,
            decision=decision,
            reviewer_fingerprint=semantic_fingerprint(
                "speech-human-reviewer-v1", reviewer
            ),
            resolved_at=utc_timestamp(),
            notes=notes,
            notes_hash=text_fingerprint(notes),
        )
        atomic_write_json(
            self._disposition_path(review_id),
            disposition.to_dict(),
            overwrite=False,
        )
        return disposition

    def _validate_bound_files(self, candidate: HumanReviewCandidate) -> None:
        issues = self._bound_file_issues(candidate)
        if issues:
            raise ValidationError("Review candidate evidence is invalid: " + issues[0])

    def _bound_file_issues(self, candidate: HumanReviewCandidate) -> tuple[str, ...]:
        issues: list[str] = []
        for item in candidate.bound_files:
            path = safe_child(
                self.evidence_root, self.evidence_root / item.relative_path
            )
            if path.is_symlink() or not path.is_file():
                issues.append(f"{item.kind}_file_missing")
            elif sha256_file(path) != item.sha256:
                issues.append(f"{item.kind}_file_changed")
        return tuple(issues)

    def _candidate_path(self, review_id: str) -> Path:
        _require_identifier(review_id)
        return safe_child(
            self.root,
            self.root / "candidates" / f"{review_id}.candidate.json",
        )

    def _disposition_path(self, review_id: str) -> Path:
        _require_identifier(review_id)
        return safe_child(
            self.root,
            self.root / "dispositions" / f"{review_id}.disposition.json",
        )

    def _load_candidate(self, review_id: str) -> HumanReviewCandidate:
        raw = _read_json(self._candidate_path(review_id), "review candidate")
        candidate = _candidate_from_dict(raw)
        if raw.get("fingerprint") != candidate.fingerprint:
            raise ValidationError("Review candidate fingerprint differs")
        return candidate

    def _load_disposition(
        self, review_id: str, *, required: bool
    ) -> HumanDisposition | None:
        path = self._disposition_path(review_id)
        if not path.exists() and not required:
            return None
        raw = _read_json(path, "human disposition")
        disposition = _disposition_from_dict(raw)
        if raw.get("fingerprint") != disposition.fingerprint:
            raise ValidationError("Human disposition fingerprint differs")
        return disposition


def human_review_status_report(
    statuses: tuple[HumanReviewStatus, ...],
) -> dict[str, object]:
    """Serialize a privacy-safe list or show result for CLI JSON output."""
    return {
        **runtime_metadata("speech-human-review-status"),
        "review_count": len(statuses),
        "reviews": [item.to_dict() for item in statuses],
    }


def _candidate_from_dict(raw: dict[str, object]) -> HumanReviewCandidate:
    _require_schema(raw, "speech-human-review-candidate", _CANDIDATE_FIELDS)
    files = tuple(_bound_file_from_dict(value) for value in _list(raw, "bound_files"))
    return HumanReviewCandidate(
        _string(raw, "review_id"),
        _string(raw, "timestamp"),
        _string(raw, "audio_digest"),
        _string(raw, "spoken_text_plan_fingerprint"),
        _string(raw, "expected_span_hash"),
        _string(raw, "policy_fingerprint"),
        _strings(raw, "evidence_fingerprints"),
        _strings(raw, "reason_codes"),
        _boolean(raw, "review_eligible"),
        _strings(raw, "hard_failure_codes"),
        files,
    )


def _bound_file_from_dict(value: object) -> BoundReviewFile:
    item = _dict(value, "bound review file")
    _require_exact_fields(item, _BOUND_FILE_FIELDS, "Bound review file")
    return BoundReviewFile(
        _string(item, "kind"),
        _string(item, "relative_path"),
        _string(item, "sha256"),
        _string(item, "evidence_fingerprint"),
    )


def _disposition_from_dict(raw: dict[str, object]) -> HumanDisposition:
    _require_schema(raw, "speech-human-disposition", _DISPOSITION_FIELDS)
    if raw.get("timestamp") != raw.get("resolved_at"):
        raise ValidationError("Human disposition timestamps differ")
    try:
        decision = HumanReviewDecision(_string(raw, "decision"))
    except ValueError as error:
        raise ValidationError("Human disposition decision is invalid") from error
    return HumanDisposition(
        _string(raw, "review_id"),
        _string(raw, "candidate_fingerprint"),
        _string(raw, "audio_digest"),
        _string(raw, "spoken_text_plan_fingerprint"),
        _string(raw, "expected_span_hash"),
        _string(raw, "policy_fingerprint"),
        _strings(raw, "evidence_fingerprints"),
        decision,
        _string(raw, "reviewer_fingerprint"),
        _string(raw, "resolved_at"),
        _string(raw, "notes", allow_empty=True),
        _string(raw, "notes_hash"),
    )


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"Cannot read {label}") from error
    return _dict(value, label)


def _require_schema(
    raw: dict[str, object],
    name: str,
    expected_fields: frozenset[str],
) -> None:
    if raw.get("$schema") != schema_uri(name) or raw.get("schema_version") != 1:
        raise ValidationError("Human review artifact schema identity is invalid")
    _require_exact_fields(raw, expected_fields, "Human review artifact")
    if not isinstance(raw.get("yakbox_version"), str) or not raw["yakbox_version"]:
        raise ValidationError("Human review artifact Yakbox version is invalid")


def _require_exact_fields(
    raw: dict[str, object], expected: frozenset[str], label: str
) -> None:
    if set(raw) != expected:
        raise ValidationError(f"{label} fields are invalid")


def _safe_relative_path(value: str) -> None:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or "\\" in value
        or ".." in posix.parts
    ):
        raise ValidationError("Bound review file path must be safe and relative")


def _validate_binding_topology(candidate: HumanReviewCandidate) -> None:
    by_kind: dict[str, list[BoundReviewFile]] = {}
    for item in candidate.bound_files:
        by_kind.setdefault(item.kind, []).append(item)
    audio = by_kind.get("context_audio", [])
    text_plan = by_kind.get("spoken_text_plan", [])
    policy = by_kind.get("policy", [])
    evidence = by_kind.get("analysis_evidence", [])
    if (
        len(audio) != 1
        or audio[0].sha256 != candidate.audio_digest
        or len(text_plan) != 1
        or text_plan[0].evidence_fingerprint != candidate.spoken_text_plan_fingerprint
        or len(policy) != 1
        or policy[0].evidence_fingerprint != candidate.policy_fingerprint
        or {item.evidence_fingerprint for item in evidence}
        != set(candidate.evidence_fingerprints)
    ):
        raise ValidationError("Human review bound-file topology is inconsistent")


def _validate_notes(notes: str) -> None:
    try:
        encoded = notes.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValidationError("Reviewer notes contain invalid Unicode") from error
    if len(encoded) > _MAXIMUM_NOTES_BYTES or "\x00" in notes:
        raise ValidationError("Reviewer notes exceed the bounded UTF-8 contract")


def _require_utc(value: str, label: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValidationError(f"{label.capitalize()} is invalid") from error
    if parsed.utcoffset() != timedelta(0):
        raise ValidationError(f"{label.capitalize()} must be UTC")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label.capitalize()} must be a SHA-256")


def _require_identifier(value: str) -> None:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError("Human review identifier is invalid")


def _dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValidationError(f"{label.capitalize()} must be an object")
    return cast(dict[str, object], value)


def _list(raw: dict[str, object], key: str) -> list[object]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise ValidationError(f"Human review {key} must be an array")
    return cast(list[object], value)


def _string(raw: dict[str, object], key: str, *, allow_empty: bool = False) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ValidationError(f"Human review {key} must be text")
    return value


def _strings(raw: dict[str, object], key: str) -> tuple[str, ...]:
    values = _list(raw, key)
    if any(not isinstance(item, str) for item in values):
        raise ValidationError(f"Human review {key} must contain text")
    return tuple(cast(list[str], values))


def _boolean(raw: dict[str, object], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValidationError(f"Human review {key} must be boolean")
    return value


__all__ = [
    "BoundReviewFile",
    "HumanDisposition",
    "HumanReviewCandidate",
    "HumanReviewDecision",
    "HumanReviewState",
    "HumanReviewStatus",
    "HumanReviewStore",
    "human_review_status_report",
]
