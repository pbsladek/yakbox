"""Validate human-approved transcript truth for the qualification corpus."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_json, sha256_bytes, sha256_file
from yakbox.contracts import runtime_metadata, utc_timestamp
from yakbox.errors import ValidationError
from yakbox.speech.analysis_corpus_sources import (
    CorpusSourceInventory,
    CorpusSourceWindow,
)
from yakbox.speech.analysis_corpus_transcripts import (
    CorpusTranscriptDraft,
    CorpusTranscriptDraftCase,
    TranscriptAgreement,
    TranscriptCandidate,
)
from yakbox.speech.analysis_fingerprints import semantic_fingerprint, text_fingerprint
from yakbox.speech.normalization import normalize_english

_SCHEMA = "https://yakbox.dev/schemas/speech-corpus-transcript-draft-v1.schema.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAXIMUM_TRANSCRIPT_BYTES = 65_536
_TOP_FIELDS = {
    "$schema",
    "schema_version",
    "yakbox_version",
    "timestamp",
    "fingerprint",
    "source_inventory_fingerprint",
    "review_status",
    "case_count",
    "cases",
}
_CASE_FIELDS = {
    "source_window_id",
    "source_passage_group",
    "voice",
    "reader",
    "relative_audio_path",
    "audio_digest",
    "agreement",
    "agreement_count",
    "proposed_text",
    "accepted_text",
    "review_status",
    "reviewer_fingerprint",
    "candidates",
    "fingerprint",
}
_CANDIDATE_FIELDS = {
    "engine",
    "text",
    "recognition_fingerprint",
    "model_fingerprint",
    "execution_fingerprint",
    "normalized_transcript_hash",
    "token_count",
    "issues",
}


@dataclass(frozen=True, slots=True)
class ApprovedTranscriptCase:
    source_window_id: str
    source_passage_group: str
    voice: str
    reader: str
    relative_audio_path: str
    audio_digest: str
    accepted_text: str
    accepted_tokens: tuple[str, ...]
    accepted_tokens_hash: str
    reviewer_fingerprint: str
    draft_case_fingerprint: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.audio_digest, "approved transcript audio digest"),
            (self.accepted_tokens_hash, "approved transcript token hash"),
            (self.reviewer_fingerprint, "transcript reviewer fingerprint"),
            (self.draft_case_fingerprint, "transcript draft case fingerprint"),
        ):
            _require_sha256(value, label)
        if (
            not self.source_window_id
            or not self.source_passage_group
            or not self.voice
            or not self.reader
            or not self.relative_audio_path
            or not self.accepted_text.strip()
            or not self.accepted_tokens
            or text_fingerprint("\u001f".join(self.accepted_tokens))
            != self.accepted_tokens_hash
        ):
            raise ValidationError("Approved transcript case is inconsistent")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-approved-transcript-case-v1", self)


@dataclass(frozen=True, slots=True)
class ApprovedCorpusTranscripts:
    source_inventory_fingerprint: str
    draft_fingerprint: str
    cases: tuple[ApprovedTranscriptCase, ...]

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_inventory_fingerprint,
            "approved transcript source inventory fingerprint",
        )
        _require_sha256(self.draft_fingerprint, "transcript draft fingerprint")
        identifiers = tuple(item.source_window_id for item in self.cases)
        if not self.cases or identifiers != tuple(sorted(set(identifiers))):
            raise ValidationError("Approved corpus transcripts are inconsistent")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-approved-transcripts-v1", self)

    def to_report_dict(self) -> dict[str, object]:
        """Serialize approved truth evidence without transcript plaintext."""
        return {
            **runtime_metadata("speech-corpus-transcript-truth"),
            "fingerprint": self.fingerprint,
            "source_inventory_fingerprint": self.source_inventory_fingerprint,
            "draft_fingerprint": self.draft_fingerprint,
            "case_count": len(self.cases),
            "cases": [
                {
                    "source_window_id": item.source_window_id,
                    "source_passage_group": item.source_passage_group,
                    "audio_digest": item.audio_digest,
                    "accepted_tokens_hash": item.accepted_tokens_hash,
                    "accepted_token_count": len(item.accepted_tokens),
                    "reviewer_fingerprint": item.reviewer_fingerprint,
                    "draft_case_fingerprint": item.draft_case_fingerprint,
                    "fingerprint": item.fingerprint,
                }
                for item in self.cases
            ],
        }


@dataclass(frozen=True, slots=True)
class CorpusTranscriptReviewProgress:
    """Text-free progress for the private transcript-authoring review."""

    draft_fingerprint: str
    source_inventory_fingerprint: str
    review_status: str
    case_count: int
    approved_count: int
    pending_count: int
    pending_case_ids: tuple[str, ...]
    agreement_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        _require_sha256(self.draft_fingerprint, "transcript draft fingerprint")
        _require_sha256(
            self.source_inventory_fingerprint,
            "transcript source inventory fingerprint",
        )
        if (
            self.review_status not in {"pending", "approved"}
            or self.case_count < 1
            or self.approved_count < 0
            or self.pending_count < 0
            or self.approved_count + self.pending_count != self.case_count
            or len(self.pending_case_ids) != self.pending_count
            or len(self.pending_case_ids) != len(set(self.pending_case_ids))
        ):
            raise ValidationError("Corpus transcript review progress is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "draft_fingerprint": self.draft_fingerprint,
            "source_inventory_fingerprint": self.source_inventory_fingerprint,
            "review_status": self.review_status,
            "case_count": self.case_count,
            "approved_count": self.approved_count,
            "pending_count": self.pending_count,
            "next_pending_case_id": (
                self.pending_case_ids[0] if self.pending_case_ids else None
            ),
            "pending_case_ids": list(self.pending_case_ids),
            "agreement_counts": dict(self.agreement_counts),
        }


def load_approved_corpus_transcripts(
    path: Path,
    *,
    inventory: CorpusSourceInventory,
) -> ApprovedCorpusTranscripts:
    """Load complete transcript truth and reject stale or partial review state."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(
            "Cannot read corpus transcript authoring source"
        ) from error
    if not isinstance(raw, dict) or set(raw) != _TOP_FIELDS:
        raise ValidationError("Corpus transcript authoring fields are invalid")
    if (
        raw.get("$schema") != _SCHEMA
        or raw.get("schema_version") != 1
        or not _utc_timestamp(raw.get("timestamp"))
        or not isinstance(raw.get("yakbox_version"), str)
        or raw.get("review_status") != "approved"
        or raw.get("source_inventory_fingerprint") != inventory.fingerprint
    ):
        raise ValidationError("Corpus transcript approval metadata is invalid")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list):
        raise ValidationError("Corpus transcript approval cases are invalid")
    windows = {item.source_window_id: item for item in inventory.windows}
    draft_cases: list[CorpusTranscriptDraftCase] = []
    approved_cases: list[ApprovedTranscriptCase] = []
    for value in cases_raw:
        draft_case, approved_case = _load_approved_case(value, windows=windows)
        draft_cases.append(draft_case)
        approved_cases.append(approved_case)
    draft = CorpusTranscriptDraft(
        inventory.fingerprint,
        tuple(sorted(draft_cases, key=lambda item: item.source_window_id)),
    )
    if (
        raw.get("case_count") != len(inventory.windows)
        or len(approved_cases) != len(inventory.windows)
        or raw.get("fingerprint") != draft.fingerprint
    ):
        raise ValidationError("Corpus transcript approval identity differs")
    return ApprovedCorpusTranscripts(
        inventory.fingerprint,
        draft.fingerprint,
        tuple(sorted(approved_cases, key=lambda item: item.source_window_id)),
    )


def write_approved_transcript_report(
    path: Path,
    approved: ApprovedCorpusTranscripts,
) -> None:
    atomic_write_json(path, approved.to_report_dict())


def corpus_transcript_review_progress(
    path: Path,
    *,
    inventory: CorpusSourceInventory,
) -> CorpusTranscriptReviewProgress:
    """Validate private authoring state and return no transcript plaintext."""
    raw, _digest = _read_authoring(path)
    draft, cases = _load_review_authoring(raw, inventory=inventory)
    return _review_progress(draft, cases, review_status=_text(raw, "review_status"))


def load_corpus_transcript_review_draft(
    path: Path,
    *,
    inventory: CorpusSourceInventory,
) -> CorpusTranscriptDraft:
    """Load and verify the immutable draft behind current review decisions."""
    raw, _digest = _read_authoring(path)
    draft, _cases = _load_review_authoring(raw, inventory=inventory)
    return draft


def approve_corpus_transcript_case(
    path: Path,
    *,
    inventory: CorpusSourceInventory,
    source_window_id: str,
    reviewer_label: str,
    accepted_text: str | None = None,
    expected_authoring_digest: str | None = None,
) -> CorpusTranscriptReviewProgress:
    """Approve one listened-to transcript while preserving immutable evidence."""
    reviewer = reviewer_label.strip()
    if not reviewer:
        raise ValidationError("Transcript reviewer label cannot be empty")
    raw, original_digest = _read_authoring(path)
    if (
        expected_authoring_digest is not None
        and expected_authoring_digest != original_digest
    ):
        raise ValidationError("Corpus transcript authoring source changed")
    _draft, cases = _load_review_authoring(raw, inventory=inventory)
    selected = _pending_review_case(cases, source_window_id=source_window_id)
    final_text = _accepted_review_text(selected, accepted_text=accepted_text)
    selected["accepted_text"] = final_text
    selected["review_status"] = "approved"
    selected["reviewer_fingerprint"] = semantic_fingerprint(
        "speech-qualification-reviewer-v1",
        reviewer,
    )
    raw["review_status"] = (
        "approved"
        if all(case.get("review_status") == "approved" for case in cases)
        else "pending"
    )
    raw["timestamp"] = utc_timestamp()
    draft, cases = _load_review_authoring(raw, inventory=inventory)
    if sha256_file(path) != original_digest:
        raise ValidationError("Corpus transcript authoring source changed")
    atomic_write_json(path, raw)
    return _review_progress(
        draft,
        cases,
        review_status=_text(raw, "review_status"),
    )


def _pending_review_case(
    cases: list[dict[str, object]],
    *,
    source_window_id: str,
) -> dict[str, object]:
    selected = next(
        (case for case in cases if case.get("source_window_id") == source_window_id),
        None,
    )
    if selected is None:
        raise ValidationError("Corpus transcript review case is unknown")
    if selected.get("review_status") != "pending":
        raise ValidationError("Corpus transcript review case is not pending")
    return selected


def _accepted_review_text(
    selected: dict[str, object],
    *,
    accepted_text: str | None,
) -> str:
    value = (
        accepted_text if accepted_text is not None else selected.get("accepted_text")
    )
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            "Corpus transcript review requires corrected text for this case"
        )
    value = value.strip()
    if len(value.encode("utf-8")) > _MAXIMUM_TRANSCRIPT_BYTES:
        raise ValidationError("Corpus transcript review text exceeds 65536 bytes")
    if not _normalized_tokens(value):
        raise ValidationError("Corpus transcript review text has no English tokens")
    return value


def _read_authoring(path: Path) -> tuple[dict[str, object], str]:
    try:
        payload = path.read_bytes()
        decoded = payload.decode("utf-8")
        raw = json.loads(decoded)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(
            "Cannot read corpus transcript authoring source"
        ) from error
    if not isinstance(raw, dict):
        raise ValidationError("Corpus transcript authoring fields are invalid")
    return cast(dict[str, object], raw), sha256_bytes(payload)


def _load_review_authoring(
    raw: dict[str, object],
    *,
    inventory: CorpusSourceInventory,
) -> tuple[CorpusTranscriptDraft, list[dict[str, object]]]:
    if set(raw) != _TOP_FIELDS:
        raise ValidationError("Corpus transcript authoring fields are invalid")
    review_status = raw.get("review_status")
    if not _valid_review_metadata(
        raw,
        inventory_fingerprint=inventory.fingerprint,
    ):
        raise ValidationError("Corpus transcript review metadata is invalid")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list):
        raise ValidationError("Corpus transcript review cases are invalid")
    windows = {item.source_window_id: item for item in inventory.windows}
    cases: list[dict[str, object]] = []
    draft_cases: list[CorpusTranscriptDraftCase] = []
    for value in cases_raw:
        if not isinstance(value, dict):
            raise ValidationError("Corpus transcript review case is invalid")
        case = cast(dict[str, object], value)
        draft_case = _load_review_draft_case(case, windows=windows)
        _validate_review_case_state(case)
        cases.append(case)
        draft_cases.append(draft_case)
    draft = CorpusTranscriptDraft(
        inventory.fingerprint,
        tuple(draft_cases),
    )
    approved_count = sum(case.get("review_status") == "approved" for case in cases)
    expected_status = "approved" if approved_count == len(cases) else "pending"
    if (
        raw.get("case_count") != len(inventory.windows)
        or len(cases) != len(inventory.windows)
        or raw.get("fingerprint") != draft.fingerprint
        or review_status != expected_status
    ):
        raise ValidationError("Corpus transcript review identity differs")
    return draft, cases


def _valid_review_metadata(
    raw: dict[str, object],
    *,
    inventory_fingerprint: str,
) -> bool:
    return (
        raw.get("$schema") == _SCHEMA
        and raw.get("schema_version") == 1
        and _utc_timestamp(raw.get("timestamp"))
        and isinstance(raw.get("yakbox_version"), str)
        and raw.get("review_status") in {"pending", "approved"}
        and raw.get("source_inventory_fingerprint") == inventory_fingerprint
    )


def _validate_review_case_state(case: dict[str, object]) -> None:
    case_status = case.get("review_status")
    reviewer = case.get("reviewer_fingerprint")
    accepted = case.get("accepted_text")
    if case_status == "pending":
        if reviewer != "" or not isinstance(accepted, str):
            raise ValidationError("Pending transcript review state is invalid")
        return
    if case_status != "approved":
        raise ValidationError("Corpus transcript review status is invalid")
    if (
        not isinstance(reviewer, str)
        or _SHA256.fullmatch(reviewer) is None
        or not isinstance(accepted, str)
        or not accepted.strip()
        or not _normalized_tokens(accepted)
    ):
        raise ValidationError("Approved transcript review state is invalid")


def _load_review_draft_case(
    raw: dict[str, object],
    *,
    windows: dict[str, CorpusSourceWindow],
) -> CorpusTranscriptDraftCase:
    if set(raw) != _CASE_FIELDS:
        raise ValidationError("Corpus transcript review case fields are invalid")
    window_id = _text(raw, "source_window_id")
    try:
        window = windows[window_id]
    except KeyError as error:
        raise ValidationError(
            "Corpus transcript review contains an unknown case"
        ) from error
    candidates_raw = raw.get("candidates")
    if not isinstance(candidates_raw, list):
        raise ValidationError("Corpus transcript candidates are invalid")
    try:
        agreement = TranscriptAgreement(_text(raw, "agreement"))
    except ValueError as error:
        raise ValidationError("Corpus transcript agreement is invalid") from error
    draft_case = CorpusTranscriptDraftCase(
        window_id,
        _text(raw, "source_passage_group"),
        _text(raw, "voice"),
        _text(raw, "reader"),
        _text(raw, "relative_audio_path"),
        _text(raw, "audio_digest"),
        tuple(_candidate(item) for item in candidates_raw),
        agreement,
        _integer(raw, "agreement_count"),
        _normalized_tokens(_text(raw, "proposed_text", allow_empty=True)),
    )
    if (
        raw.get("fingerprint") != draft_case.fingerprint
        or draft_case.source_passage_group != window.source_passage_group
        or draft_case.voice != window.voice
        or draft_case.reader != window.reader
        or draft_case.relative_audio_path != window.relative_audio_path
        or draft_case.audio_digest != window.audio_digest
    ):
        raise ValidationError("Corpus transcript review case identity differs")
    return draft_case


def _review_progress(
    draft: CorpusTranscriptDraft,
    cases: list[dict[str, object]],
    *,
    review_status: str,
) -> CorpusTranscriptReviewProgress:
    agreement_by_id = {item.source_window_id: item.agreement for item in draft.cases}
    priority = {
        TranscriptAgreement.DISSENT: 0,
        TranscriptAgreement.MAJORITY: 1,
        TranscriptAgreement.UNANIMOUS: 2,
        TranscriptAgreement.UNUSABLE: 3,
    }
    pending = tuple(
        sorted(
            (
                _text(case, "source_window_id")
                for case in cases
                if case.get("review_status") == "pending"
            ),
            key=lambda case_id: (priority[agreement_by_id[case_id]], case_id),
        )
    )
    counts = Counter(item.agreement.value for item in draft.cases)
    return CorpusTranscriptReviewProgress(
        draft.fingerprint,
        draft.source_inventory_fingerprint,
        review_status,
        len(cases),
        len(cases) - len(pending),
        len(pending),
        pending,
        tuple(sorted(counts.items())),
    )


def _load_approved_case(
    value: object,
    *,
    windows: dict[str, CorpusSourceWindow],
) -> tuple[CorpusTranscriptDraftCase, ApprovedTranscriptCase]:
    if not isinstance(value, dict) or set(value) != _CASE_FIELDS:
        raise ValidationError("Corpus transcript approval case fields are invalid")
    raw = cast(dict[str, object], value)
    window_id = _text(raw, "source_window_id")
    try:
        window = windows[window_id]
    except KeyError as error:
        raise ValidationError(
            "Corpus transcript approval contains an unknown case"
        ) from error
    candidates_raw = raw.get("candidates")
    if not isinstance(candidates_raw, list):
        raise ValidationError("Corpus transcript candidates are invalid")
    candidates = tuple(_candidate(item) for item in candidates_raw)
    try:
        agreement = TranscriptAgreement(_text(raw, "agreement"))
    except ValueError as error:
        raise ValidationError("Corpus transcript agreement is invalid") from error
    proposed = _normalized_tokens(_text(raw, "proposed_text", allow_empty=True))
    draft_case = CorpusTranscriptDraftCase(
        window_id,
        _text(raw, "source_passage_group"),
        _text(raw, "voice"),
        _text(raw, "reader"),
        _text(raw, "relative_audio_path"),
        _text(raw, "audio_digest"),
        candidates,
        agreement,
        _integer(raw, "agreement_count"),
        proposed,
    )
    if (
        raw.get("fingerprint") != draft_case.fingerprint
        or draft_case.source_passage_group != window.source_passage_group
        or draft_case.voice != window.voice
        or draft_case.reader != window.reader
        or draft_case.relative_audio_path != window.relative_audio_path
        or draft_case.audio_digest != window.audio_digest
        or raw.get("review_status") != "approved"
    ):
        raise ValidationError("Corpus transcript approval case identity differs")
    accepted_text = _text(raw, "accepted_text")
    accepted_tokens = _normalized_tokens(accepted_text)
    reviewer = _text(raw, "reviewer_fingerprint")
    return draft_case, ApprovedTranscriptCase(
        window_id,
        window.source_passage_group,
        window.voice,
        window.reader,
        window.relative_audio_path,
        window.audio_digest,
        accepted_text,
        accepted_tokens,
        text_fingerprint("\u001f".join(accepted_tokens)),
        reviewer,
        draft_case.fingerprint,
    )


def _candidate(value: object) -> TranscriptCandidate:
    if not isinstance(value, dict) or set(value) != _CANDIDATE_FIELDS:
        raise ValidationError("Corpus transcript candidate fields are invalid")
    raw = cast(dict[str, object], value)
    candidate_text = _text(raw, "text", allow_empty=True)
    tokens = _normalized_tokens(candidate_text)
    issues_raw = raw.get("issues")
    if not isinstance(issues_raw, list) or not all(
        isinstance(item, str) for item in issues_raw
    ):
        raise ValidationError("Corpus transcript candidate issues are invalid")
    candidate = TranscriptCandidate(
        _text(raw, "engine"),
        _text(raw, "recognition_fingerprint"),
        _text(raw, "model_fingerprint"),
        _text(raw, "execution_fingerprint"),
        _text(raw, "normalized_transcript_hash"),
        tokens,
        tuple(cast(list[str], issues_raw)),
    )
    if raw.get("token_count") != len(tokens):
        raise ValidationError("Corpus transcript candidate token count differs")
    return candidate


def _normalized_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.text for token in normalize_english(value).tokens)


def _text(
    raw: dict[str, object],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValidationError(f"Corpus transcript {key} must be text")
    return value


def _integer(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"Corpus transcript {key} must be an integer")
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
    "ApprovedCorpusTranscripts",
    "ApprovedTranscriptCase",
    "CorpusTranscriptReviewProgress",
    "approve_corpus_transcript_case",
    "corpus_transcript_review_progress",
    "load_approved_corpus_transcripts",
    "load_corpus_transcript_review_draft",
    "write_approved_transcript_report",
]
