from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from yakbox._files import sha256_file
from yakbox.errors import ArtifactError, ValidationError
from yakbox.schemas import load_schema
from yakbox.speech.analysis_disposition import (
    BoundReviewFile,
    HumanReviewCandidate,
    HumanReviewDecision,
    HumanReviewState,
    HumanReviewStore,
    human_review_status_report,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _candidate(root: Path, *, hard: bool = False) -> HumanReviewCandidate:
    values = {
        "context_audio": (b"RIFF-context", SHA_A),
        "spoken_text_plan": (b'{"plan": 1}', SHA_A),
        "policy": (b'{"policy": 1}', SHA_B),
        "analysis_evidence": (b'{"evidence": 1}', SHA_C),
    }
    files: list[BoundReviewFile] = []
    for index, (kind, (content, fingerprint)) in enumerate(values.items()):
        path = root / f"evidence/{index}-{kind}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files.append(
            BoundReviewFile(
                kind,
                path.relative_to(root).as_posix(),
                sha256_file(path),
                fingerprint,
            )
        )
    return HumanReviewCandidate(
        review_id="review-1",
        created_at="2026-08-14T00:00:00+00:00",
        audio_digest=files[0].sha256,
        spoken_text_plan_fingerprint=SHA_A,
        expected_span_hash=SHA_B,
        policy_fingerprint=SHA_B,
        evidence_fingerprints=(SHA_C,),
        reason_codes=("persistent_valid_dissent",),
        review_eligible=True,
        hard_failure_codes=("clipping",) if hard else (),
        bound_files=tuple(files),
    )


def test_review_resolution_is_atomic_bound_and_schema_valid(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    store = HumanReviewStore(tmp_path / "reviews", evidence_root=tmp_path)
    store.register(candidate)

    pending = store.show(candidate.review_id)
    disposition = store.resolve(
        candidate.review_id,
        expected_candidate_fingerprint=pending.candidate_fingerprint,
        decision=HumanReviewDecision.ACCEPT,
        reviewer_identifier="local-reviewer",
        notes="The contextual audio is acceptable.",
    )
    accepted = store.show(candidate.review_id)

    assert pending.state is HumanReviewState.PENDING
    assert accepted.state is HumanReviewState.ACCEPTED
    assert disposition.candidate_fingerprint == candidate.fingerprint
    assert disposition.reviewer_fingerprint not in {"", "local-reviewer"}
    assert "local-reviewer" not in str(disposition.to_dict())
    Draft202012Validator(load_schema("speech-human-review-candidate")).validate(
        candidate.to_dict()
    )
    Draft202012Validator(load_schema("speech-human-disposition")).validate(
        disposition.to_dict()
    )
    report = human_review_status_report(store.list())
    Draft202012Validator(load_schema("speech-human-review-status")).validate(report)
    assert report["review_count"] == 1

    with pytest.raises(ArtifactError, match="already exists"):
        store.resolve(
            candidate.review_id,
            expected_candidate_fingerprint=candidate.fingerprint,
            decision=HumanReviewDecision.REJECT,
            reviewer_identifier="local-reviewer",
            notes="Changed decision",
        )


def test_changed_evidence_or_candidate_binding_is_stale(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    store = HumanReviewStore(tmp_path / "reviews", evidence_root=tmp_path)
    store.register(candidate)
    shown = store.show(candidate.review_id)
    evidence_path = tmp_path / candidate.bound_files[-1].relative_path
    evidence_path.write_bytes(b"changed")

    stale = store.show(candidate.review_id)

    assert stale.state is HumanReviewState.STALE
    assert stale.issues == ("analysis_evidence_file_changed",)
    with pytest.raises(ValidationError, match="evidence is stale"):
        store.resolve(
            candidate.review_id,
            expected_candidate_fingerprint=shown.candidate_fingerprint,
            decision=HumanReviewDecision.ACCEPT,
            reviewer_identifier="reviewer",
            notes="",
        )

    replacement = _candidate(tmp_path)
    replacement = replace(replacement, expected_span_hash=SHA_C)
    store.register(replacement, overwrite=True)
    with pytest.raises(ValidationError, match="changed after it was shown"):
        store.resolve(
            candidate.review_id,
            expected_candidate_fingerprint=shown.candidate_fingerprint,
            decision=HumanReviewDecision.ACCEPT,
            reviewer_identifier="reviewer",
            notes="",
        )


def test_hard_failure_and_unbounded_notes_cannot_be_overridden(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, hard=True)
    store = HumanReviewStore(tmp_path / "reviews", evidence_root=tmp_path)
    store.register(candidate)

    with pytest.raises(ValidationError, match="soft review-eligible"):
        store.resolve(
            candidate.review_id,
            expected_candidate_fingerprint=candidate.fingerprint,
            decision=HumanReviewDecision.ACCEPT,
            reviewer_identifier="reviewer",
            notes="",
        )

    soft = replace(candidate, hard_failure_codes=())
    store.register(soft, overwrite=True)
    with pytest.raises(ValidationError, match="bounded UTF-8"):
        store.resolve(
            soft.review_id,
            expected_candidate_fingerprint=soft.fingerprint,
            decision=HumanReviewDecision.ACCEPT,
            reviewer_identifier="reviewer",
            notes="x" * 4_097,
        )


def test_bound_files_reject_path_escape_and_mismatched_topology(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="safe and relative"):
        BoundReviewFile("context_audio", "../outside.wav", SHA_A, SHA_B)

    candidate = _candidate(tmp_path)
    with pytest.raises(ValidationError, match="topology"):
        replace(candidate, audio_digest=SHA_A)


def test_review_fingerprints_ignore_timestamps_and_file_report_order(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)
    reordered = replace(
        candidate,
        created_at="2026-08-14T01:00:00+00:00",
        bound_files=tuple(reversed(candidate.bound_files)),
    )

    assert reordered.fingerprint == candidate.fingerprint

    store = HumanReviewStore(tmp_path / "reviews", evidence_root=tmp_path)
    store.register(candidate)
    disposition = store.resolve(
        candidate.review_id,
        expected_candidate_fingerprint=candidate.fingerprint,
        decision=HumanReviewDecision.ACCEPT,
        reviewer_identifier="reviewer",
        notes="Accepted in context.",
    )

    assert (
        replace(disposition, resolved_at="2026-08-15T00:00:00+00:00").fingerprint
        == disposition.fingerprint
    )


def test_review_store_rejects_unregistered_fields_and_incoherent_timestamps(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)
    store = HumanReviewStore(tmp_path / "reviews", evidence_root=tmp_path)
    store.register(candidate)
    candidate_path = tmp_path / "reviews/candidates/review-1.candidate.json"
    candidate_document = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_document["unexpected"] = True
    candidate_path.write_text(json.dumps(candidate_document), encoding="utf-8")

    with pytest.raises(ValidationError, match="fields are invalid"):
        store.show(candidate.review_id)

    store.register(candidate, overwrite=True)
    disposition = store.resolve(
        candidate.review_id,
        expected_candidate_fingerprint=candidate.fingerprint,
        decision=HumanReviewDecision.REJECT,
        reviewer_identifier="reviewer",
        notes="Rejected.",
    )
    disposition_path = tmp_path / "reviews/dispositions/review-1.disposition.json"
    disposition_document = json.loads(disposition_path.read_text(encoding="utf-8"))
    disposition_document["timestamp"] = "2026-08-13T00:00:00+00:00"
    disposition_path.write_text(json.dumps(disposition_document), encoding="utf-8")

    assert disposition.resolved_at != disposition_document["timestamp"]
    with pytest.raises(ValidationError, match="timestamps differ"):
        store.show(candidate.review_id)
