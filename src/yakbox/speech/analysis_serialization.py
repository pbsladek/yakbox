"""Schema-backed serialization for generic speech-analysis evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import cast

from yakbox.contracts import runtime_metadata
from yakbox.errors import WorkerProtocolError
from yakbox.speech.analysis_models import (
    AlignmentPurpose,
    AppliedLexicalEquivalence,
    AudioSpan,
    ConsensusOutcome,
    ConsensusResult,
    ConversionIdentity,
    ExecutionIdentity,
    ForcedAlignmentResult,
    ForcedAlignmentUnit,
    LexicalSpan,
    ModelArtifactIdentity,
    ParakeetEvidence,
    QwenEvidence,
    RecognitionResult,
    RecognitionToken,
    ScoreKind,
    SpeechVerification,
    TokenVote,
    VerificationScope,
    VoteState,
    WhisperEvidence,
)


def recognition_report(result: RecognitionResult) -> dict[str, object]:
    """Serialize independent recognition without transcript plaintext."""
    evidence = result.evidence
    if isinstance(evidence, WhisperEvidence):
        evidence_value: dict[str, object] = {
            "kind": "whisper",
            **asdict(evidence),
        }
    elif isinstance(evidence, ParakeetEvidence):
        evidence_value = {"kind": "parakeet", **asdict(evidence)}
    elif isinstance(evidence, QwenEvidence):
        evidence_value = {"kind": "qwen", **asdict(evidence)}
    return {
        **runtime_metadata("speech-recognition"),
        "fingerprint": result.fingerprint,
        "engine": result.engine,
        "model": _identity(result.model),
        "execution": _identity(result.execution),
        "span": asdict(result.span),
        "requested_language": result.requested_language,
        "detected_language": result.detected_language,
        "normalized_transcript_hash": result.normalized_transcript_hash,
        "raw_transcript_hash": result.raw_transcript_hash,
        "score_calibration_fingerprint": result.score_calibration_fingerprint,
        "tokens": [asdict(token) for token in result.tokens],
        "evidence": evidence_value,
        "issues": list(result.issues),
    }


def forced_alignment_report(result: ForcedAlignmentResult) -> dict[str, object]:
    """Serialize timing evidence without granting transcript authority."""
    return {
        **runtime_metadata("speech-forced-alignment"),
        "fingerprint": result.fingerprint,
        "engine": result.engine,
        "model": _identity(result.model),
        "execution": _identity(result.execution),
        "span": asdict(result.span),
        "purpose": result.purpose.value,
        "aligner_text_hash": result.aligner_text_hash,
        "expected_lexical_span_hash": result.expected_lexical_span_hash,
        "units": [asdict(unit) for unit in result.units],
        "coverage_ratio": result.coverage_ratio,
        "issues": list(result.issues),
    }


def consensus_report(result: ConsensusResult) -> dict[str, object]:
    """Serialize deterministic votes, disagreement spans, and outcome."""
    return {
        **runtime_metadata("speech-consensus"),
        "fingerprint": result.fingerprint,
        "policy_fingerprint": result.policy_fingerprint,
        "expected_tokens_hash": result.expected_tokens_hash,
        "equivalence_set_fingerprint": result.equivalence_set_fingerprint,
        "recognition_fingerprints": list(result.recognition_fingerprints),
        "applied_equivalences": [
            asdict(application) for application in result.applied_equivalences
        ],
        "votes": [
            {
                **asdict(vote),
                "state": vote.state.value,
                "reason_codes": list(vote.reason_codes),
            }
            for vote in result.votes
        ],
        "accepted_spans": [_span(span) for span in result.accepted_spans],
        "rejected_spans": [_span(span) for span in result.rejected_spans],
        "disagreement_spans": [_span(span) for span in result.disagreement_spans],
        "high_risk_spans": [_span(span) for span in result.high_risk_spans],
        "escalation_reason": result.escalation_reason,
        "outcome": result.outcome.value,
        "reason_codes": list(result.reason_codes),
    }


def verification_report(result: SpeechVerification) -> dict[str, object]:
    """Serialize final common acceptance evidence."""
    return {
        **runtime_metadata("speech-verification"),
        "fingerprint": result.fingerprint,
        "policy_fingerprint": result.policy_fingerprint,
        "consensus_fingerprint": result.consensus_fingerprint,
        "forced_alignment_fingerprint": result.forced_alignment_fingerprint,
        "signal_evidence_fingerprint": result.signal_evidence_fingerprint,
        "artifact_digest": result.artifact_digest,
        "scope": result.scope.value,
        "accepted": result.accepted,
        "reason_codes": list(result.reason_codes),
    }


def recognition_from_report(value: object) -> RecognitionResult:
    """Rebuild and validate recognition evidence returned by a worker."""
    raw = _mapping(value, "recognition report")
    evidence_raw = _mapping(raw.get("evidence"), "recognition evidence")
    kind = _string(evidence_raw, "kind")
    if kind == "whisper":
        evidence = WhisperEvidence(
            _optional_float(evidence_raw, "average_log_probability"),
            _optional_float(evidence_raw, "compression_ratio"),
            _optional_float(evidence_raw, "no_speech_probability"),
            _optional_float(evidence_raw, "temperature"),
        )
    elif kind == "parakeet":
        evidence = ParakeetEvidence(
            _optional_float(evidence_raw, "sentence_confidence"),
            _string(evidence_raw, "decoding"),
            _integer(evidence_raw, "chunk_duration_frames"),
            _integer(evidence_raw, "overlap_frames"),
        )
    elif kind == "qwen":
        evidence = QwenEvidence(
            _string(evidence_raw, "finish_reason"),
            _integer(evidence_raw, "prompt_tokens"),
            _integer(evidence_raw, "generation_tokens"),
        )
    else:
        raise WorkerProtocolError("Unknown recognition evidence kind")
    tokens = tuple(
        _recognition_token(_mapping(item, "recognition token"))
        for item in _sequence(raw, "tokens")
    )
    result = RecognitionResult(
        engine=_string(raw, "engine"),
        model=_model_identity(_mapping(raw.get("model"), "model identity")),
        execution=_execution_identity(
            _mapping(raw.get("execution"), "execution identity")
        ),
        span=_audio_span(_mapping(raw.get("span"), "audio span")),
        requested_language=_string(raw, "requested_language"),
        detected_language=_optional_string(raw, "detected_language"),
        normalized_transcript_hash=_string(raw, "normalized_transcript_hash"),
        raw_transcript_hash=_string(raw, "raw_transcript_hash"),
        score_calibration_fingerprint=_string(raw, "score_calibration_fingerprint"),
        tokens=tokens,
        evidence=evidence,
        issues=_string_tuple(raw, "issues"),
    )
    if raw.get("fingerprint") != result.fingerprint:
        raise WorkerProtocolError("Recognition result fingerprint does not match")
    return result


def forced_alignment_from_report(value: object) -> ForcedAlignmentResult:
    """Rebuild and validate forced-alignment evidence returned by a worker."""
    raw = _mapping(value, "forced-alignment report")
    try:
        purpose = AlignmentPurpose(_string(raw, "purpose"))
    except ValueError as error:
        raise WorkerProtocolError("Unknown forced-alignment purpose") from error
    units = tuple(
        ForcedAlignmentUnit(
            _string(item_raw, "text_hash"),
            _integer(item_raw, "start_frame"),
            _integer(item_raw, "end_frame"),
        )
        for item in _sequence(raw, "units")
        for item_raw in (_mapping(item, "forced-alignment unit"),)
    )
    result = ForcedAlignmentResult(
        engine=_string(raw, "engine"),
        model=_model_identity(_mapping(raw.get("model"), "model identity")),
        execution=_execution_identity(
            _mapping(raw.get("execution"), "execution identity")
        ),
        span=_audio_span(_mapping(raw.get("span"), "audio span")),
        purpose=purpose,
        aligner_text_hash=_string(raw, "aligner_text_hash"),
        expected_lexical_span_hash=_string(raw, "expected_lexical_span_hash"),
        units=units,
        coverage_ratio=_float(raw, "coverage_ratio"),
        issues=_string_tuple(raw, "issues"),
    )
    if raw.get("fingerprint") != result.fingerprint:
        raise WorkerProtocolError("Forced-alignment fingerprint does not match")
    return result


def consensus_from_report(value: object) -> ConsensusResult:
    """Rebuild and validate deterministic cached consensus evidence."""
    raw = _mapping(value, "consensus report")
    try:
        outcome = ConsensusOutcome(_string(raw, "outcome"))
    except ValueError as error:
        raise WorkerProtocolError("Unknown consensus outcome") from error
    result = ConsensusResult(
        policy_fingerprint=_string(raw, "policy_fingerprint"),
        expected_tokens_hash=_string(raw, "expected_tokens_hash"),
        equivalence_set_fingerprint=_string(raw, "equivalence_set_fingerprint"),
        recognition_fingerprints=_string_tuple(raw, "recognition_fingerprints"),
        applied_equivalences=tuple(
            _applied_equivalence(_mapping(item, "applied equivalence"))
            for item in _sequence(raw, "applied_equivalences")
        ),
        votes=tuple(
            _token_vote(_mapping(item, "token vote"))
            for item in _sequence(raw, "votes")
        ),
        accepted_spans=_lexical_spans(raw, "accepted_spans"),
        rejected_spans=_lexical_spans(raw, "rejected_spans"),
        disagreement_spans=_lexical_spans(raw, "disagreement_spans"),
        high_risk_spans=_lexical_spans(raw, "high_risk_spans"),
        escalation_reason=_string(raw, "escalation_reason", allow_empty=True),
        outcome=outcome,
        reason_codes=_string_tuple(raw, "reason_codes"),
    )
    if raw.get("fingerprint") != result.fingerprint:
        raise WorkerProtocolError("Consensus result fingerprint does not match")
    return result


def verification_from_report(value: object) -> SpeechVerification:
    """Rebuild and validate cached terminal verification evidence."""
    raw = _mapping(value, "verification report")
    try:
        scope = VerificationScope(_string(raw, "scope"))
    except ValueError as error:
        raise WorkerProtocolError("Unknown verification scope") from error
    result = SpeechVerification(
        policy_fingerprint=_string(raw, "policy_fingerprint"),
        consensus_fingerprint=_string(raw, "consensus_fingerprint"),
        forced_alignment_fingerprint=_optional_string(
            raw, "forced_alignment_fingerprint"
        ),
        signal_evidence_fingerprint=_optional_string(
            raw, "signal_evidence_fingerprint"
        ),
        artifact_digest=_string(raw, "artifact_digest"),
        scope=scope,
        accepted=_boolean(raw, "accepted"),
        reason_codes=_string_tuple(raw, "reason_codes"),
    )
    if raw.get("fingerprint") != result.fingerprint:
        raise WorkerProtocolError("Verification result fingerprint does not match")
    return result


def _applied_equivalence(raw: Mapping[str, object]) -> AppliedLexicalEquivalence:
    return AppliedLexicalEquivalence(
        engine=_string(raw, "engine"),
        recognition_fingerprint=_string(raw, "recognition_fingerprint"),
        rule_fingerprint=_string(raw, "rule_fingerprint"),
        reason_code=_string(raw, "reason_code"),
        expected_start=_integer(raw, "expected_start"),
        expected_end=_integer(raw, "expected_end"),
        recognized_start=_integer(raw, "recognized_start"),
        recognized_end=_integer(raw, "recognized_end"),
        recognized_sequence_hash=_string(raw, "recognized_sequence_hash"),
    )


def _token_vote(raw: Mapping[str, object]) -> TokenVote:
    try:
        state = VoteState(_string(raw, "state"))
    except ValueError as error:
        raise WorkerProtocolError("Unknown token-vote state") from error
    return TokenVote(
        expected_index=_integer(raw, "expected_index"),
        engine=_string(raw, "engine"),
        recognition_fingerprint=_string(raw, "recognition_fingerprint"),
        state=state,
        recognized_token_hash=_optional_string(raw, "recognized_token_hash"),
        reason_codes=_string_tuple(raw, "reason_codes"),
    )


def _lexical_spans(raw: Mapping[str, object], key: str) -> tuple[LexicalSpan, ...]:
    return tuple(
        LexicalSpan(
            start=_integer(item_raw, "start"),
            end=_integer(item_raw, "end"),
            reason_codes=_string_tuple(item_raw, "reason_codes"),
            review_eligible=_boolean(item_raw, "review_eligible"),
        )
        for item in _sequence(raw, key)
        for item_raw in (_mapping(item, "lexical span"),)
    )


def _identity(
    value: ModelArtifactIdentity | ExecutionIdentity,
) -> dict[str, object]:
    raw = asdict(value)
    return {key: _json_value(item) for key, item in raw.items()}


def _json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def _span(value: LexicalSpan) -> dict[str, object]:
    raw = asdict(value)
    raw["reason_codes"] = list(raw["reason_codes"])
    return raw


def _model_identity(raw: Mapping[str, object]) -> ModelArtifactIdentity:
    conversion_raw = _mapping(raw.get("conversion"), "conversion identity")
    conversion = ConversionIdentity(
        _string(conversion_raw, "source"),
        _string(conversion_raw, "tool"),
        _string(conversion_raw, "tool_version"),
        _string(conversion_raw, "recipe_fingerprint"),
        _string(conversion_raw, "precision_policy"),
        _boolean(conversion_raw, "verified"),
    )
    return ModelArtifactIdentity(
        engine=_string(raw, "engine"),
        backend_package=_string(raw, "backend_package"),
        backend_version=_string(raw, "backend_version"),
        adapter_version=_integer(raw, "adapter_version"),
        worker_protocol_version=_integer(raw, "worker_protocol_version"),
        converted_repository=_string(raw, "converted_repository"),
        converted_revision=_string(raw, "converted_revision"),
        converted_directory_fingerprint=_string(raw, "converted_directory_fingerprint"),
        upstream_repository=_string(raw, "upstream_repository"),
        upstream_revision=_string(raw, "upstream_revision"),
        conversion=conversion,
        precision=_string(raw, "precision"),
        decode_fingerprint=_string(raw, "decode_fingerprint"),
    )


def _execution_identity(raw: Mapping[str, object]) -> ExecutionIdentity:
    return ExecutionIdentity(
        worker_artifact_digest=_string(raw, "worker_artifact_digest"),
        lock_digest=_string(raw, "lock_digest"),
        python_version=_string(raw, "python_version"),
        os_family=_string(raw, "os_family"),
        os_version=_string(raw, "os_version", allow_empty=True),
        architecture=_string(raw, "architecture"),
        mlx_version=_optional_string(raw, "mlx_version"),
        metal_version=_optional_string(raw, "metal_version"),
        device_class=_string(raw, "device_class"),
        determinism_mode=_string(raw, "determinism_mode"),
        decode_seeds=tuple(
            _checked_integer(item, "decode seed")
            for item in _sequence(raw, "decode_seeds")
        ),
    )


def _recognition_token(raw: Mapping[str, object]) -> RecognitionToken:
    try:
        score_kind = ScoreKind(_string(raw, "score_kind"))
    except ValueError as error:
        raise WorkerProtocolError("Unknown recognition score kind") from error
    return RecognitionToken(
        text=_string(raw, "text"),
        start_frame=_optional_integer(raw, "start_frame"),
        end_frame=_optional_integer(raw, "end_frame"),
        score=_optional_float(raw, "score"),
        score_kind=score_kind,
        calibration_fingerprint=_string(raw, "calibration_fingerprint"),
    )


def _audio_span(raw: Mapping[str, object]) -> AudioSpan:
    return AudioSpan(
        _string(raw, "audio_digest"),
        _integer(raw, "start_frame"),
        _integer(raw, "end_frame"),
        _integer(raw, "sample_rate"),
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise WorkerProtocolError(f"Worker {label} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(raw: Mapping[str, object], key: str) -> tuple[object, ...]:
    value = raw.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise WorkerProtocolError(f"Worker {key} must be an array")
    return tuple(value)


def _string(raw: Mapping[str, object], key: str, *, allow_empty: bool = False) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or (not value and not allow_empty):
        raise WorkerProtocolError(f"Worker {key} must be text")
    return value


def _optional_string(raw: Mapping[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise WorkerProtocolError(f"Worker {key} must be text or null")
    return value


def _checked_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerProtocolError(f"Worker {label} must be an integer")
    return value


def _integer(raw: Mapping[str, object], key: str) -> int:
    return _checked_integer(raw.get(key), key)


def _optional_integer(raw: Mapping[str, object], key: str) -> int | None:
    value = raw.get(key)
    return None if value is None else _checked_integer(value, key)


def _float(raw: Mapping[str, object], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WorkerProtocolError(f"Worker {key} must be numeric")
    return float(value)


def _optional_float(raw: Mapping[str, object], key: str) -> float | None:
    value = raw.get(key)
    return None if value is None else _float(raw, key)


def _boolean(raw: Mapping[str, object], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise WorkerProtocolError(f"Worker {key} must be boolean")
    return value


def _string_tuple(raw: Mapping[str, object], key: str) -> tuple[str, ...]:
    values = _sequence(raw, key)
    if not all(isinstance(item, str) for item in values):
        raise WorkerProtocolError(f"Worker {key} entries must be text")
    return cast(tuple[str, ...], values)


__all__ = [
    "consensus_from_report",
    "consensus_report",
    "forced_alignment_from_report",
    "forced_alignment_report",
    "recognition_from_report",
    "recognition_report",
    "verification_from_report",
    "verification_report",
]
