from __future__ import annotations

import asyncio
import hashlib
import json
import os
import queue
import subprocess
import sys
import time
import wave
from collections.abc import AsyncIterator
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema import Draft202012Validator

from yakbox._files import sha256_file
from yakbox.errors import ModelIntegrityError, ValidationError, WorkerProtocolError
from yakbox.schemas import load_schema
from yakbox.speech.analysis_cache import RecognitionCacheIdentity
from yakbox.speech.analysis_calibration import (
    CalibrationObservation,
    fit_calibration_table,
)
from yakbox.speech.analysis_fingerprints import text_fingerprint
from yakbox.speech.analysis_models import (
    AlignmentPurpose,
    AudioSpan,
    ClipClass,
    ConsensusOutcome,
    ConversionIdentity,
    ExecutionIdentity,
    ForcedAlignmentResult,
    ForcedAlignmentUnit,
    ModelArtifactIdentity,
    ParakeetEvidence,
    QwenEvidence,
    RecognitionResult,
    RecognitionToken,
    ScoreKind,
    SpeechVerification,
    VerificationScope,
    WhisperEvidence,
)
from yakbox.speech.analysis_policy import (
    CalibrationTable,
    CalibrationThreshold,
    EnginePolicy,
    SpeechAnalysisPolicy,
)
from yakbox.speech.analysis_protocol import (
    AnalysisWorkerRequest,
    AnalysisWorkerResponse,
    CancellationWorkerRequest,
    RecognitionBatchItem,
    RecognitionManyWorkerRequest,
    RecognitionWorkerRequest,
    ShutdownWorkerRequest,
    StatusWorkerRequest,
    UnloadWorkerRequest,
    WorkerBatchFinished,
    WorkerErrorCode,
    WorkerFailure,
    WorkerItemSuccess,
    WorkerModelLoad,
    WorkerStatus,
    WorkerSuccess,
    encode_worker_handshake,
    encode_worker_request,
    encode_worker_response,
    parse_worker_handshake,
    parse_worker_request,
    parse_worker_response,
)
from yakbox.speech.analysis_runtime import (
    BUILT_IN_WORKERS,
    IsolatedAnalysisWorker,
    ProtocolSupervisedWorker,
)
from yakbox.speech.analysis_scheduler import (
    BatchOperation,
    BatchWorkItem,
    OperationMetrics,
    OperationTerminalStatus,
    build_worker_handshake,
)
from yakbox.speech.analysis_serialization import (
    consensus_report,
    forced_alignment_report,
    recognition_report,
    verification_report,
)
from yakbox.speech.analysis_services import (
    FakeForcedAligner,
    FakeSpeechRecognizer,
    ForcedAligner,
    SpeechRecognizer,
)
from yakbox.speech.analysis_worker import (
    AnalysisWorkerApplication,
    EngineFactory,
    _serve,
)
from yakbox.speech.consensus import evaluate_consensus
from yakbox.speech.normalization import (
    MAXIMUM_EQUIVALENCE_RULES,
    DirectionalEquivalence,
    EquivalenceSet,
    align_token_sequences,
    normalize_english,
)
from yakbox.speech.shadow import (
    ShadowClassification,
    ShadowGroundTruth,
    compare_shadow_decision,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _execution() -> ExecutionIdentity:
    return ExecutionIdentity(
        worker_artifact_digest=SHA_A,
        lock_digest=SHA_B,
        python_version="3.14.0",
        os_family="Darwin",
        os_version="26.0",
        architecture="arm64",
        mlx_version="1.0",
        metal_version=None,
        device_class="m5-64gb",
        determinism_mode="greedy",
        decode_seeds=(),
    )


def _model(engine: str) -> ModelArtifactIdentity:
    return ModelArtifactIdentity(
        engine=engine,
        backend_package=f"{engine}-package",
        backend_version="1.0",
        adapter_version=1,
        worker_protocol_version=1,
        converted_repository=f"example/{engine}",
        converted_revision="1" * 40,
        converted_directory_fingerprint=SHA_A,
        upstream_repository=f"upstream/{engine}",
        upstream_revision="2" * 40,
        conversion=ConversionIdentity(
            source="upstream",
            tool="converter",
            tool_version="1",
            recipe_fingerprint=SHA_B,
            precision_policy="bf16",
            verified=True,
        ),
        precision="bf16",
        decode_fingerprint=SHA_C,
    )


def _evidence(engine: str) -> WhisperEvidence | ParakeetEvidence | QwenEvidence:
    if engine == "whisper":
        return WhisperEvidence(-0.1, 1.0, 0.01, 0.0)
    if engine == "parakeet":
        return ParakeetEvidence(0.99, "greedy", 1_920_000, 240_000)
    return QwenEvidence("stop", 0, 2)


def _recognition(
    engine: str,
    tokens: tuple[str, ...],
    *,
    calibration_fingerprint: str = SHA_C,
    issues: tuple[str, ...] = (),
    span: AudioSpan | None = None,
) -> RecognitionResult:
    audio_span = span or AudioSpan(SHA_A, 0, 16_000, 16_000)
    recognized = tuple(
        RecognitionToken(
            token,
            index * 100,
            (index + 1) * 100,
            0.99,
            ScoreKind.PROBABILITY,
            calibration_fingerprint,
        )
        for index, token in enumerate(tokens)
    )
    joined = "\u001f".join(tokens)
    return RecognitionResult(
        engine=engine,
        model=_model(engine),
        execution=_execution(),
        span=audio_span,
        requested_language="en",
        detected_language="en" if engine == "whisper" else None,
        normalized_transcript_hash=text_fingerprint(joined),
        raw_transcript_hash=text_fingerprint(" ".join(tokens)),
        score_calibration_fingerprint=calibration_fingerprint,
        tokens=recognized,
        evidence=_evidence(engine),
        issues=issues,
    )


def _alignment(span: AudioSpan | None = None) -> ForcedAlignmentResult:
    audio_span = span or AudioSpan(SHA_A, 0, 16_000, 16_000)
    return ForcedAlignmentResult(
        engine="qwen-forced",
        model=_model("qwen-forced"),
        execution=_execution(),
        span=audio_span,
        purpose=AlignmentPurpose.NON_AUTHORITATIVE,
        aligner_text_hash=SHA_A,
        expected_lexical_span_hash=SHA_B,
        units=(ForcedAlignmentUnit(SHA_C, 10, 100),),
        coverage_ratio=1.0,
        issues=(),
    )


def _policy() -> SpeechAnalysisPolicy:
    return SpeechAnalysisPolicy(
        version=1,
        preset="strict",
        language="en",
        baseline_recognizers=("whisper", "parakeet"),
        escalation_recognizer="qwen",
        forced_aligner="qwen-forced",
        always_escalate_clip_classes=(ClipClass.ONE_WORD,),
        always_escalate_repairs=True,
        reject_unresolved_disagreement=True,
        reject_unexpected_speech=True,
        missing_required_engine="error",
        valid_dissent="retry_then_reject",
        engines=tuple(
            EnginePolicy(
                engine,
                "test",
                f"example/{engine}",
                "1" * 40,
                10,
                "greedy",
            )
            for engine in ("whisper", "parakeet", "qwen", "qwen-forced")
        ),
    )


def _calibration() -> CalibrationTable:
    return CalibrationTable(
        version=1,
        language="en",
        corpus_fingerprint=SHA_A,
        approved=True,
        reviewer_disposition_fingerprint=SHA_B,
        execution_class_fingerprints=(_execution().fingerprint,),
        thresholds=tuple(
            CalibrationThreshold(engine, clip_class, SHA_C)
            for engine in ("whisper", "parakeet", "qwen")
            for clip_class in (ClipClass.ONE_WORD, ClipClass.SENTENCE)
        ),
    )


def _calibrated_recognition(
    engine: str,
    tokens: tuple[str, ...],
    *,
    issues: tuple[str, ...] = (),
) -> RecognitionResult:
    return _recognition(
        engine,
        tokens,
        calibration_fingerprint=SHA_C,
        issues=issues,
    )


def test_recognition_protocol_has_no_expected_text_authority() -> None:
    span = AudioSpan(SHA_A, 0, 1_600, 16_000)
    request = RecognitionWorkerRequest(
        "recognize-1",
        "whisper",
        SHA_B,
        "windows/one.wav",
        SHA_A,
        "en",
        span,
    )
    encoded = encode_worker_request(request)

    assert b"expected_text" not in encoded
    assert parse_worker_request(encoded) == request
    assert "expected_text" not in {
        field.name for field in fields(RecognitionWorkerRequest)
    }
    assert "expected_text" not in {
        field.name for field in fields(RecognitionCacheIdentity)
    }

    value = json.loads(encoded)
    value["expected_text"] = "the answer"
    with pytest.raises(WorkerProtocolError, match="Unexpected"):
        parse_worker_request(json.dumps(value).encode())

    without_span = json.loads(encoded)
    without_span["span"] = None
    with pytest.raises(WorkerProtocolError, match="requires a sample span"):
        parse_worker_request(json.dumps(without_span).encode())

    without_engine_fingerprint = json.loads(encoded)
    del without_engine_fingerprint["engine_fingerprint"]
    with pytest.raises(WorkerProtocolError, match=r"Missing.*engine_fingerprint"):
        parse_worker_request(json.dumps(without_engine_fingerprint).encode())

    mismatched_span = json.loads(encoded)
    mismatched_span["span"]["audio_digest"] = SHA_C
    with pytest.raises(WorkerProtocolError, match="span and audio digest"):
        parse_worker_request(json.dumps(mismatched_span).encode())


def test_batch_protocol_is_deadlined_item_framed_and_round_trips() -> None:
    span = AudioSpan(SHA_A, 0, 1_600, 16_000)
    request = RecognitionManyWorkerRequest(
        request_id="operation-1",
        batch_id="batch-1",
        engine="whisper",
        engine_fingerprint=SHA_B,
        timeout_milliseconds=12_500,
        items=(
            RecognitionBatchItem(
                index=0,
                request_fingerprint=SHA_C,
                relative_audio_path="windows/one.wav",
                audio_digest=SHA_A,
                language="en",
                span=span,
            ),
        ),
    )

    encoded = encode_worker_request(request)

    assert parse_worker_request(encoded) == request
    assert b'"operation":"recognize_many"' in encoded
    assert b'"timeout_milliseconds":12500' in encoded
    assert b"expected_text" not in encoded

    metrics = OperationMetrics(1_600, 1_600, 10, 20, 30, False, 1)
    response = WorkerItemSuccess(
        request_id="operation-1",
        batch_id="batch-1",
        index=0,
        request_fingerprint=SHA_C,
        result=_recognition("whisper", ("wren", "asked"), span=span),
        metrics=metrics,
    )
    assert parse_worker_response(encode_worker_response(response)) == response
    assert parse_worker_response(
        encode_worker_response(WorkerBatchFinished("operation-1", "batch-1", 1))
    ) == WorkerBatchFinished("operation-1", "batch-1", 1)


def test_worker_handshake_round_trip_binds_every_planned_identity() -> None:
    handshake = build_worker_handshake(
        family="whisper",
        engines=("whisper",),
        worker_artifact_fingerprint=SHA_A,
        environment_lock_fingerprint=SHA_B,
        adapter_fingerprint=SHA_C,
    )

    assert parse_worker_handshake(encode_worker_handshake(handshake)) == handshake
    raw = json.loads(encode_worker_handshake(handshake))
    raw["environment_lock_fingerprint"] = SHA_A
    assert parse_worker_handshake(json.dumps(raw).encode()) != handshake


def test_worker_responses_round_trip_typed_evidence() -> None:
    recognition = _recognition("whisper", ("wren", "asked"))
    alignment = _alignment()

    status = WorkerStatus(
        pid=1,
        family="whisper",
        python_version="3.14.4",
        environment_fingerprint=SHA_A,
        ready=True,
        loaded_engines=(),
        active_request_id=None,
        completed_requests=2,
        failed_requests=0,
        model_loads=(),
        peak_resident_memory_bytes=128,
        metal_active_memory_bytes=None,
        metal_peak_memory_bytes=None,
    )
    for result in (recognition, alignment, status):
        response = WorkerSuccess("request-1", result)
        assert parse_worker_response(encode_worker_response(response)) == response


def test_fake_reports_validate_generic_schemas() -> None:
    recognition = _recognition("whisper", ("wren", "asked"))
    alignment = _alignment()
    verification = SpeechVerification(
        policy_fingerprint=SHA_A,
        consensus_fingerprint=SHA_B,
        forced_alignment_fingerprint=alignment.fingerprint,
        signal_evidence_fingerprint=None,
        artifact_digest=SHA_C,
        scope=VerificationScope.CANDIDATE,
        accepted=True,
        reason_codes=(),
    )
    consensus = evaluate_consensus(
        expected_tokens=("wren", "asked"),
        recognitions=(
            _calibrated_recognition("whisper", ("wren", "asked")),
            _calibrated_recognition("parakeet", ("wren", "asked")),
        ),
        clip_class=ClipClass.SENTENCE,
        policy=_policy(),
        calibration=_calibration(),
        equivalences=EquivalenceSet(1, ()),
    ).result
    assert consensus is not None

    reports = (
        ("speech-recognition", recognition_report(recognition)),
        ("speech-forced-alignment", forced_alignment_report(alignment)),
        ("speech-consensus", consensus_report(consensus)),
        ("speech-verification", verification_report(verification)),
    )
    for schema_name, report in reports:
        Draft202012Validator(load_schema(schema_name)).validate(report)


def test_consensus_is_order_independent_and_strict_about_valid_dissent() -> None:
    expected = ("wren", "asked")
    whisper = _calibrated_recognition("whisper", expected)
    parakeet = _calibrated_recognition("parakeet", expected)
    policy = _policy()
    calibration = _calibration()
    equivalences = EquivalenceSet(1, ())

    first = evaluate_consensus(
        expected_tokens=expected,
        recognitions=(whisper, parakeet),
        clip_class=ClipClass.SENTENCE,
        policy=policy,
        calibration=calibration,
        equivalences=equivalences,
    )
    second = evaluate_consensus(
        expected_tokens=expected,
        recognitions=(parakeet, whisper),
        clip_class=ClipClass.SENTENCE,
        policy=policy,
        calibration=calibration,
        equivalences=equivalences,
    )

    assert first.result == second.result
    assert first.result is not None
    assert first.result.outcome is ConsensusOutcome.ACCEPTED
    assert all(vote.recognized_token_hash for vote in first.result.votes)

    dissent = _calibrated_recognition("parakeet", ("wren", "acts"))
    escalation = evaluate_consensus(
        expected_tokens=expected,
        recognitions=(whisper, dissent),
        clip_class=ClipClass.SENTENCE,
        policy=policy,
        calibration=calibration,
        equivalences=equivalences,
    )
    assert escalation.result is None
    assert escalation.escalation is not None
    assert escalation.escalation.reason_codes == ("baseline_disagreement",)

    rejected = evaluate_consensus(
        expected_tokens=expected,
        recognitions=(
            whisper,
            dissent,
            _calibrated_recognition("qwen", expected),
        ),
        clip_class=ClipClass.SENTENCE,
        policy=policy,
        calibration=calibration,
        equivalences=equivalences,
    )
    assert rejected.result is not None
    assert rejected.result.outcome is ConsensusOutcome.REJECTED
    assert rejected.result.reason_codes == ("persistent_valid_dissent",)

    invalid = evaluate_consensus(
        expected_tokens=expected,
        recognitions=(
            whisper,
            _calibrated_recognition(
                "parakeet", expected, issues=("invalid_engine_result",)
            ),
            _calibrated_recognition(
                "qwen", expected, issues=("invalid_engine_result",)
            ),
        ),
        clip_class=ClipClass.SENTENCE,
        policy=policy,
        calibration=calibration,
        equivalences=equivalences,
    )
    assert invalid.result is not None
    assert invalid.result.reason_codes == ("missing_required_engine",)
    assert invalid.result.rejected_spans[0].reason_codes == ("engine_decode_invalid",)


@given(
    st.lists(
        st.sampled_from(("alpha", "bravo", "charlie", "delta")),
        min_size=1,
        max_size=8,
    )
)
def test_sequence_consensus_is_order_independent_for_normalized_tokens(
    values: list[str],
) -> None:
    expected = tuple(values)
    whisper = _calibrated_recognition("whisper", expected)
    parakeet = _calibrated_recognition("parakeet", expected)

    forward = evaluate_consensus(
        expected_tokens=expected,
        recognitions=(whisper, parakeet),
        clip_class=ClipClass.SENTENCE,
        policy=_policy(),
        calibration=_calibration(),
        equivalences=EquivalenceSet(1, ()),
    )
    reverse = evaluate_consensus(
        expected_tokens=expected,
        recognitions=(parakeet, whisper),
        clip_class=ClipClass.SENTENCE,
        policy=_policy(),
        calibration=_calibration(),
        equivalences=EquivalenceSet(1, ()),
    )

    assert forward == reverse
    assert forward.result is not None
    assert forward.result.outcome is ConsensusOutcome.ACCEPTED


@given(
    st.lists(
        st.sampled_from(("Alpha", "BRAVO", "café", "twenty-one")),
        min_size=0,
        max_size=20,
    )
)
def test_english_normalization_is_lexically_idempotent(values: list[str]) -> None:
    source = " — ".join(values)
    first = normalize_english(source)
    second = normalize_english(" ".join(item.text for item in first.tokens))

    assert tuple(item.text for item in second.tokens) == tuple(
        item.text for item in first.tokens
    )


@given(
    st.lists(st.sampled_from(("a", "b", "c")), max_size=10),
    st.lists(st.sampled_from(("a", "b", "c")), max_size=10),
)
def test_sequence_alignment_ties_are_deterministic_and_bounded(
    expected_values: list[str],
    recognized_values: list[str],
) -> None:
    expected = tuple(expected_values)
    recognized = tuple(recognized_values)
    equivalences = EquivalenceSet(1, ())

    first = align_token_sequences(expected, recognized, equivalences)
    second = align_token_sequences(expected, recognized, equivalences)

    assert first == second
    assert all(
        0 <= item.expected_start <= item.expected_end <= len(expected)
        and 0 <= item.recognized_start <= item.recognized_end <= len(recognized)
        for item in first
    )


def test_equivalence_rules_reject_cycles_oversized_forms_and_adversarial_counts() -> (
    None
):
    with pytest.raises(ValidationError, match="directional chains"):
        EquivalenceSet(
            1,
            (
                DirectionalEquivalence(("a",), ("b",), "spelling"),
                DirectionalEquivalence(("b",), ("a",), "spelling"),
            ),
        )
    with pytest.raises(ValidationError, match="token limit"):
        DirectionalEquivalence(tuple("abcdefghi"), ("alias",), "spelling")
    with pytest.raises(ValidationError, match="rule limit"):
        EquivalenceSet(
            1,
            tuple(
                DirectionalEquivalence(
                    (f"expected-{index}",),
                    (f"recognized-{index}",),
                    "spelling",
                )
                for index in range(MAXIMUM_EQUIVALENCE_RULES + 1)
            ),
        )


def test_multi_token_equivalence_applies_once_and_ambiguous_occurrences_do_not() -> (
    None
):
    equivalences = EquivalenceSet(
        1,
        (
            DirectionalEquivalence(
                ("twenty", "one"),
                ("twenty-one",),
                "number_form",
            ),
        ),
    )

    unique = align_token_sequences(
        ("chapter", "twenty", "one"),
        ("chapter", "twenty-one"),
        equivalences,
    )
    ambiguous = align_token_sequences(
        ("twenty", "one", "then", "twenty", "one"),
        ("twenty-one", "then", "twenty-one"),
        equivalences,
    )

    assert any(item.operation == "equivalent" for item in unique)
    assert all(item.operation != "equivalent" for item in ambiguous)


def test_calibration_is_fitted_independently_per_engine_and_clip_class() -> None:
    clip_class = ClipClass.SHORT_PHRASE
    good_whisper = _recognition("whisper", ("wren", "asked"))
    bad_whisper = replace(
        good_whisper,
        evidence=WhisperEvidence(-2.0, 1.0, 0.01, 0.0),
    )
    good_parakeet = _recognition("parakeet", ("wren", "asked"))
    bad_parakeet = replace(
        good_parakeet,
        evidence=ParakeetEvidence(0.25, "greedy", 1_920_000, 240_000),
    )
    good_qwen = _recognition("qwen", ("wren", "asked"))
    bad_qwen = replace(good_qwen, evidence=QwenEvidence("length", 0, 100))
    observations = tuple(
        CalibrationObservation(identifier, clip_class, result, usable)
        for identifier, result, usable in (
            ("whisper-good", good_whisper, True),
            ("whisper-bad", bad_whisper, False),
            ("parakeet-good", good_parakeet, True),
            ("parakeet-bad", bad_parakeet, False),
            ("qwen-good", good_qwen, True),
            ("qwen-bad", bad_qwen, False),
        )
    )

    table = fit_calibration_table(
        observations,
        corpus_fingerprint=SHA_A,
        required_keys=tuple(
            (engine, clip_class) for engine in ("whisper", "parakeet", "qwen")
        ),
    )

    assert table.approved is False
    assert (
        table.threshold("whisper", clip_class).minimum_average_log_probability == -0.1
    )
    assert table.threshold("parakeet", clip_class).minimum_sentence_score == 0.99
    assert table.threshold("qwen", clip_class).allowed_finish_reasons == ("stop",)
    assert all(
        threshold.score_calibration_fingerprint == SHA_C
        for threshold in table.thresholds
    )
    decision = evaluate_consensus(
        expected_tokens=("wren", "asked"),
        recognitions=(good_whisper, good_parakeet),
        clip_class=clip_class,
        policy=_policy(),
        calibration=table,
        equivalences=EquivalenceSet(1, ()),
    )
    assert decision.result is not None
    assert decision.result.outcome is ConsensusOutcome.ACCEPTED


def test_calibration_fails_when_decode_evidence_is_not_separable() -> None:
    result = _recognition("qwen", ("wren", "asked"))
    observations = (
        CalibrationObservation("usable", ClipClass.SENTENCE, result, True),
        CalibrationObservation("unusable", ClipClass.SENTENCE, result, False),
    )

    with pytest.raises(ValidationError, match="cannot separate"):
        fit_calibration_table(
            observations,
            corpus_fingerprint=SHA_A,
            required_keys=(("qwen", ClipClass.SENTENCE),),
        )


def test_calibration_rejects_mixed_score_provenance() -> None:
    usable = _recognition("qwen", ("wren", "asked"))
    unusable = replace(
        usable,
        score_calibration_fingerprint=SHA_B,
        tokens=tuple(
            replace(token, calibration_fingerprint=SHA_B) for token in usable.tokens
        ),
        evidence=QwenEvidence("length", 0, 100),
    )

    with pytest.raises(ValidationError, match="score calibration identities"):
        fit_calibration_table(
            (
                CalibrationObservation("usable", ClipClass.SENTENCE, usable, True),
                CalibrationObservation("unusable", ClipClass.SENTENCE, unusable, False),
            ),
            corpus_fingerprint=SHA_A,
            required_keys=(("qwen", ClipClass.SENTENCE),),
        )


def test_consensus_records_bounded_directional_equivalences() -> None:
    expected = ("room", "609")
    equivalences = EquivalenceSet(
        1,
        (
            DirectionalEquivalence(
                expected=("609",),
                recognized=("six", "oh", "nine"),
                reason_code="spoken_room_number",
            ),
        ),
    )
    result = evaluate_consensus(
        expected_tokens=expected,
        recognitions=(
            _calibrated_recognition("whisper", ("room", "six", "oh", "nine")),
            _calibrated_recognition("parakeet", expected),
        ),
        clip_class=ClipClass.SENTENCE,
        policy=_policy(),
        calibration=_calibration(),
        equivalences=equivalences,
    ).result

    assert result is not None
    assert result.outcome is ConsensusOutcome.ACCEPTED
    assert result.equivalence_set_fingerprint == equivalences.fingerprint
    assert len(result.applied_equivalences) == 1
    application = result.applied_equivalences[0]
    assert application.engine == "whisper"
    assert (application.expected_start, application.expected_end) == (1, 2)
    assert (application.recognized_start, application.recognized_end) == (1, 4)
    assert application.reason_code == "spoken_room_number"
    Draft202012Validator(load_schema("speech-consensus")).validate(
        consensus_report(result)
    )


def test_invalid_decode_can_be_outvoted_but_forced_alignment_cannot_vote() -> None:
    expected = ("wren", "asked")
    result = evaluate_consensus(
        expected_tokens=expected,
        recognitions=(
            _calibrated_recognition("whisper", expected),
            _calibrated_recognition(
                "parakeet", expected, issues=("engine_decode_invalid",)
            ),
            _calibrated_recognition("qwen", expected),
        ),
        clip_class=ClipClass.SENTENCE,
        policy=_policy(),
        calibration=_calibration(),
        equivalences=EquivalenceSet(1, ()),
    ).result
    assert result is not None
    assert result.outcome is ConsensusOutcome.ACCEPTED

    forced_vote = cast(tuple[RecognitionResult, ...], (_alignment(),))
    with pytest.raises(TypeError, match="RecognitionResult"):
        evaluate_consensus(
            expected_tokens=expected,
            recognitions=forced_vote,
            clip_class=ClipClass.SENTENCE,
            policy=_policy(),
            calibration=_calibration(),
            equivalences=EquivalenceSet(1, ()),
        )


def test_shadow_classifies_known_defects_without_changing_baseline() -> None:
    consensus = evaluate_consensus(
        expected_tokens=("no",),
        recognitions=(
            _calibrated_recognition("whisper", ("no",)),
            _calibrated_recognition("parakeet", ("now",)),
            _calibrated_recognition("qwen", ("no",)),
        ),
        clip_class=ClipClass.ONE_WORD,
        policy=_policy(),
        calibration=_calibration(),
        equivalences=EquivalenceSet(1, ()),
    ).result
    assert consensus is not None
    assert consensus.outcome is ConsensusOutcome.REJECTED

    caught = compare_shadow_decision(
        audio_digest=SHA_A,
        baseline_accepted=True,
        baseline_reason_codes=(),
        consensus=consensus,
        ground_truth=ShadowGroundTruth.KNOWN_DEFECTIVE,
    )
    unknown = compare_shadow_decision(
        audio_digest=SHA_A,
        baseline_accepted=True,
        baseline_reason_codes=(),
        consensus=consensus,
    )

    assert caught.classification is ShadowClassification.ADDITIONAL_DEFECT
    assert unknown.classification is ShadowClassification.FALSE_REJECTION_CANDIDATE
    Draft202012Validator(load_schema("speech-analysis-shadow")).validate(
        caught.to_dict()
    )


class _FakeFactory(EngineFactory):
    def __init__(
        self,
        recognizer: SpeechRecognizer,
        forced_aligner: ForcedAligner,
    ) -> None:
        self._recognizer = recognizer
        self._forced_aligner = forced_aligner

    def recognizer(self, engine: str) -> SpeechRecognizer:
        if engine != "whisper":
            raise WorkerProtocolError("unexpected recognizer")
        return self._recognizer

    def forced_aligner(self, engine: str) -> ForcedAligner:
        if engine != "qwen-forced":
            raise WorkerProtocolError("unexpected forced aligner")
        return self._forced_aligner


def _write_wav(path: Path) -> AudioSpan:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\0\0" * 1_600)
    return AudioSpan(sha256_file(path), 0, 1_600, 16_000)


@pytest.mark.asyncio
async def test_worker_application_matches_direct_fake_adapter_and_unloads(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "window.wav"
    span = _write_wav(audio)
    recognizer = FakeSpeechRecognizer(
        SHA_B,
        lambda _path, _language, actual_span: _recognition(
            "whisper", ("wren", "asked"), span=actual_span
        ),
    )
    aligner = FakeForcedAligner(
        SHA_C,
        lambda _path, _text, _language, _purpose, _verified, actual_span: _alignment(
            actual_span
        ),
    )
    application = AnalysisWorkerApplication(
        family="whisper",
        audio_root=tmp_path,
        factory=_FakeFactory(recognizer, aligner),
        pid=42,
    )
    request = RecognitionWorkerRequest(
        "recognize-1",
        "whisper",
        recognizer.fingerprint,
        audio.name,
        span.audio_digest,
        "en",
        span,
    )

    direct = await recognizer.recognize(audio, language="en", span=span)
    response = await application.handle(request)
    assert response.result == direct
    assert application.status().loaded_engines == ("whisper",)
    assert application.status().model_loads == (WorkerModelLoad("whisper", 1),)

    await application.handle(UnloadWorkerRequest("unload-1", "whisper"))
    assert application.status().loaded_engines == ()

    with pytest.raises(ModelIntegrityError, match="digest"):
        await application.handle(
            RecognitionWorkerRequest(
                "recognize-2",
                "whisper",
                recognizer.fingerprint,
                audio.name,
                SHA_A,
                "en",
                span,
            )
        )


@pytest.mark.asyncio
async def test_worker_batch_yields_committable_items_and_one_model_load(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "window.wav"
    span = _write_wav(audio)
    recognizer = FakeSpeechRecognizer(
        SHA_B,
        lambda _path, _language, actual_span: _recognition(
            "whisper", ("wren", "asked"), span=actual_span
        ),
    )
    application = AnalysisWorkerApplication(
        family="whisper",
        audio_root=tmp_path,
        factory=_FakeFactory(
            recognizer,
            FakeForcedAligner(
                SHA_C,
                lambda _path, _text, _language, _purpose, _verified, actual_span: (
                    _alignment(actual_span)
                ),
            ),
        ),
        pid=42,
    )
    request = RecognitionManyWorkerRequest(
        request_id="operation-1",
        batch_id="batch-1",
        engine="whisper",
        engine_fingerprint=SHA_B,
        timeout_milliseconds=10_000,
        items=tuple(
            RecognitionBatchItem(
                index=index,
                request_fingerprint=f"{index + 1:064x}",
                relative_audio_path=audio.name,
                audio_digest=span.audio_digest,
                language="en",
                span=span,
            )
            for index in range(2)
        ),
    )

    frames = [frame async for frame in application.handle_batch(request)]

    completions = [frame for frame in frames if isinstance(frame, WorkerItemSuccess)]
    assert [frame.index for frame in completions] == [0, 1]
    assert all(
        frame.terminal_status is OperationTerminalStatus.COMPLETED
        for frame in completions
    )
    assert [frame.metrics.model_loads for frame in completions] == [1, 0]
    assert frames[-1] == WorkerBatchFinished("operation-1", "batch-1", 2)
    assert application.status().model_loads == (WorkerModelLoad("whisper", 1),)


class _ProtocolWorkerDouble:
    def __init__(self, span: AudioSpan) -> None:
        self.span = span
        self.batch: RecognitionManyWorkerRequest | None = None
        self.soft_cancelled: str | None = None
        self.closed = False

    async def start(self) -> object:
        return build_worker_handshake(
            family="whisper",
            engines=("whisper",),
            worker_artifact_fingerprint=SHA_A,
            environment_lock_fingerprint=SHA_B,
            adapter_fingerprint=SHA_C,
        )

    async def request_many(
        self,
        request: RecognitionManyWorkerRequest,
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[WorkerItemSuccess]:
        assert timeout_seconds > 0
        self.batch = request
        for item in request.items:
            yield WorkerItemSuccess(
                request.request_id,
                request.batch_id,
                item.index,
                item.request_fingerprint,
                _recognition("whisper", ("wren", "asked"), span=self.span),
                OperationMetrics(1_600, 1_600, 10, 20, 30, False, 1),
            )

    async def soft_cancel(self, request_id: str) -> WorkerFailure:
        self.soft_cancelled = request_id
        return WorkerFailure(
            request_id,
            WorkerErrorCode.CANCELLED,
            "Analysis worker request was cancelled",
            True,
        )

    async def wait_idle(self) -> None:
        return

    async def close(self) -> None:
        self.closed = True

    async def request(
        self,
        request: AnalysisWorkerRequest,
        *,
        timeout_seconds: float,
        restart_once: bool,
    ) -> AnalysisWorkerResponse:
        del timeout_seconds, restart_once
        if isinstance(request, UnloadWorkerRequest):
            return WorkerSuccess(request.request_id, None)
        return WorkerSuccess(
            request.request_id,
            WorkerStatus(
                pid=1,
                family="whisper",
                python_version="3.14.4",
                environment_fingerprint=SHA_A,
                ready=True,
                loaded_engines=(),
                active_request_id=None,
                completed_requests=1,
                failed_requests=0,
                model_loads=(),
                peak_resident_memory_bytes=100,
                metal_active_memory_bytes=20,
                metal_peak_memory_bytes=30,
                resident_memory_bytes=40,
            ),
        )


@pytest.mark.asyncio
async def test_protocol_supervisor_bridge_executes_and_reports_current_unload_memory(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "window.wav"
    span = _write_wav(audio)
    worker_double = _ProtocolWorkerDouble(span)

    def request_factory(operation: BatchOperation) -> RecognitionManyWorkerRequest:
        return RecognitionManyWorkerRequest(
            "placeholder",
            "placeholder",
            "placeholder",
            SHA_C,
            30_000,
            tuple(
                RecognitionBatchItem(
                    item.index,
                    item.request_fingerprint,
                    audio.name,
                    span.audio_digest,
                    "en",
                    span,
                )
                for item in operation.items
            ),
        )

    bridge = ProtocolSupervisedWorker(
        cast(IsolatedAnalysisWorker, worker_double),
        request_factory=request_factory,
    )
    operation = BatchOperation(
        "operation-1",
        "batch-1",
        "whisper",
        SHA_B,
        "recognize_many",
        time.monotonic_ns() + 10_000_000_000,
        (BatchWorkItem(0, SHA_C, audio.name, 0, 1_600),),
    )

    completions = [item async for item in bridge.execute(operation)]
    watermark = await bridge.unload("whisper")

    assert await bridge.handshake() == await worker_double.start()
    assert len(completions) == 1
    assert completions[0].terminal_status is OperationTerminalStatus.COMPLETED
    assert completions[0].evidence_fingerprint is not None
    assert worker_double.batch is not None
    assert worker_double.batch.request_id == "operation-1"
    assert worker_double.batch.batch_id == "batch-1"
    assert worker_double.batch.engine_fingerprint == SHA_B
    assert watermark.resident_bytes == 40
    assert watermark.accelerator_bytes == 20


class _BlockingRecognizer:
    fingerprint = SHA_B

    def __init__(self, span: AudioSpan) -> None:
        self.span = span
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def recognize(
        self,
        audio: Path,
        *,
        language: str,
        span: AudioSpan | None = None,
    ) -> RecognitionResult:
        del audio
        assert language == "en"
        assert span == self.span
        self.started.set()
        await self.release.wait()
        return _recognition("whisper", ("wren", "asked"), span=span)


@pytest.mark.asyncio
async def test_worker_serves_status_and_cancellation_while_inference_is_active(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "window.wav"
    span = _write_wav(audio)
    recognizer = _BlockingRecognizer(span)
    application = AnalysisWorkerApplication(
        family="whisper",
        audio_root=tmp_path,
        factory=_FakeFactory(
            recognizer,
            FakeForcedAligner(
                SHA_C,
                lambda _path, _text, _language, _purpose, _verified, actual_span: (
                    _alignment(actual_span)
                ),
            ),
        ),
        pid=42,
    )
    inputs: queue.Queue[bytes] = queue.Queue()
    outputs: list[bytes] = []

    def reader(_maximum: int) -> bytes:
        return inputs.get(timeout=5)

    def writer(value: bytes) -> None:
        outputs.append(value)

    handshake = build_worker_handshake(
        family="whisper",
        engines=("whisper",),
        worker_artifact_fingerprint=SHA_A,
        environment_lock_fingerprint=SHA_B,
        adapter_fingerprint=SHA_C,
    )
    serving = asyncio.create_task(_serve(application, handshake, reader, writer))
    inputs.put(
        encode_worker_request(
            RecognitionWorkerRequest(
                "recognize-1",
                "whisper",
                SHA_B,
                audio.name,
                span.audio_digest,
                "en",
                span,
            )
        )
        + b"\n"
    )
    await recognizer.started.wait()
    inputs.put(encode_worker_request(StatusWorkerRequest("status-1")) + b"\n")
    inputs.put(
        encode_worker_request(CancellationWorkerRequest("cancel-1", "recognize-1"))
        + b"\n"
    )
    await _wait_for_output_count(outputs, 3)

    status = parse_worker_response(outputs[1].rstrip(b"\n"))
    cancellation = parse_worker_response(outputs[2].rstrip(b"\n"))
    assert isinstance(status, WorkerSuccess)
    assert isinstance(status.result, WorkerStatus)
    assert status.result.active_request_id == "recognize-1"
    assert cancellation == WorkerSuccess("cancel-1", None)

    recognizer.release.set()
    await _wait_for_output_count(outputs, 4)
    inputs.put(encode_worker_request(ShutdownWorkerRequest("shutdown-1")) + b"\n")
    assert await asyncio.wait_for(serving, timeout=5) == 0
    inference = parse_worker_response(outputs[3].rstrip(b"\n"))
    assert isinstance(inference, WorkerSuccess)
    assert isinstance(inference.result, RecognitionResult)


async def _wait_for_output_count(outputs: list[bytes], count: int) -> None:
    for _attempt in range(1_000):
        if len(outputs) >= count:
            return
        await asyncio.sleep(0.001)
    raise AssertionError("Worker did not emit the expected control response")


@pytest.mark.asyncio
async def test_worker_family_processes_are_independent(tmp_path: Path) -> None:
    def worker(family: str) -> IsolatedAnalysisWorker:
        return IsolatedAnalysisWorker(
            BUILT_IN_WORKERS[family],
            audio_root=tmp_path / "audio",
            model_root=tmp_path / "models",
            calibration_fingerprint=SHA_A,
            worker_artifact_digest=SHA_B,
            lock_digest=SHA_C,
            python_executable=Path(sys.executable),
            environment=os.environ,
        )

    whisper = worker("whisper")
    parakeet = worker("parakeet")
    try:
        whisper_status = await whisper.request(
            StatusWorkerRequest("status-whisper"), timeout_seconds=10
        )
        parakeet_status = await parakeet.request(
            StatusWorkerRequest("status-parakeet"), timeout_seconds=10
        )
        assert isinstance(whisper_status, WorkerSuccess)
        assert isinstance(parakeet_status, WorkerSuccess)
        parakeet_pid = parakeet.pid

        await whisper.cancel("active-whisper-request")
        still_ready = await parakeet.request(
            StatusWorkerRequest("status-parakeet-again"), timeout_seconds=10
        )
        assert isinstance(still_ready, WorkerSuccess)
        assert parakeet.pid == parakeet_pid

        with pytest.raises(WorkerProtocolError, match="outside worker family"):
            await whisper.request(
                RecognitionWorkerRequest(
                    "wrong-family",
                    "qwen",
                    SHA_B,
                    "window.wav",
                    SHA_A,
                    "en",
                    AudioSpan(SHA_A, 0, 1, 16_000),
                ),
                timeout_seconds=10,
            )
    finally:
        await whisper.close()
        await parakeet.close()


class _BlockingSupervisorWorker(IsolatedAnalysisWorker):
    def __init__(self, tmp_path: Path) -> None:
        super().__init__(
            BUILT_IN_WORKERS["whisper"],
            audio_root=tmp_path / "audio",
            model_root=tmp_path / "models",
            calibration_fingerprint=SHA_A,
            worker_artifact_digest=SHA_B,
            lock_digest=SHA_C,
        )
        self.started = asyncio.Event()
        self.terminated = asyncio.Event()

    async def _exchange(self, request: AnalysisWorkerRequest) -> AnalysisWorkerResponse:
        del request
        self.started.set()
        await self.terminated.wait()
        raise EOFError

    async def _terminate(self) -> None:
        self.terminated.set()


class _TerminatedSupervisorWorker(_BlockingSupervisorWorker):
    async def _exchange(self, request: AnalysisWorkerRequest) -> AnalysisWorkerResponse:
        del request
        raise EOFError

    async def _terminate(self) -> None:
        return


@pytest.mark.asyncio
async def test_unexpected_worker_exit_is_not_misreported_as_timeout(
    tmp_path: Path,
) -> None:
    worker = _TerminatedSupervisorWorker(tmp_path)

    response = await worker.request(
        StatusWorkerRequest("terminated-request"),
        timeout_seconds=10,
    )

    assert isinstance(response, WorkerFailure)
    assert response.code.value == "worker_terminated"
    assert response.retryable is True


@pytest.mark.asyncio
async def test_active_cancellation_is_not_replayed_by_restart(
    tmp_path: Path,
) -> None:
    worker = _BlockingSupervisorWorker(tmp_path)
    request = StatusWorkerRequest("active-request")
    pending = asyncio.create_task(worker.request(request, timeout_seconds=10))
    await worker.started.wait()

    wrong = await worker.cancel("different-request")
    assert wrong.code.value == "invalid_request"
    cancelled = await worker.cancel(request.request_id)
    response = await pending

    assert cancelled.code.value == "cancelled"
    assert isinstance(response, WorkerFailure)
    assert response.code.value == "cancelled"


@pytest.mark.asyncio
async def test_soft_cancellation_precedes_forced_termination_for_blocked_inference(
    tmp_path: Path,
) -> None:
    worker = _BlockingSupervisorWorker(tmp_path)
    request = StatusWorkerRequest("active-request")
    pending = asyncio.create_task(worker.request(request, timeout_seconds=10))
    await worker.started.wait()

    soft = await worker.soft_cancel(request.request_id)
    idle = asyncio.create_task(worker.wait_idle())
    await asyncio.sleep(0)

    assert soft.code.value == "cancelled"
    assert not idle.done()
    await worker.cancel(request.request_id)
    response = await pending
    await idle
    assert isinstance(response, WorkerFailure)
    assert response.code.value == "cancelled"


@pytest.mark.asyncio
async def test_cancellation_during_handshake_returns_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def interrupted_handshake(
        _self: IsolatedAnalysisWorker,
        _request: AnalysisWorkerRequest,
    ) -> AnalysisWorkerResponse:
        started.set()
        await release.wait()
        raise WorkerProtocolError("Analysis worker handshake is incomplete")

    worker = IsolatedAnalysisWorker(
        BUILT_IN_WORKERS["whisper"],
        audio_root=tmp_path / "audio",
        model_root=tmp_path / "models",
        calibration_fingerprint=SHA_A,
        worker_artifact_digest=SHA_B,
        lock_digest=SHA_C,
    )
    monkeypatch.setattr(worker, "_exchange", interrupted_handshake.__get__(worker))
    request = StatusWorkerRequest("handshake-cancellation")
    pending = asyncio.create_task(worker.request(request, timeout_seconds=10))
    await started.wait()

    cancellation = await worker.cancel(request.request_id)
    release.set()
    response = await pending

    assert cancellation.code.value == "cancelled"
    assert isinstance(response, WorkerFailure)
    assert response.code.value == "cancelled"


@pytest.mark.asyncio
async def test_framed_transport_rejects_truncated_unexpected_and_oversized_stdout(
    tmp_path: Path,
) -> None:
    worker = IsolatedAnalysisWorker(
        BUILT_IN_WORKERS["whisper"],
        audio_root=tmp_path / "audio",
        model_root=tmp_path / "models",
        calibration_fingerprint=SHA_A,
        worker_artifact_digest=SHA_B,
        lock_digest=SHA_C,
    )

    truncated = asyncio.StreamReader()
    truncated.feed_data(b'{"protocol_version":2')
    truncated.feed_eof()
    with pytest.raises(EOFError):
        await worker._read_response(
            cast(asyncio.subprocess.Process, SimpleNamespace(stdout=truncated))
        )

    unexpected = asyncio.StreamReader()
    unexpected.feed_data(b"upstream debug output\n")
    with pytest.raises(WorkerProtocolError):
        await worker._read_response(
            cast(asyncio.subprocess.Process, SimpleNamespace(stdout=unexpected))
        )

    oversized = asyncio.StreamReader(limit=64)
    oversized.feed_data((b"x" * 128) + b"\n")
    with pytest.raises(WorkerProtocolError, match="too large"):
        await worker._read_response(
            cast(asyncio.subprocess.Process, SimpleNamespace(stdout=oversized))
        )


@pytest.mark.asyncio
async def test_worker_discards_stderr_instead_of_buffering_unbounded_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = BUILT_IN_WORKERS["whisper"]
    handshake = build_worker_handshake(
        family=definition.family,
        engines=definition.engines,
        worker_artifact_fingerprint=SHA_B,
        environment_lock_fingerprint=SHA_C,
        adapter_fingerprint=definition.fingerprint,
    )
    stdout = asyncio.StreamReader()
    stdout.feed_data(encode_worker_handshake(handshake) + b"\n")
    stopped = asyncio.Event()

    class Stdin:
        def write(self, _value: bytes) -> None:
            return

        async def drain(self) -> None:
            return

    class Process:
        stdin = Stdin()
        returncode: int | None = None
        pid = 42

        def __init__(self) -> None:
            self.stdout = stdout

        def terminate(self) -> None:
            self.returncode = -15
            stopped.set()

        def kill(self) -> None:
            self.returncode = -9
            stopped.set()

        async def wait(self) -> int:
            await stopped.wait()
            return self.returncode or 0

    options: dict[str, object] = {}

    async def spawn(*_arguments: object, **kwargs: object) -> Process:
        options.update(kwargs)
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    worker = IsolatedAnalysisWorker(
        definition,
        audio_root=tmp_path / "audio",
        model_root=tmp_path / "models",
        calibration_fingerprint=SHA_A,
        worker_artifact_digest=SHA_B,
        lock_digest=SHA_C,
    )

    await worker.start()
    await worker._terminate()

    assert options["stderr"] is asyncio.subprocess.DEVNULL


def test_worker_uses_only_a_digest_bound_explicit_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "analysis-worker.pyz"
    artifact.write_bytes(b"worker")
    digest = hashlib.sha256(b"worker").hexdigest()
    worker = IsolatedAnalysisWorker(
        BUILT_IN_WORKERS["whisper"],
        audio_root=tmp_path / "audio",
        model_root=tmp_path / "models",
        calibration_fingerprint=SHA_A,
        worker_artifact_digest=digest,
        lock_digest=SHA_C,
        worker_artifact_path=artifact,
    )

    assert worker._worker_entrypoint() == (str(artifact),)

    artifact.write_bytes(b"changed")
    with pytest.raises(WorkerProtocolError, match="digest differs"):
        worker._worker_entrypoint()


@pytest.mark.asyncio
async def test_graceful_close_terminates_child_that_never_closes_its_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped = asyncio.Event()

    class Process:
        returncode: int | None = None

        def terminate(self) -> None:
            self.returncode = -15
            stopped.set()

        def kill(self) -> None:
            self.returncode = -9
            stopped.set()

        async def wait(self) -> int:
            await stopped.wait()
            return self.returncode or 0

    async def successful_shutdown(
        _self: IsolatedAnalysisWorker,
        request: AnalysisWorkerRequest,
    ) -> AnalysisWorkerResponse:
        return WorkerSuccess(request.request_id, None)

    monkeypatch.setattr(
        "yakbox.speech.analysis_runtime._GRACEFUL_SHUTDOWN_SECONDS", 0.001
    )
    monkeypatch.setattr(IsolatedAnalysisWorker, "_exchange", successful_shutdown)
    worker = IsolatedAnalysisWorker(
        BUILT_IN_WORKERS["whisper"],
        audio_root=tmp_path / "audio",
        model_root=tmp_path / "models",
        calibration_fingerprint=SHA_A,
        worker_artifact_digest=SHA_B,
        lock_digest=SHA_C,
    )
    process = Process()
    worker._process = cast(asyncio.subprocess.Process, process)

    await asyncio.wait_for(worker.close(), timeout=1)

    assert stopped.is_set()
    assert worker.pid is None


def test_worker_environment_drops_credentials_proxies_indexes_and_python_paths(
    tmp_path: Path,
) -> None:
    source = {
        "PATH": "/usr/bin",
        "LANG": "en_US.UTF-8",
        "TMPDIR": str(tmp_path),
        "AWS_SECRET_ACCESS_KEY": "secret",
        "HF_TOKEN": "secret",
        "HTTP_PROXY": "https://proxy.invalid",
        "HTTPS_PROXY": "https://proxy.invalid",
        "PIP_INDEX_URL": "https://index.invalid",
        "UV_INDEX": "https://index.invalid",
        "PYTHONPATH": "/untrusted",
        "UNRELATED_PARENT_VALUE": "private",
    }

    worker = IsolatedAnalysisWorker(
        BUILT_IN_WORKERS["whisper"],
        audio_root=tmp_path / "audio",
        model_root=tmp_path / "models",
        calibration_fingerprint=SHA_A,
        worker_artifact_digest=SHA_B,
        lock_digest=SHA_C,
        environment=source,
    )

    assert worker.environment == {
        "PATH": "/usr/bin",
        "LANG": "en_US.UTF-8",
        "TMPDIR": str(tmp_path),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def test_worker_process_audit_guard_blocks_network_but_preserves_local_ipc() -> None:
    script = """
import socket
from yakbox.speech.analysis_worker import _install_offline_audit_guard

_install_offline_audit_guard()
left, right = socket.socketpair()
left.close()
right.close()
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except PermissionError:
    print("offline")
else:
    raise SystemExit("internet socket unexpectedly allowed")
"""

    completed = subprocess.run(  # noqa: S603 - fixed interpreter and test script
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert completed.stdout == b"offline\n"


def test_importing_yakbox_does_not_import_model_runtimes() -> None:
    script = """
import json
import sys
import yakbox
import yakbox.speech.analysis_adapters
print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in {
    'mlx', 'mlx_audio', 'mlx_whisper', 'parakeet_mlx', 'transformers'
})))
"""
    completed = subprocess.run(  # noqa: S603 - fixed current-interpreter argv
        [sys.executable, "-I", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(completed.stdout) == []
