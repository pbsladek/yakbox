from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from jsonschema import Draft202012Validator

from yakbox.schemas import load_schema
from yakbox.speech.analysis_cutover import (
    CutoverEvidence,
    CutoverEvidenceKind,
    SpeechAnalysisCutoverReadiness,
    evaluate_cutover_readiness,
)
from yakbox.speech.analysis_evaluation import (
    ReviewDecision,
    ReviewKind,
    SpeechAnalysisEvaluation,
)
from yakbox.speech.analysis_models import ClipClass
from yakbox.speech.analysis_performance_qualification import (
    SpeechPerformanceQualification,
)
from yakbox.speech.analysis_policy import CalibrationTable, CalibrationThreshold
from yakbox.speech.analysis_review import (
    QualificationReview,
    QualificationReviewEntry,
    QualificationReviewStatus,
)
from yakbox.speech.analysis_runtime_install import (
    AnalysisRuntimeReport,
    InstalledAnalysisRuntime,
)
from yakbox.speech.model_registry import ModelStatus

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _evidence() -> tuple[
    SpeechAnalysisEvaluation,
    CalibrationTable,
    QualificationReview,
    SpeechPerformanceQualification,
    AnalysisRuntimeReport,
    tuple[ModelStatus, ...],
    tuple[CutoverEvidence, ...],
]:
    evaluation = SpeechAnalysisEvaluation(
        corpus_fingerprint=SHA_A,
        policy_fingerprint=SHA_B,
        protocol_fingerprint=SHA_C,
        cases=(),
        slices=(),
        candidate_false_accept_cluster_upper_bound=0.01,
        workflow_false_accept_cluster_upper_bound=0.01,
        candidate_false_acceptance_safe=True,
        workflow_false_acceptance_safe=True,
        false_rejection_noninferior=True,
        name_number_noninferior=True,
        forced_boundary_improved=True,
        crop_safety_noninferior=True,
        mandatory_defects_rejected=True,
        automatic_acceptances_explainable=True,
        passed=True,
    )
    review = QualificationReview(
        schema_version=1,
        yakbox_version="1.0.0",
        timestamp="2026-08-14T00:00:00+00:00",
        status=QualificationReviewStatus.APPROVED,
        evaluation_fingerprint=evaluation.fingerprint,
        corpus_fingerprint=evaluation.corpus_fingerprint,
        policy_fingerprint=evaluation.policy_fingerprint,
        protocol_fingerprint=evaluation.protocol_fingerprint,
        order_seed_fingerprint=SHA_D,
        entries=(
            QualificationReviewEntry(
                "case-1",
                ReviewKind.AUTOMATIC_ACCEPTANCE,
                1,
                SHA_E,
                ReviewDecision.APPROVED,
            ),
        ),
    )
    calibration = CalibrationTable(
        version=1,
        language="en",
        corpus_fingerprint=evaluation.corpus_fingerprint,
        approved=True,
        reviewer_disposition_fingerprint=review.fingerprint,
        execution_class_fingerprints=(SHA_A,),
        thresholds=(CalibrationThreshold("whisper", ClipClass.CHAPTER, SHA_B),),
    )
    performance = SpeechPerformanceQualification(
        protocol_fingerprint=SHA_A,
        observation_fingerprint=SHA_B,
        slices=(),
        coverage_complete=True,
        repeat_uses_zero_inference=True,
        localized_repair_within_fraction=True,
        multi_repair_is_single_pass_and_scoped=True,
        model_loads_bounded=True,
        qwen_is_policy_scoped=True,
        offline_operation_passed=True,
        delivery_verification_separate=True,
        passed=True,
    )
    runtimes = AnalysisRuntimeReport(
        Path("/managed/runtimes"),
        tuple(
            InstalledAnalysisRuntime(
                family,
                fingerprint,
                Path("/managed/runtimes") / family,
                Path("/managed/runtimes") / family / "python",
                SHA_D,
                SHA_E,
                True,
                (),
            )
            for family, fingerprint in (
                ("whisper", SHA_A),
                ("parakeet", SHA_B),
                ("qwen", SHA_C),
            )
        ),
    )
    models = tuple(
        ModelStatus(
            engine,
            True,
            True,
            Path("/managed/models") / engine,
            1,
            1,
            fingerprint,
            True,
            "1.0.0",
            (),
        )
        for engine, fingerprint in (
            ("whisper", SHA_A),
            ("parakeet", SHA_B),
            ("qwen", SHA_C),
            ("qwen-forced", SHA_D),
        )
    )
    independent = tuple(
        CutoverEvidence(kind, SHA_D, True) for kind in CutoverEvidenceKind
    )
    return (
        evaluation,
        calibration,
        review,
        performance,
        runtimes,
        models,
        independent,
    )


def _readiness(
    evidence: tuple[
        SpeechAnalysisEvaluation,
        CalibrationTable,
        QualificationReview,
        SpeechPerformanceQualification,
        AnalysisRuntimeReport,
        tuple[ModelStatus, ...],
        tuple[CutoverEvidence, ...],
    ],
) -> SpeechAnalysisCutoverReadiness:
    evaluation, calibration, review, performance, runtimes, models, independent = (
        evidence
    )
    return evaluate_cutover_readiness(
        evaluation=evaluation,
        calibration=calibration,
        review=review,
        performance=performance,
        runtimes=runtimes,
        models=models,
        independent_evidence=independent,
    )


def test_cutover_requires_all_bound_evidence_and_emits_schema_valid_report() -> None:
    readiness = _readiness(_evidence())

    assert readiness.ready
    assert len(readiness.gates) == 6
    assert all(gate.passed for gate in readiness.gates)
    Draft202012Validator(load_schema("speech-analysis-cutover-readiness")).validate(
        readiness.to_dict()
    )


def test_cutover_fails_closed_for_missing_or_failed_independent_evidence() -> None:
    values = _evidence()
    incomplete = (*values[:-1], values[-1][:-1])

    readiness = _readiness(incomplete)

    gate = next(
        item for item in readiness.gates if item.name == "independent_release_evidence"
    )
    assert not readiness.ready
    assert not gate.passed
    assert "independent_evidence_incomplete" in gate.issues


def test_cutover_requires_the_separate_forced_aligner_model() -> None:
    evaluation, calibration, review, performance, runtimes, models, independent = (
        _evidence()
    )

    readiness = _readiness(
        (
            evaluation,
            calibration,
            review,
            performance,
            runtimes,
            tuple(item for item in models if item.engine != "qwen-forced"),
            independent,
        )
    )

    gate = next(item for item in readiness.gates if item.name == "pinned_models")
    assert not readiness.ready
    assert "model_engine_coverage_incomplete" in gate.issues


def test_cutover_rejects_unbound_review_and_unverified_runtime_or_model() -> None:
    evaluation, calibration, review, performance, runtimes, models, independent = (
        _evidence()
    )
    changed_review = replace(review, evaluation_fingerprint=SHA_A)
    changed_calibration = replace(
        calibration,
        reviewer_disposition_fingerprint=changed_review.fingerprint,
    )
    changed_runtimes = replace(
        runtimes,
        runtimes=(
            replace(runtimes.runtimes[0], verified=False),
            *runtimes.runtimes[1:],
        ),
    )
    changed_models = (replace(models[0], verified=False), *models[1:])

    readiness = _readiness(
        (
            evaluation,
            changed_calibration,
            changed_review,
            performance,
            changed_runtimes,
            changed_models,
            independent,
        )
    )

    assert not readiness.ready
    failed = {item.name: item.issues for item in readiness.gates if not item.passed}
    assert "qualification_review_binding_mismatch" in failed["calibration_and_review"]
    assert "runtime_verification_failed" in failed["frozen_runtimes"]
    assert "model_verification_failed" in failed["pinned_models"]


def test_cutover_fingerprint_is_independent_of_input_order() -> None:
    values = _evidence()
    evaluation, calibration, review, performance, runtimes, models, independent = values
    reordered = (
        evaluation,
        calibration,
        review,
        performance,
        replace(runtimes, runtimes=tuple(reversed(runtimes.runtimes))),
        tuple(reversed(models)),
        tuple(reversed(independent)),
    )

    assert _readiness(values).fingerprint == _readiness(reordered).fingerprint
