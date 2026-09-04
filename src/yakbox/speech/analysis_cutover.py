"""Fail-closed readiness decision for the speech-analysis public cutover."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from yakbox.contracts import runtime_metadata
from yakbox.errors import ValidationError
from yakbox.speech.analysis_evaluation import ReviewDecision, SpeechAnalysisEvaluation
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_performance_qualification import (
    SpeechPerformanceQualification,
)
from yakbox.speech.analysis_policy import CalibrationTable
from yakbox.speech.analysis_review import (
    QualificationReview,
    QualificationReviewStatus,
)
from yakbox.speech.analysis_runtime_install import AnalysisRuntimeReport
from yakbox.speech.model_registry import ModelStatus

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ISSUE = re.compile(r"[a-z][a-z0-9_]{1,127}")
_REQUIRED_RUNTIME_FAMILIES = frozenset({"whisper", "parakeet", "qwen"})
_REQUIRED_MODEL_ENGINES = frozenset({"whisper", "parakeet", "qwen", "qwen-forced"})


class CutoverEvidenceKind(StrEnum):
    """Independent proofs that are not represented by quality artifacts."""

    APPLE_SILICON_REAL_MODELS = "apple_silicon_real_models"
    AUTOMATED_TESTS = "automated_tests"
    DEPENDENCY_SBOM_AUDIT = "dependency_sbom_audit"
    INSTALLED_WHEEL_RUNTIME_MATRIX = "installed_wheel_runtime_matrix"
    LISTENING_REVIEW = "listening_review"
    PACKAGE_RELEASE_PREFLIGHT = "package_release_preflight"
    REPEATED_CALL_ENDURANCE = "repeated_call_endurance"


@dataclass(frozen=True, slots=True)
class CutoverEvidence:
    kind: CutoverEvidenceKind
    fingerprint: str
    passed: bool

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.fingerprint) is None:
            raise ValidationError("Cutover evidence fingerprint must be a SHA-256")


@dataclass(frozen=True, slots=True)
class CutoverGate:
    name: str
    passed: bool
    evidence_fingerprints: tuple[str, ...]
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        if _ISSUE.fullmatch(self.name) is None:
            raise ValidationError("Cutover gate name is invalid")
        if (
            not self.evidence_fingerprints
            or tuple(sorted(set(self.evidence_fingerprints)))
            != self.evidence_fingerprints
            or any(
                _SHA256.fullmatch(item) is None for item in self.evidence_fingerprints
            )
            or any(_ISSUE.fullmatch(item) is None for item in self.issues)
            or self.passed == bool(self.issues)
        ):
            raise ValidationError("Cutover gate evidence is inconsistent")


@dataclass(frozen=True, slots=True)
class SpeechAnalysisCutoverReadiness:
    gates: tuple[CutoverGate, ...]
    ready: bool

    def __post_init__(self) -> None:
        names = tuple(item.name for item in self.gates)
        if (
            not self.gates
            or names != tuple(sorted(set(names)))
            or self.ready != all(item.passed for item in self.gates)
        ):
            raise ValidationError("Cutover readiness gates are inconsistent")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-analysis-cutover-readiness-v1", self)

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("speech-analysis-cutover-readiness"),
            "fingerprint": self.fingerprint,
            "ready": self.ready,
            "gates": [
                {
                    "name": item.name,
                    "passed": item.passed,
                    "evidence_fingerprints": list(item.evidence_fingerprints),
                    "issues": list(item.issues),
                }
                for item in self.gates
            ],
        }


def evaluate_cutover_readiness(
    *,
    evaluation: SpeechAnalysisEvaluation,
    calibration: CalibrationTable,
    review: QualificationReview,
    performance: SpeechPerformanceQualification,
    runtimes: AnalysisRuntimeReport,
    models: tuple[ModelStatus, ...],
    independent_evidence: tuple[CutoverEvidence, ...],
) -> SpeechAnalysisCutoverReadiness:
    """Require every preregistered proof before public cutover can be ready."""
    gates = (
        _quality_gate(evaluation),
        _review_gate(evaluation, calibration, review),
        _performance_gate(performance),
        _runtime_gate(runtimes),
        _model_gate(models),
        _independent_gate(independent_evidence),
    )
    ordered = tuple(sorted(gates, key=lambda item: item.name))
    return SpeechAnalysisCutoverReadiness(
        ordered,
        all(item.passed for item in ordered),
    )


def _quality_gate(evaluation: SpeechAnalysisEvaluation) -> CutoverGate:
    issues = () if evaluation.passed else ("quality_evaluation_failed",)
    return _gate("quality_evaluation", (evaluation.fingerprint,), issues)


def _review_gate(
    evaluation: SpeechAnalysisEvaluation,
    calibration: CalibrationTable,
    review: QualificationReview,
) -> CutoverGate:
    issues: list[str] = []
    if review.status is not QualificationReviewStatus.APPROVED:
        issues.append("qualification_review_not_approved")
    if any(entry.decision is not ReviewDecision.APPROVED for entry in review.entries):
        issues.append("qualification_review_rejected")
    if (
        review.evaluation_fingerprint != evaluation.fingerprint
        or review.corpus_fingerprint != evaluation.corpus_fingerprint
        or review.policy_fingerprint != evaluation.policy_fingerprint
        or review.protocol_fingerprint != evaluation.protocol_fingerprint
    ):
        issues.append("qualification_review_binding_mismatch")
    if not calibration.approved:
        issues.append("calibration_not_approved")
    if calibration.corpus_fingerprint != evaluation.corpus_fingerprint:
        issues.append("calibration_corpus_mismatch")
    if calibration.reviewer_disposition_fingerprint != review.fingerprint:
        issues.append("calibration_review_binding_mismatch")
    return _gate(
        "calibration_and_review",
        (
            evaluation.fingerprint,
            calibration.disposition_fingerprint,
            review.fingerprint,
        ),
        tuple(issues),
    )


def _performance_gate(
    performance: SpeechPerformanceQualification,
) -> CutoverGate:
    issues = () if performance.passed else ("performance_qualification_failed",)
    return _gate("performance_qualification", (performance.fingerprint,), issues)


def _runtime_gate(runtimes: AnalysisRuntimeReport) -> CutoverGate:
    families = tuple(item.family for item in runtimes.runtimes)
    issues: list[str] = []
    if len(families) != len(set(families)):
        issues.append("runtime_family_duplicate")
    if set(families) != _REQUIRED_RUNTIME_FAMILIES:
        issues.append("runtime_family_coverage_incomplete")
    if not runtimes.verified:
        issues.append("runtime_verification_failed")
    fingerprints = tuple(item.install_fingerprint for item in runtimes.runtimes)
    return _gate("frozen_runtimes", fingerprints or ("0" * 64,), tuple(issues))


def _model_gate(models: tuple[ModelStatus, ...]) -> CutoverGate:
    engines = tuple(item.engine for item in models)
    issues: list[str] = []
    if len(engines) != len(set(engines)):
        issues.append("model_engine_duplicate")
    if set(engines) != _REQUIRED_MODEL_ENGINES:
        issues.append("model_engine_coverage_incomplete")
    if not models or any(not item.verified for item in models):
        issues.append("model_verification_failed")
    fingerprints = tuple(
        item.directory_fingerprint
        for item in models
        if item.directory_fingerprint is not None
    )
    if len(fingerprints) != len(models):
        issues.append("model_fingerprint_missing")
    return _gate("pinned_models", fingerprints or ("0" * 64,), tuple(issues))


def _independent_gate(evidence: tuple[CutoverEvidence, ...]) -> CutoverGate:
    kinds = tuple(item.kind for item in evidence)
    issues: list[str] = []
    if len(kinds) != len(set(kinds)):
        issues.append("independent_evidence_duplicate")
    if set(kinds) != set(CutoverEvidenceKind):
        issues.append("independent_evidence_incomplete")
    if not evidence or any(not item.passed for item in evidence):
        issues.append("independent_evidence_failed")
    return _gate(
        "independent_release_evidence",
        tuple(item.fingerprint for item in evidence) or ("0" * 64,),
        tuple(issues),
    )


def _gate(
    name: str,
    fingerprints: tuple[str, ...],
    issues: tuple[str, ...],
) -> CutoverGate:
    return CutoverGate(
        name,
        not issues,
        tuple(sorted(set(fingerprints))),
        tuple(sorted(set(issues))),
    )


__all__ = [
    "CutoverEvidence",
    "CutoverEvidenceKind",
    "CutoverGate",
    "SpeechAnalysisCutoverReadiness",
    "evaluate_cutover_readiness",
]
