"""Frozen-corpus evaluation, safety metrics, and reviewer qualification."""

from __future__ import annotations

import math
import re
import tomllib
import wave
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from statistics import NormalDist, median
from typing import cast

from yakbox._files import safe_child, sha256_file
from yakbox.contracts import runtime_metadata
from yakbox.errors import ValidationError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint, text_fingerprint
from yakbox.speech.analysis_models import ClipClass, ConsensusOutcome, ConsensusResult
from yakbox.speech.analysis_policy import CalibrationTable
from yakbox.speech.normalization import normalize_english
from yakbox.speech.shadow import (
    ShadowComparison,
    ShadowGroundTruth,
    compare_shadow_decision,
)

_MINIMUM_ONE_SIDED_CONFIDENCE = 0.5
_MEDIAN_PROBABILITY = 0.5
_SHA256_LENGTH = 64
_MINIMUM_WAV_BYTES = 44
_PCM16_SAMPLE_WIDTH = 2
_RISK_CLASS = re.compile(r"[a-z][a-z0-9_]{1,63}")


class CorpusPartition(StrEnum):
    CALIBRATION = "calibration"
    HELD_OUT = "held_out"
    REGRESSION = "regression"


class CorpusTruth(StrEnum):
    CLEAN = "clean"
    DEFECTIVE = "defective"


class ReviewKind(StrEnum):
    AUTOMATIC_ACCEPTANCE = "automatic_acceptance"
    VALID_DISSENT = "valid_dissent"
    BOUNDARY = "boundary"


class ReviewDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class EvaluationCorpusCase:
    """Licensed ground truth without manuscript text in generated reports."""

    case_id: str
    source_passage_group: str
    voice: str
    partition: CorpusPartition
    clip_class: ClipClass
    risk_class: str
    truth: CorpusTruth
    expected_tokens: tuple[str, ...]
    expected_tokens_hash: str
    expected_token_count: int
    expected_character_count: int
    audio_digest: str
    audio_frame_count: int
    sample_rate: int
    rights_id: str
    source_url: str
    boundary_review_required: bool = False
    relative_audio_path: str = ""
    accent: str = "unspecified"
    gender: str = "unspecified"
    speaking_rate: str = "unspecified"
    acoustic_condition: str = "clean"

    def __post_init__(self) -> None:
        if not all(
            (
                self.case_id,
                self.source_passage_group,
                self.voice,
                self.risk_class,
                self.rights_id,
                self.source_url,
                self.accent,
                self.gender,
                self.speaking_rate,
                self.acoustic_condition,
            )
        ):
            raise ValidationError("Evaluation corpus case identity is incomplete")
        if _RISK_CLASS.fullmatch(self.risk_class) is None:
            raise ValidationError("Evaluation case risk class is invalid")
        if self.expected_token_count <= 0 or self.expected_character_count <= 0:
            raise ValidationError("Evaluation case requires expected lexical tokens")
        if (
            not self.expected_tokens
            or len(self.expected_tokens) != self.expected_token_count
            or any(
                not token or token != token.casefold() for token in self.expected_tokens
            )
            or text_fingerprint("\u001f".join(self.expected_tokens))
            != self.expected_tokens_hash
        ):
            raise ValidationError("Evaluation expected-token identity is inconsistent")
        if self.audio_frame_count <= 0 or self.sample_rate <= 0:
            raise ValidationError("Evaluation case requires a positive audio shape")
        _require_sha256(self.expected_tokens_hash, "expected tokens hash")
        _require_sha256(self.audio_digest, "evaluation audio digest")
        if self.gender not in {"female", "male", "unspecified"}:
            raise ValidationError("Evaluation case gender is invalid")
        if self.relative_audio_path:
            candidate = Path(self.relative_audio_path)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValidationError("Evaluation audio path must be relative and safe")

    @property
    def cluster(self) -> tuple[str, str]:
        return self.source_passage_group, self.voice


@dataclass(frozen=True, slots=True)
class EvaluationCorpus:
    version: int
    language: str
    cases: tuple[EvaluationCorpusCase, ...]

    def __post_init__(self) -> None:
        if self.version < 1 or self.language != "en" or not self.cases:
            raise ValidationError("Evaluation corpus must be versioned for English")
        identifiers = tuple(case.case_id for case in self.cases)
        if len(identifiers) != len(set(identifiers)):
            raise ValidationError("Evaluation corpus case IDs must be unique")
        partitions: dict[tuple[str, str], set[CorpusPartition]] = defaultdict(set)
        for case in self.cases:
            partitions[case.cluster].add(case.partition)
        if any(len(values) != 1 for values in partitions.values()):
            raise ValidationError(
                "Source passage and voice clusters cannot cross corpus partitions"
            )

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "speech-evaluation-corpus-v1",
            {
                "version": self.version,
                "language": self.language,
                "cases": tuple(sorted(self.cases, key=lambda case: case.case_id)),
            },
        )


@dataclass(frozen=True, slots=True)
class ObservedCase:
    """Model-independent measurements collected by a qualification harness."""

    case_id: str
    baseline_accepted: bool
    baseline_reason_codes: tuple[str, ...]
    consensus: ConsensusResult
    insertion_count: int
    deletion_count: int
    substitution_count: int
    character_error_count: int
    exact_token_match: bool
    baseline_name_number_correct: bool | None
    proposed_name_number_correct: bool | None
    hallucinated_on_silence: bool
    baseline_boundary_errors_ms: tuple[float, ...]
    proposed_boundary_errors_ms: tuple[float, ...]
    baseline_crop_contaminated: bool
    proposed_crop_contaminated: bool
    baseline_clipped_word: bool
    proposed_clipped_word: bool
    workflow_accepted: bool
    workflow_evidence_fingerprint: str
    cold_runtime_seconds: float | None
    warm_runtime_seconds: float | None
    peak_memory_bytes: int | None
    model_load_count: int
    model_switch_seconds: float | None
    batch_size: int = 1

    def __post_init__(self) -> None:
        counts = (
            self.insertion_count,
            self.deletion_count,
            self.substitution_count,
            self.character_error_count,
            self.model_load_count,
        )
        if (
            not self.case_id
            or self.batch_size < 1
            or any(value < 0 for value in counts)
        ):
            raise ValidationError("Observed evaluation measurements are invalid")
        boundary_errors = (
            *self.baseline_boundary_errors_ms,
            *self.proposed_boundary_errors_ms,
        )
        if any(not math.isfinite(value) or value < 0 for value in boundary_errors):
            raise ValidationError("Boundary errors must be finite and non-negative")
        _require_sha256(
            self.workflow_evidence_fingerprint,
            "workflow evidence fingerprint",
        )
        for runtime in (
            self.cold_runtime_seconds,
            self.warm_runtime_seconds,
            self.model_switch_seconds,
        ):
            if runtime is not None and (not math.isfinite(runtime) or runtime < 0):
                raise ValidationError(
                    "Evaluation runtime must be finite and non-negative"
                )
        if self.peak_memory_bytes is not None and self.peak_memory_bytes < 0:
            raise ValidationError("Evaluation memory cannot be negative")


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    """Pre-registered paired non-inferiority thresholds."""

    version: int
    partition: CorpusPartition
    confidence_level: float
    maximum_false_accept_cluster_rate: float
    maximum_false_rejection_increase: int
    maximum_crop_contamination_increase: int
    maximum_clipped_word_increase: int
    require_forced_boundary_improvement: bool
    mandatory_rejection_risks: tuple[str, ...]
    name_number_risks: tuple[str, ...]
    minimum_defect_clusters_by_risk: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if (
            self.version < 1
            or not _MINIMUM_ONE_SIDED_CONFIDENCE < self.confidence_level < 1
        ):
            raise ValidationError("Evaluation protocol confidence is invalid")
        if not 0 <= self.maximum_false_accept_cluster_rate <= 1:
            raise ValidationError("False-acceptance margin must be a rate")
        margins = (
            self.maximum_false_rejection_increase,
            self.maximum_crop_contamination_increase,
            self.maximum_clipped_word_increase,
        )
        if any(value < 0 for value in margins):
            raise ValidationError("Evaluation count margins cannot be negative")
        risks = tuple(
            name for name, count in self.minimum_defect_clusters_by_risk if count > 0
        )
        if (
            not risks
            or len(risks) != len(self.minimum_defect_clusters_by_risk)
            or len(risks) != len(set(risks))
        ):
            raise ValidationError("Risk-class sample requirements are invalid")
        if any(_RISK_CLASS.fullmatch(name) is None for name in risks):
            raise ValidationError("Risk-class name is invalid")
        registered = set(risks)
        for label, required_risks in (
            ("mandatory rejection", self.mandatory_rejection_risks),
            ("name and number", self.name_number_risks),
        ):
            if (
                not required_risks
                or len(required_risks) != len(set(required_risks))
                or any(name not in registered for name in required_risks)
            ):
                raise ValidationError(
                    f"Evaluation {label} risks must be unique registered classes"
                )
        required = minimum_zero_failure_clusters(
            self.confidence_level,
            self.maximum_false_accept_cluster_rate,
        )
        if any(
            count < required for _name, count in self.minimum_defect_clusters_by_risk
        ):
            raise ValidationError(
                "Risk-class cluster count cannot prove the false-acceptance bound"
            )

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-evaluation-protocol-v1", self)


@dataclass(frozen=True, slots=True)
class MetricSlice:
    dimension: str
    value: str
    case_count: int
    clean_count: int
    defective_count: int
    baseline_false_accepts: int
    candidate_false_accepts: int
    workflow_false_accepts: int
    baseline_false_rejects: int
    candidate_false_rejects: int
    workflow_false_rejects: int
    additional_defects_caught: int
    new_false_accepts: int
    defect_cluster_count: int
    candidate_false_accept_cluster_count: int
    candidate_false_accept_cluster_upper_bound: float
    workflow_false_accept_cluster_count: int
    workflow_false_accept_cluster_upper_bound: float
    word_error_rate: float
    character_error_rate: float
    exact_token_accuracy: float
    baseline_name_number_accuracy: float | None
    proposed_name_number_accuracy: float | None
    hallucination_count: int
    baseline_boundary_median_absolute_error_ms: float | None
    proposed_boundary_median_absolute_error_ms: float | None
    baseline_boundary_p95_error_ms: float | None
    proposed_boundary_p95_error_ms: float | None
    baseline_crop_contamination_count: int
    proposed_crop_contamination_count: int
    baseline_clipped_word_count: int
    proposed_clipped_word_count: int
    cold_real_time_factor: float | None
    warm_real_time_factor: float | None
    batch_throughput_items_per_second: float | None
    peak_memory_bytes: int | None
    model_load_count: int
    mean_model_switch_seconds: float | None


@dataclass(frozen=True, slots=True)
class EvaluatedCase:
    case: EvaluationCorpusCase
    observed: ObservedCase
    shadow: ShadowComparison


@dataclass(frozen=True, slots=True)
class SpeechAnalysisEvaluation:
    corpus_fingerprint: str
    policy_fingerprint: str
    protocol_fingerprint: str
    cases: tuple[EvaluatedCase, ...]
    slices: tuple[MetricSlice, ...]
    candidate_false_accept_cluster_upper_bound: float
    workflow_false_accept_cluster_upper_bound: float
    candidate_false_acceptance_safe: bool
    workflow_false_acceptance_safe: bool
    false_rejection_noninferior: bool
    name_number_noninferior: bool
    forced_boundary_improved: bool
    crop_safety_noninferior: bool
    mandatory_defects_rejected: bool
    automatic_acceptances_explainable: bool
    passed: bool

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "speech-analysis-evaluation-v1",
            {
                "corpus_fingerprint": self.corpus_fingerprint,
                "policy_fingerprint": self.policy_fingerprint,
                "protocol_fingerprint": self.protocol_fingerprint,
                "cases": tuple(
                    (
                        item.case.case_id,
                        item.shadow.classification.value,
                        item.shadow.consensus.fingerprint,
                    )
                    for item in self.cases
                ),
                "slices": tuple(_quality_metric_slice(item) for item in self.slices),
                "candidate_false_accept_cluster_upper_bound": (
                    self.candidate_false_accept_cluster_upper_bound
                ),
                "workflow_false_accept_cluster_upper_bound": (
                    self.workflow_false_accept_cluster_upper_bound
                ),
                "candidate_false_acceptance_safe": (
                    self.candidate_false_acceptance_safe
                ),
                "workflow_false_acceptance_safe": (self.workflow_false_acceptance_safe),
                "false_rejection_noninferior": self.false_rejection_noninferior,
                "name_number_noninferior": self.name_number_noninferior,
                "forced_boundary_improved": self.forced_boundary_improved,
                "crop_safety_noninferior": self.crop_safety_noninferior,
                "mandatory_defects_rejected": self.mandatory_defects_rejected,
                "automatic_acceptances_explainable": (
                    self.automatic_acceptances_explainable
                ),
                "passed": self.passed,
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("speech-analysis-evaluation"),
            "fingerprint": self.fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "protocol_fingerprint": self.protocol_fingerprint,
            "case_count": len(self.cases),
            "candidate_false_accept_cluster_upper_bound": (
                self.candidate_false_accept_cluster_upper_bound
            ),
            "workflow_false_accept_cluster_upper_bound": (
                self.workflow_false_accept_cluster_upper_bound
            ),
            "candidate_false_acceptance_safe": self.candidate_false_acceptance_safe,
            "workflow_false_acceptance_safe": self.workflow_false_acceptance_safe,
            "false_rejection_noninferior": self.false_rejection_noninferior,
            "name_number_noninferior": self.name_number_noninferior,
            "forced_boundary_improved": self.forced_boundary_improved,
            "crop_safety_noninferior": self.crop_safety_noninferior,
            "mandatory_defects_rejected": self.mandatory_defects_rejected,
            "automatic_acceptances_explainable": (
                self.automatic_acceptances_explainable
            ),
            "passed": self.passed,
            "slices": [_metric_slice_dict(item) for item in self.slices],
            "cases": [
                {
                    "case_id": item.case.case_id,
                    "partition": item.case.partition.value,
                    "clip_class": item.case.clip_class.value,
                    "risk_class": item.case.risk_class,
                    "truth": item.case.truth.value,
                    "classification": item.shadow.classification.value,
                    "baseline_accepted": item.observed.baseline_accepted,
                    "proposed_accepted": (
                        item.observed.consensus.outcome is ConsensusOutcome.ACCEPTED
                    ),
                    "workflow_accepted": item.observed.workflow_accepted,
                    "consensus_fingerprint": item.observed.consensus.fingerprint,
                    "workflow_evidence_fingerprint": (
                        item.observed.workflow_evidence_fingerprint
                    ),
                }
                for item in self.cases
            ],
        }


@dataclass(frozen=True, slots=True)
class ReviewerDisposition:
    case_id: str
    policy_fingerprint: str
    reviewer_fingerprint: str
    pass_index: int
    kind: ReviewKind
    decision: ReviewDecision

    def __post_init__(self) -> None:
        if not self.case_id or self.pass_index < 1:
            raise ValidationError("Reviewer disposition identity is invalid")
        _require_sha256(self.policy_fingerprint, "review policy fingerprint")
        _require_sha256(self.reviewer_fingerprint, "reviewer fingerprint")


def evaluate_shadow_corpus(
    corpus: EvaluationCorpus,
    observations: tuple[ObservedCase, ...],
    *,
    protocol: EvaluationProtocol,
) -> SpeechAnalysisEvaluation:
    """Evaluate paired baseline/proposed decisions in stable corpus order."""
    selected = tuple(
        case for case in corpus.cases if case.partition is protocol.partition
    )
    observed = {item.case_id: item for item in observations}
    if set(observed) != {case.case_id for case in selected}:
        raise ValidationError("Evaluation observations do not match the partition")
    policy_fingerprints = {item.consensus.policy_fingerprint for item in observations}
    if len(policy_fingerprints) != 1:
        raise ValidationError("Evaluation consensus policies must match")
    policy_fingerprint = next(iter(policy_fingerprints))
    evaluated = tuple(
        _evaluated_case(case, observed[case.case_id])
        for case in sorted(selected, key=lambda item: item.case_id)
    )
    _require_risk_samples(evaluated, protocol.minimum_defect_clusters_by_risk)
    _require_qualification_measurements(evaluated, protocol)
    slices = _metric_slices(evaluated, confidence=protocol.confidence_level)
    overall = next(
        item for item in slices if item.dimension == "all" and item.value == "all"
    )
    candidate_failures, defect_clusters = _false_accept_clusters(
        evaluated,
        workflow=False,
    )
    workflow_failures, _workflow_clusters = _false_accept_clusters(
        evaluated,
        workflow=True,
    )
    candidate_upper = _wilson_upper(
        candidate_failures,
        defect_clusters,
        protocol.confidence_level,
    )
    workflow_upper = _wilson_upper(
        workflow_failures,
        defect_clusters,
        protocol.confidence_level,
    )
    required_risks = {name for name, _count in protocol.minimum_defect_clusters_by_risk}
    risk_slices = tuple(
        item
        for item in slices
        if item.dimension == "risk_class" and item.value in required_risks
    )
    candidate_false_acceptance_safe = (
        overall.candidate_false_accepts == 0
        and candidate_upper <= protocol.maximum_false_accept_cluster_rate
        and all(
            item.candidate_false_accepts == 0
            and item.candidate_false_accept_cluster_upper_bound
            <= protocol.maximum_false_accept_cluster_rate
            for item in risk_slices
        )
    )
    workflow_false_acceptance_safe = (
        overall.workflow_false_accepts == 0
        and workflow_upper <= protocol.maximum_false_accept_cluster_rate
        and all(
            item.workflow_false_accepts == 0
            and item.workflow_false_accept_cluster_upper_bound
            <= protocol.maximum_false_accept_cluster_rate
            for item in risk_slices
        )
    )
    false_rejection_noninferior = (
        overall.workflow_false_rejects - overall.baseline_false_rejects
        <= protocol.maximum_false_rejection_increase
    )
    name_number_slice = tuple(
        item
        for item in evaluated
        if item.case.risk_class in set(protocol.name_number_risks)
    )
    baseline_name_number_accuracy = _boolean_accuracy(
        tuple(item.observed.baseline_name_number_correct for item in name_number_slice)
    )
    proposed_name_number_accuracy = _boolean_accuracy(
        tuple(item.observed.proposed_name_number_correct for item in name_number_slice)
    )
    name_number_noninferior = (
        baseline_name_number_accuracy is not None
        and proposed_name_number_accuracy is not None
        and proposed_name_number_accuracy >= baseline_name_number_accuracy
    )
    forced_boundary_improved = _forced_boundary_improved(
        evaluated,
        required=protocol.require_forced_boundary_improvement,
    )
    crop_safety_noninferior = (
        overall.proposed_crop_contamination_count
        - overall.baseline_crop_contamination_count
        <= protocol.maximum_crop_contamination_increase
        and overall.proposed_clipped_word_count - overall.baseline_clipped_word_count
        <= protocol.maximum_clipped_word_increase
    )
    mandatory_risks = set(protocol.mandatory_rejection_risks)
    mandatory_defects_rejected = all(
        item.observed.consensus.outcome is ConsensusOutcome.REJECTED
        and not item.observed.workflow_accepted
        for item in evaluated
        if item.case.truth is CorpusTruth.DEFECTIVE
        and item.case.risk_class in mandatory_risks
    )
    automatic_acceptances_explainable = all(
        _is_sha256(item.observed.consensus.fingerprint)
        and _is_sha256(item.observed.workflow_evidence_fingerprint)
        for item in evaluated
        if item.observed.consensus.outcome is ConsensusOutcome.ACCEPTED
        or item.observed.workflow_accepted
    )
    passed = all(
        (
            candidate_false_acceptance_safe,
            workflow_false_acceptance_safe,
            false_rejection_noninferior,
            name_number_noninferior,
            forced_boundary_improved,
            crop_safety_noninferior,
            mandatory_defects_rejected,
            automatic_acceptances_explainable,
        )
    )
    return SpeechAnalysisEvaluation(
        corpus.fingerprint,
        policy_fingerprint,
        protocol.fingerprint,
        evaluated,
        slices,
        candidate_upper,
        workflow_upper,
        candidate_false_acceptance_safe,
        workflow_false_acceptance_safe,
        false_rejection_noninferior,
        name_number_noninferior,
        forced_boundary_improved,
        crop_safety_noninferior,
        mandatory_defects_rejected,
        automatic_acceptances_explainable,
        passed,
    )


def load_evaluation_corpus(path: Path, *, repository_root: Path) -> EvaluationCorpus:
    """Load licensed ground truth while retaining no transcript plaintext."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValidationError("Cannot read speech-analysis corpus metadata") from error
    if not isinstance(raw, dict):
        raise ValidationError("Speech-analysis corpus must be a TOML table")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValidationError("Speech-analysis corpus requires case tables")
    cases = tuple(
        _load_corpus_case(cast(dict[str, object], item), repository_root)
        for item in cases_raw
        if isinstance(item, dict)
    )
    if len(cases) != len(cases_raw):
        raise ValidationError("Speech-analysis corpus case must be a table")
    return EvaluationCorpus(
        version=_corpus_integer(raw, "version"),
        language=_corpus_string(raw, "language"),
        cases=cases,
    )


def load_evaluation_protocol(path: Path) -> EvaluationProtocol:
    """Load a frozen qualification protocol without accepting implicit defaults."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValidationError("Cannot read speech-analysis protocol") from error
    if not isinstance(raw, dict):
        raise ValidationError("Speech-analysis protocol must be a TOML table")
    expected_keys = {
        "version",
        "partition",
        "confidence_level",
        "maximum_false_accept_cluster_rate",
        "maximum_false_rejection_increase",
        "maximum_crop_contamination_increase",
        "maximum_clipped_word_increase",
        "require_forced_boundary_improvement",
        "mandatory_rejection_risks",
        "name_number_risks",
        "minimum_defect_clusters_by_risk",
    }
    if set(raw) != expected_keys:
        raise ValidationError("Speech-analysis protocol fields are invalid")
    risk_counts = raw["minimum_defect_clusters_by_risk"]
    if not isinstance(risk_counts, dict) or not risk_counts:
        raise ValidationError("Speech-analysis protocol requires risk classes")
    requirements: list[tuple[str, int]] = []
    for name, count in sorted(risk_counts.items()):
        if (
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
        ):
            raise ValidationError("Speech-analysis protocol risk count is invalid")
        requirements.append((name, count))
    try:
        partition = CorpusPartition(_protocol_string(raw, "partition"))
    except ValueError as error:
        raise ValidationError(
            "Speech-analysis protocol partition is invalid"
        ) from error
    return EvaluationProtocol(
        version=_protocol_integer(raw, "version"),
        partition=partition,
        confidence_level=_protocol_number(raw, "confidence_level"),
        maximum_false_accept_cluster_rate=_protocol_number(
            raw, "maximum_false_accept_cluster_rate"
        ),
        maximum_false_rejection_increase=_protocol_integer(
            raw, "maximum_false_rejection_increase"
        ),
        maximum_crop_contamination_increase=_protocol_integer(
            raw, "maximum_crop_contamination_increase"
        ),
        maximum_clipped_word_increase=_protocol_integer(
            raw, "maximum_clipped_word_increase"
        ),
        require_forced_boundary_improvement=_protocol_boolean(
            raw, "require_forced_boundary_improvement"
        ),
        mandatory_rejection_risks=_protocol_string_tuple(
            raw, "mandatory_rejection_risks"
        ),
        name_number_risks=_protocol_string_tuple(raw, "name_number_risks"),
        minimum_defect_clusters_by_risk=tuple(requirements),
    )


def load_default_evaluation_protocol() -> EvaluationProtocol:
    """Load Yakbox's preregistered English qualification protocol."""
    return load_evaluation_protocol(
        Path(__file__).parents[1] / "data" / "speech-evaluation-protocol-v1.toml"
    )


def _load_corpus_case(
    raw: dict[str, object], repository_root: Path
) -> EvaluationCorpusCase:
    relative = _corpus_string(raw, "audio_path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValidationError("Evaluation audio path must be relative and safe")
    unresolved = repository_root.resolve() / relative_path
    if unresolved.is_symlink():
        raise ValidationError("Evaluation corpus rejects symlink audio")
    audio = safe_child(repository_root.resolve(), unresolved)
    if not audio.is_file() or audio.stat().st_size <= _MINIMUM_WAV_BYTES:
        raise ValidationError("Evaluation corpus audio is missing or empty")
    digest = sha256_file(audio)
    if digest != _corpus_string(raw, "audio_sha256"):
        raise ValidationError("Evaluation corpus audio digest does not match")
    sample_rate, frame_count = _wave_shape(audio)
    expected = normalize_english(_corpus_string(raw, "expected_text")).tokens
    if not expected:
        raise ValidationError("Evaluation corpus case requires expected text")
    expected_values = tuple(token.text for token in expected)
    try:
        partition = CorpusPartition(_corpus_string(raw, "partition"))
        clip_class = ClipClass(_corpus_string(raw, "clip_class"))
        truth = CorpusTruth(_corpus_string(raw, "truth"))
    except ValueError as error:
        raise ValidationError("Evaluation corpus enum value is invalid") from error
    return EvaluationCorpusCase(
        case_id=_corpus_string(raw, "case_id"),
        source_passage_group=_corpus_string(raw, "source_passage_group"),
        voice=_corpus_string(raw, "voice"),
        partition=partition,
        clip_class=clip_class,
        risk_class=_corpus_string(raw, "risk_class"),
        truth=truth,
        expected_tokens=expected_values,
        expected_tokens_hash=text_fingerprint("\u001f".join(expected_values)),
        expected_token_count=len(expected_values),
        expected_character_count=sum(len(item) for item in expected_values),
        audio_digest=digest,
        audio_frame_count=frame_count,
        sample_rate=sample_rate,
        rights_id=_corpus_string(raw, "rights_id"),
        source_url=_corpus_string(raw, "source_url"),
        boundary_review_required=_corpus_boolean(raw, "boundary_review_required"),
        relative_audio_path=relative,
        accent=_corpus_string(raw, "accent"),
        gender=_corpus_string(raw, "gender"),
        speaking_rate=_corpus_string(raw, "speaking_rate"),
        acoustic_condition=_corpus_string(raw, "acoustic_condition"),
    )


def _wave_shape(path: Path) -> tuple[int, int]:
    try:
        with wave.open(str(path), "rb") as reader:
            if (
                reader.getnchannels() != 1
                or reader.getsampwidth() != _PCM16_SAMPLE_WIDTH
            ):
                raise ValidationError("Evaluation corpus WAV must be mono PCM16")
            return reader.getframerate(), reader.getnframes()
    except (EOFError, OSError, wave.Error) as error:
        raise ValidationError("Evaluation corpus audio is not a valid WAV") from error


def _corpus_string(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Evaluation corpus {key!r} must be text")
    return value


def _corpus_integer(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"Evaluation corpus {key!r} must be an integer")
    return value


def _corpus_boolean(raw: dict[str, object], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValidationError(f"Evaluation corpus {key!r} must be boolean")
    return value


def _protocol_string(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Evaluation protocol {key!r} must be text")
    return value


def _protocol_integer(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"Evaluation protocol {key!r} must be an integer")
    return value


def _protocol_number(raw: dict[str, object], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"Evaluation protocol {key!r} must be a number")
    return float(value)


def _protocol_boolean(raw: dict[str, object], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValidationError(f"Evaluation protocol {key!r} must be boolean")
    return value


def _protocol_string_tuple(raw: dict[str, object], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValidationError(f"Evaluation protocol {key!r} must be a text array")
    return tuple(cast(list[str], value))


def approve_calibration(
    table: CalibrationTable,
    evaluation: SpeechAnalysisEvaluation,
    dispositions: tuple[ReviewerDisposition, ...],
) -> CalibrationTable:
    """Bind thresholds to the minimum explicit reviews required by the plan."""
    if table.corpus_fingerprint != evaluation.corpus_fingerprint:
        raise ValidationError("Calibration table and evaluation corpus differ")
    if not evaluation.passed:
        raise ValidationError("Failed safety evaluation cannot approve calibration")
    required: dict[tuple[str, ReviewKind], int] = {}
    for item in evaluation.cases:
        proposed_accepted = item.observed.consensus.outcome is ConsensusOutcome.ACCEPTED
        if proposed_accepted:
            required[(item.case.case_id, ReviewKind.AUTOMATIC_ACCEPTANCE)] = 1
        if "persistent_valid_dissent" in item.observed.consensus.reason_codes:
            required[(item.case.case_id, ReviewKind.VALID_DISSENT)] = 1
        if item.case.boundary_review_required:
            required[(item.case.case_id, ReviewKind.BOUNDARY)] = 2
    relevant = tuple(
        item
        for item in dispositions
        if item.policy_fingerprint == evaluation.policy_fingerprint
    )
    for (case_id, kind), minimum in required.items():
        approved = {
            (item.reviewer_fingerprint, item.pass_index)
            for item in relevant
            if item.case_id == case_id
            and item.kind is kind
            and item.decision is ReviewDecision.APPROVED
        }
        rejected = any(
            item.case_id == case_id
            and item.kind is kind
            and item.decision is ReviewDecision.REJECTED
            for item in relevant
        )
        if rejected or len(approved) < minimum:
            raise ValidationError(
                f"Required {kind.value} review is incomplete for {case_id}"
            )
    disposition_fingerprint = semantic_fingerprint(
        "speech-calibration-review-v1",
        tuple(
            sorted(
                relevant,
                key=lambda item: (
                    item.case_id,
                    item.kind.value,
                    item.reviewer_fingerprint,
                    item.pass_index,
                ),
            )
        ),
    )
    return replace(
        table,
        approved=True,
        reviewer_disposition_fingerprint=disposition_fingerprint,
    )


def _evaluated_case(
    case: EvaluationCorpusCase, observed: ObservedCase
) -> EvaluatedCase:
    ground_truth = (
        ShadowGroundTruth.KNOWN_CLEAN
        if case.truth is CorpusTruth.CLEAN
        else ShadowGroundTruth.KNOWN_DEFECTIVE
    )
    shadow = compare_shadow_decision(
        audio_digest=case.audio_digest,
        baseline_accepted=observed.baseline_accepted,
        baseline_reason_codes=observed.baseline_reason_codes,
        consensus=observed.consensus,
        ground_truth=ground_truth,
    )
    return EvaluatedCase(case, observed, shadow)


def _metric_slices(
    cases: tuple[EvaluatedCase, ...], *, confidence: float
) -> tuple[MetricSlice, ...]:
    dimensions: list[tuple[str, str, tuple[EvaluatedCase, ...]]] = [
        ("all", "all", cases)
    ]
    for name, getter in (
        ("clip_class", lambda item: item.case.clip_class.value),
        ("risk_class", lambda item: item.case.risk_class),
        ("voice", lambda item: item.case.voice),
        ("accent", lambda item: item.case.accent),
        ("gender", lambda item: item.case.gender),
        ("speaking_rate", lambda item: item.case.speaking_rate),
        ("acoustic_condition", lambda item: item.case.acoustic_condition),
    ):
        values = sorted({getter(item) for item in cases})
        dimensions.extend(
            (name, value, tuple(item for item in cases if getter(item) == value))
            for value in values
        )
    return tuple(
        _metric_slice(name, value, selected, confidence=confidence)
        for name, value, selected in dimensions
    )


def _metric_slice(
    dimension: str,
    value: str,
    cases: tuple[EvaluatedCase, ...],
    *,
    confidence: float,
) -> MetricSlice:
    clean = tuple(item for item in cases if item.case.truth is CorpusTruth.CLEAN)
    defective = tuple(
        item for item in cases if item.case.truth is CorpusTruth.DEFECTIVE
    )
    expected_words = sum(item.case.expected_token_count for item in cases)
    expected_characters = sum(item.case.expected_character_count for item in cases)
    word_errors = sum(
        item.observed.insertion_count
        + item.observed.deletion_count
        + item.observed.substitution_count
        for item in cases
    )
    baseline_name_number = tuple(
        item.observed.baseline_name_number_correct
        for item in cases
        if item.observed.baseline_name_number_correct is not None
    )
    proposed_name_number = tuple(
        item.observed.proposed_name_number_correct
        for item in cases
        if item.observed.proposed_name_number_correct is not None
    )
    baseline_boundary_errors = tuple(
        error for item in cases for error in item.observed.baseline_boundary_errors_ms
    )
    proposed_boundary_errors = tuple(
        error for item in cases for error in item.observed.proposed_boundary_errors_ms
    )
    total_duration = sum(
        item.case.audio_frame_count / item.case.sample_rate for item in cases
    )
    cold_runtime = tuple(
        item.observed.cold_runtime_seconds
        for item in cases
        if item.observed.cold_runtime_seconds is not None
    )
    warm_runtime = tuple(
        item.observed.warm_runtime_seconds
        for item in cases
        if item.observed.warm_runtime_seconds is not None
    )
    switch_times = tuple(
        item.observed.model_switch_seconds
        for item in cases
        if item.observed.model_switch_seconds is not None
    )
    memories = tuple(
        item.observed.peak_memory_bytes
        for item in cases
        if item.observed.peak_memory_bytes is not None
    )
    candidate_false_accept_clusters, defect_clusters = _false_accept_clusters(
        cases,
        workflow=False,
    )
    workflow_false_accept_clusters, _ = _false_accept_clusters(
        cases,
        workflow=True,
    )
    return MetricSlice(
        dimension,
        value,
        len(cases),
        len(clean),
        len(defective),
        sum(item.observed.baseline_accepted for item in defective),
        sum(
            item.observed.consensus.outcome is ConsensusOutcome.ACCEPTED
            for item in defective
        ),
        sum(item.observed.workflow_accepted for item in defective),
        sum(not item.observed.baseline_accepted for item in clean),
        sum(
            item.observed.consensus.outcome is ConsensusOutcome.REJECTED
            for item in clean
        ),
        sum(not item.observed.workflow_accepted for item in clean),
        sum(
            item.observed.baseline_accepted
            and item.observed.consensus.outcome is ConsensusOutcome.REJECTED
            for item in defective
        ),
        sum(
            not item.observed.baseline_accepted
            and item.observed.consensus.outcome is ConsensusOutcome.ACCEPTED
            for item in defective
        ),
        defect_clusters,
        candidate_false_accept_clusters,
        _wilson_upper(candidate_false_accept_clusters, defect_clusters, confidence),
        workflow_false_accept_clusters,
        _wilson_upper(workflow_false_accept_clusters, defect_clusters, confidence),
        word_errors / max(1, expected_words),
        sum(item.observed.character_error_count for item in cases)
        / max(1, expected_characters),
        sum(item.observed.exact_token_match for item in cases) / max(1, len(cases)),
        (
            sum(bool(result) for result in baseline_name_number)
            / len(baseline_name_number)
            if baseline_name_number
            else None
        ),
        (
            sum(bool(result) for result in proposed_name_number)
            / len(proposed_name_number)
            if proposed_name_number
            else None
        ),
        sum(item.observed.hallucinated_on_silence for item in cases),
        _percentile(baseline_boundary_errors, 0.5),
        _percentile(proposed_boundary_errors, 0.5),
        _percentile(baseline_boundary_errors, 0.95),
        _percentile(proposed_boundary_errors, 0.95),
        sum(item.observed.baseline_crop_contaminated for item in cases),
        sum(item.observed.proposed_crop_contaminated for item in cases),
        sum(item.observed.baseline_clipped_word for item in cases),
        sum(item.observed.proposed_clipped_word for item in cases),
        (
            sum(cold_runtime) / total_duration
            if cold_runtime and total_duration
            else None
        ),
        (
            sum(warm_runtime) / total_duration
            if warm_runtime and total_duration
            else None
        ),
        (
            sum(item.observed.batch_size for item in cases) / sum(warm_runtime)
            if warm_runtime and sum(warm_runtime) > 0
            else None
        ),
        max(memories) if memories else None,
        sum(item.observed.model_load_count for item in cases),
        sum(switch_times) / len(switch_times) if switch_times else None,
    )


def _metric_slice_dict(item: MetricSlice) -> dict[str, object]:
    names = (
        "dimension",
        "value",
        "case_count",
        "clean_count",
        "defective_count",
        "baseline_false_accepts",
        "candidate_false_accepts",
        "workflow_false_accepts",
        "baseline_false_rejects",
        "candidate_false_rejects",
        "workflow_false_rejects",
        "additional_defects_caught",
        "new_false_accepts",
        "defect_cluster_count",
        "candidate_false_accept_cluster_count",
        "candidate_false_accept_cluster_upper_bound",
        "workflow_false_accept_cluster_count",
        "workflow_false_accept_cluster_upper_bound",
        "word_error_rate",
        "character_error_rate",
        "exact_token_accuracy",
        "baseline_name_number_accuracy",
        "proposed_name_number_accuracy",
        "hallucination_count",
        "baseline_boundary_median_absolute_error_ms",
        "proposed_boundary_median_absolute_error_ms",
        "baseline_boundary_p95_error_ms",
        "proposed_boundary_p95_error_ms",
        "baseline_crop_contamination_count",
        "proposed_crop_contamination_count",
        "baseline_clipped_word_count",
        "proposed_clipped_word_count",
        "cold_real_time_factor",
        "warm_real_time_factor",
        "batch_throughput_items_per_second",
        "peak_memory_bytes",
        "model_load_count",
        "mean_model_switch_seconds",
    )
    return {name: getattr(item, name) for name in names}


def _quality_metric_slice(item: MetricSlice) -> dict[str, object]:
    """Return decision-bearing metrics without operational observations."""
    excluded = {
        "batch_throughput_items_per_second",
        "cold_real_time_factor",
        "mean_model_switch_seconds",
        "model_load_count",
        "peak_memory_bytes",
        "warm_real_time_factor",
    }
    return {
        name: value
        for name, value in _metric_slice_dict(item).items()
        if name not in excluded
    }


def _percentile(values: tuple[float, ...], probability: float) -> float | None:
    if not values:
        return None
    if probability == _MEDIAN_PROBABILITY:
        return float(median(values))
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _false_accept_clusters(
    cases: tuple[EvaluatedCase, ...],
    *,
    workflow: bool,
) -> tuple[int, int]:
    defective = tuple(
        item for item in cases if item.case.truth is CorpusTruth.DEFECTIVE
    )
    clusters = {item.case.cluster for item in defective}
    failed = {
        item.case.cluster
        for item in defective
        if (
            item.observed.workflow_accepted
            if workflow
            else item.observed.consensus.outcome is ConsensusOutcome.ACCEPTED
        )
    }
    return len(failed), len(clusters)


def _wilson_upper(failures: int, count: int, confidence: float) -> float:
    if count <= 0:
        return 1.0
    probability = failures / count
    z = NormalDist().inv_cdf(confidence)
    denominator = 1 + z * z / count
    center = probability + z * z / (2 * count)
    radius = z * math.sqrt(
        probability * (1 - probability) / count + z * z / (4 * count * count)
    )
    return min(1.0, (center + radius) / denominator)


def _require_risk_samples(
    cases: tuple[EvaluatedCase, ...],
    requirements: tuple[tuple[str, int], ...],
) -> None:
    required_risks = {name for name, _minimum in requirements}
    observed_risks = {
        item.case.risk_class
        for item in cases
        if item.case.truth is CorpusTruth.DEFECTIVE
    }
    if observed_risks != required_risks:
        raise ValidationError(
            "Evaluation protocol must preregister every defective risk class"
        )
    clusters: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for item in cases:
        if item.case.truth is CorpusTruth.DEFECTIVE:
            clusters[item.case.risk_class].add(item.case.cluster)
    missing = tuple(
        name for name, minimum in requirements if len(clusters[name]) < minimum
    )
    if missing:
        raise ValidationError(
            f"Evaluation risk class {missing[0]!r} lacks preregistered samples"
        )


def _require_qualification_measurements(
    cases: tuple[EvaluatedCase, ...],
    protocol: EvaluationProtocol,
) -> None:
    name_number_risks = set(protocol.name_number_risks)
    name_number_cases = tuple(
        item for item in cases if item.case.risk_class in name_number_risks
    )
    if not name_number_cases or any(
        item.observed.baseline_name_number_correct is None
        or item.observed.proposed_name_number_correct is None
        for item in name_number_cases
    ):
        raise ValidationError(
            "Name and number cases require paired baseline and proposed truth"
        )
    if not protocol.require_forced_boundary_improvement:
        return
    boundary_cases = tuple(item for item in cases if item.case.boundary_review_required)
    if not boundary_cases or any(
        not item.observed.baseline_boundary_errors_ms
        or not item.observed.proposed_boundary_errors_ms
        or len(item.observed.baseline_boundary_errors_ms)
        != len(item.observed.proposed_boundary_errors_ms)
        for item in boundary_cases
    ):
        raise ValidationError(
            "Boundary-review cases require paired baseline and proposed errors"
        )


def _boolean_accuracy(values: tuple[bool | None, ...]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return sum(bool(value) for value in values) / len(values)


def _forced_boundary_improved(
    cases: tuple[EvaluatedCase, ...],
    *,
    required: bool,
) -> bool:
    if not required:
        return True
    selected = tuple(item for item in cases if item.case.boundary_review_required)
    baseline = tuple(
        error
        for item in selected
        for error in item.observed.baseline_boundary_errors_ms
    )
    proposed = tuple(
        error
        for item in selected
        for error in item.observed.proposed_boundary_errors_ms
    )
    baseline_median = _percentile(baseline, 0.5)
    proposed_median = _percentile(proposed, 0.5)
    baseline_p95 = _percentile(baseline, 0.95)
    proposed_p95 = _percentile(proposed, 0.95)
    return (
        baseline_median is not None
        and proposed_median is not None
        and baseline_p95 is not None
        and proposed_p95 is not None
        and proposed_median < baseline_median
        and proposed_p95 < baseline_p95
    )


def minimum_zero_failure_clusters(confidence: float, maximum_rate: float) -> int:
    """Return the minimum independent clusters needed for a zero-failure bound."""
    if not _MINIMUM_ONE_SIDED_CONFIDENCE < confidence < 1:
        raise ValidationError("Evaluation confidence is invalid")
    if not 0 < maximum_rate <= 1:
        raise ValidationError("False-acceptance rate must be greater than zero")
    count = 1
    while _wilson_upper(0, count, confidence) > maximum_rate:
        count += 1
    return count


def _require_sha256(value: str, label: str) -> None:
    if not _is_sha256(value):
        raise ValidationError(f"{label.capitalize()} must be a SHA-256")


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "CorpusPartition",
    "CorpusTruth",
    "EvaluatedCase",
    "EvaluationCorpus",
    "EvaluationCorpusCase",
    "EvaluationProtocol",
    "MetricSlice",
    "ObservedCase",
    "ReviewDecision",
    "ReviewKind",
    "ReviewerDisposition",
    "SpeechAnalysisEvaluation",
    "approve_calibration",
    "evaluate_shadow_corpus",
    "load_default_evaluation_protocol",
    "load_evaluation_corpus",
    "load_evaluation_protocol",
    "minimum_zero_failure_clusters",
]
