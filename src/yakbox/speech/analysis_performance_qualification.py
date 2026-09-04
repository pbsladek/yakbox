"""Preregistered performance gates for the strict speech-analysis workflow."""

from __future__ import annotations

import math
import re
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import cast

from yakbox.contracts import runtime_metadata
from yakbox.errors import ValidationError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}")


class BenchmarkWorkflow(StrEnum):
    DELIVERY_VERIFICATION = "delivery_verification"
    FULL_CHAPTER_VERIFICATION = "full_chapter_verification"
    LOCALIZED_REPAIR = "localized_repair"
    MASTERED_RELEASE_PROMOTION = "mastered_release_promotion"
    MULTI_REPAIR = "multi_repair"
    UNCHANGED_BUILD = "unchanged_build"


class BenchmarkCacheState(StrEnum):
    COLD_PROCESS_COLD_CACHE = "cold_process_cold_cache"
    FULLY_REPEATED_CACHE_HIT = "fully_repeated_cache_hit"
    WARM_PROCESS_COLD_EVIDENCE = "warm_process_cold_evidence"


class DeliveryVerificationMode(StrEnum):
    FULL_RECOGNITION = "full_recognition"
    QUALIFIED_EQUIVALENCE = "qualified_equivalence"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class EngineLoadCount:
    engine: str
    count: int

    def __post_init__(self) -> None:
        if _IDENTIFIER.fullmatch(self.engine) is None or self.count < 0:
            raise ValidationError("Performance model-load observation is invalid")


@dataclass(frozen=True, slots=True)
class PerformanceObservation:
    observation_id: str
    comparison_group: str
    workflow: BenchmarkWorkflow
    cache_state: BenchmarkCacheState
    run_index: int
    measured: bool
    quality_policy_fingerprint: str
    terminal_artifact_state_fingerprint: str
    audio_seconds: float
    speech_analysis_seconds: float
    peak_memory_bytes: int
    inference_calls: int
    model_loads: tuple[EngineLoadCount, ...]
    qwen_windows: int
    policy_required_qwen_windows: int
    raw_assembly_passes: int
    mastering_passes: int
    affected_scope_audio_seconds: float
    analyzed_scope_audio_seconds: float
    offline: bool
    faulted: bool
    delivery_verification_mode: DeliveryVerificationMode

    def __post_init__(self) -> None:
        if (
            _IDENTIFIER.fullmatch(self.observation_id) is None
            or _IDENTIFIER.fullmatch(self.comparison_group) is None
            or self.run_index < 0
            or _SHA256.fullmatch(self.quality_policy_fingerprint) is None
            or _SHA256.fullmatch(self.terminal_artifact_state_fingerprint) is None
        ):
            raise ValidationError("Performance observation identity is invalid")
        counts = (
            self.peak_memory_bytes,
            self.inference_calls,
            self.qwen_windows,
            self.policy_required_qwen_windows,
            self.raw_assembly_passes,
            self.mastering_passes,
        )
        durations = (
            self.audio_seconds,
            self.speech_analysis_seconds,
            self.affected_scope_audio_seconds,
            self.analyzed_scope_audio_seconds,
        )
        if (
            min(counts) < 0
            or self.audio_seconds <= 0
            or any(not math.isfinite(value) or value < 0 for value in durations)
        ):
            raise ValidationError("Performance observation measurement is invalid")
        engines = tuple(item.engine for item in self.model_loads)
        if engines != tuple(sorted(set(engines))):
            raise ValidationError("Performance model loads must be unique and ordered")
        delivery = self.workflow is BenchmarkWorkflow.DELIVERY_VERIFICATION
        if delivery == (
            self.delivery_verification_mode is DeliveryVerificationMode.NOT_APPLICABLE
        ):
            raise ValidationError(
                "Delivery verification mode must match the measured workflow"
            )


@dataclass(frozen=True, slots=True)
class PerformanceProtocol:
    version: int
    minimum_measured_runs: int
    minimum_p95_runs: int
    maximum_localized_repair_fraction: float
    maximum_model_loads_per_engine: int
    required_workflows: tuple[BenchmarkWorkflow, ...]
    required_cache_states: tuple[BenchmarkCacheState, ...]
    p95_workflows: tuple[BenchmarkWorkflow, ...]

    def __post_init__(self) -> None:
        if (
            self.version < 1
            or self.minimum_measured_runs < 1
            or self.minimum_p95_runs < self.minimum_measured_runs
            or not 0 < self.maximum_localized_repair_fraction < 1
            or self.maximum_model_loads_per_engine < 1
        ):
            raise ValidationError("Speech performance protocol limits are invalid")
        if (
            set(self.required_workflows) != set(BenchmarkWorkflow)
            or set(self.required_cache_states) != set(BenchmarkCacheState)
            or not self.p95_workflows
            or not set(self.p95_workflows).issubset(self.required_workflows)
        ):
            raise ValidationError("Speech performance protocol coverage is incomplete")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-performance-protocol-v1", self)


@dataclass(frozen=True, slots=True)
class PerformanceSlice:
    comparison_group: str
    workflow: BenchmarkWorkflow
    cache_state: BenchmarkCacheState
    measured_runs: int
    median_speech_analysis_seconds: float
    minimum_speech_analysis_seconds: float
    maximum_speech_analysis_seconds: float
    p95_speech_analysis_seconds: float | None
    median_audio_seconds: float
    peak_memory_bytes: int


@dataclass(frozen=True, slots=True)
class SpeechPerformanceQualification:
    protocol_fingerprint: str
    observation_fingerprint: str
    slices: tuple[PerformanceSlice, ...]
    coverage_complete: bool
    repeat_uses_zero_inference: bool
    localized_repair_within_fraction: bool
    multi_repair_is_single_pass_and_scoped: bool
    model_loads_bounded: bool
    qwen_is_policy_scoped: bool
    offline_operation_passed: bool
    delivery_verification_separate: bool
    passed: bool

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-performance-qualification-v1", self)

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("speech-performance-qualification"),
            "fingerprint": self.fingerprint,
            "protocol_fingerprint": self.protocol_fingerprint,
            "observation_fingerprint": self.observation_fingerprint,
            "coverage_complete": self.coverage_complete,
            "repeat_uses_zero_inference": self.repeat_uses_zero_inference,
            "localized_repair_within_fraction": (self.localized_repair_within_fraction),
            "multi_repair_is_single_pass_and_scoped": (
                self.multi_repair_is_single_pass_and_scoped
            ),
            "model_loads_bounded": self.model_loads_bounded,
            "qwen_is_policy_scoped": self.qwen_is_policy_scoped,
            "offline_operation_passed": self.offline_operation_passed,
            "delivery_verification_separate": self.delivery_verification_separate,
            "passed": self.passed,
            "slices": [_slice_dict(item) for item in self.slices],
        }


def evaluate_performance_observations(
    observations: tuple[PerformanceObservation, ...],
    *,
    protocol: PerformanceProtocol,
) -> SpeechPerformanceQualification:
    """Evaluate frozen benchmark observations without silently weakening gates."""
    _validate_observation_set(observations)
    grouped = _group_observations(observations)
    slices = _performance_slices(grouped, protocol)
    coverage_complete = _coverage_complete(grouped, protocol)
    measured = tuple(item for item in observations if item.measured)
    repeat_uses_zero_inference = all(
        item.inference_calls == 0
        for item in measured
        if item.workflow is BenchmarkWorkflow.UNCHANGED_BUILD
    )
    localized_repair_within_fraction = _localized_repair_within_fraction(
        slices,
        protocol.maximum_localized_repair_fraction,
    )
    multi_repair = tuple(
        item for item in measured if item.workflow is BenchmarkWorkflow.MULTI_REPAIR
    )
    multi_repair_is_single_pass_and_scoped = bool(multi_repair) and all(
        item.raw_assembly_passes == 1
        and item.mastering_passes == 1
        and item.analyzed_scope_audio_seconds <= item.affected_scope_audio_seconds
        for item in multi_repair
    )
    model_loads_bounded = all(
        item.faulted
        or all(
            load.count <= protocol.maximum_model_loads_per_engine
            for load in item.model_loads
        )
        for item in measured
    )
    qwen_is_policy_scoped = all(
        item.qwen_windows == item.policy_required_qwen_windows for item in measured
    )
    offline_operation_passed = bool(measured) and all(item.offline for item in measured)
    delivery = tuple(
        item
        for item in measured
        if item.workflow is BenchmarkWorkflow.DELIVERY_VERIFICATION
    )
    release = tuple(
        item
        for item in measured
        if item.workflow is BenchmarkWorkflow.MASTERED_RELEASE_PROMOTION
    )
    delivery_verification_separate = bool(delivery and release) and all(
        item.delivery_verification_mode is not DeliveryVerificationMode.NOT_APPLICABLE
        for item in delivery
    )
    gates = (
        coverage_complete,
        repeat_uses_zero_inference,
        localized_repair_within_fraction,
        multi_repair_is_single_pass_and_scoped,
        model_loads_bounded,
        qwen_is_policy_scoped,
        offline_operation_passed,
        delivery_verification_separate,
    )
    return SpeechPerformanceQualification(
        protocol_fingerprint=protocol.fingerprint,
        observation_fingerprint=semantic_fingerprint(
            "speech-performance-observations-v1",
            tuple(sorted(observations, key=lambda item: item.observation_id)),
        ),
        slices=slices,
        coverage_complete=coverage_complete,
        repeat_uses_zero_inference=repeat_uses_zero_inference,
        localized_repair_within_fraction=localized_repair_within_fraction,
        multi_repair_is_single_pass_and_scoped=(multi_repair_is_single_pass_and_scoped),
        model_loads_bounded=model_loads_bounded,
        qwen_is_policy_scoped=qwen_is_policy_scoped,
        offline_operation_passed=offline_operation_passed,
        delivery_verification_separate=delivery_verification_separate,
        passed=all(gates),
    )


def load_default_performance_protocol() -> PerformanceProtocol:
    return load_performance_protocol(
        Path(__file__).parents[1] / "data" / "speech-performance-protocol-v1.toml"
    )


def load_performance_protocol(path: Path) -> PerformanceProtocol:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValidationError("Cannot read speech performance protocol") from error
    expected = {
        "version",
        "minimum_measured_runs",
        "minimum_p95_runs",
        "maximum_localized_repair_fraction",
        "maximum_model_loads_per_engine",
        "required_workflows",
        "required_cache_states",
        "p95_workflows",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValidationError("Speech performance protocol fields are invalid")
    try:
        workflows = tuple(
            BenchmarkWorkflow(item) for item in _text_array(raw, "required_workflows")
        )
        cache_states = tuple(
            BenchmarkCacheState(item)
            for item in _text_array(raw, "required_cache_states")
        )
        p95_workflows = tuple(
            BenchmarkWorkflow(item) for item in _text_array(raw, "p95_workflows")
        )
    except ValueError as error:
        raise ValidationError("Speech performance protocol enum is invalid") from error
    return PerformanceProtocol(
        version=_integer(raw, "version"),
        minimum_measured_runs=_integer(raw, "minimum_measured_runs"),
        minimum_p95_runs=_integer(raw, "minimum_p95_runs"),
        maximum_localized_repair_fraction=_number(
            raw, "maximum_localized_repair_fraction"
        ),
        maximum_model_loads_per_engine=_integer(raw, "maximum_model_loads_per_engine"),
        required_workflows=workflows,
        required_cache_states=cache_states,
        p95_workflows=p95_workflows,
    )


def _validate_observation_set(
    observations: tuple[PerformanceObservation, ...],
) -> None:
    if not observations:
        raise ValidationError("Speech performance qualification requires observations")
    identifiers = tuple(item.observation_id for item in observations)
    if len(identifiers) != len(set(identifiers)):
        raise ValidationError("Speech performance observation IDs must be unique")
    groups: dict[str, set[tuple[str, str]]] = defaultdict(set)
    run_indices: set[tuple[str, BenchmarkWorkflow, BenchmarkCacheState, int, bool]] = (
        set()
    )
    for item in observations:
        groups[item.comparison_group].add(
            (
                item.quality_policy_fingerprint,
                item.terminal_artifact_state_fingerprint,
            )
        )
        identity = (
            item.comparison_group,
            item.workflow,
            item.cache_state,
            item.run_index,
            item.measured,
        )
        if identity in run_indices:
            raise ValidationError("Speech performance run identity is duplicated")
        run_indices.add(identity)
    if any(len(identities) != 1 for identities in groups.values()):
        raise ValidationError(
            "Compared performance runs must bind one quality policy and terminal state"
        )


type _GroupKey = tuple[str, BenchmarkWorkflow, BenchmarkCacheState]


def _group_observations(
    observations: tuple[PerformanceObservation, ...],
) -> dict[_GroupKey, tuple[PerformanceObservation, ...]]:
    grouped: dict[_GroupKey, list[PerformanceObservation]] = defaultdict(list)
    for item in observations:
        grouped[(item.comparison_group, item.workflow, item.cache_state)].append(item)
    return {
        key: tuple(sorted(values, key=lambda item: (item.measured, item.run_index)))
        for key, values in grouped.items()
    }


def _coverage_complete(
    grouped: dict[_GroupKey, tuple[PerformanceObservation, ...]],
    protocol: PerformanceProtocol,
) -> bool:
    comparison_groups = {group for group, _workflow, _cache in grouped}
    if not comparison_groups:
        return False
    for group in comparison_groups:
        for workflow in protocol.required_workflows:
            for cache_state in protocol.required_cache_states:
                values = grouped.get((group, workflow, cache_state), ())
                measured_count = sum(item.measured for item in values)
                minimum = (
                    protocol.minimum_p95_runs
                    if workflow in protocol.p95_workflows
                    else protocol.minimum_measured_runs
                )
                if measured_count < minimum or not any(
                    not item.measured for item in values
                ):
                    return False
    return True


def _performance_slices(
    grouped: dict[_GroupKey, tuple[PerformanceObservation, ...]],
    protocol: PerformanceProtocol,
) -> tuple[PerformanceSlice, ...]:
    slices: list[PerformanceSlice] = []
    for (group, workflow, cache_state), values in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1].value,
            item[0][2].value,
        ),
    ):
        measured = tuple(item for item in values if item.measured)
        if not measured:
            continue
        durations = tuple(item.speech_analysis_seconds for item in measured)
        audio = tuple(item.audio_seconds for item in measured)
        p95 = (
            _percentile(durations, 0.95)
            if len(measured) >= protocol.minimum_p95_runs
            else None
        )
        slices.append(
            PerformanceSlice(
                group,
                workflow,
                cache_state,
                len(measured),
                float(median(durations)),
                min(durations),
                max(durations),
                p95,
                float(median(audio)),
                max(item.peak_memory_bytes for item in measured),
            )
        )
    return tuple(slices)


def _localized_repair_within_fraction(
    slices: tuple[PerformanceSlice, ...],
    maximum_fraction: float,
) -> bool:
    indexed = {
        (item.comparison_group, item.workflow, item.cache_state): item
        for item in slices
    }
    comparisons = 0
    for item in slices:
        if item.workflow is not BenchmarkWorkflow.LOCALIZED_REPAIR:
            continue
        baseline = indexed.get(
            (
                item.comparison_group,
                BenchmarkWorkflow.FULL_CHAPTER_VERIFICATION,
                item.cache_state,
            )
        )
        if baseline is None or baseline.median_speech_analysis_seconds <= 0:
            return False
        comparisons += 1
        if (
            item.median_speech_analysis_seconds
            >= baseline.median_speech_analysis_seconds * maximum_fraction
        ):
            return False
    return comparisons > 0


def _slice_dict(item: PerformanceSlice) -> dict[str, object]:
    return {
        "comparison_group": item.comparison_group,
        "workflow": item.workflow.value,
        "cache_state": item.cache_state.value,
        "measured_runs": item.measured_runs,
        "median_speech_analysis_seconds": item.median_speech_analysis_seconds,
        "minimum_speech_analysis_seconds": item.minimum_speech_analysis_seconds,
        "maximum_speech_analysis_seconds": item.maximum_speech_analysis_seconds,
        "p95_speech_analysis_seconds": item.p95_speech_analysis_seconds,
        "median_audio_seconds": item.median_audio_seconds,
        "peak_memory_bytes": item.peak_memory_bytes,
    }


def _percentile(values: tuple[float, ...], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _integer(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"Speech performance protocol {key!r} must be integer")
    return value


def _number(raw: dict[str, object], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"Speech performance protocol {key!r} must be numeric")
    return float(value)


def _text_array(raw: dict[str, object], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValidationError(
            f"Speech performance protocol {key!r} must be a text array"
        )
    return tuple(cast(list[str], value))


__all__ = [
    "BenchmarkCacheState",
    "BenchmarkWorkflow",
    "DeliveryVerificationMode",
    "EngineLoadCount",
    "PerformanceObservation",
    "PerformanceProtocol",
    "PerformanceSlice",
    "SpeechPerformanceQualification",
    "evaluate_performance_observations",
    "load_default_performance_protocol",
    "load_performance_protocol",
]
