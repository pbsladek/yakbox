"""Fail-fast resource and capability checks for strict speech analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass

from yakbox.contracts import runtime_metadata
from yakbox.errors import SpeechAnalysisError, ValidationError
from yakbox.speech.analysis_capabilities import CapabilityMatrix
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_models import ClipClass
from yakbox.speech.analysis_policy import CalibrationTable, SpeechAnalysisPolicy
from yakbox.speech.analysis_runtime_install import AnalysisRuntimeReport
from yakbox.speech.model_registry import ModelStatus

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ISSUE = re.compile(r"[a-z][a-z0-9_]{1,127}")
_ENGINE_FAMILY = {
    "whisper": "whisper",
    "parakeet": "parakeet",
    "qwen": "qwen",
    "qwen-forced": "qwen",
}


@dataclass(frozen=True, slots=True)
class SpeechAnalysisPreflightRequest:
    """Exact planned work and host capacity checked before synthesis."""

    language: str
    engine_window_frames: tuple[tuple[str, int], ...]
    clip_classes: tuple[ClipClass, ...]
    execution_fingerprint: str
    required_disk_bytes: int
    available_disk_bytes: int
    required_memory_bytes: int
    available_memory_bytes: int

    def __post_init__(self) -> None:
        engines = tuple(engine for engine, _frames in self.engine_window_frames)
        if (
            not self.language
            or not self.engine_window_frames
            or engines != tuple(sorted(set(engines)))
            or any(frames <= 0 for _engine, frames in self.engine_window_frames)
            or not self.clip_classes
            or tuple(sorted(set(self.clip_classes), key=lambda item: item.value))
            != self.clip_classes
            or _SHA256.fullmatch(self.execution_fingerprint) is None
            or min(
                self.required_disk_bytes,
                self.available_disk_bytes,
                self.required_memory_bytes,
                self.available_memory_bytes,
            )
            < 0
        ):
            raise ValidationError("Speech-analysis preflight request is invalid")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-analysis-preflight-request-v1", self)


@dataclass(frozen=True, slots=True)
class SpeechAnalysisPreflight:
    """Privacy-safe terminal preflight result with stable issue codes."""

    request_fingerprint: str
    policy_fingerprint: str
    capability_fingerprint: str
    calibration_disposition_fingerprint: str
    runtime_install_fingerprints: tuple[str, ...]
    model_directory_fingerprints: tuple[str, ...]
    issue_codes: tuple[str, ...]
    ready: bool

    def __post_init__(self) -> None:
        fingerprints = (
            self.request_fingerprint,
            self.policy_fingerprint,
            self.capability_fingerprint,
            self.calibration_disposition_fingerprint,
            *self.runtime_install_fingerprints,
            *self.model_directory_fingerprints,
        )
        if (
            any(_SHA256.fullmatch(item) is None for item in fingerprints)
            or tuple(sorted(set(self.runtime_install_fingerprints)))
            != self.runtime_install_fingerprints
            or tuple(sorted(set(self.model_directory_fingerprints)))
            != self.model_directory_fingerprints
            or tuple(sorted(set(self.issue_codes))) != self.issue_codes
            or any(_ISSUE.fullmatch(item) is None for item in self.issue_codes)
            or self.ready == bool(self.issue_codes)
        ):
            raise ValidationError("Speech-analysis preflight result is inconsistent")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-analysis-preflight-v1", self)

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("speech-analysis-preflight"),
            "fingerprint": self.fingerprint,
            "ready": self.ready,
            "request_fingerprint": self.request_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "capability_fingerprint": self.capability_fingerprint,
            "calibration_disposition_fingerprint": (
                self.calibration_disposition_fingerprint
            ),
            "runtime_install_fingerprints": list(self.runtime_install_fingerprints),
            "model_directory_fingerprints": list(self.model_directory_fingerprints),
            "issue_codes": list(self.issue_codes),
        }


def evaluate_speech_analysis_preflight(
    request: SpeechAnalysisPreflightRequest,
    *,
    policy: SpeechAnalysisPolicy,
    capabilities: CapabilityMatrix,
    calibration: CalibrationTable,
    runtimes: AnalysisRuntimeReport,
    models: tuple[ModelStatus, ...],
) -> SpeechAnalysisPreflight:
    """Evaluate every strict prerequisite without loading a synthesis backend."""
    issues: list[str] = []
    required_engines = tuple(engine.engine for engine in policy.engines)
    if request.language != policy.language:
        issues.append("language_policy_mismatch")
    window_frames = dict(request.engine_window_frames)
    for engine in required_engines:
        if engine not in window_frames:
            issues.append(f"window_plan_missing_{_issue_name(engine)}")
            continue
        try:
            resolved = (
                capabilities.resolve_forced_aligner(engine, request.language)
                if engine == policy.forced_aligner
                else capabilities.resolve_recognizer(engine, request.language)
            )
        except ValidationError:
            issues.append(f"capability_missing_{_issue_name(engine)}")
            continue
        configured = next(item for item in policy.engines if item.engine == engine)
        maximum = resolved.capabilities.maximum_duration_frames
        if configured.maximum_window_seconds is not None:
            maximum = min(
                maximum,
                configured.maximum_window_seconds * resolved.capabilities.sample_rate,
            )
        if window_frames[engine] > maximum:
            issues.append(f"duration_exceeded_{_issue_name(engine)}")
    extra_windows = set(window_frames) - set(required_engines)
    if extra_windows:
        issues.append("window_plan_engine_unexpected")
    _runtime_issues(required_engines, runtimes, issues)
    _model_issues(required_engines, models, issues)
    _calibration_issues(request, policy, calibration, issues)
    if request.available_disk_bytes < request.required_disk_bytes:
        issues.append("disk_capacity_insufficient")
    if request.available_memory_bytes < request.required_memory_bytes:
        issues.append("memory_capacity_insufficient")
    runtime_fingerprints = tuple(
        sorted(item.install_fingerprint for item in runtimes.runtimes)
    )
    model_fingerprints = tuple(
        sorted(
            item.directory_fingerprint
            for item in models
            if item.directory_fingerprint is not None
        )
    )
    ordered_issues = tuple(sorted(set(issues)))
    return SpeechAnalysisPreflight(
        request.fingerprint,
        policy.fingerprint,
        capabilities.fingerprint,
        calibration.disposition_fingerprint,
        runtime_fingerprints,
        model_fingerprints,
        ordered_issues,
        not ordered_issues,
    )


def require_speech_analysis_preflight(
    request: SpeechAnalysisPreflightRequest,
    *,
    policy: SpeechAnalysisPolicy,
    capabilities: CapabilityMatrix,
    calibration: CalibrationTable,
    runtimes: AnalysisRuntimeReport,
    models: tuple[ModelStatus, ...],
) -> SpeechAnalysisPreflight:
    """Return a ready preflight or fail before synthesis can begin."""
    result = evaluate_speech_analysis_preflight(
        request,
        policy=policy,
        capabilities=capabilities,
        calibration=calibration,
        runtimes=runtimes,
        models=models,
    )
    if not result.ready:
        raise SpeechAnalysisError(
            "Speech-analysis preflight failed: " + ", ".join(result.issue_codes)
        )
    return result


def _runtime_issues(
    engines: tuple[str, ...],
    runtimes: AnalysisRuntimeReport,
    issues: list[str],
) -> None:
    by_family = {item.family: item for item in runtimes.runtimes}
    for family in sorted({_ENGINE_FAMILY.get(engine, "") for engine in engines}):
        if not family:
            issues.append("runtime_family_unknown")
        elif family not in by_family:
            issues.append(f"runtime_missing_{_issue_name(family)}")
        elif not by_family[family].verified:
            issues.append(f"runtime_unverified_{_issue_name(family)}")


def _model_issues(
    engines: tuple[str, ...],
    models: tuple[ModelStatus, ...],
    issues: list[str],
) -> None:
    by_engine = {item.engine: item for item in models}
    if len(by_engine) != len(models):
        issues.append("model_engine_duplicate")
    for engine in engines:
        status = by_engine.get(engine)
        if status is None:
            issues.append(f"model_missing_{_issue_name(engine)}")
        elif not status.verified or status.directory_fingerprint is None:
            issues.append(f"model_unverified_{_issue_name(engine)}")


def _calibration_issues(
    request: SpeechAnalysisPreflightRequest,
    policy: SpeechAnalysisPolicy,
    calibration: CalibrationTable,
    issues: list[str],
) -> None:
    if not calibration.approved:
        issues.append("calibration_not_approved")
    if calibration.language != request.language:
        issues.append("calibration_language_mismatch")
    if request.execution_fingerprint not in calibration.execution_class_fingerprints:
        issues.append("calibration_execution_unqualified")
    calibrated = {(item.engine, item.clip_class) for item in calibration.thresholds}
    recognizers = {
        *policy.baseline_recognizers,
        policy.escalation_recognizer,
    }
    if any(
        (engine, clip_class) not in calibrated
        for engine in recognizers
        for clip_class in request.clip_classes
    ):
        issues.append("calibration_threshold_missing")


def _issue_name(value: str) -> str:
    return value.replace("-", "_")


__all__ = [
    "SpeechAnalysisPreflight",
    "SpeechAnalysisPreflightRequest",
    "evaluate_speech_analysis_preflight",
    "require_speech_analysis_preflight",
]
