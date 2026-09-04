from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from yakbox.errors import SpeechAnalysisError
from yakbox.schemas import load_schema
from yakbox.speech.analysis_capabilities import default_capability_matrix
from yakbox.speech.analysis_manifest import default_draft_speech_analysis_config
from yakbox.speech.analysis_models import ClipClass
from yakbox.speech.analysis_policy import CalibrationTable, CalibrationThreshold
from yakbox.speech.analysis_preflight import (
    SpeechAnalysisPreflight,
    SpeechAnalysisPreflightRequest,
    evaluate_speech_analysis_preflight,
    require_speech_analysis_preflight,
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


def _inputs() -> tuple[
    SpeechAnalysisPreflightRequest,
    CalibrationTable,
    AnalysisRuntimeReport,
    tuple[ModelStatus, ...],
]:
    request = SpeechAnalysisPreflightRequest(
        language="en",
        engine_window_frames=(
            ("parakeet", 16_000),
            ("qwen", 16_000),
            ("qwen-forced", 16_000),
            ("whisper", 16_000),
        ),
        clip_classes=(ClipClass.SENTENCE,),
        execution_fingerprint=SHA_A,
        required_disk_bytes=1_000,
        available_disk_bytes=2_000,
        required_memory_bytes=1_000,
        available_memory_bytes=2_000,
    )
    calibration = CalibrationTable(
        1,
        "en",
        SHA_B,
        True,
        SHA_C,
        (SHA_A,),
        tuple(
            CalibrationThreshold(engine, ClipClass.SENTENCE, SHA_D)
            for engine in ("whisper", "parakeet", "qwen")
        ),
    )
    runtimes = AnalysisRuntimeReport(
        Path("/managed/runtimes"),
        tuple(
            InstalledAnalysisRuntime(
                family,
                fingerprint,
                Path("/managed/runtimes") / family,
                Path("/managed/runtimes") / family / "python",
                SHA_A,
                SHA_B,
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
    return request, calibration, runtimes, models


def _evaluate(
    request: SpeechAnalysisPreflightRequest,
    calibration: CalibrationTable,
    runtimes: AnalysisRuntimeReport,
    models: tuple[ModelStatus, ...],
) -> SpeechAnalysisPreflight:
    return evaluate_speech_analysis_preflight(
        request,
        policy=default_draft_speech_analysis_config().policy,
        capabilities=default_capability_matrix(),
        calibration=calibration,
        runtimes=runtimes,
        models=models,
    )


def test_preflight_accepts_complete_strict_pipeline_and_validates_schema() -> None:
    result = _evaluate(*_inputs())

    assert result.ready
    assert not result.issue_codes
    Draft202012Validator(load_schema("speech-analysis-preflight")).validate(
        result.to_dict()
    )


def test_preflight_reports_every_missing_resource_before_synthesis() -> None:
    request, calibration, runtimes, models = _inputs()
    broken_request = replace(
        request,
        engine_window_frames=tuple(
            (engine, 5_000_001 if engine == "qwen-forced" else frames)
            for engine, frames in request.engine_window_frames
        ),
        available_disk_bytes=999,
        available_memory_bytes=999,
    )
    broken_calibration = replace(calibration, approved=False)
    broken_runtimes = replace(runtimes, runtimes=runtimes.runtimes[:-1])
    broken_models = tuple(
        replace(item, verified=False) if item.engine == "whisper" else item
        for item in models
        if item.engine != "qwen-forced"
    )

    result = _evaluate(
        broken_request,
        broken_calibration,
        broken_runtimes,
        broken_models,
    )

    assert not result.ready
    assert set(result.issue_codes) >= {
        "calibration_not_approved",
        "disk_capacity_insufficient",
        "duration_exceeded_qwen_forced",
        "memory_capacity_insufficient",
        "model_missing_qwen_forced",
        "model_unverified_whisper",
        "runtime_missing_qwen",
    }
    with pytest.raises(SpeechAnalysisError, match="preflight failed"):
        require_speech_analysis_preflight(
            broken_request,
            policy=default_draft_speech_analysis_config().policy,
            capabilities=default_capability_matrix(),
            calibration=broken_calibration,
            runtimes=broken_runtimes,
            models=broken_models,
        )


def test_preflight_rejects_unsupported_language_and_calibration_gaps() -> None:
    request, calibration, runtimes, models = _inputs()
    unsupported = replace(request, language="fr")
    incomplete = replace(calibration, thresholds=calibration.thresholds[:-1])

    language_result = _evaluate(unsupported, calibration, runtimes, models)
    calibration_result = _evaluate(request, incomplete, runtimes, models)

    assert "language_policy_mismatch" in language_result.issue_codes
    assert "capability_missing_whisper" in language_result.issue_codes
    assert "calibration_language_mismatch" in language_result.issue_codes
    assert calibration_result.issue_codes == ("calibration_threshold_missing",)
