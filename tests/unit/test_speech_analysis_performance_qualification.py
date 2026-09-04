from __future__ import annotations

from dataclasses import replace

from jsonschema import Draft202012Validator

from yakbox.schemas import load_schema
from yakbox.speech.analysis_performance_qualification import (
    BenchmarkCacheState,
    BenchmarkWorkflow,
    DeliveryVerificationMode,
    EngineLoadCount,
    PerformanceObservation,
    evaluate_performance_observations,
    load_default_performance_protocol,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _observations() -> tuple[PerformanceObservation, ...]:
    protocol = load_default_performance_protocol()
    values: list[PerformanceObservation] = []
    identifier = 0
    for workflow in protocol.required_workflows:
        for cache_state in protocol.required_cache_states:
            run_count = (
                protocol.minimum_p95_runs
                if workflow in protocol.p95_workflows
                else protocol.minimum_measured_runs
            )
            for run_index in range(run_count + 1):
                measured = run_index > 0
                identifier += 1
                duration = _duration(workflow)
                values.append(
                    PerformanceObservation(
                        observation_id=f"observation-{identifier}",
                        comparison_group="reference-m5",
                        workflow=workflow,
                        cache_state=cache_state,
                        run_index=run_index,
                        measured=measured,
                        quality_policy_fingerprint=SHA_A,
                        terminal_artifact_state_fingerprint=SHA_B,
                        audio_seconds=600.0,
                        speech_analysis_seconds=duration,
                        peak_memory_bytes=1_000,
                        inference_calls=(
                            0 if workflow is BenchmarkWorkflow.UNCHANGED_BUILD else 1
                        ),
                        model_loads=(EngineLoadCount("whisper", 1),),
                        qwen_windows=1,
                        policy_required_qwen_windows=1,
                        raw_assembly_passes=(
                            1 if workflow is BenchmarkWorkflow.MULTI_REPAIR else 0
                        ),
                        mastering_passes=(
                            1 if workflow is BenchmarkWorkflow.MULTI_REPAIR else 0
                        ),
                        affected_scope_audio_seconds=30.0,
                        analyzed_scope_audio_seconds=(
                            20.0 if workflow is BenchmarkWorkflow.MULTI_REPAIR else 0.0
                        ),
                        offline=True,
                        faulted=False,
                        delivery_verification_mode=(
                            DeliveryVerificationMode.FULL_RECOGNITION
                            if workflow is BenchmarkWorkflow.DELIVERY_VERIFICATION
                            else DeliveryVerificationMode.NOT_APPLICABLE
                        ),
                    )
                )
    return tuple(values)


def _duration(workflow: BenchmarkWorkflow) -> float:
    if workflow is BenchmarkWorkflow.FULL_CHAPTER_VERIFICATION:
        return 100.0
    if workflow is BenchmarkWorkflow.LOCALIZED_REPAIR:
        return 20.0
    if workflow is BenchmarkWorkflow.UNCHANGED_BUILD:
        return 0.0
    return 30.0


def test_performance_protocol_enforces_every_release_criterion() -> None:
    protocol = load_default_performance_protocol()

    qualification = evaluate_performance_observations(
        _observations(),
        protocol=protocol,
    )

    assert qualification.passed is True
    assert qualification.coverage_complete is True
    assert len(qualification.slices) == 18
    unchanged = next(
        item
        for item in qualification.slices
        if item.workflow is BenchmarkWorkflow.UNCHANGED_BUILD
    )
    assert unchanged.measured_runs == 20
    assert unchanged.p95_speech_analysis_seconds == 0
    Draft202012Validator(load_schema("speech-performance-qualification")).validate(
        qualification.to_dict()
    )


def test_localized_repair_must_be_strictly_less_than_quarter_chapter_cost() -> None:
    protocol = load_default_performance_protocol()
    changed = tuple(
        replace(item, speech_analysis_seconds=25.0)
        if item.workflow is BenchmarkWorkflow.LOCALIZED_REPAIR
        else item
        for item in _observations()
    )

    qualification = evaluate_performance_observations(changed, protocol=protocol)

    assert qualification.passed is False
    assert qualification.localized_repair_within_fraction is False


def test_repeat_model_load_qwen_and_multi_repair_failures_are_independent() -> None:
    protocol = load_default_performance_protocol()
    base = _observations()
    repeat = tuple(
        replace(item, inference_calls=1)
        if item.measured and item.workflow is BenchmarkWorkflow.UNCHANGED_BUILD
        else item
        for item in base
    )
    loads = tuple(
        replace(item, model_loads=(EngineLoadCount("whisper", 2),))
        if item.measured
        else item
        for item in base
    )
    qwen = tuple(
        replace(item, qwen_windows=2) if item.measured else item for item in base
    )
    multi = tuple(
        replace(item, mastering_passes=2)
        if item.measured and item.workflow is BenchmarkWorkflow.MULTI_REPAIR
        else item
        for item in base
    )

    assert not evaluate_performance_observations(
        repeat, protocol=protocol
    ).repeat_uses_zero_inference
    assert not evaluate_performance_observations(
        loads, protocol=protocol
    ).model_loads_bounded
    assert not evaluate_performance_observations(
        qwen, protocol=protocol
    ).qwen_is_policy_scoped
    assert not evaluate_performance_observations(
        multi, protocol=protocol
    ).multi_repair_is_single_pass_and_scoped


def test_missing_warmup_or_measured_runs_cannot_claim_coverage() -> None:
    protocol = load_default_performance_protocol()
    incomplete = tuple(
        item
        for item in _observations()
        if not (
            item.workflow is BenchmarkWorkflow.LOCALIZED_REPAIR
            and item.cache_state is BenchmarkCacheState.COLD_PROCESS_COLD_CACHE
            and not item.measured
        )
    )

    qualification = evaluate_performance_observations(incomplete, protocol=protocol)

    assert qualification.coverage_complete is False
    assert qualification.passed is False
