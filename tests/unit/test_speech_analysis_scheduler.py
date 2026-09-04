from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import cast

import pytest

from yakbox.errors import SpeechAnalysisError, ValidationError, WorkerProtocolError
from yakbox.speech.analysis_scheduler import (
    AcceleratorLease,
    AnalysisPerformanceCollector,
    AnalysisSupervisor,
    BatchCompletion,
    BatchOperation,
    BatchWorkItem,
    BoundedCandidateSearch,
    CandidateStageResult,
    CandidateWork,
    MemoryWatermark,
    ModelMajorRepairScheduler,
    OperationMetrics,
    OperationStageTimings,
    OperationTerminalStatus,
    PlannedWorkerIdentity,
    ResidentModelManager,
    ScheduledCandidate,
    WorkerHandshake,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def _handshake(**changes: object) -> WorkerHandshake:
    baseline = WorkerHandshake(2, SHA_A, SHA_B, SHA_C, SHA_D, SHA_E, SHA_F)
    return replace(baseline, **changes)


def _planned() -> PlannedWorkerIdentity:
    return PlannedWorkerIdentity(SHA_A, SHA_B, SHA_C, SHA_D, SHA_E, SHA_F)


def _metrics(*, wall: int = 10, memory: int = 100) -> OperationMetrics:
    return OperationMetrics(100, 100, wall, memory, 50, False, 1)


def _operation(*, timeout_seconds: float = 10) -> BatchOperation:
    return BatchOperation(
        "operation-1",
        "batch-1",
        "whisper",
        SHA_A,
        "recognize_many",
        time.monotonic_ns() + round(timeout_seconds * 1_000_000_000),
        (
            BatchWorkItem(0, SHA_A, "one.wav", 0, 100),
            BatchWorkItem(1, SHA_B, "two.wav", 0, 100),
        ),
    )


class _Worker:
    def __init__(
        self,
        completions: tuple[BatchCompletion, ...] = (),
        *,
        handshake: WorkerHandshake | None = None,
        crash_after: int | None = None,
        block: bool = False,
        watermark: MemoryWatermark | None = None,
    ) -> None:
        self.completions = completions
        self.handshake_value = handshake or _handshake()
        self.crash_after = crash_after
        self.block = block
        self.watermark = watermark or MemoryWatermark(10, 10)
        self.terminated = 0
        self.cancelled = 0
        self.unloaded: list[str] = []
        self.requested_indices: list[tuple[int, ...]] = []
        self.started = asyncio.Event()
        self._idle = asyncio.Event()
        if not block:
            self._idle.set()

    async def handshake(self) -> WorkerHandshake:
        return self.handshake_value

    async def execute(
        self, operation: BatchOperation
    ) -> AsyncIterator[BatchCompletion]:
        self.requested_indices.append(tuple(item.index for item in operation.items))
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        selected = {item.index for item in operation.items}
        for position, completion in enumerate(self.completions):
            if completion.index not in selected:
                continue
            if self.crash_after is not None and position == self.crash_after:
                raise EOFError
            yield completion
        if self.crash_after is not None and self.crash_after >= len(self.completions):
            raise EOFError

    async def soft_cancel(self, operation_id: str) -> None:
        del operation_id
        self.cancelled += 1

    async def wait_idle(self) -> None:
        await self._idle.wait()

    async def terminate(self) -> None:
        self.terminated += 1
        self._idle.set()

    async def unload(self, engine: str) -> MemoryWatermark:
        self.unloaded.append(engine)
        return self.watermark

    async def current_memory(self) -> MemoryWatermark:
        return self.watermark


def _completion(
    index: int,
    *,
    evidence: str,
    metrics: OperationMetrics | None = None,
) -> BatchCompletion:
    return BatchCompletion(
        "batch-1",
        index,
        SHA_A if index == 0 else SHA_B,
        OperationTerminalStatus.COMPLETED,
        evidence,
        None,
        metrics or _metrics(),
    )


@pytest.mark.parametrize(
    "unsafe_path",
    ("../outside.wav", "/absolute.wav", r"C:\outside.wav", r"folder\clip.wav"),
)
def test_batch_items_reject_cross_platform_path_escapes(unsafe_path: str) -> None:
    with pytest.raises(ValidationError, match="batch item is invalid"):
        BatchWorkItem(0, SHA_A, unsafe_path, 0, 100)


def test_batch_operations_require_cryptographic_fingerprints() -> None:
    with pytest.raises(ValidationError, match="batch item is invalid"):
        BatchWorkItem(0, "not-a-fingerprint", "one.wav", 0, 100)

    with pytest.raises(ValidationError, match="operation is incomplete"):
        BatchOperation(
            "operation-1",
            "batch-1",
            "whisper",
            "not-a-fingerprint",
            "recognize_many",
            time.monotonic_ns() + 1_000_000,
            (BatchWorkItem(0, SHA_A, "one.wav", 0, 100),),
        )


def test_completion_semantic_identity_ignores_batch_position_and_observability() -> (
    None
):
    first = _completion(0, evidence=SHA_C, metrics=_metrics(wall=1, memory=2))
    second = replace(
        first,
        batch_id="different-batch",
        index=99,
        metrics=OperationMetrics(100, 100, 999, 888, 777, True, 0),
    )

    assert second.semantic_fingerprint == first.semantic_fingerprint


@pytest.mark.asyncio
async def test_supervisor_reorders_and_coalesces_duplicate_completions() -> None:
    second = _completion(1, evidence=SHA_D)
    duplicate_with_new_metrics = _completion(
        1, evidence=SHA_D, metrics=_metrics(wall=999, memory=999)
    )
    first = _completion(0, evidence=SHA_C)
    worker = _Worker((second, duplicate_with_new_metrics, first))
    committed: list[int] = []

    async def factory() -> _Worker:
        return worker

    async def commit(value: BatchCompletion) -> None:
        committed.append(value.index)

    result = await AnalysisSupervisor(
        worker_factory=factory,
        planned_identity=_planned(),
        accelerator_lease=AcceleratorLease(),
        commit=commit,
    ).execute(_operation())

    assert tuple(item.index for item in result.completions) == (0, 1)
    assert committed == [1, 0]
    assert result.worker_restarts == 0


@pytest.mark.asyncio
async def test_supervisor_reports_privacy_safe_per_engine_stage_performance() -> None:
    first_metrics = OperationMetrics(
        100,
        100,
        2_000_000_000,
        1_000,
        500,
        False,
        1,
        "worker_inference_required",
        OperationStageTimings(
            preparing_audio_ns=100,
            inference_ns=200,
            normalizing_tokens_ns=300,
            writing_evidence_ns=400,
        ),
    )
    second_metrics = OperationMetrics(
        100,
        100,
        1_000_000_000,
        900,
        400,
        True,
        0,
    )
    worker = _Worker(
        (
            _completion(0, evidence=SHA_C, metrics=first_metrics),
            _completion(1, evidence=SHA_D, metrics=second_metrics),
        )
    )
    collector = AnalysisPerformanceCollector()

    async def factory() -> _Worker:
        return worker

    async def commit(_value: BatchCompletion) -> None:
        return

    await AnalysisSupervisor(
        worker_factory=factory,
        planned_identity=_planned(),
        accelerator_lease=AcceleratorLease(),
        commit=commit,
        performance=collector,
    ).execute(_operation())
    collector.record_lifecycle(
        engine="whisper",
        stage="recognize_many",
        unloads=1,
        evictions=1,
    )

    report = collector.to_dict()
    engines = cast(dict[str, dict[str, object]], report["engines"])
    stages = cast(dict[str, dict[str, object]], engines["whisper"]["stages"])
    stage = stages["recognize_many"]
    assert stage["cold_loads"] == 1
    assert stage["cache_hits"] == 1
    assert stage["cache_misses"] == 1
    assert stage["cache_miss_reasons"] == {"worker_inference_required": 1}
    assert stage["maximum_batch_size"] == 2
    assert stage["survivor_count"] == 2
    assert stage["unloads"] == 1
    assert stage["evictions"] == 1
    assert stage["requested_audio_seconds"] == pytest.approx(0.0125)
    assert stage["real_time_factor"] == pytest.approx(240.0)
    assert "transcript" not in str(report).casefold()


@pytest.mark.asyncio
async def test_conflicting_duplicate_is_protocol_violation() -> None:
    worker = _Worker(
        (
            _completion(0, evidence=SHA_C),
            _completion(0, evidence=SHA_D),
            _completion(1, evidence=SHA_E),
        )
    )

    async def factory() -> _Worker:
        return worker

    async def commit(_value: BatchCompletion) -> None:
        return

    with pytest.raises(WorkerProtocolError, match="conflicts"):
        await AnalysisSupervisor(
            worker_factory=factory,
            planned_identity=_planned(),
            accelerator_lease=AcceleratorLease(),
            commit=commit,
        ).execute(_operation())


@pytest.mark.asyncio
async def test_crash_restarts_only_uncommitted_idempotent_items() -> None:
    first = _Worker(
        (_completion(0, evidence=SHA_C), _completion(1, evidence=SHA_D)),
        crash_after=1,
    )
    second = _Worker((_completion(1, evidence=SHA_D),))
    workers = iter((first, second))
    committed: list[int] = []

    async def factory() -> _Worker:
        return next(workers)

    async def commit(value: BatchCompletion) -> None:
        committed.append(value.index)

    result = await AnalysisSupervisor(
        worker_factory=factory,
        planned_identity=_planned(),
        accelerator_lease=AcceleratorLease(),
        commit=commit,
    ).execute(_operation())

    assert first.requested_indices == [(0, 1)]
    assert second.requested_indices == [(1,)]
    assert committed == [0, 1]
    assert result.worker_restarts == 1
    assert all(
        item.terminal_status is OperationTerminalStatus.COMPLETED
        for item in result.completions
    )


@pytest.mark.asyncio
async def test_deadline_soft_cancels_then_forces_blocked_worker_termination() -> None:
    worker = _Worker(block=True)
    collector = AnalysisPerformanceCollector()

    async def factory() -> _Worker:
        return worker

    async def commit(_value: BatchCompletion) -> None:
        raise AssertionError("timed-out work cannot commit")

    result = await AnalysisSupervisor(
        worker_factory=factory,
        planned_identity=_planned(),
        accelerator_lease=AcceleratorLease(),
        commit=commit,
        performance=collector,
        cancellation_grace_seconds=0.01,
    ).execute(_operation(timeout_seconds=0.01))

    assert worker.cancelled == 1
    assert worker.terminated == 1
    assert result.soft_cancellations == 1
    assert result.forced_terminations == 1
    engines = cast(dict[str, dict[str, object]], collector.to_dict()["engines"])
    stages = cast(dict[str, dict[str, object]], engines["whisper"]["stages"])
    assert stages["recognize_many"]["soft_cancellations"] == 1
    assert stages["recognize_many"]["forced_terminations"] == 1
    assert all(
        item.terminal_status is OperationTerminalStatus.TIMED_OUT
        for item in result.completions
    )


@pytest.mark.asyncio
async def test_user_cancellation_soft_cancels_then_forces_blocked_worker() -> None:
    worker = _Worker(block=True)

    async def factory() -> _Worker:
        return worker

    async def commit(_value: BatchCompletion) -> None:
        raise AssertionError("cancelled work cannot commit")

    task = asyncio.create_task(
        AnalysisSupervisor(
            worker_factory=factory,
            planned_identity=_planned(),
            accelerator_lease=AcceleratorLease(),
            commit=commit,
            cancellation_grace_seconds=0.01,
        ).execute(_operation())
    )
    await worker.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert worker.cancelled == 1
    assert worker.terminated == 1


@pytest.mark.asyncio
async def test_stale_handshake_is_terminated_without_downgrade() -> None:
    worker = _Worker(handshake=_handshake(protocol_version=1))

    async def factory() -> _Worker:
        return worker

    async def commit(_value: BatchCompletion) -> None:
        return

    with pytest.raises(WorkerProtocolError, match="differs"):
        await AnalysisSupervisor(
            worker_factory=factory,
            planned_identity=_planned(),
            accelerator_lease=AcceleratorLease(),
            commit=commit,
        ).execute(_operation())
    assert worker.terminated == 1


@pytest.mark.asyncio
async def test_accelerator_lease_serializes_tts_and_analysis() -> None:
    lease = AcceleratorLease()
    entered: list[str] = []
    release_tts = asyncio.Event()
    tts_entered = asyncio.Event()

    async def tts() -> None:
        async with lease.hold("tts"):
            entered.append("tts")
            tts_entered.set()
            await release_tts.wait()

    async def analysis() -> None:
        async with lease.hold("analysis"):
            entered.append("analysis")

    tts_task = asyncio.create_task(tts())
    await tts_entered.wait()
    analysis_task = asyncio.create_task(analysis())
    await asyncio.sleep(0)
    assert entered == ["tts"]
    release_tts.set()
    await asyncio.gather(tts_task, analysis_task)
    assert entered == ["tts", "analysis"]


@pytest.mark.asyncio
async def test_unload_above_watermark_recycles_instead_of_false_eviction() -> None:
    worker = _Worker(watermark=MemoryWatermark(1_000, 1_000))
    manager = ResidentModelManager(
        maximum_resident_models=1,
        maximum_resident_memory_bytes=2_000,
        post_unload_resident_watermark_bytes=100,
        post_unload_accelerator_watermark_bytes=100,
    )
    manager.loaded("qwen", 500)

    recycled = await manager.evict(worker, "qwen")

    assert recycled
    assert worker.terminated == 1
    assert manager.metrics.evictions == 0
    assert manager.metrics.restarts == 1


@pytest.mark.asyncio
async def test_unload_uses_fresh_post_grace_memory_before_recording_eviction() -> None:
    class ReleasedWorker(_Worker):
        async def unload(self, engine: str) -> MemoryWatermark:
            self.unloaded.append(engine)
            return MemoryWatermark(1_000, 1_000)

        async def current_memory(self) -> MemoryWatermark:
            return MemoryWatermark(10, 10)

    worker = ReleasedWorker()
    manager = ResidentModelManager(
        maximum_resident_models=1,
        maximum_resident_memory_bytes=2_000,
        post_unload_resident_watermark_bytes=100,
        post_unload_accelerator_watermark_bytes=100,
    )
    manager.loaded("qwen", 500)

    recycled = await manager.evict(worker, "qwen")

    assert recycled is False
    assert worker.terminated == 0
    assert manager.metrics.evictions == 1


@pytest.mark.asyncio
async def test_failed_post_unload_measurement_recycles_uncertain_worker() -> None:
    class UnmeasurableWorker(_Worker):
        async def current_memory(self) -> MemoryWatermark:
            raise WorkerProtocolError("status unavailable")

    worker = UnmeasurableWorker()
    manager = ResidentModelManager(
        maximum_resident_models=1,
        maximum_resident_memory_bytes=2_000,
        post_unload_resident_watermark_bytes=100,
        post_unload_accelerator_watermark_bytes=100,
    )
    manager.loaded("qwen", 500)

    with pytest.raises(WorkerProtocolError, match="status unavailable"):
        await manager.evict(worker, "qwen")

    assert worker.terminated == 1
    assert manager.models() == ()
    assert manager.workers() == ()
    assert manager.metrics.evictions == 0
    assert manager.metrics.restarts == 1


def test_resident_policy_tracks_last_use_and_enforces_worker_budget_atomically() -> (
    None
):
    ticks = iter((10, 20, 30))
    manager = ResidentModelManager(
        maximum_resident_workers=1,
        maximum_resident_models=2,
        maximum_resident_memory_bytes=2_000,
        post_unload_resident_watermark_bytes=100,
        post_unload_accelerator_watermark_bytes=100,
        clock=lambda: next(ticks),
    )

    manager.loaded("whisper", 500, worker_family="whisper")
    manager.loaded("whisper", 450, worker_family="whisper")

    assert manager.models()[0].last_use_monotonic_ns == 20
    assert manager.workers()[0].last_use_monotonic_ns == 20
    assert manager.metrics.cold_loads == 1
    assert manager.metrics.warm_reuses == 1
    with pytest.raises(SpeechAnalysisError, match="budget"):
        manager.loaded("qwen", 500, worker_family="qwen")
    assert tuple(item.engine for item in manager.models()) == ("whisper",)
    assert tuple(item.family for item in manager.workers()) == ("whisper",)


@pytest.mark.asyncio
async def test_model_major_round_preserves_authority_and_input_order() -> None:
    candidates = (
        CandidateWork(2, "high-risk", high_risk=True),
        CandidateWork(0, "clean"),
        CandidateWork(3, "hard", hard_failure_reason="clipping"),
        CandidateWork(1, "disputed"),
    )
    calls: list[tuple[str, tuple[str, ...]]] = []

    async def recognize(
        engine: str, items: tuple[CandidateWork, ...]
    ) -> tuple[CandidateStageResult, ...]:
        calls.append((engine, tuple(item.candidate_id for item in items)))
        values: list[CandidateStageResult] = []
        for item in items:
            match = not (item.candidate_id == "disputed" and engine == "parakeet")
            values.append(CandidateStageResult(item.candidate_id, engine, match))
        return tuple(values)

    async def force(
        items: tuple[CandidateWork, ...],
    ) -> tuple[CandidateStageResult, ...]:
        calls.append(("qwen-forced", tuple(item.candidate_id for item in items)))
        return tuple(
            CandidateStageResult(item.candidate_id, "qwen-forced", True)
            for item in items
        )

    results = await ModelMajorRepairScheduler(recognize, force).run_round(candidates)

    assert tuple(item.candidate_id for item in results) == (
        "clean",
        "disputed",
        "high-risk",
        "hard",
    )
    assert calls[0] == ("parakeet", ("clean", "disputed", "high-risk"))
    assert calls[1] == ("whisper", ("clean", "disputed", "high-risk"))
    assert calls[2] == ("qwen", ("disputed", "high-risk"))
    # A Parakeet mismatch did not suppress required Whisper analysis.
    assert "whisper" in results[1].stages_run
    assert not results[1].accepted
    assert results[0].accepted
    assert results[2].accepted
    assert results[3].stages_run == ()
    assert all(reason for _stage, reason in results[3].skipped_stages)


@pytest.mark.asyncio
async def test_model_major_scheduler_rejects_malformed_batch_shape() -> None:
    candidates = (CandidateWork(0, "first"), CandidateWork(1, "second"))

    async def recognize(
        engine: str,
        items: tuple[CandidateWork, ...],
    ) -> tuple[CandidateStageResult, ...]:
        return tuple(
            CandidateStageResult(item.candidate_id, engine, True)
            for item in reversed(items)
        )

    async def force(
        items: tuple[CandidateWork, ...],
    ) -> tuple[CandidateStageResult, ...]:
        return tuple(
            CandidateStageResult(item.candidate_id, "qwen-forced", True)
            for item in items
        )

    with pytest.raises(WorkerProtocolError, match="shape or ordering"):
        await ModelMajorRepairScheduler(recognize, force).run_round(candidates)


@pytest.mark.asyncio
async def test_bounded_candidate_search_stops_early_only_for_verified_take() -> None:
    generated_rounds: list[int] = []

    async def generate(round_index: int, limit: int) -> tuple[CandidateWork, ...]:
        generated_rounds.append(round_index)
        assert limit == 2
        offset = (round_index - 1) * limit
        return tuple(
            CandidateWork(offset + item, f"round-{round_index}-{item}")
            for item in range(limit)
        )

    async def recognize(
        engine: str, items: tuple[CandidateWork, ...]
    ) -> tuple[CandidateStageResult, ...]:
        return tuple(
            CandidateStageResult(
                item.candidate_id,
                engine,
                round_number == 2,
            )
            for item in items
            for round_number in (int(item.candidate_id.split("-")[1]),)
        )

    async def force(
        items: tuple[CandidateWork, ...],
    ) -> tuple[CandidateStageResult, ...]:
        return tuple(
            CandidateStageResult(item.candidate_id, "qwen-forced", True)
            for item in items
        )

    result = await BoundedCandidateSearch(
        ModelMajorRepairScheduler(recognize, force),
        candidates_per_round=2,
        maximum_rounds=3,
        generate_round=generate,
        rank=lambda values: values[-1],
    ).run()

    assert generated_rounds == [1, 2]
    assert result.rounds_run == 2
    assert result.generated_candidates == 4
    assert result.winner is not None
    assert result.winner.candidate_id == "round-2-1"
    assert all(not item.accepted for item in result.candidates[:2])


@pytest.mark.asyncio
async def test_bounded_search_exhausts_limits_without_false_accept() -> None:
    generated = 0

    async def generate(round_index: int, limit: int) -> tuple[CandidateWork, ...]:
        nonlocal generated
        offset = (round_index - 1) * limit
        values = tuple(
            CandidateWork(offset + item, f"candidate-{offset + item}")
            for item in range(limit)
        )
        generated += len(values)
        return values

    async def reject(
        engine: str, items: tuple[CandidateWork, ...]
    ) -> tuple[CandidateStageResult, ...]:
        return tuple(
            CandidateStageResult(item.candidate_id, engine, False) for item in items
        )

    async def force(
        items: tuple[CandidateWork, ...],
    ) -> tuple[CandidateStageResult, ...]:
        assert not items
        return ()

    result = await BoundedCandidateSearch(
        ModelMajorRepairScheduler(reject, force),
        candidates_per_round=3,
        maximum_rounds=4,
        generate_round=generate,
        rank=lambda values: values[0],
    ).run()

    assert generated == 12
    assert result.rounds_run == 4
    assert result.generated_candidates == 12
    assert result.winner is None
    assert not any(item.accepted for item in result.candidates)
    assert result.terminal_reason == "candidate_round_limit_exhausted"


@pytest.mark.asyncio
async def test_bounded_candidate_ranker_cannot_admit_rejected_take() -> None:
    accepted = CandidateWork(0, "accepted")
    rejected = CandidateWork(1, "rejected", hard_failure_reason="clipping")

    async def generate(_round_index: int, _limit: int) -> tuple[CandidateWork, ...]:
        return accepted, rejected

    async def recognize(
        engine: str, items: tuple[CandidateWork, ...]
    ) -> tuple[CandidateStageResult, ...]:
        return tuple(
            CandidateStageResult(item.candidate_id, engine, True) for item in items
        )

    async def force(
        items: tuple[CandidateWork, ...],
    ) -> tuple[CandidateStageResult, ...]:
        return tuple(
            CandidateStageResult(item.candidate_id, "qwen-forced", True)
            for item in items
        )

    injected = ScheduledCandidate(
        rejected.index,
        rejected.candidate_id,
        (),
        (),
        False,
        "clipping",
    )
    search = BoundedCandidateSearch(
        ModelMajorRepairScheduler(recognize, force),
        candidates_per_round=2,
        maximum_rounds=1,
        generate_round=generate,
        rank=lambda _values: injected,
    )

    with pytest.raises(WorkerProtocolError, match="outside the accepted cohort"):
        await search.run()
