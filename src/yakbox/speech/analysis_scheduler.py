"""Supervised, model-major scheduling for isolated speech-analysis workers."""

from __future__ import annotations

import asyncio
import platform
import re
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Protocol

from yakbox.errors import SpeechAnalysisError, ValidationError, WorkerProtocolError
from yakbox.speech.accelerator import AcceleratorLease
from yakbox.speech.analysis_fingerprints import semantic_fingerprint

SUPERVISOR_PROTOCOL_VERSION = 2
_MINIMUM_ACCEPTING_VOTES = 2
_MAXIMUM_REASON_CODE_LENGTH = 128
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}")
_SUPERVISED_OPERATIONS = (
    "recognize",
    "recognize_many",
    "force_align",
    "force_align_many",
    "status",
    "unload",
    "shutdown",
)


class OperationTerminalStatus(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    WORKER_CRASHED = "worker_crashed"
    INVALID_RESULT = "invalid_result"


@dataclass(frozen=True, slots=True)
class WorkerHandshake:
    """Exact child identity; the supervisor never negotiates it downward."""

    protocol_version: int
    worker_artifact_fingerprint: str
    environment_lock_fingerprint: str
    adapter_fingerprint: str
    python_fingerprint: str
    execution_fingerprint: str
    capability_fingerprint: str

    def __post_init__(self) -> None:
        if self.protocol_version < 1:
            raise WorkerProtocolError("Analysis worker handshake protocol is invalid")
        if any(
            _SHA256.fullmatch(value) is None
            for value in (
                self.worker_artifact_fingerprint,
                self.environment_lock_fingerprint,
                self.adapter_fingerprint,
                self.python_fingerprint,
                self.execution_fingerprint,
                self.capability_fingerprint,
            )
        ):
            raise WorkerProtocolError("Analysis worker handshake identity is invalid")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("analysis-worker-handshake-v1", self)


@dataclass(frozen=True, slots=True)
class PlannedWorkerIdentity:
    worker_artifact_fingerprint: str
    environment_lock_fingerprint: str
    adapter_fingerprint: str
    python_fingerprint: str
    execution_fingerprint: str
    capability_fingerprint: str
    protocol_version: int = SUPERVISOR_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        WorkerHandshake(
            self.protocol_version,
            self.worker_artifact_fingerprint,
            self.environment_lock_fingerprint,
            self.adapter_fingerprint,
            self.python_fingerprint,
            self.execution_fingerprint,
            self.capability_fingerprint,
        )

    def validate(self, actual: WorkerHandshake) -> None:
        expected = WorkerHandshake(
            self.protocol_version,
            self.worker_artifact_fingerprint,
            self.environment_lock_fingerprint,
            self.adapter_fingerprint,
            self.python_fingerprint,
            self.execution_fingerprint,
            self.capability_fingerprint,
        )
        if actual != expected:
            raise WorkerProtocolError(
                "Analysis worker handshake differs from the planned runtime"
            )


def build_worker_handshake(
    *,
    family: str,
    engines: tuple[str, ...],
    worker_artifact_fingerprint: str,
    environment_lock_fingerprint: str,
    adapter_fingerprint: str,
) -> WorkerHandshake:
    """Derive supervisor and child identity from the same reviewed inputs."""
    python_fingerprint = semantic_fingerprint(
        "analysis-python-contract-v1",
        {"python": platform.python_version()},
    )
    execution_fingerprint = semantic_fingerprint(
        "analysis-worker-execution-contract-v1",
        {
            "family": family,
            "worker_artifact": worker_artifact_fingerprint,
            "environment_lock": environment_lock_fingerprint,
            "adapter": adapter_fingerprint,
        },
    )
    capability_fingerprint = semantic_fingerprint(
        "analysis-worker-capabilities-v1",
        {
            "family": family,
            "engines": engines,
            "languages": ("en",),
            "operations": _SUPERVISED_OPERATIONS,
        },
    )
    return WorkerHandshake(
        SUPERVISOR_PROTOCOL_VERSION,
        worker_artifact_fingerprint,
        environment_lock_fingerprint,
        adapter_fingerprint,
        python_fingerprint,
        execution_fingerprint,
        capability_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class BatchWorkItem:
    """One idempotent item in stable user-visible input order."""

    index: int
    request_fingerprint: str
    relative_audio_path: str
    start_frame: int
    end_frame: int
    sample_rate: int = 16_000

    def __post_init__(self) -> None:
        if (
            self.index < 0
            or _SHA256.fullmatch(self.request_fingerprint) is None
            or not _safe_relative_path(self.relative_audio_path)
            or self.start_frame < 0
            or self.end_frame <= self.start_frame
            or self.sample_rate <= 0
        ):
            raise ValidationError("Analysis batch item is invalid")


@dataclass(frozen=True, slots=True)
class BatchOperation:
    """One bounded worker operation using supervisor-relative time."""

    operation_id: str
    batch_id: str
    engine: str
    engine_fingerprint: str
    operation: str
    deadline_monotonic_ns: int
    items: tuple[BatchWorkItem, ...]

    def __post_init__(self) -> None:
        if (
            _IDENTIFIER.fullmatch(self.operation_id) is None
            or _IDENTIFIER.fullmatch(self.batch_id) is None
            or not self.engine
            or _SHA256.fullmatch(self.engine_fingerprint) is None
            or self.operation not in {"recognize_many", "force_align_many"}
            or self.deadline_monotonic_ns <= 0
            or not self.items
        ):
            raise ValidationError("Analysis batch operation is incomplete")
        indices = tuple(item.index for item in self.items)
        if indices != tuple(sorted(set(indices))):
            raise ValidationError("Analysis batch indices must be unique and ordered")

    def remaining_seconds(self, *, now_ns: int | None = None) -> float:
        now = time.monotonic_ns() if now_ns is None else now_ns
        return max(0.0, (self.deadline_monotonic_ns - now) / 1_000_000_000)


def _safe_relative_path(value: str) -> bool:
    if not value or "\\" in value or value in {"", "."}:
        return False
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return (
        not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and ".." not in posix.parts
    )


@dataclass(frozen=True, slots=True)
class OperationStageTimings:
    """Privacy-safe stage timing attached to one worker item."""

    preparing_audio_ns: int = 0
    inference_ns: int = 0
    normalizing_tokens_ns: int = 0
    consensus_ns: int = 0
    forced_alignment_ns: int = 0
    writing_evidence_ns: int = 0

    def __post_init__(self) -> None:
        if (
            min(
                self.preparing_audio_ns,
                self.inference_ns,
                self.normalizing_tokens_ns,
                self.consensus_ns,
                self.forced_alignment_ns,
                self.writing_evidence_ns,
            )
            < 0
        ):
            raise ValidationError("Analysis stage timings cannot be negative")


@dataclass(frozen=True, slots=True)
class OperationMetrics:
    requested_audio_frames: int
    analyzed_audio_frames: int
    wall_time_ns: int
    peak_resident_memory_bytes: int
    peak_accelerator_memory_bytes: int | None
    cache_hit: bool
    model_loads: int
    cache_miss_reason: str | None = None
    stage_timings: OperationStageTimings = field(default_factory=OperationStageTimings)

    def __post_init__(self) -> None:
        values = (
            self.requested_audio_frames,
            self.analyzed_audio_frames,
            self.wall_time_ns,
            self.peak_resident_memory_bytes,
            self.model_loads,
        )
        if min(values) < 0 or (
            self.peak_accelerator_memory_bytes is not None
            and self.peak_accelerator_memory_bytes < 0
        ):
            raise ValidationError("Analysis operation metrics cannot be negative")
        if self.cache_hit and self.cache_miss_reason is not None:
            raise ValidationError("A cache hit cannot carry a miss reason")
        if self.cache_miss_reason is not None and (
            not self.cache_miss_reason
            or len(self.cache_miss_reason) > _MAXIMUM_REASON_CODE_LENGTH
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.cache_miss_reason) is None
        ):
            raise ValidationError("Analysis cache miss reason is invalid")


@dataclass(frozen=True, slots=True)
class BatchCompletion:
    """Item-framed completion. Duplicate identical frames are idempotent."""

    batch_id: str
    index: int
    request_fingerprint: str
    terminal_status: OperationTerminalStatus
    evidence_fingerprint: str | None
    reason_code: str | None
    metrics: OperationMetrics

    @property
    def semantic_fingerprint(self) -> str:
        # Timing, memory, and cache state are observability—not decision identity.
        return semantic_fingerprint(
            "analysis-batch-completion-v1",
            {
                "request_fingerprint": self.request_fingerprint,
                "terminal_status": self.terminal_status,
                "evidence_fingerprint": self.evidence_fingerprint,
                "reason_code": self.reason_code,
            },
        )


class SupervisedWorker(Protocol):
    async def handshake(self) -> WorkerHandshake: ...

    def execute(self, operation: BatchOperation) -> AsyncIterator[BatchCompletion]: ...

    async def soft_cancel(self, operation_id: str) -> None: ...

    async def wait_idle(self) -> None: ...

    async def terminate(self) -> None: ...

    async def unload(self, engine: str) -> MemoryWatermark: ...

    async def current_memory(self) -> MemoryWatermark: ...


@dataclass(frozen=True, slots=True)
class SupervisedBatchResult:
    operation_id: str
    completions: tuple[BatchCompletion, ...]
    worker_restarts: int
    soft_cancellations: int
    forced_terminations: int


@dataclass(slots=True)
class _MutableStagePerformance:
    cold_loads: int = 0
    warm_reuses: int = 0
    unloads: int = 0
    evictions: int = 0
    restarts: int = 0
    soft_cancellations: int = 0
    forced_terminations: int = 0
    requested_audio_seconds: float = 0.0
    analyzed_audio_seconds: float = 0.0
    wall_seconds: float = 0.0
    batches: int = 0
    batch_items: int = 0
    maximum_batch_size: int = 0
    survivor_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    miss_reasons: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    peak_resident_memory_bytes: int = 0
    peak_accelerator_memory_bytes: int | None = None
    preparing_audio_seconds: float = 0.0
    inference_seconds: float = 0.0
    normalizing_tokens_seconds: float = 0.0
    consensus_seconds: float = 0.0
    forced_alignment_seconds: float = 0.0
    writing_evidence_seconds: float = 0.0


class AnalysisPerformanceCollector:
    """Aggregate privacy-safe per-engine and per-stage operational metrics."""

    def __init__(self) -> None:
        self._stages: dict[tuple[str, str], _MutableStagePerformance] = {}

    def record_batch(
        self,
        operation: BatchOperation,
        result: SupervisedBatchResult,
    ) -> None:
        stage = self._stage(operation.engine, operation.operation)
        sample_rates = {item.index: item.sample_rate for item in operation.items}
        stage.batches += 1
        stage.batch_items += len(operation.items)
        stage.maximum_batch_size = max(stage.maximum_batch_size, len(operation.items))
        stage.restarts += result.worker_restarts
        stage.soft_cancellations += result.soft_cancellations
        stage.forced_terminations += result.forced_terminations
        for completion in result.completions:
            metrics = completion.metrics
            sample_rate = sample_rates[completion.index]
            stage.requested_audio_seconds += (
                metrics.requested_audio_frames / sample_rate
            )
            stage.analyzed_audio_seconds += metrics.analyzed_audio_frames / sample_rate
            stage.wall_seconds += metrics.wall_time_ns / 1_000_000_000
            stage.cold_loads += metrics.model_loads
            if (
                completion.terminal_status is OperationTerminalStatus.COMPLETED
                and metrics.model_loads == 0
                and not metrics.cache_hit
            ):
                stage.warm_reuses += 1
            stage.survivor_count += (
                completion.terminal_status is OperationTerminalStatus.COMPLETED
            )
            self._record_cache(stage, metrics)
            self._record_memory(stage, metrics)
            self._record_timings(stage, metrics.stage_timings)

    def record_lifecycle(
        self,
        *,
        engine: str,
        stage: str,
        unloads: int = 0,
        evictions: int = 0,
        restarts: int = 0,
    ) -> None:
        if min(unloads, evictions, restarts) < 0:
            raise ValidationError("Analysis lifecycle metric cannot be negative")
        target = self._stage(engine, stage)
        target.unloads += unloads
        target.evictions += evictions
        target.restarts += restarts

    def to_dict(self) -> dict[str, object]:
        by_engine: dict[str, dict[str, dict[str, object]]] = {}
        for engine, stage_name in sorted(self._stages):
            by_engine.setdefault(engine, {})[stage_name] = self._stage_dict(
                self._stages[(engine, stage_name)]
            )
        return {
            "engines": {
                engine: {"stages": stages}
                for engine, stages in sorted(by_engine.items())
            }
        }

    def _stage(self, engine: str, stage: str) -> _MutableStagePerformance:
        if (
            re.fullmatch(r"[a-z0-9][a-z0-9_-]*", engine) is None
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", stage) is None
        ):
            raise ValidationError("Analysis performance identity is invalid")
        return self._stages.setdefault((engine, stage), _MutableStagePerformance())

    @staticmethod
    def _record_cache(
        stage: _MutableStagePerformance, metrics: OperationMetrics
    ) -> None:
        if metrics.cache_hit:
            stage.cache_hits += 1
            return
        stage.cache_misses += 1
        if metrics.cache_miss_reason is not None:
            stage.miss_reasons[metrics.cache_miss_reason] += 1

    @staticmethod
    def _record_memory(
        stage: _MutableStagePerformance, metrics: OperationMetrics
    ) -> None:
        stage.peak_resident_memory_bytes = max(
            stage.peak_resident_memory_bytes,
            metrics.peak_resident_memory_bytes,
        )
        if metrics.peak_accelerator_memory_bytes is not None:
            stage.peak_accelerator_memory_bytes = max(
                stage.peak_accelerator_memory_bytes or 0,
                metrics.peak_accelerator_memory_bytes,
            )

    @staticmethod
    def _record_timings(
        stage: _MutableStagePerformance, timings: OperationStageTimings
    ) -> None:
        divisor = 1_000_000_000
        stage.preparing_audio_seconds += timings.preparing_audio_ns / divisor
        stage.inference_seconds += timings.inference_ns / divisor
        stage.normalizing_tokens_seconds += timings.normalizing_tokens_ns / divisor
        stage.consensus_seconds += timings.consensus_ns / divisor
        stage.forced_alignment_seconds += timings.forced_alignment_ns / divisor
        stage.writing_evidence_seconds += timings.writing_evidence_ns / divisor

    @staticmethod
    def _stage_dict(stage: _MutableStagePerformance) -> dict[str, object]:
        real_time_factor = (
            stage.wall_seconds / stage.analyzed_audio_seconds
            if stage.analyzed_audio_seconds > 0
            else 0.0
        )
        return {
            "cold_loads": stage.cold_loads,
            "warm_reuses": stage.warm_reuses,
            "unloads": stage.unloads,
            "evictions": stage.evictions,
            "restarts": stage.restarts,
            "soft_cancellations": stage.soft_cancellations,
            "forced_terminations": stage.forced_terminations,
            "requested_audio_seconds": stage.requested_audio_seconds,
            "analyzed_audio_seconds": stage.analyzed_audio_seconds,
            "wall_seconds": stage.wall_seconds,
            "real_time_factor": real_time_factor,
            "batches": stage.batches,
            "batch_items": stage.batch_items,
            "maximum_batch_size": stage.maximum_batch_size,
            "survivor_count": stage.survivor_count,
            "cache_hits": stage.cache_hits,
            "cache_misses": stage.cache_misses,
            "cache_miss_reasons": dict(sorted(stage.miss_reasons.items())),
            "peak_resident_memory_bytes": stage.peak_resident_memory_bytes,
            "peak_accelerator_memory_bytes": stage.peak_accelerator_memory_bytes,
            "stage_seconds": {
                "preparing_audio": stage.preparing_audio_seconds,
                "inference": stage.inference_seconds,
                "normalizing_tokens": stage.normalizing_tokens_seconds,
                "consensus": stage.consensus_seconds,
                "forced_alignment": stage.forced_alignment_seconds,
                "writing_evidence": stage.writing_evidence_seconds,
            },
        }


@dataclass(slots=True)
class _CompletionAccumulator:
    operation: BatchOperation
    completed: dict[int, BatchCompletion] = field(default_factory=dict)

    def add(self, completion: BatchCompletion) -> bool:
        expected = {item.index: item for item in self.operation.items}.get(
            completion.index
        )
        if (
            completion.batch_id != self.operation.batch_id
            or expected is None
            or completion.request_fingerprint != expected.request_fingerprint
        ):
            raise WorkerProtocolError("Analysis batch completion identity is invalid")
        prior = self.completed.get(completion.index)
        if prior is None:
            self.completed[completion.index] = completion
            return True
        if prior.semantic_fingerprint != completion.semantic_fingerprint:
            raise WorkerProtocolError(
                "Analysis batch completion conflicts with prior evidence"
            )
        return False

    def ordered(self) -> tuple[BatchCompletion, ...]:
        return tuple(self.completed[index] for index in sorted(self.completed))


class AnalysisSupervisor:
    """Validate workers, commit item results, and recover only safe work."""

    def __init__(
        self,
        *,
        worker_factory: Callable[[], Awaitable[SupervisedWorker]],
        planned_identity: PlannedWorkerIdentity,
        accelerator_lease: AcceleratorLease,
        commit: Callable[[BatchCompletion], Awaitable[None]],
        performance: AnalysisPerformanceCollector | None = None,
        cancellation_grace_seconds: float = 0.25,
        maximum_restarts: int = 1,
    ) -> None:
        if cancellation_grace_seconds < 0 or maximum_restarts < 0:
            raise ValidationError("Analysis supervisor lifecycle limits are invalid")
        self._worker_factory = worker_factory
        self._planned_identity = planned_identity
        self._accelerator_lease = accelerator_lease
        self._commit = commit
        self._performance = performance
        self._grace = cancellation_grace_seconds
        self._maximum_restarts = maximum_restarts

    async def execute(self, operation: BatchOperation) -> SupervisedBatchResult:
        accumulator = _CompletionAccumulator(operation)
        restarts = soft_cancellations = forced_terminations = 0
        worker = await self._verified_worker()
        while True:
            remaining_items = _remaining_items(operation, accumulator)
            if not remaining_items:
                break
            current = _remaining_operation(operation, remaining_items)
            try:
                await self._execute_attempt(worker, current, accumulator)
                break
            except asyncio.CancelledError:
                cleanup = asyncio.create_task(self._cancel_timed_out(worker, operation))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                raise
            except TimeoutError:
                soft_cancellations += 1
                forced_terminations += await self._cancel_timed_out(worker, operation)
                break
            except EOFError, BrokenPipeError:
                await worker.terminate()
                if restarts >= self._maximum_restarts:
                    break
                restarts += 1
                worker = await self._verified_worker()
        terminal = (
            OperationTerminalStatus.TIMED_OUT
            if soft_cancellations
            else OperationTerminalStatus.WORKER_CRASHED
        )
        result = SupervisedBatchResult(
            operation.operation_id,
            _terminal_completions(operation, accumulator, terminal),
            restarts,
            soft_cancellations,
            forced_terminations,
        )
        if self._performance is not None:
            self._performance.record_batch(operation, result)
        return result

    async def _execute_attempt(
        self,
        worker: SupervisedWorker,
        operation: BatchOperation,
        accumulator: _CompletionAccumulator,
    ) -> None:
        remaining = operation.remaining_seconds()
        if remaining <= 0:
            raise TimeoutError
        async with self._accelerator_lease.hold(f"analysis:{operation.engine}"):
            async with asyncio.timeout(remaining):
                async for completion in worker.execute(operation):
                    if operation.remaining_seconds() <= 0:
                        raise TimeoutError
                    if (
                        accumulator.add(completion)
                        and completion.terminal_status
                        is OperationTerminalStatus.COMPLETED
                    ):
                        await self._commit(completion)

    async def _cancel_timed_out(
        self, worker: SupervisedWorker, operation: BatchOperation
    ) -> int:
        await worker.soft_cancel(operation.operation_id)
        try:
            await asyncio.wait_for(worker.wait_idle(), timeout=self._grace)
        except TimeoutError:
            await worker.terminate()
            return 1
        return 0

    async def _verified_worker(self) -> SupervisedWorker:
        worker = await self._worker_factory()
        try:
            self._planned_identity.validate(await worker.handshake())
        except Exception:
            await worker.terminate()
            raise
        return worker


def _remaining_items(
    operation: BatchOperation, accumulator: _CompletionAccumulator
) -> tuple[BatchWorkItem, ...]:
    return tuple(
        item for item in operation.items if item.index not in accumulator.completed
    )


def _remaining_operation(
    operation: BatchOperation, items: tuple[BatchWorkItem, ...]
) -> BatchOperation:
    return BatchOperation(
        operation.operation_id,
        operation.batch_id,
        operation.engine,
        operation.engine_fingerprint,
        operation.operation,
        operation.deadline_monotonic_ns,
        items,
    )


def _terminal_completions(
    operation: BatchOperation,
    accumulator: _CompletionAccumulator,
    terminal: OperationTerminalStatus,
) -> tuple[BatchCompletion, ...]:
    completions = list(accumulator.ordered())
    present = {item.index for item in completions}
    completions.extend(
        BatchCompletion(
            operation.batch_id,
            item.index,
            item.request_fingerprint,
            terminal,
            None,
            terminal.value,
            OperationMetrics(
                item.end_frame - item.start_frame,
                0,
                0,
                0,
                None,
                False,
                0,
            ),
        )
        for item in operation.items
        if item.index not in present
    )
    return tuple(sorted(completions, key=lambda item: item.index))


@dataclass(frozen=True, slots=True)
class MemoryWatermark:
    resident_bytes: int
    accelerator_bytes: int | None


@dataclass(slots=True)
class LifecycleMetrics:
    cold_loads: int = 0
    warm_reuses: int = 0
    unloads: int = 0
    evictions: int = 0
    restarts: int = 0


@dataclass(frozen=True, slots=True)
class ResidentModelState:
    engine: str
    worker_family: str
    peak_memory_bytes: int
    last_use_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class ResidentWorkerState:
    family: str
    last_use_monotonic_ns: int
    engines: tuple[str, ...]


class ResidentModelManager:
    """Treat unload as advisory and recycle above calibrated watermarks."""

    def __init__(
        self,
        *,
        maximum_resident_models: int,
        maximum_resident_memory_bytes: int,
        post_unload_resident_watermark_bytes: int,
        post_unload_accelerator_watermark_bytes: int,
        maximum_resident_workers: int = 3,
        cleanup_grace_seconds: float = 0.0,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        limits = (
            maximum_resident_workers,
            maximum_resident_models,
            maximum_resident_memory_bytes,
            post_unload_resident_watermark_bytes,
            post_unload_accelerator_watermark_bytes,
        )
        if min(limits) < 1 or cleanup_grace_seconds < 0:
            raise ValidationError("Resident-model policy limits must be positive")
        self.maximum_resident_workers = maximum_resident_workers
        self.maximum_resident_models = maximum_resident_models
        self.maximum_resident_memory_bytes = maximum_resident_memory_bytes
        self.resident_watermark = post_unload_resident_watermark_bytes
        self.accelerator_watermark = post_unload_accelerator_watermark_bytes
        self.cleanup_grace_seconds = cleanup_grace_seconds
        self._clock = clock
        self.metrics = LifecycleMetrics()
        self._resident: dict[str, ResidentModelState] = {}
        self._worker_last_use: dict[str, int] = {}

    def loaded(
        self,
        engine: str,
        peak_bytes: int,
        *,
        worker_family: str = "default",
    ) -> None:
        if not engine or not worker_family or peak_bytes < 0:
            raise ValidationError("Resident model identity or memory is invalid")
        now = self._clock()
        existing = self._resident.get(engine)
        if existing is not None and existing.worker_family != worker_family:
            raise ValidationError("A resident engine cannot change worker family")
        proposed = dict(self._resident)
        proposed[engine] = ResidentModelState(
            engine,
            worker_family,
            peak_bytes,
            now,
        )
        proposed_workers = {**self._worker_last_use, worker_family: now}
        if (
            len(proposed_workers) > self.maximum_resident_workers
            or len(proposed) > self.maximum_resident_models
            or sum(item.peak_memory_bytes for item in proposed.values())
            > self.maximum_resident_memory_bytes
        ):
            raise SpeechAnalysisError("Resident analysis model budget was exceeded")
        self._resident = proposed
        self._worker_last_use = proposed_workers
        if existing is not None:
            self.metrics.warm_reuses += 1
        else:
            self.metrics.cold_loads += 1

    def models(self) -> tuple[ResidentModelState, ...]:
        return tuple(self._resident[key] for key in sorted(self._resident))

    def workers(self) -> tuple[ResidentWorkerState, ...]:
        return tuple(
            ResidentWorkerState(
                family,
                last_use,
                tuple(
                    sorted(
                        item.engine
                        for item in self._resident.values()
                        if item.worker_family == family
                    )
                ),
            )
            for family, last_use in sorted(self._worker_last_use.items())
        )

    async def evict(
        self,
        worker: SupervisedWorker,
        engine: str,
        *,
        worker_family: str = "default",
    ) -> bool:
        try:
            await worker.unload(engine)
            self.metrics.unloads += 1
            if self.cleanup_grace_seconds:
                await asyncio.sleep(self.cleanup_grace_seconds)
            watermark = await worker.current_memory()
        except asyncio.CancelledError:
            await asyncio.shield(self._recycle(worker, worker_family))
            raise
        except Exception:
            await self._recycle(worker, worker_family)
            raise
        self._resident.pop(engine, None)
        accelerator = watermark.accelerator_bytes or 0
        if (
            watermark.resident_bytes > self.resident_watermark
            or accelerator > self.accelerator_watermark
        ):
            await self._recycle(worker, worker_family)
            return True
        self.metrics.evictions += 1
        return False

    async def _recycle(
        self,
        worker: SupervisedWorker,
        worker_family: str,
    ) -> None:
        await worker.terminate()
        self.metrics.restarts += 1
        self._drop_worker(worker_family)

    def _drop_worker(self, family: str) -> None:
        self._worker_last_use.pop(family, None)
        self._resident = {
            engine: item
            for engine, item in self._resident.items()
            if item.worker_family != family
        }


@dataclass(frozen=True, slots=True)
class CandidateWork:
    index: int
    candidate_id: str
    hard_failure_reason: str | None = None
    high_risk: bool = False


@dataclass(frozen=True, slots=True)
class CandidateStageResult:
    candidate_id: str
    engine: str
    lexical_match: bool
    valid: bool = True
    terminal_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ScheduledCandidate:
    index: int
    candidate_id: str
    stages_run: tuple[str, ...]
    skipped_stages: tuple[tuple[str, str], ...]
    accepted: bool
    terminal_reason: str


@dataclass(frozen=True, slots=True)
class CandidateSearchResult:
    """Terminal result of one strictly bounded candidate search."""

    rounds_run: int
    generated_candidates: int
    candidates: tuple[ScheduledCandidate, ...]
    winner: ScheduledCandidate | None
    terminal_reason: str


class ModelMajorRepairScheduler:
    """Run bounded repair rounds by model without weakening the truth table."""

    def __init__(
        self,
        recognize_many: Callable[
            [str, tuple[CandidateWork, ...]],
            Awaitable[tuple[CandidateStageResult, ...]],
        ],
        force_align_many: Callable[
            [tuple[CandidateWork, ...]],
            Awaitable[tuple[CandidateStageResult, ...]],
        ],
    ) -> None:
        self._recognize_many = recognize_many
        self._force_align_many = force_align_many

    async def run_round(
        self, candidates: Sequence[CandidateWork]
    ) -> tuple[ScheduledCandidate, ...]:
        ordered = tuple(sorted(candidates, key=lambda item: item.index))
        if tuple(item.index for item in ordered) != tuple(
            sorted({item.index for item in ordered})
        ):
            raise ValidationError("Repair candidate indices must be unique")
        hard_failed = {
            item.candidate_id: item.hard_failure_reason
            for item in ordered
            if item.hard_failure_reason is not None
        }
        survivors = tuple(
            item for item in ordered if item.candidate_id not in hard_failed
        )
        stage_results: dict[str, dict[str, CandidateStageResult]] = {}
        parakeet = await self._stage("parakeet", survivors)
        stage_results["parakeet"] = parakeet
        # Parakeet never filters the same baseline cohort required by Whisper.
        whisper = await self._stage("whisper", survivors)
        stage_results["whisper"] = whisper
        qwen_items = tuple(
            item
            for item in survivors
            if item.high_risk
            or not parakeet[item.candidate_id].valid
            or not whisper[item.candidate_id].valid
            or (
                parakeet[item.candidate_id].lexical_match
                != whisper[item.candidate_id].lexical_match
            )
            or not parakeet[item.candidate_id].lexical_match
        )
        qwen = await self._stage("qwen", qwen_items)
        stage_results["qwen"] = qwen
        lexical_accepted = tuple(
            item
            for item in survivors
            if _lexically_accepted(item, parakeet, whisper, qwen)
        )
        forced_values = await self._force_align_many(lexical_accepted)
        forced = _indexed_results("qwen-forced", lexical_accepted, forced_values)
        stage_results["qwen-forced"] = forced
        results: list[ScheduledCandidate] = []
        for item in ordered:
            if item.candidate_id in hard_failed:
                reason = hard_failed[item.candidate_id]
                if reason is None:
                    raise AssertionError("hard-failure reason disappeared")
                results.append(
                    ScheduledCandidate(
                        item.index,
                        item.candidate_id,
                        (),
                        tuple(
                            (stage, "engine_independent_hard_failure")
                            for stage in ("parakeet", "whisper", "qwen", "qwen-forced")
                        ),
                        False,
                        reason,
                    )
                )
                continue
            stages = tuple(
                stage
                for stage in ("parakeet", "whisper", "qwen", "qwen-forced")
                if item.candidate_id in stage_results[stage]
            )
            skipped = tuple(
                (stage, _skip_reason(stage, item, lexical_accepted, qwen_items))
                for stage in ("qwen", "qwen-forced")
                if item.candidate_id not in stage_results[stage]
            )
            forced_result = forced.get(item.candidate_id)
            accepted = (
                forced_result is not None
                and forced_result.valid
                and forced_result.lexical_match
            )
            results.append(
                ScheduledCandidate(
                    item.index,
                    item.candidate_id,
                    stages,
                    skipped,
                    accepted,
                    "accepted_all_required_gates"
                    if accepted
                    else "lexical_or_forced_gate_rejected",
                )
            )
        return tuple(results)

    async def _stage(
        self, engine: str, items: tuple[CandidateWork, ...]
    ) -> dict[str, CandidateStageResult]:
        if not items:
            return {}
        return _indexed_results(
            engine,
            items,
            await self._recognize_many(engine, items),
        )


class BoundedCandidateSearch:
    """Generate, verify, and rank candidates within immutable round limits."""

    def __init__(
        self,
        scheduler: ModelMajorRepairScheduler,
        *,
        candidates_per_round: int,
        maximum_rounds: int,
        generate_round: Callable[[int, int], Awaitable[tuple[CandidateWork, ...]]],
        rank: Callable[[tuple[ScheduledCandidate, ...]], ScheduledCandidate],
    ) -> None:
        if candidates_per_round < 1 or maximum_rounds < 1:
            raise ValidationError("Candidate search limits must be positive")
        self._scheduler = scheduler
        self._candidates_per_round = candidates_per_round
        self._maximum_rounds = maximum_rounds
        self._generate_round = generate_round
        self._rank = rank

    async def run(self) -> CandidateSearchResult:
        """Stop at the first round with a verified winner or at the fixed limit."""
        scheduled: list[ScheduledCandidate] = []
        seen_ids: set[str] = set()
        seen_indices: set[int] = set()
        rounds_run = 0
        for round_index in range(1, self._maximum_rounds + 1):
            candidates = await self._generate_round(
                round_index, self._candidates_per_round
            )
            if len(candidates) > self._candidates_per_round:
                raise ValidationError("Candidate generator exceeded its round limit")
            if not candidates:
                break
            identifiers = {item.candidate_id for item in candidates}
            indices = {item.index for item in candidates}
            if (
                len(identifiers) != len(candidates)
                or len(indices) != len(candidates)
                or identifiers & seen_ids
                or indices & seen_indices
            ):
                raise ValidationError(
                    "Candidate identities must be unique across search rounds"
                )
            seen_ids.update(identifiers)
            seen_indices.update(indices)
            rounds_run = round_index
            round_results = await self._scheduler.run_round(candidates)
            scheduled.extend(round_results)
            accepted = tuple(item for item in round_results if item.accepted)
            if not accepted:
                continue
            winner = self._rank(accepted)
            if winner not in accepted or not winner.accepted:
                raise WorkerProtocolError(
                    "Candidate ranker selected evidence outside the accepted cohort"
                )
            return CandidateSearchResult(
                rounds_run,
                len(scheduled),
                tuple(scheduled),
                winner,
                "accepted_verified_candidate",
            )
        return CandidateSearchResult(
            rounds_run,
            len(scheduled),
            tuple(scheduled),
            None,
            "candidate_round_limit_exhausted",
        )


def _indexed_results(
    engine: str,
    requested: tuple[CandidateWork, ...],
    values: tuple[CandidateStageResult, ...],
) -> dict[str, CandidateStageResult]:
    expected = tuple(item.candidate_id for item in requested)
    actual = tuple(item.candidate_id for item in values)
    if actual != expected or any(item.engine != engine for item in values):
        raise WorkerProtocolError(
            f"{engine} batch result shape or ordering differs from its request"
        )
    return {item.candidate_id: item for item in values}


def _lexically_accepted(
    item: CandidateWork,
    parakeet: Mapping[str, CandidateStageResult],
    whisper: Mapping[str, CandidateStageResult],
    qwen: Mapping[str, CandidateStageResult],
) -> bool:
    baseline = (parakeet[item.candidate_id], whisper[item.candidate_id])
    if any(not value.valid for value in baseline):
        return False
    qwen_value = qwen.get(item.candidate_id)
    if qwen_value is None:
        return all(value.lexical_match for value in baseline)
    if not qwen_value.valid:
        return False
    votes = (*baseline, qwen_value)
    matches = sum(value.lexical_match for value in votes)
    valid_dissent = any(value.valid and not value.lexical_match for value in votes)
    return matches >= _MINIMUM_ACCEPTING_VOTES and not valid_dissent


def _skip_reason(
    stage: str,
    item: CandidateWork,
    lexical_accepted: tuple[CandidateWork, ...],
    qwen_items: tuple[CandidateWork, ...],
) -> str:
    if stage == "qwen":
        return "baseline_agreement_not_high_risk"
    if item in qwen_items or item not in lexical_accepted:
        return "lexical_consensus_not_accepted"
    return "terminal_policy_decision"


__all__ = [
    "SUPERVISOR_PROTOCOL_VERSION",
    "AcceleratorLease",
    "AnalysisPerformanceCollector",
    "AnalysisSupervisor",
    "BatchCompletion",
    "BatchOperation",
    "BatchWorkItem",
    "BoundedCandidateSearch",
    "CandidateSearchResult",
    "CandidateStageResult",
    "CandidateWork",
    "LifecycleMetrics",
    "MemoryWatermark",
    "ModelMajorRepairScheduler",
    "OperationMetrics",
    "OperationStageTimings",
    "OperationTerminalStatus",
    "PlannedWorkerIdentity",
    "ResidentModelManager",
    "ResidentModelState",
    "ResidentWorkerState",
    "ScheduledCandidate",
    "SupervisedBatchResult",
    "SupervisedWorker",
    "WorkerHandshake",
    "build_worker_handshake",
]
