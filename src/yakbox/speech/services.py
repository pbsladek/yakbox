from __future__ import annotations

import io
import math
import struct
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Protocol, runtime_checkable

from yakbox._files import atomic_write_bytes, sha256_file
from yakbox.errors import BackendUnavailableError
from yakbox.speech.accelerator import AcceleratorLease
from yakbox.speech.capabilities import BackendCapabilities
from yakbox.speech.models import (
    AudioFormat,
    HostedUsageBudget,
    HostedUsageSnapshot,
    SpeechArtifact,
    SpeechBackendOptions,
    SpeechSynthesisRequest,
    SpeechTransformationRequest,
)


@runtime_checkable
class TextToSpeechService(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities:
        """Describe synthesis, streaming, format, and hosted capabilities."""
        ...

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        """Synthesize one request and atomically commit its destination."""
        ...


@runtime_checkable
class BatchTextToSpeechService(Protocol):
    async def synthesize_many_to_files(
        self,
        requests: tuple[tuple[SpeechSynthesisRequest, Path], ...],
        *,
        overwrite: bool = False,
    ) -> tuple[SpeechArtifact, ...]:
        """Synthesize ordered request/destination pairs as one batch."""
        ...


@runtime_checkable
class SpeechTransformationService(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities:
        """Describe the backend's transformation capabilities."""
        ...

    async def transform_to_file(
        self,
        request: SpeechTransformationRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        """Transform one input and atomically commit its destination."""
        ...


@runtime_checkable
class HostedUsageReportingService(Protocol):
    async def usage_snapshot(self) -> HostedUsageSnapshot | None:
        """Return current hosted usage when the service tracks it."""
        ...


type HostedUsageRecorder = Callable[
    [HostedUsageSnapshot, int],
    Awaitable[None],
]


@runtime_checkable
class HostedUsageJournalingService(Protocol):
    def set_usage_recorder(self, recorder: HostedUsageRecorder | None) -> None:
        """Install a durable callback invoked before provider sends."""
        ...

    async def restore_usage(self, snapshot: HostedUsageSnapshot) -> None:
        """Restore durable counters before resumed provider work."""
        ...


class FakeSpeechService:
    """Deterministic backend for planning, tests, and zero-cost examples."""

    capabilities = BackendCapabilities(
        name="fake",
        synthesis=True,
        transformation=True,
        streaming=False,
        hosted=False,
        output_formats=("wav",),
        supports_reference_voice=True,
    )

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        """Generate deterministic WAV audio for an accepted request."""
        if request.output_format is not AudioFormat.WAV:
            raise BackendUnavailableError("The fake backend produces WAV only")
        rate = request.sample_rate or 16_000
        seconds = max(0.05, min(0.5, len(request.text) / 200.0))
        audio = _tone_wav(request.text, rate=rate, seconds=seconds)
        atomic_write_bytes(destination, audio, overwrite=overwrite)
        return SpeechArtifact(
            path=destination.resolve(),
            backend="fake",
            voice=request.voice,
            output_format=AudioFormat.WAV,
            bytes_written=len(audio),
            sha256=sha256_file(destination),
            duration_seconds=seconds,
            sample_rate=rate,
        )

    async def transform_to_file(
        self,
        request: SpeechTransformationRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        """Copy an input deterministically as a fake transformation."""
        audio = request.input_path.read_bytes()
        atomic_write_bytes(destination, audio, overwrite=overwrite)
        return SpeechArtifact(
            path=destination.resolve(),
            backend="fake",
            voice=request.voice,
            output_format=AudioFormat.WAV,
            bytes_written=len(audio),
            sha256=sha256_file(destination),
        )


def _tone_wav(seed: str, *, rate: int, seconds: float) -> bytes:
    frequency = 180 + (sum(seed.encode()) % 120)
    frames = int(rate * seconds)
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        samples = bytearray()
        for index in range(frames):
            value = int(1000 * math.sin(2 * math.pi * frequency * index / rate))
            samples.extend(struct.pack("<h", value))
        writer.writeframes(bytes(samples))
    return output.getvalue()


@asynccontextmanager
async def open_speech_backend(
    name: str,
    *,
    api_key: str | None = None,
    isolated_local: bool = False,
    hosted_budget: HostedUsageBudget | None = None,
    price_per_character: Decimal | None = None,
    max_connections: int | None = None,
    device: str | None = None,
    local_worker_timeout_seconds: float = 3_600,
    local_threads_per_process: int = 1,
    local_worker_log_path: Path | None = None,
    local_runtime_workspace: Path | None = None,
    local_runtime_idle_timeout_seconds: float = 900.0,
    local_conditioning_cache_size: int = 8,
    local_runtime_maximum_memory_bytes: int | None = None,
    accelerator_lease: AcceleratorLease | None = None,
) -> AsyncIterator[TextToSpeechService]:
    """Open one backend service and close resources when the context exits."""
    normalized = name.casefold()
    if normalized == "fake":
        yield FakeSpeechService()
        return
    if normalized in {"local", "chatterbox", "chatterbox-local"}:
        if isolated_local and local_runtime_workspace is not None:
            from yakbox.local_runtime import (  # noqa: PLC0415 - optional backend
                LocalRuntimeOptions,
                PersistentLocalSpeechService,
            )

            service = PersistentLocalSpeechService(
                local_runtime_workspace,
                device=device or "cpu",
                options=LocalRuntimeOptions(
                    idle_timeout_seconds=local_runtime_idle_timeout_seconds,
                    conditioning_cache_size=local_conditioning_cache_size,
                    maximum_memory_bytes=local_runtime_maximum_memory_bytes,
                ),
                request_timeout_seconds=local_worker_timeout_seconds,
                accelerator_lease=accelerator_lease,
            )
            yield service
        elif isolated_local:
            from yakbox.speech.workers import (  # noqa: PLC0415 - optional backend
                IsolatedLocalSpeechService,
            )

            service = IsolatedLocalSpeechService(
                device=device or "cpu",
                timeout_seconds=local_worker_timeout_seconds,
                threads_per_process=local_threads_per_process,
                log_path=local_worker_log_path,
                accelerator_lease=accelerator_lease,
            )
            try:
                yield service
            finally:
                await service.aclose()
        else:
            from yakbox.local import (  # noqa: PLC0415 - optional heavy dependency
                LocalChatterboxService,
            )

            yield LocalChatterboxService(device=device or "cpu")
        return
    if normalized in {"resemble", "cloud"}:
        if not api_key:
            raise BackendUnavailableError(
                "Resemble API key is required; set RESEMBLE_API_KEY"
            )
        from yakbox.cloud import (  # noqa: PLC0415 - optional hosted backend
            ClientOptions,
            HostedUsageGate,
            ResembleClient,
            ResembleSpeechService,
        )

        usage = HostedUsageGate(
            hosted_budget or HostedUsageBudget(),
            price_per_character=price_per_character,
        )
        connections = max(20, max_connections or 20)
        options = ClientOptions(
            max_connections=connections,
            max_keepalive_connections=connections,
        )
        async with ResembleClient(
            api_key,
            options=options,
            usage_gate=usage,
        ) as client:
            service = ResembleSpeechService(
                client,
                concurrency=max_connections or 1,
            )
            try:
                yield service
            finally:
                await service.aclose()
        return
    if normalized in {"remote", "chatterbox-remote"}:
        raise BackendUnavailableError(
            "Remote Chatterbox is not available: no verified service contract exists"
        )
    raise BackendUnavailableError(f"Unknown speech backend: {name}")


@asynccontextmanager
async def open_transformation_backend(
    name: str,
) -> AsyncIterator[SpeechTransformationService]:
    """Open a transformation backend for the lifetime of the async context."""
    normalized = name.casefold()
    if normalized == "fake":
        yield FakeSpeechService()
        return
    if normalized in {"local", "chatterbox", "chatterbox-local"}:
        from yakbox.local import (  # noqa: PLC0415 - optional heavy dependency
            LocalChatterboxService,
        )

        yield LocalChatterboxService()
        return
    raise BackendUnavailableError(
        f"Backend {name!r} does not support speech transformation"
    )


@asynccontextmanager
async def open_configured_speech_backend(
    name: str,
    options: SpeechBackendOptions,
) -> AsyncIterator[TextToSpeechService]:
    """Open a speech backend from one evolvable typed options object."""
    async with open_speech_backend(
        name,
        api_key=options.api_key,
        isolated_local=options.isolated_local,
        hosted_budget=options.hosted_budget,
        price_per_character=options.price_per_character,
        max_connections=options.max_connections,
        device=options.device,
        local_worker_timeout_seconds=options.local_worker_timeout_seconds,
        local_threads_per_process=options.local_threads_per_process,
        local_worker_log_path=options.local_worker_log_path,
    ) as service:
        yield service
