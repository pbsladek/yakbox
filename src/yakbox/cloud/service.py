from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from yakbox._files import sha256_file
from yakbox.cloud.client import ResembleClient
from yakbox.cloud.models import AudioFormat as CloudAudioFormat
from yakbox.cloud.models import Precision, SynthesisRequest
from yakbox.speech.capabilities import BackendCapabilities
from yakbox.speech.models import (
    AudioFormat,
    HostedUsageSnapshot,
    SpeechArtifact,
    SpeechSynthesisRequest,
)
from yakbox.speech.services import HostedUsageRecorder


@dataclass(slots=True)
class _SynthesisJob:
    request: SpeechSynthesisRequest
    destination: Path
    overwrite: bool
    future: asyncio.Future[SpeechArtifact]


class ResembleSpeechService:
    capabilities = BackendCapabilities(
        name="resemble",
        synthesis=True,
        transformation=False,
        streaming=True,
        hosted=True,
        output_formats=("wav", "mp3"),
        max_text_characters=3_000,
        supports_hd=True,
    )

    def __init__(self, client: ResembleClient, *, concurrency: int = 1) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self.client = client
        self.concurrency = concurrency
        self._queue: asyncio.Queue[_SynthesisJob] = asyncio.Queue(
            maxsize=2 * concurrency
        )
        self._workers: tuple[asyncio.Task[None], ...] = ()
        self._closed = False

    async def usage_snapshot(self) -> HostedUsageSnapshot | None:
        return await self.client.usage_snapshot()

    def set_usage_recorder(self, recorder: HostedUsageRecorder | None) -> None:
        self.client.set_usage_recorder(recorder)

    async def restore_usage(self, snapshot: HostedUsageSnapshot) -> None:
        await self.client.restore_usage(snapshot)

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        return await self._synthesize_direct(
            request,
            destination,
            overwrite=overwrite,
        )

    async def synthesize_many_to_files(
        self,
        requests: tuple[tuple[SpeechSynthesisRequest, Path], ...],
        *,
        overwrite: bool = False,
    ) -> tuple[SpeechArtifact, ...]:
        futures = [
            await self._enqueue(request, destination, overwrite=overwrite)
            for request, destination in requests
        ]
        try:
            results = await asyncio.gather(*futures, return_exceptions=True)
        except asyncio.CancelledError:
            await self.aclose()
            raise
        error = next(
            (item for item in results if isinstance(item, BaseException)), None
        )
        if error is not None:
            raise error
        return tuple(item for item in results if isinstance(item, SpeechArtifact))

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = ()
        while not self._queue.empty():
            job = self._queue.get_nowait()
            if not job.future.done():
                job.future.cancel()
            self._queue.task_done()

    async def _enqueue(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool,
    ) -> asyncio.Future[SpeechArtifact]:
        if self._closed:
            raise RuntimeError("Resemble speech service is closed")
        if not self._workers:
            self._workers = tuple(
                asyncio.create_task(self._worker()) for _ in range(self.concurrency)
            )
        future = asyncio.get_running_loop().create_future()
        await self._queue.put(
            _SynthesisJob(
                request=request,
                destination=destination,
                overwrite=overwrite,
                future=future,
            )
        )
        return future

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job.future.cancelled():
                    continue
                artifact = await self._synthesize_direct(
                    job.request,
                    job.destination,
                    overwrite=job.overwrite,
                )
                if not job.future.done():
                    job.future.set_result(artifact)
            except asyncio.CancelledError:
                if not job.future.done():
                    job.future.cancel()
                raise
            except Exception as error:
                if not job.future.done():
                    job.future.set_exception(error)
            finally:
                self._queue.task_done()

    async def _synthesize_direct(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool,
    ) -> SpeechArtifact:
        provider_request = SynthesisRequest(
            text=request.text,
            voice_uuid=request.voice,
            project_uuid=request.project,
            title=request.title,
            output_format=CloudAudioFormat(request.output_format.value),
            sample_rate=request.sample_rate,
            use_hd=request.use_hd,
            precision=Precision(request.precision or Precision.PCM_32.value),
            apply_custom_pronunciations=request.apply_custom_pronunciations,
        )
        result = await self.client.synthesize_to_file(
            provider_request, destination, overwrite=overwrite
        )
        return SpeechArtifact(
            path=result.path,
            backend="resemble",
            voice=request.voice,
            output_format=AudioFormat(request.output_format.value),
            bytes_written=result.bytes_written,
            sha256=sha256_file(result.path),
            duration_seconds=result.duration_seconds,
            sample_rate=request.sample_rate,
            request_id=result.request_id,
            attempts=result.attempts,
        )
