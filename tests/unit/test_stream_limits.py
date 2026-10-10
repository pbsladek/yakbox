from __future__ import annotations

import asyncio
import gzip
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from pathlib import Path
from typing import cast

import httpx
import pytest

from yakbox.cloud import ClientOptions, ResembleClient
from yakbox.cloud.errors import ProviderError, ProviderProtocolError
from yakbox.errors import BuildError, ValidationError, WorkerProtocolError
from yakbox.speech.analysis_corpus_text_sources import _bounded_get
from yakbox.speech.analysis_models import AudioSpan
from yakbox.speech.analysis_protocol import (
    MAXIMUM_WORKER_FRAME_BYTES,
    RecognitionBatchItem,
    RecognitionManyWorkerRequest,
    StatusWorkerRequest,
)
from yakbox.speech.analysis_runtime import BUILT_IN_WORKERS, IsolatedAnalysisWorker
from yakbox.speech.models import SpeechSynthesisRequest
from yakbox.speech.workers import IsolatedLocalSpeechService


class _AsyncBody(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: tuple[bytes, ...],
        *,
        error: httpx.TransportError | None = None,
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.read_count = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.read_count += 1
            yield chunk
        if self.error is not None:
            raise self.error

    async def aclose(self) -> None:
        self.closed = True


class _Body(httpx.SyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self.chunks = chunks
        self.read_count = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.chunks:
            self.read_count += 1
            yield chunk

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["list", "upload"])
@pytest.mark.parametrize(
    ("headers", "expected_reads"),
    [
        ({}, 2),
        ({"content-length": "1000"}, 0),
        ({"content-length": "invalid"}, 0),
        ({"content-length": "-1"}, 0),
    ],
)
async def test_provider_json_limit_stops_body_consumption(
    tmp_path: Path, operation: str, headers: dict[str, str], expected_reads: int
) -> None:
    body = _AsyncBody((b"x" * 16, b"x", b"unread tail"))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers, stream=body)

    options = ClientOptions(max_json_response_bytes=16)
    audio = tmp_path / "upload.wav"
    audio.write_bytes(b"RIFF")
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http,
        ResembleClient("secret", options=options, http_client=http) as client,
    ):
        with pytest.raises(ProviderProtocolError):
            if operation == "list":
                await client.list_voices()
            else:
                await client.create_recording("voice", audio, name="Read", text="Hi")

    assert body.read_count == expected_reads
    assert body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("compressed", [False, True])
async def test_provider_accepts_exact_json_limit(compressed: bool) -> None:
    payload = b'{"items": []}'
    headers = {"content-encoding": "gzip"} if compressed else {}
    body = _AsyncBody((gzip.compress(payload) if compressed else payload,))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers, stream=body)

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http,
        ResembleClient(
            "secret",
            options=ClientOptions(max_json_response_bytes=len(payload)),
            http_client=http,
        ) as client,
    ):
        page = await client.list_voices()

    assert page.items == ()
    assert body.closed


@pytest.mark.asyncio
async def test_provider_limit_counts_decompressed_bytes() -> None:
    body = _AsyncBody((gzip.compress(b"x" * 32), b"unread tail"))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-encoding": "gzip"}, stream=body)

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http,
        ResembleClient(
            "secret",
            options=ClientOptions(max_json_response_bytes=16),
            http_client=http,
        ) as client,
    ):
        with pytest.raises(ProviderProtocolError, match="too large"):
            await client.list_voices()

    assert body.read_count == 1
    assert body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("broken", [False, True])
async def test_provider_error_body_is_bounded_and_closed(broken: bool) -> None:
    body = _AsyncBody(
        () if broken else (b"secret " + b"x" * 4096, b"unread tail"),
        error=httpx.ReadError("unreadable error body") if broken else None,
    )
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, stream=body)

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http,
        ResembleClient("secret", http_client=http) as client,
    ):
        with pytest.raises(ProviderError) as raised:
            await client.list_voices()

    assert raised.value.status_code == 401
    assert "secret" not in str(raised.value)
    assert body.read_count == (0 if broken else 1)
    assert body.closed
    assert calls == 1


@pytest.mark.parametrize(
    ("headers", "expected_reads"),
    [
        ({}, 2),
        ({"content-length": "1000"}, 0),
        ({"content-length": "invalid"}, 0),
        ({"content-length": "-1"}, 0),
    ],
)
def test_source_text_limit_stops_body_consumption(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str], expected_reads: int
) -> None:
    monkeypatch.setattr(
        "yakbox.speech.analysis_corpus_text_sources._MAXIMUM_RESPONSE_BYTES", 16
    )
    body = _Body((b"x" * 16, b"x", b"unread tail"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers, stream=body, request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ValidationError),
    ):
        _bounded_get(client, "https://source.test/text")

    assert body.read_count == expected_reads
    assert body.closed


@pytest.mark.parametrize("compressed", [False, True])
def test_source_text_accepts_exact_limit(
    monkeypatch: pytest.MonkeyPatch, compressed: bool
) -> None:
    payload = b"x" * 16
    monkeypatch.setattr(
        "yakbox.speech.analysis_corpus_text_sources._MAXIMUM_RESPONSE_BYTES", 16
    )
    body = _Body((gzip.compress(payload) if compressed else payload,))
    headers = {"content-encoding": "gzip"} if compressed else {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers, stream=body, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = _bounded_get(client, "https://source.test/text")

    assert response.content == payload
    assert str(response.url) == "https://source.test/text"
    assert body.closed


class _BlockedWriter:
    def write(self, _payload: bytes) -> None:
        return

    async def drain(self) -> None:
        await asyncio.Event().wait()


class _ReadyWriter(_BlockedWriter):
    async def drain(self) -> None:
        return


class _BlockedProcess:
    pid = 42
    stdin = _BlockedWriter()
    stderr = None

    def __init__(self) -> None:
        self.stdout = asyncio.StreamReader()
        self.returncode: int | None = None
        self.finished = asyncio.Event()

    def terminate(self) -> None:
        self.returncode = -15
        self.finished.set()

    def kill(self) -> None:
        self.returncode = -9
        self.finished.set()

    async def wait(self) -> int:
        await self.finished.wait()
        return self.returncode or 0


@pytest.mark.asyncio
async def test_local_worker_timeout_covers_blocked_request_write(
    tmp_path: Path,
) -> None:
    process = _BlockedProcess()
    service = IsolatedLocalSpeechService(timeout_seconds=0.03)
    service._process = cast(asyncio.subprocess.Process, process)

    with pytest.raises(BuildError, match="exceeded"):
        await asyncio.wait_for(
            service.synthesize_to_file(
                SpeechSynthesisRequest(text="Hi.", voice="narrator"),
                tmp_path / "speech.wav",
            ),
            timeout=1,
        )

    assert process.returncode == -15
    assert service._process is None


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["handshake", "write"])
async def test_analysis_batch_timeout_covers_startup_and_request_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    digest = "a" * 64
    process = _BlockedProcess()
    worker = IsolatedAnalysisWorker(
        BUILT_IN_WORKERS["whisper"],
        audio_root=tmp_path,
        model_root=tmp_path,
        calibration_fingerprint=digest,
        worker_artifact_digest=digest,
        lock_digest=digest,
    )

    async def ensure_process() -> asyncio.subprocess.Process:
        worker._process = cast(asyncio.subprocess.Process, process)
        if phase == "handshake":
            await asyncio.Event().wait()
        return cast(asyncio.subprocess.Process, process)

    monkeypatch.setattr(worker, "_ensure_process", ensure_process)
    request = RecognitionManyWorkerRequest(
        "request",
        "batch",
        "whisper",
        digest,
        30,
        (
            RecognitionBatchItem(
                0, digest, "audio.wav", digest, "en", AudioSpan(digest, 0, 1600, 16000)
            ),
        ),
    )

    async def run_batch() -> None:
        async for _item in worker.request_many(request, timeout_seconds=0.03):
            pytest.fail("Blocked worker unexpectedly returned a batch item")

    pending = asyncio.create_task(run_batch())
    try:
        done, _ = await asyncio.wait({pending}, timeout=1)
        assert pending in done, "Batch deadline did not cover startup/write"
        with pytest.raises(TimeoutError):
            await pending
        assert worker.pid is None
        assert process.returncode == -15
    finally:
        if not pending.done():
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["handshake", "response"])
@pytest.mark.parametrize("oversized", [False, True])
async def test_rejected_analysis_frame_discards_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, oversized: bool
) -> None:
    process = _BlockedProcess()
    process.stdin = _ReadyWriter()
    process.stdout = asyncio.StreamReader(limit=MAXIMUM_WORKER_FRAME_BYTES + 2)
    frame = b"x" * (MAXIMUM_WORKER_FRAME_BYTES + 1) if oversized else b"invalid JSON"
    process.stdout.feed_data(frame + b"\n")

    async def spawn(*_args: object, **_kwargs: object) -> asyncio.subprocess.Process:
        return cast(asyncio.subprocess.Process, process)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    digest = "a" * 64
    worker = IsolatedAnalysisWorker(
        BUILT_IN_WORKERS["whisper"],
        audio_root=tmp_path,
        model_root=tmp_path,
        calibration_fingerprint=digest,
        worker_artifact_digest=digest,
        lock_digest=digest,
    )
    with pytest.raises(WorkerProtocolError):
        if phase == "handshake":
            await worker.start()
        else:
            worker._process = cast(asyncio.subprocess.Process, process)
            await worker.request(StatusWorkerRequest("status"), timeout_seconds=1)

    assert worker.pid is None
    assert process.returncode == -15
