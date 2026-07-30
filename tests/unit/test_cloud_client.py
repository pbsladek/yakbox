from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from yakbox.cloud import (
    AudioFormat,
    ClientOptions,
    ResembleClient,
    RetryPolicy,
    StreamRequest,
    SynthesisRequest,
)
from yakbox.cloud.errors import (
    AmbiguousMutationError,
    ClientStateError,
    ProviderError,
    ProviderProtocolError,
    RetryExhaustedError,
)
from yakbox.cloud.usage import HostedUsageGate
from yakbox.errors import ValidationError
from yakbox.speech import HostedUsageBudget, HostedUsageSnapshot


@pytest.mark.asyncio
async def test_synthesis_retries_and_decodes(tmp_path: Path) -> None:
    calls = 0
    audio = b"RIFF-test-audio"

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["authorization"] == "Bearer secret"
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            headers={"x-request-id": "req-1"},
            json={
                "success": True,
                "audio_content": base64.b64encode(audio).decode(),
                "output_format": "wav",
                "duration": 1.25,
            },
        )

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer secret"},
    )
    options = ClientOptions(
        synthesis_base_url="https://example.test",
        retry=RetryPolicy(max_attempts=2, base_delay=0),
    )
    async with ResembleClient("secret", options=options, http_client=http) as client:
        result = await client.synthesize_to_file(
            SynthesisRequest(text="hello", voice_uuid="voice"),
            tmp_path / "line.wav",
        )
    assert not http.is_closed
    await http.aclose()

    assert calls == 2
    assert result.path.read_bytes() == audio
    assert result.attempts == 2
    assert result.request_id == "req-1"


@pytest.mark.asyncio
async def test_usage_reservation_is_durable_before_provider_send() -> None:
    recorded_attempts: list[int] = []

    async def recorder(snapshot: HostedUsageSnapshot, _characters: int) -> None:
        recorded_attempts.append(snapshot.provider_attempts)

    async def handler(_request: httpx.Request) -> httpx.Response:
        assert recorded_attempts == [1]
        return httpx.Response(
            200,
            json={
                "audio_content": base64.b64encode(b"audio").decode(),
                "output_format": "wav",
            },
        )

    gate = HostedUsageGate(HostedUsageBudget(max_provider_requests=1))
    gate.set_recorder(recorder)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with ResembleClient(
        "secret",
        http_client=http,
        usage_gate=gate,
    ) as client:
        await client.synthesize(SynthesisRequest(text="hello", voice_uuid="voice"))
    await http.aclose()

    snapshot = await gate.snapshot()
    assert snapshot.provider_attempts == 1
    assert snapshot.submitted_characters == 5


@pytest.mark.asyncio
async def test_stream_writes_incrementally(tmp_path: Path) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"RIFF-stream")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    options = ClientOptions(synthesis_base_url="https://example.test")
    async with ResembleClient("secret", options=options, http_client=http) as client:
        result = await client.stream_to_file(
            StreamRequest(text="hello", voice_uuid="voice"),
            tmp_path / "stream.wav",
        )
    await http.aclose()
    assert result.bytes_written == len(b"RIFF-stream")
    assert result.path.read_bytes() == b"RIFF-stream"


@pytest.mark.asyncio
async def test_auth_failure_is_not_retried() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="bad token")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    options = ClientOptions(synthesis_base_url="https://example.test")
    async with ResembleClient("secret", options=options, http_client=http) as client:
        with pytest.raises(ProviderError, match="bad token"):
            await client.synthesize(
                SynthesisRequest(
                    text="hello",
                    voice_uuid="voice",
                    output_format=AudioFormat.WAV,
                )
            )
    await http.aclose()
    assert calls == 1


@pytest.mark.asyncio
async def test_client_requires_context() -> None:
    client = ResembleClient("secret")
    with pytest.raises(ClientStateError):
        await client.synthesize(SynthesisRequest(text="hello", voice_uuid="voice"))


@pytest.mark.asyncio
async def test_owned_client_is_constructed_once_reused_and_closed() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "audio_content": base64.b64encode(b"audio").decode(),
                "output_format": "wav",
            },
        )

    owned = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("yakbox.cloud.client.httpx.AsyncClient", return_value=owned) as factory:
        async with ResembleClient("secret") as client:
            await client.synthesize(SynthesisRequest(text="one", voice_uuid="voice"))
            await client.synthesize(SynthesisRequest(text="two", voice_uuid="voice"))
        assert factory.call_count == 1
        constructor = factory.call_args.kwargs
        assert constructor["headers"]["Authorization"] == "Bearer secret"
        assert str(constructor["headers"]["User-Agent"]).startswith("yakbox/")
    assert owned.is_closed


@pytest.mark.asyncio
async def test_owned_client_closes_after_operation_failure() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="invalid")

    owned = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with (
        patch("yakbox.cloud.client.httpx.AsyncClient", return_value=owned),
        pytest.raises(ProviderError),
    ):
        async with ResembleClient("secret") as client:
            await client.synthesize(SynthesisRequest(text="one", voice_uuid="voice"))
    assert owned.is_closed


@pytest.mark.asyncio
async def test_urls_serialization_and_safe_provider_error() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json={"items": []})
        if len(requests) == 2:
            return httpx.Response(
                200,
                json={
                    "audio_content": base64.b64encode(b"audio").decode(),
                    "output_format": "wav",
                },
            )
        return httpx.Response(
            400,
            headers={"x-request-id": "req-safe"},
            text="Bearer secret " + ("x" * 3_000),
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    options = ClientOptions(
        management_base_url="https://management.test/v2",
        synthesis_base_url="https://synthesis.test",
    )
    async with ResembleClient("secret", options=options, http_client=http) as client:
        await client.list_voices(page=2, page_size=3)
        await client.synthesize(
            SynthesisRequest(
                text="hello",
                voice_uuid="voice",
                project_uuid=None,
                title=None,
            )
        )
        with pytest.raises(ProviderError) as raised:
            await client.create_project("bad")
    await http.aclose()

    assert requests[0].url == "https://management.test/v2/voices?page=2&page_size=3"
    assert requests[1].url == "https://synthesis.test/synthesize"
    body = requests[1].content.decode()
    assert '"data":"hello"' in body
    assert "project_uuid" not in body
    assert "title" not in body
    assert requests[2].url == "https://management.test/v2/projects"
    assert "secret" not in str(raised.value)
    assert len(raised.value.message) <= 2_048
    assert raised.value.request_id == "req-safe"


@pytest.mark.asyncio
async def test_management_mutations_use_current_contract_and_retry_429(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []
    project_calls = 0
    recording_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal project_calls, recording_calls
        requests.append(request)
        if request.url.path.endswith("/projects"):
            project_calls += 1
            if project_calls == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(
                201,
                json={
                    "item": {
                        "uuid": "project-1",
                        "name": "Book",
                        "description": "Description",
                        "is_collaborative": True,
                        "is_archived": False,
                    }
                },
            )
        recording_calls += 1
        if recording_calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            201,
            json={"item": {"uuid": "recording-1", "name": "Read", "text": "Hi"}},
        )

    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF-audio")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    options = ClientOptions(
        management_base_url="https://management.test/v2",
        retry=RetryPolicy(max_attempts=2, base_delay=0),
    )
    async with ResembleClient("secret", options=options, http_client=http) as client:
        project = await client.create_project(
            "Book",
            description="Description",
            is_collaborative=True,
        )
        recording = await client.create_recording(
            "voice-1",
            audio,
            name="Read",
            text="Hi",
            emotion="neutral",
        )
    await http.aclose()

    assert project.uuid == "project-1"
    assert recording.uuid == "recording-1"
    assert project_calls == recording_calls == 2
    project_payloads = [
        request.content.decode()
        for request in requests
        if request.url.path.endswith("/projects")
    ]
    assert all('"is_public"' not in payload for payload in project_payloads)
    assert all('"is_collaborative":true' in payload for payload in project_payloads)
    recording_requests = [
        request for request in requests if request.url.path.endswith("/recordings")
    ]
    assert all(b"RIFF-audio" in request.content for request in recording_requests)
    assert all(b'name="is_active"' in request.content for request in recording_requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_calls"),
    [
        (httpx.ReadTimeout("ambiguous read"), 1),
        (httpx.WriteError("ambiguous write"), 1),
    ],
)
async def test_management_mutation_does_not_retry_ambiguous_transport_failure(
    failure: httpx.TransportError,
    expected_calls: int,
) -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise failure

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    options = ClientOptions(
        management_base_url="https://management.test/v2",
        retry=RetryPolicy(max_attempts=3, base_delay=0),
    )
    async with ResembleClient("secret", options=options, http_client=http) as client:
        with pytest.raises(
            AmbiguousMutationError,
            match="verify provider state",
        ) as raised:
            await client.create_project("Book")
    await http.aclose()

    assert calls == expected_calls
    assert raised.value.cause is failure


@pytest.mark.asyncio
async def test_management_mutation_does_not_retry_ambiguous_server_error() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="maybe accepted")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    options = ClientOptions(
        management_base_url="https://management.test/v2",
        retry=RetryPolicy(max_attempts=3, base_delay=0),
    )
    async with ResembleClient("secret", options=options, http_client=http) as client:
        with pytest.raises(AmbiguousMutationError) as raised:
            await client.create_project("Book")
    await http.aclose()

    assert calls == 1
    assert isinstance(raised.value.cause, ProviderError)
    assert raised.value.cause.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "text", "emotion", "message"),
    [
        ("x" * 257, "text", None, "name exceeds"),
        ("name", "x" * 1_025, None, "text exceeds"),
        ("name", "text", "x" * 65, "emotion exceeds"),
    ],
)
async def test_recording_limits_fail_before_network(
    tmp_path: Path,
    name: str,
    text: str,
    emotion: str | None,
    message: str,
) -> None:
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF")
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(201, json={"item": {"uuid": "unexpected"}})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with ResembleClient("secret", http_client=http) as client:
        with pytest.raises(ValidationError, match=message):
            await client.create_recording(
                "voice",
                audio,
                name=name,
                text=text,
                emotion=emotion,
            )
    await http.aclose()
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "description", "message"),
    [
        ("x" * 257, None, "name exceeds"),
        ("name", "x" * 1_025, "description exceeds"),
    ],
)
async def test_project_limits_fail_before_network(
    name: str,
    description: str | None,
    message: str,
) -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(201, json={"item": {"uuid": "unexpected"}})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with ResembleClient("secret", http_client=http) as client:
        with pytest.raises(ValidationError, match=message):
            await client.create_project(name, description=description)
    await http.aclose()
    assert calls == 0


class _FailingStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"partial"
        raise httpx.ReadError("interrupted")


@pytest.mark.asyncio
async def test_stream_to_file_retries_from_zero_and_raw_stream_does_not(
    tmp_path: Path,
) -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, stream=_FailingStream())
        return httpx.Response(200, content=b"complete")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    options = ClientOptions(
        synthesis_base_url="https://example.test",
        retry=RetryPolicy(max_attempts=2, base_delay=0),
    )
    async with ResembleClient("secret", options=options, http_client=http) as client:
        result = await client.stream_to_file(
            StreamRequest(text="hello", voice_uuid="voice"),
            tmp_path / "stream.wav",
        )
    assert result.path.read_bytes() == b"complete"
    assert calls == 2
    assert not tuple(tmp_path.glob("*.part"))
    await http.aclose()

    raw_calls = 0

    async def raw_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal raw_calls
        raw_calls += 1
        return httpx.Response(200, stream=_FailingStream())

    raw_http = httpx.AsyncClient(transport=httpx.MockTransport(raw_handler))
    async with ResembleClient(
        "secret", options=options, http_client=raw_http
    ) as client:
        with pytest.raises(httpx.ReadError):
            async with client.stream(
                StreamRequest(text="hello", voice_uuid="voice")
            ) as chunks:
                assert await anext(chunks) == b"partial"
                await anext(chunks)
    assert raw_calls == 1
    await raw_http.aclose()


@pytest.mark.asyncio
async def test_stream_retries_status_before_yield_and_decode_validation() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, content=b"ok")

    options = ClientOptions(
        synthesis_base_url="https://example.test",
        retry=RetryPolicy(max_attempts=2, base_delay=0),
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with (
        ResembleClient("secret", options=options, http_client=http) as client,
        client.stream(StreamRequest(text="hello", voice_uuid="voice")) as chunks,
    ):
        assert b"".join([chunk async for chunk in chunks]) == b"ok"
    assert calls == 2
    await http.aclose()

    async def invalid(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"audio_content": "not base64"})

    invalid_http = httpx.AsyncClient(transport=httpx.MockTransport(invalid))
    async with ResembleClient("secret", http_client=invalid_http) as client:
        with pytest.raises(ProviderProtocolError, match="base64"):
            await client.synthesize(SynthesisRequest(text="hello", voice_uuid="voice"))
    await invalid_http.aclose()


@pytest.mark.asyncio
async def test_retry_exhaustion_retains_status_request_and_attempts() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            headers={"x-request-id": "last-request"},
            text="busy",
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    options = ClientOptions(retry=RetryPolicy(max_attempts=2, base_delay=0))
    async with ResembleClient("secret", options=options, http_client=http) as client:
        with pytest.raises(RetryExhaustedError) as raised:
            await client.synthesize(SynthesisRequest(text="hello", voice_uuid="voice"))
    await http.aclose()
    assert raised.value.attempts == 2
    assert raised.value.status_code == 503
    assert raised.value.request_id == "last-request"
