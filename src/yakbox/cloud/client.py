from __future__ import annotations

import base64
import binascii
import json
import os
import re
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from pathlib import Path
from typing import Self, cast

import httpx

from yakbox import __version__
from yakbox.cloud.errors import (
    AmbiguousMutationError,
    ClientStateError,
    ProviderError,
    ProviderProtocolError,
)
from yakbox.cloud.models import (
    AudioFormat,
    ClientOptions,
    FileSynthesisResult,
    Page,
    Project,
    Recording,
    StreamRequest,
    SynthesisRequest,
    SynthesisResult,
    Voice,
)
from yakbox.cloud.output import atomic_commit_bytes, commit_file
from yakbox.cloud.rate_limit import RateLimitGate
from yakbox.cloud.retry import parse_retry_after, retry_operation
from yakbox.cloud.usage import HostedUsageGate
from yakbox.errors import ValidationError
from yakbox.speech.models import HostedUsageSnapshot
from yakbox.speech.services import HostedUsageRecorder

MAX_PAGE_SIZE = 100
HTTP_TOO_MANY_REQUESTS = 429
_AMBIGUOUS_MUTATION_STATUSES = frozenset({408, 425, 500, 502, 503, 504})
MAX_RECORDING_NAME_CHARACTERS = 256
MAX_RECORDING_TEXT_CHARACTERS = 1_024
MAX_RECORDING_EMOTION_CHARACTERS = 64
MAX_PROJECT_NAME_CHARACTERS = 256
MAX_PROJECT_DESCRIPTION_CHARACTERS = 1_024
_DEFINITELY_UNSENT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)


class ResembleClient:
    """Async Resemble client with bounded retries, responses, and lifecycle."""

    def __init__(
        self,
        api_key: str,
        *,
        options: ClientOptions | None = None,
        http_client: httpx.AsyncClient | None = None,
        usage_gate: HostedUsageGate | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValidationError("Resemble API key must not be empty")
        self.options = options or ClientOptions()
        if self.options.max_json_response_bytes <= 0:
            raise ValidationError("max_json_response_bytes must be positive")
        self._api_key = api_key
        self._injected = http_client
        self._http: httpx.AsyncClient | None = None
        self._owns_http = http_client is None
        self._closed = False
        self._entered = False
        self._gate = RateLimitGate()
        self._usage_gate = usage_gate

    async def __aenter__(self) -> Self:
        if self._closed:
            raise ClientStateError("ResembleClient is closed")
        if self._entered:
            raise ClientStateError("ResembleClient is already open")
        self._entered = True
        if self._injected is not None:
            self._http = self._injected
        else:
            self._http = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                    "User-Agent": f"yakbox/{__version__}",
                },
                timeout=httpx.Timeout(
                    connect=self.options.connect_timeout,
                    read=self.options.read_timeout,
                    write=self.options.write_timeout,
                    pool=self.options.pool_timeout,
                ),
                limits=httpx.Limits(
                    max_connections=self.options.max_connections,
                    max_keepalive_connections=self.options.max_keepalive_connections,
                ),
            )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close owned HTTP resources; repeated calls are safe."""
        if self._closed:
            return
        self._closed = True
        http = self._http
        self._http = None
        if http is not None and self._owns_http:
            await http.aclose()

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Submit one bounded synthesis request and decode its audio response."""
        attempts = 0
        if self._usage_gate is not None:
            await self._usage_gate.add_logical_item()

        async def operation(attempt: int) -> httpx.Response:
            nonlocal attempts
            attempts = attempt
            if self._usage_gate is not None:
                await self._usage_gate.reserve_attempt(len(request.text))
            try:
                return await self._json_request(
                    "POST",
                    f"{self.options.synthesis_base_url}/synthesize",
                    json_body=_synthesis_payload(request),
                )
            except httpx.TransportError:
                if self._usage_gate is not None:
                    await self._usage_gate.mark_ambiguous()
                raise

        response = await retry_operation(
            operation,
            policy=self.options.retry,
            rate_limit_gate=self._gate,
        )
        raw = self._decode_json(response)
        encoded = raw.get("audio_content")
        if not isinstance(encoded, str):
            raise ProviderProtocolError("Synthesis response lacks audio_content")
        try:
            audio = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ProviderProtocolError(
                "Synthesis audio_content is not base64"
            ) from error
        if not audio:
            raise ProviderProtocolError("Synthesis response contains empty audio")
        response_format = raw.get("output_format", request.output_format.value)
        try:
            output_format = AudioFormat(str(response_format).casefold())
        except ValueError as error:
            raise ProviderProtocolError(
                f"Unsupported response output format: {response_format!r}"
            ) from error
        issues_raw = raw.get("issues", [])
        issues = (
            tuple(str(item) for item in issues_raw)
            if isinstance(issues_raw, list)
            else ()
        )
        return SynthesisResult(
            audio=audio,
            duration_seconds=_optional_float(raw.get("duration")),
            synthesis_seconds=_optional_float(raw.get("synth_duration")),
            output_format=output_format,
            sample_rate=_optional_int(raw.get("sample_rate")),
            title=_string(raw.get("title")),
            issues=issues,
            request_id=_request_id(response),
            attempts=attempts,
        )

    async def synthesize_to_file(
        self,
        request: SynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> FileSynthesisResult:
        """Synthesize one request and atomically commit it to a destination."""
        result = await self.synthesize(request)
        atomic_commit_bytes(destination, result.audio, overwrite=overwrite)
        return FileSynthesisResult(
            path=destination.resolve(),
            bytes_written=len(result.audio),
            duration_seconds=result.duration_seconds,
            issues=result.issues,
            request_id=result.request_id,
            attempts=result.attempts,
        )

    def stream(
        self, request: StreamRequest
    ) -> AbstractAsyncContextManager[AsyncIterator[bytes]]:
        """Open a one-shot streaming response context for a bounded request."""
        return self._stream_once(request)

    @asynccontextmanager
    async def _stream_once(
        self, request: StreamRequest
    ) -> AsyncIterator[AsyncIterator[bytes]]:
        if self._usage_gate is not None:
            await self._usage_gate.add_logical_item()

        async def operation(_attempt: int) -> httpx.Response:
            if self._usage_gate is not None:
                await self._usage_gate.reserve_attempt(len(request.text))
            http = self._require_http()
            request_object = http.build_request(
                "POST",
                f"{self.options.synthesis_base_url}/stream",
                json=_stream_payload(request),
            )
            try:
                response = await http.send(request_object, stream=True)
            except httpx.TransportError:
                if self._usage_gate is not None:
                    await self._usage_gate.mark_ambiguous()
                raise
            if not response.is_success:
                await self._raise_for_response(response)
            return response

        response = await retry_operation(
            operation,
            policy=self.options.retry,
            rate_limit_gate=self._gate,
        )
        try:
            yield response.aiter_bytes()
        finally:
            await response.aclose()

    async def stream_to_file(
        self,
        request: StreamRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> FileSynthesisResult:
        """Stream one request into an atomic destination file with retries."""
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        attempts = 0
        request_id: str | None = None
        if self._usage_gate is not None:
            await self._usage_gate.add_logical_item()

        async def operation(attempt: int) -> int:
            nonlocal attempts, request_id
            attempts = attempt
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".part",
                dir=destination.parent,
            )
            temporary = Path(temporary_name)
            try:
                if self._usage_gate is not None:
                    await self._usage_gate.reserve_attempt(len(request.text))
                with os.fdopen(descriptor, "wb") as output:
                    http = self._require_http()
                    request_object = http.build_request(
                        "POST",
                        f"{self.options.synthesis_base_url}/stream",
                        json=_stream_payload(request),
                    )
                    try:
                        response = await http.send(request_object, stream=True)
                    except httpx.TransportError:
                        if self._usage_gate is not None:
                            await self._usage_gate.mark_ambiguous()
                        raise
                    request_id = _request_id(response)
                    try:
                        if not response.is_success:
                            await self._raise_for_response(response)
                        async for chunk in response.aiter_bytes():
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    finally:
                        await response.aclose()
                if temporary.stat().st_size == 0:
                    raise ProviderProtocolError("Streaming response was empty")
                bytes_written = temporary.stat().st_size
                commit_file(temporary, destination, overwrite=overwrite)
                return bytes_written
            finally:
                with suppress(OSError):
                    os.close(descriptor)
                temporary.unlink(missing_ok=True)

        bytes_written = await retry_operation(
            operation,
            policy=self.options.retry,
            rate_limit_gate=self._gate,
        )
        return FileSynthesisResult(
            path=destination,
            bytes_written=bytes_written,
            duration_seconds=None,
            issues=(),
            request_id=request_id,
            attempts=attempts,
        )

    async def list_voices(self, *, page: int = 1, page_size: int = 10) -> Page[Voice]:
        """Return one validated page of provider voices."""
        _validate_page(page, page_size)
        response = await self._retried_json_request(
            "GET",
            f"{self.options.management_base_url}/voices",
            params={"page": page, "page_size": page_size},
        )
        raw = self._decode_json(response)
        return _page(
            raw,
            lambda item: Voice(
                uuid=_identifier(item),
                name=str(item.get("name", "")),
                status=_string(item.get("status")),
                created_at=_string(item.get("created_at")),
            ),
        )

    async def create_recording(
        self,
        voice_uuid: str,
        audio_path: Path,
        *,
        name: str,
        text: str,
        emotion: str | None = None,
        is_active: bool = True,
        fill: bool = False,
    ) -> Recording:
        """Upload one recording through ambiguity-safe mutation handling."""
        if not audio_path.is_file():
            raise ValidationError(f"Recording audio does not exist: {audio_path}")
        if not voice_uuid.strip():
            raise ValidationError("voice_uuid must not be empty")
        if not name.strip():
            raise ValidationError("Recording name must not be empty")
        if len(name) > MAX_RECORDING_NAME_CHARACTERS:
            raise ValidationError(
                "Recording name exceeds the provider limit of 256 characters"
            )
        if not text.strip():
            raise ValidationError("Recording text must not be empty")
        if len(text) > MAX_RECORDING_TEXT_CHARACTERS:
            raise ValidationError(
                "Recording text exceeds the provider limit of 1024 characters"
            )
        if emotion is not None and len(emotion) > MAX_RECORDING_EMOTION_CHARACTERS:
            raise ValidationError(
                "Recording emotion exceeds the provider limit of 64 characters"
            )

        async def operation(_attempt: int) -> httpx.Response:
            http = self._require_http()
            with audio_path.open("rb") as audio:
                response = await http.post(
                    (
                        f"{self.options.management_base_url}/voices/"
                        f"{voice_uuid}/recordings"
                    ),
                    files={
                        "file": (
                            audio_path.name,
                            audio,
                            "application/octet-stream",
                        )
                    },
                    data={
                        "name": name,
                        "text": text,
                        "emotion": emotion or "",
                        "is_active": str(is_active).lower(),
                        "fill": str(fill).lower(),
                    },
                )
            if not response.is_success:
                await self._raise_for_response(
                    response,
                    ambiguous=(response.status_code in _AMBIGUOUS_MUTATION_STATUSES),
                )
            return response

        response = await self._management_mutation(
            "Recording creation",
            operation,
        )
        raw = self._decode_json(response)
        item = _payload_item(raw)
        return Recording(
            uuid=_identifier(item),
            name=str(item.get("name", name)),
            text=_string(item.get("text")),
            status=_string(item.get("status")),
        )

    async def list_projects(
        self, *, page: int = 1, page_size: int = 10
    ) -> Page[Project]:
        """Return one validated page of provider projects."""
        _validate_page(page, page_size)
        response = await self._retried_json_request(
            "GET",
            f"{self.options.management_base_url}/projects",
            params={"page": page, "page_size": page_size},
        )
        raw = self._decode_json(response)
        return _page(raw, _project)

    async def create_project(
        self,
        name: str,
        *,
        description: str | None = None,
        is_collaborative: bool = False,
        is_archived: bool = False,
    ) -> Project:
        """Create one project through ambiguity-safe mutation handling."""
        if not name.strip():
            raise ValidationError("Project name must not be empty")
        if len(name) > MAX_PROJECT_NAME_CHARACTERS:
            raise ValidationError(
                "Project name exceeds the provider limit of 256 characters"
            )
        if (
            description is not None
            and len(description) > MAX_PROJECT_DESCRIPTION_CHARACTERS
        ):
            raise ValidationError(
                "Project description exceeds the provider limit of 1024 characters"
            )

        async def operation(_attempt: int) -> httpx.Response:
            return await self._json_request(
                "POST",
                f"{self.options.management_base_url}/projects",
                json_body={
                    "name": name,
                    "description": description,
                    "is_collaborative": is_collaborative,
                    "is_archived": is_archived,
                },
                ambiguous_mutation=True,
            )

        response = await self._management_mutation(
            "Project creation",
            operation,
        )
        raw = self._decode_json(response)
        return _project(_payload_item(raw))

    async def _retried_json_request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str | int | float | None] | None = None,
    ) -> httpx.Response:
        async def operation(_attempt: int) -> httpx.Response:
            return await self._json_request(method, url, params=params)

        return await retry_operation(
            operation,
            policy=self.options.retry,
            rate_limit_gate=self._gate,
        )

    async def _json_request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, object] | None = None,
        params: dict[str, str | int | float | None] | None = None,
        ambiguous_mutation: bool = False,
    ) -> httpx.Response:
        response = await self._require_http().request(
            method, url, json=json_body, params=params
        )
        if not response.is_success:
            await self._raise_for_response(
                response,
                ambiguous=(
                    ambiguous_mutation
                    and response.status_code in _AMBIGUOUS_MUTATION_STATUSES
                ),
            )
        length = response.headers.get("content-length")
        if length and int(length) > self.options.max_json_response_bytes:
            await response.aclose()
            raise ProviderProtocolError("Provider JSON response is too large")
        if len(response.content) > self.options.max_json_response_bytes:
            await response.aclose()
            raise ProviderProtocolError("Provider JSON response is too large")
        return response

    async def _management_mutation[T](
        self,
        operation_name: str,
        operation: Callable[[int], Awaitable[T]],
    ) -> T:
        def retryable(error: Exception) -> bool:
            return isinstance(error, _DEFINITELY_UNSENT_ERRORS) or (
                isinstance(error, ProviderError)
                and error.status_code == HTTP_TOO_MANY_REQUESTS
            )

        try:
            return await retry_operation(
                operation,
                policy=self.options.retry,
                is_retryable=retryable,
                rate_limit_gate=self._gate,
            )
        except ProviderError as error:
            if error.ambiguous:
                raise AmbiguousMutationError(
                    operation_name,
                    cause=error,
                ) from error
            raise
        except httpx.TransportError as error:
            if not isinstance(error, _DEFINITELY_UNSENT_ERRORS):
                raise AmbiguousMutationError(
                    operation_name,
                    cause=error,
                ) from error
            raise

    async def _raise_for_response(
        self, response: httpx.Response, *, ambiguous: bool = False
    ) -> None:
        body = self._safe_provider_message(
            (await response.aread())[:4096].decode(errors="replace").strip()
        )
        request_id = _request_id(response)
        retry_after = parse_retry_after(response.headers.get("retry-after"))
        await response.aclose()
        raise ProviderError(
            status_code=response.status_code,
            message=body or response.reason_phrase,
            request_id=request_id,
            retry_after=retry_after,
            ambiguous=ambiguous,
        )

    def _safe_provider_message(self, value: str) -> str:
        redacted = value.replace(self._api_key, "[REDACTED]")
        redacted = re.sub(
            r"(?i)bearer\s+[a-z0-9._~+/=-]+",
            "Bearer [REDACTED]",
            redacted,
        )
        return redacted[:2048]

    def _decode_json(self, response: httpx.Response) -> dict[str, object]:
        try:
            raw = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ProviderProtocolError("Provider returned invalid JSON") from error
        if not isinstance(raw, dict):
            raise ProviderProtocolError("Provider JSON response must be an object")
        return cast(dict[str, object], raw)

    def _require_http(self) -> httpx.AsyncClient:
        if not self._entered or self._closed or self._http is None:
            raise ClientStateError(
                "Use ResembleClient inside 'async with ResembleClient(...)'"
            )
        return self._http

    async def usage_snapshot(self) -> HostedUsageSnapshot | None:
        """Return the current hosted usage snapshot when accounting is enabled."""
        if self._usage_gate is None:
            return None
        return await self._usage_gate.snapshot()

    def set_usage_recorder(self, recorder: HostedUsageRecorder | None) -> None:
        """Install or clear the durable usage reservation callback."""
        if self._usage_gate is not None:
            self._usage_gate.set_recorder(recorder)

    async def restore_usage(self, snapshot: HostedUsageSnapshot) -> None:
        """Restore durable hosted usage counters before resumed work."""
        if self._usage_gate is None:
            return
        await self._usage_gate.restore_prior_usage(
            logical_items=snapshot.logical_items,
            provider_attempts=snapshot.provider_attempts,
            submitted_characters=snapshot.submitted_characters,
            ambiguous_attempts=snapshot.ambiguous_attempts,
        )


def _synthesis_payload(request: SynthesisRequest) -> dict[str, object]:
    payload: dict[str, object] = {
        "voice_uuid": request.voice_uuid,
        "data": request.text,
        "precision": request.precision.value,
        "output_format": request.output_format.value,
        "use_hd": request.use_hd,
        "apply_custom_pronunciations": request.apply_custom_pronunciations,
    }
    _optional(payload, "project_uuid", request.project_uuid)
    _optional(payload, "title", request.title)
    _optional(payload, "sample_rate", request.sample_rate)
    return payload


def _stream_payload(request: StreamRequest) -> dict[str, object]:
    payload: dict[str, object] = {
        "voice_uuid": request.voice_uuid,
        "data": request.text,
        "precision": request.precision.value,
        "use_hd": request.use_hd,
        "apply_custom_pronunciations": request.apply_custom_pronunciations,
    }
    _optional(payload, "project_uuid", request.project_uuid)
    _optional(payload, "sample_rate", request.sample_rate)
    return payload


def _optional(payload: dict[str, object], key: str, value: object | None) -> None:
    if value is not None:
        payload[key] = value


def _validate_page(page: int, page_size: int) -> None:
    if page < 1 or page_size < 1 or page_size > MAX_PAGE_SIZE:
        raise ValidationError("page must be >= 1 and page_size must be 1..100")


def _request_id(response: httpx.Response) -> str | None:
    for name in ("x-request-id", "request-id", "resemble-request-id"):
        value = response.headers.get(name)
        if value:
            return value
    return None


def _payload_item(raw: dict[str, object]) -> dict[str, object]:
    item = raw.get("item", raw)
    if not isinstance(item, dict):
        raise ProviderProtocolError("Provider response item must be an object")
    return cast(dict[str, object], item)


def _identifier(item: dict[str, object]) -> str:
    value = item.get("uuid", item.get("id"))
    if not isinstance(value, str) or not value:
        raise ProviderProtocolError("Provider item lacks an identifier")
    return value


def _project(item: dict[str, object]) -> Project:
    return Project(
        uuid=_identifier(item),
        name=str(item.get("name", "")),
        description=_string(item.get("description")),
        is_collaborative=bool(item.get("is_collaborative", False)),
        is_archived=bool(item.get("is_archived", False)),
        created_at=_string(item.get("created_at")),
    )


def _page[T](
    raw: dict[str, object], factory: Callable[[dict[str, object]], T]
) -> Page[T]:
    items_raw = raw.get("items", raw.get("results", []))
    if not isinstance(items_raw, list):
        raise ProviderProtocolError("Provider page items must be an array")
    items = tuple(
        factory(cast(dict[str, object], item))
        for item in items_raw
        if isinstance(item, dict)
    )
    return Page(
        items=items,
        page=_optional_int(raw.get("page")),
        page_count=_optional_int(raw.get("page_count")),
        total_results=_optional_int(raw.get("total_results")),
    )


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, int | str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None
