"""Workspace-managed persistent runtime for local speech and alignment models."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_json, safe_child
from yakbox.audio.crop import SpeechRegion
from yakbox.contracts import runtime_metadata, schema_uri
from yakbox.errors import BuildError, ValidationError
from yakbox.speech.accelerator import AcceleratorLease, accelerator_operation
from yakbox.speech.alignment import (
    AlignmentResult,
    AlignmentSegment,
    AlignmentToken,
    DecodePassEvidence,
    WindowSpeechAligner,
)
from yakbox.speech.capabilities import BackendCapabilities
from yakbox.speech.models import (
    AudioFormat,
    ChatterboxSynthesisOptions,
    SpeechArtifact,
    SpeechSynthesisRequest,
)
from yakbox.speech.services import TextToSpeechService

RUNTIME_PROTOCOL_VERSION = 1
_START_TIMEOUT_SECONDS = 15.0
_CONNECT_TIMEOUT_SECONDS = 5.0
_MAXIMUM_MESSAGE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class LocalRuntimeOptions:
    """Lifecycle and bounded-cache settings for one workspace runtime."""

    idle_timeout_seconds: float = 900.0
    conditioning_cache_size: int = 8
    maximum_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.idle_timeout_seconds <= 0:
            raise ValidationError("Runtime idle timeout must be positive")
        if self.conditioning_cache_size < 1:
            raise ValidationError("Runtime conditioning cache size must be positive")
        if self.maximum_memory_bytes is not None and self.maximum_memory_bytes < 1:
            raise ValidationError("Runtime memory limit must be positive")


@dataclass(frozen=True, slots=True)
class LocalRuntimeStatus:
    """Observable health and resident-cache state for a managed runtime."""

    running: bool
    pid: int | None = None
    port: int | None = None
    idle_timeout_seconds: float | None = None
    idle_seconds: float | None = None
    tts_model_loaded: bool = False
    whisper_models: int = 0
    conditioning_cache_entries: int = 0
    resident_memory_bytes: int | None = None
    maximum_memory_bytes: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize runtime health without exposing its authentication token."""
        return {
            **runtime_metadata("local-runtime-status"),
            **asdict(self),
        }


class PersistentLocalSpeechService:
    """Send local Chatterbox work to a workspace-managed persistent runtime."""

    capabilities = BackendCapabilities(
        name="chatterbox-local",
        synthesis=True,
        transformation=False,
        streaming=False,
        hosted=False,
        output_formats=("wav",),
        supports_reference_voice=True,
    )

    def __init__(
        self,
        workspace: Path,
        *,
        device: str,
        options: LocalRuntimeOptions,
        request_timeout_seconds: float,
        accelerator_lease: AcceleratorLease | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.device = device
        self.options = options
        self.request_timeout_seconds = request_timeout_seconds
        self.accelerator_lease = accelerator_lease

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        """Synthesize one request through the persistent runtime."""
        values = await self.synthesize_many_to_files(
            ((request, destination),),
            overwrite=overwrite,
        )
        return values[0]

    async def synthesize_many_to_files(
        self,
        requests: tuple[tuple[SpeechSynthesisRequest, Path], ...],
        *,
        overwrite: bool = False,
    ) -> tuple[SpeechArtifact, ...]:
        """Synthesize an ordered request batch without reloading local models."""
        if not requests:
            return ()
        items = []
        for request, destination in requests:
            resolved_destination = safe_child(self.workspace, destination)
            resolved_destination.parent.mkdir(parents=True, exist_ok=True)
            reference = (
                safe_child(self.workspace, request.reference_audio)
                if request.reference_audio is not None
                else None
            )
            items.append(
                {
                    "text": request.text,
                    "voice": request.voice,
                    "destination": str(resolved_destination),
                    "sample_rate": request.sample_rate,
                    "reference_audio": str(reference) if reference else None,
                    "chatterbox": (
                        asdict(request.chatterbox)
                        if request.chatterbox is not None
                        else None
                    ),
                }
            )
        async with accelerator_operation(
            self.accelerator_lease,
            owner="tts:chatterbox",
            enabled=self.device.casefold() != "cpu",
        ):
            response = await runtime_request(
                self.workspace,
                {
                    "operation": "synthesize_many",
                    "device": self.device,
                    "overwrite": overwrite,
                    "items": items,
                },
                options=self.options,
                timeout_seconds=self.request_timeout_seconds,
                ensure=True,
            )
        values = response.get("items")
        if not isinstance(values, list) or len(values) != len(items):
            raise BuildError("Persistent runtime returned an invalid synthesis batch")
        return tuple(_artifact_from_runtime(item) for item in values)

    async def aclose(self) -> None:
        """Release the client while intentionally leaving the runtime warm."""


class PersistentMlxWhisperAligner:
    """Run MLX Whisper inside the workspace-managed persistent runtime."""

    def __init__(
        self,
        workspace: Path,
        *,
        model: str,
        revision: str | None,
        timeout_seconds: float,
        prompted_timing: bool,
        decode_consensus: bool,
        prompt_sensitivity: bool,
        maximum_consensus_timing_delta_ms: int,
        hallucination_silence_threshold: float,
        runtime_options: LocalRuntimeOptions,
    ) -> None:
        from yakbox.local_alignment import MlxWhisperAligner  # noqa: PLC0415 - lazy

        self.workspace = workspace.resolve()
        self.model = model
        self.revision = revision
        self.timeout_seconds = timeout_seconds
        self.prompted_timing = prompted_timing
        self.decode_consensus = decode_consensus
        self.prompt_sensitivity = prompt_sensitivity
        self.maximum_consensus_timing_delta_ms = maximum_consensus_timing_delta_ms
        self.hallucination_silence_threshold = hallucination_silence_threshold
        self.runtime_options = runtime_options
        self._fingerprint = MlxWhisperAligner(
            model=model,
            revision=revision,
            timeout_seconds=timeout_seconds,
            prompted_timing=prompted_timing,
            decode_consensus=decode_consensus,
            prompt_sensitivity=prompt_sensitivity,
            maximum_consensus_timing_delta_ms=maximum_consensus_timing_delta_ms,
            hallucination_silence_threshold=hallucination_silence_threshold,
        ).fingerprint

    @property
    def fingerprint(self) -> str:
        """Return the same stable identity as the direct MLX adapter."""
        return self._fingerprint

    async def align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
    ) -> AlignmentResult:
        """Align a complete workspace WAV through the persistent runtime."""
        return await self._align(
            audio,
            expected_text,
            language=language,
            start_seconds=None,
            end_seconds=None,
        )

    async def align_window(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
        start_seconds: float,
        end_seconds: float,
    ) -> AlignmentResult:
        """Align one bounded WAV range through the persistent runtime."""
        return await self._align(
            audio,
            expected_text,
            language=language,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )

    async def _align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
        start_seconds: float | None,
        end_seconds: float | None,
    ) -> AlignmentResult:
        response = await runtime_request(
            self.workspace,
            {
                "operation": "align",
                "audio": str(safe_child(self.workspace, audio)),
                "expected_text": expected_text,
                "language": language,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "aligner": {
                    "model": self.model,
                    "revision": self.revision,
                    "timeout_seconds": self.timeout_seconds,
                    "prompted_timing": self.prompted_timing,
                    "decode_consensus": self.decode_consensus,
                    "prompt_sensitivity": self.prompt_sensitivity,
                    "maximum_consensus_timing_delta_ms": (
                        self.maximum_consensus_timing_delta_ms
                    ),
                    "hallucination_silence_threshold": (
                        self.hallucination_silence_threshold
                    ),
                },
            },
            options=self.runtime_options,
            timeout_seconds=self.timeout_seconds + 10,
            ensure=True,
        )
        alignment = response.get("alignment")
        if not isinstance(alignment, dict):
            raise BuildError("Persistent runtime returned invalid alignment evidence")
        return _alignment_from_dict(cast(dict[str, object], alignment))


async def ensure_local_runtime(
    workspace: Path,
    *,
    options: LocalRuntimeOptions | None = None,
) -> LocalRuntimeStatus:
    """Return a healthy runtime, starting one safely when needed."""
    root = workspace.resolve()
    resolved_options = options or LocalRuntimeOptions()
    current = await local_runtime_status(root)
    if current.running:
        return current
    runtime_root = _runtime_root(root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    lock = runtime_root / "starting.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            return await _wait_for_runtime(root, timeout_seconds=2.0)
        except BuildError:
            if not _stale_start_lock(lock):
                raise
            lock.unlink(missing_ok=True)
            return await ensure_local_runtime(root, options=resolved_options)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        log = (runtime_root / "runtime.log").open("ab")
        try:
            arguments = [
                sys.executable,
                "-m",
                "yakbox.local_runtime",
                "serve",
                "--workspace",
                str(root),
                "--idle-timeout",
                str(resolved_options.idle_timeout_seconds),
                "--conditioning-cache-size",
                str(resolved_options.conditioning_cache_size),
            ]
            if resolved_options.maximum_memory_bytes is not None:
                arguments.extend(
                    (
                        "--maximum-memory-bytes",
                        str(resolved_options.maximum_memory_bytes),
                    )
                )
            await asyncio.to_thread(
                subprocess.Popen,
                arguments,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            log.close()
        return await _wait_for_runtime(root)
    finally:
        lock.unlink(missing_ok=True)


async def local_runtime_status(workspace: Path) -> LocalRuntimeStatus:
    """Read and actively verify the workspace runtime endpoint."""
    endpoint = _read_endpoint(workspace.resolve())
    if endpoint is None:
        return LocalRuntimeStatus(running=False)
    try:
        response = await _send_endpoint(
            endpoint,
            {"operation": "status"},
            timeout_seconds=_CONNECT_TIMEOUT_SECONDS,
        )
    except BuildError, OSError, TimeoutError:
        return LocalRuntimeStatus(running=False)
    return _status_from_response(response)


async def stop_local_runtime(workspace: Path) -> bool:
    """Request graceful shutdown of the managed runtime when it is running."""
    endpoint = _read_endpoint(workspace.resolve())
    if endpoint is None:
        return False
    try:
        await _send_endpoint(
            endpoint,
            {"operation": "stop"},
            timeout_seconds=_CONNECT_TIMEOUT_SECONDS,
        )
    except BuildError, OSError, TimeoutError:
        return False
    return True


async def runtime_request(
    workspace: Path,
    request: dict[str, object],
    *,
    options: LocalRuntimeOptions,
    timeout_seconds: float,
    ensure: bool,
) -> dict[str, object]:
    """Send one authenticated request to a managed runtime."""
    root = workspace.resolve()
    if ensure:
        await ensure_local_runtime(root, options=options)
    endpoint = _read_endpoint(root)
    if endpoint is None:
        raise BuildError("Persistent local runtime has no endpoint")
    return await _send_endpoint(endpoint, request, timeout_seconds=timeout_seconds)


async def _wait_for_runtime(
    workspace: Path,
    *,
    timeout_seconds: float = _START_TIMEOUT_SECONDS,
) -> LocalRuntimeStatus:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = await local_runtime_status(workspace)
        if status.running:
            return status
        await asyncio.sleep(0.05)
    raise BuildError("Persistent local runtime did not become healthy")


def _stale_start_lock(path: Path) -> bool:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except OSError, ValueError:
        return True
    if pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError, PermissionError:
        return False
    return False


def _runtime_root(workspace: Path) -> Path:
    return workspace / ".yakbox" / "runtime"


def _endpoint_path(workspace: Path) -> Path:
    return _runtime_root(workspace) / "endpoint.json"


def _read_endpoint(workspace: Path) -> dict[str, object] | None:
    path = _endpoint_path(workspace)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    if (
        not isinstance(raw, dict)
        or raw.get("$schema") != schema_uri("local-runtime-endpoint")
        or raw.get("protocol_version") != RUNTIME_PROTOCOL_VERSION
        or raw.get("workspace") != str(workspace)
        or not isinstance(raw.get("token"), str)
        or not isinstance(raw.get("port"), int)
    ):
        return None
    return cast(dict[str, object], raw)


async def _send_endpoint(
    endpoint: dict[str, object],
    request: dict[str, object],
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    async def exchange() -> dict[str, object]:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", int(cast(int, endpoint["port"]))
        )
        try:
            payload = {
                "protocol_version": RUNTIME_PROTOCOL_VERSION,
                "token": endpoint["token"],
                **request,
            }
            writer.write(json.dumps(payload, ensure_ascii=False).encode() + b"\n")
            await writer.drain()
            line = await reader.readline()
            if not line or len(line) > _MAXIMUM_MESSAGE_BYTES:
                raise BuildError("Persistent runtime returned no valid response")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BuildError("Persistent runtime returned invalid JSON")
            response = cast(dict[str, object], value)
            if response.get("protocol_version") != RUNTIME_PROTOCOL_VERSION:
                raise BuildError("Unsupported persistent runtime protocol")
            if response.get("ok") is not True:
                raise BuildError(str(response.get("error", "runtime request failed")))
            return response
        finally:
            writer.close()
            await writer.wait_closed()

    try:
        return await asyncio.wait_for(exchange(), timeout=timeout_seconds)
    except (ConnectionError, OSError, json.JSONDecodeError, TimeoutError) as error:
        raise BuildError(f"Persistent local runtime request failed: {error}") from error


class _RuntimeServer:
    def __init__(self, workspace: Path, options: LocalRuntimeOptions) -> None:
        self.workspace = workspace.resolve()
        self.options = options
        self.token = secrets.token_urlsafe(32)
        self.started = time.monotonic()
        self.last_activity = self.started
        self.stop_event = asyncio.Event()
        self.operation_lock = asyncio.Lock()
        self.tts_services: dict[str, TextToSpeechService] = {}
        self.aligners: dict[str, WindowSpeechAligner] = {}
        self.port: int | None = None

    async def serve(self) -> None:
        server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        socket = server.sockets[0]
        port = int(socket.getsockname()[1])
        self.port = port
        path = _endpoint_path(self.workspace)
        atomic_write_json(
            path,
            {
                **runtime_metadata("local-runtime-endpoint"),
                "protocol_version": RUNTIME_PROTOCOL_VERSION,
                "workspace": str(self.workspace),
                "pid": os.getpid(),
                "port": port,
                "token": self.token,
                "idle_timeout_seconds": self.options.idle_timeout_seconds,
            },
        )
        path.chmod(0o600)
        try:
            while not self.stop_event.is_set():
                remaining = self.options.idle_timeout_seconds - (
                    time.monotonic() - self.last_activity
                )
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=remaining)
                except TimeoutError:
                    break
        finally:
            server.close()
            await server.wait_closed()
            path.unlink(missing_ok=True)

    async def handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            line = await reader.readline()
            if not line or len(line) > _MAXIMUM_MESSAGE_BYTES:
                raise ValidationError("Runtime request is empty or too large")
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValidationError("Runtime request must be an object")
            request = cast(dict[str, object], raw)
            if request.get("protocol_version") != RUNTIME_PROTOCOL_VERSION:
                raise ValidationError("Unsupported runtime protocol")
            token = request.get("token")
            if not isinstance(token, str) or not secrets.compare_digest(
                token, self.token
            ):
                raise ValidationError("Runtime authentication failed")
            self.last_activity = time.monotonic()
            operation = request.get("operation")
            if operation == "status":
                response = self.status()
            elif operation == "stop":
                response = {"stopping": True}
                self.stop_event.set()
            else:
                async with self.operation_lock:
                    response = await self.execute(request)
                    self.last_activity = time.monotonic()
            await self._write(writer, {"ok": True, **response})
        except Exception as error:  # noqa: BLE001 - protocol returns safe errors
            await self._write(
                writer,
                {"ok": False, "error": f"{type(error).__name__}: runtime failed"},
            )
        finally:
            writer.close()
            await writer.wait_closed()

    async def execute(self, request: dict[str, object]) -> dict[str, object]:
        self._enforce_memory_limit()
        operation = request.get("operation")
        if operation == "synthesize_many":
            response = await self._synthesize(request)
        elif operation == "align":
            response = await self._align(request)
        else:
            raise ValidationError("Unsupported runtime operation")
        if self._memory_limit_exceeded():
            self.stop_event.set()
        return response

    async def _synthesize(self, request: dict[str, object]) -> dict[str, object]:
        from yakbox.local import LocalChatterboxService  # noqa: PLC0415 - heavy

        device = str(request.get("device", "cpu"))
        service = self.tts_services.get(device)
        if service is None:
            if self.tts_services:
                raise ValidationError("Runtime supports one resident TTS device")
            service = LocalChatterboxService(
                device=device,
                conditioning_cache_size=self.options.conditioning_cache_size,
            )
            self.tts_services[device] = service
        values = request.get("items")
        if not isinstance(values, list) or not values:
            raise ValidationError("Runtime synthesis batch must not be empty")
        artifacts = []
        for value in values:
            item = _runtime_item(self.workspace, value)
            artifact = await service.synthesize_to_file(
                item[0], item[1], overwrite=bool(request.get("overwrite", False))
            )
            artifacts.append(_artifact_to_runtime(artifact))
        return {"items": artifacts}

    async def _align(self, request: dict[str, object]) -> dict[str, object]:
        from yakbox.local_alignment import open_local_aligner  # noqa: PLC0415 - heavy

        raw = request.get("aligner")
        if not isinstance(raw, dict):
            raise ValidationError("Runtime aligner configuration is invalid")
        config = cast(dict[str, object], raw)
        key = json.dumps(config, sort_keys=True, separators=(",", ":"))
        aligner = self.aligners.get(key)
        if aligner is None:
            aligner = open_local_aligner(
                "mlx-whisper",
                model=str(config["model"]),
                revision=_optional_string(config.get("revision")),
                timeout_seconds=float(cast(float, config["timeout_seconds"])),
                prompted_timing=bool(config["prompted_timing"]),
                decode_consensus=bool(config["decode_consensus"]),
                prompt_sensitivity=bool(config["prompt_sensitivity"]),
                maximum_consensus_timing_delta_ms=int(
                    cast(int, config["maximum_consensus_timing_delta_ms"])
                ),
                hallucination_silence_threshold=float(
                    cast(float, config["hallucination_silence_threshold"])
                ),
            )
            self.aligners[key] = aligner
        audio = safe_child(self.workspace, Path(str(request["audio"])))
        expected = str(request.get("expected_text", ""))
        language = str(request.get("language", "en"))
        start = request.get("start_seconds")
        end = request.get("end_seconds")
        if start is None or end is None:
            result = await aligner.align(audio, expected, language=language)
        else:
            result = await aligner.align_window(
                audio,
                expected,
                language=language,
                start_seconds=float(cast(float, start)),
                end_seconds=float(cast(float, end)),
            )
        return {"alignment": _alignment_to_dict(result)}

    def status(self) -> dict[str, object]:
        service = next(iter(self.tts_services.values()), None)
        return {
            "running": True,
            "pid": os.getpid(),
            "port": self.port,
            "idle_timeout_seconds": self.options.idle_timeout_seconds,
            "idle_seconds": max(0.0, time.monotonic() - self.last_activity),
            "tts_model_loaded": bool(
                service is not None and getattr(service, "model_loaded", False)
            ),
            "whisper_models": len(self.aligners),
            "conditioning_cache_entries": (
                int(getattr(service, "conditioning_cache_entries", 0))
                if service is not None
                else 0
            ),
            "resident_memory_bytes": _resident_memory_bytes(),
            "maximum_memory_bytes": self.options.maximum_memory_bytes,
        }

    def _enforce_memory_limit(self) -> None:
        if self._memory_limit_exceeded():
            self.stop_event.set()
            raise BuildError("Persistent runtime exceeded its memory limit")

    def _memory_limit_exceeded(self) -> bool:
        maximum = self.options.maximum_memory_bytes
        current = _resident_memory_bytes()
        return maximum is not None and current is not None and current > maximum

    async def _write(
        self, writer: asyncio.StreamWriter, payload: dict[str, object]
    ) -> None:
        writer.write(
            json.dumps(
                {"protocol_version": RUNTIME_PROTOCOL_VERSION, **payload},
                ensure_ascii=False,
            ).encode()
            + b"\n"
        )
        await writer.drain()


def _runtime_item(
    workspace: Path, value: object
) -> tuple[SpeechSynthesisRequest, Path]:
    if not isinstance(value, dict):
        raise ValidationError("Runtime synthesis item is invalid")
    raw = cast(dict[str, object], value)
    destination = safe_child(workspace, Path(str(raw["destination"])))
    reference_value = raw.get("reference_audio")
    reference = (
        safe_child(workspace, Path(str(reference_value)))
        if reference_value is not None
        else None
    )
    chatterbox_raw = raw.get("chatterbox")
    chatterbox = None
    if chatterbox_raw is not None:
        if not isinstance(chatterbox_raw, dict):
            raise ValidationError("Runtime Chatterbox options are invalid")
        values = cast(dict[str, object], chatterbox_raw)
        chatterbox = ChatterboxSynthesisOptions(
            cfg_weight=_optional_float(values.get("cfg_weight")),
            exaggeration=_optional_float(values.get("exaggeration")),
            seed=_optional_int(values.get("seed")),
        )
    return (
        SpeechSynthesisRequest(
            text=str(raw["text"]),
            voice=str(raw["voice"]),
            backend="chatterbox-local",
            output_format=AudioFormat.WAV,
            sample_rate=_optional_int(raw.get("sample_rate")),
            reference_audio=reference,
            chatterbox=chatterbox,
        ),
        destination,
    )


def _artifact_to_runtime(artifact: SpeechArtifact) -> dict[str, object]:
    return {
        "path": str(artifact.path),
        "backend": artifact.backend,
        "voice": artifact.voice,
        "output_format": artifact.output_format.value,
        "bytes_written": artifact.bytes_written,
        "sha256": artifact.sha256,
        "duration_seconds": artifact.duration_seconds,
        "sample_rate": artifact.sample_rate,
    }


def _artifact_from_runtime(value: object) -> SpeechArtifact:
    if not isinstance(value, dict):
        raise BuildError("Persistent runtime synthesis item is invalid")
    raw = cast(dict[str, object], value)
    try:
        return SpeechArtifact(
            path=Path(str(raw["path"])).resolve(),
            backend=str(raw["backend"]),
            voice=str(raw["voice"]),
            output_format=AudioFormat(str(raw["output_format"])),
            bytes_written=int(cast(int, raw["bytes_written"])),
            sha256=str(raw["sha256"]),
            duration_seconds=_optional_float(raw.get("duration_seconds")),
            sample_rate=_optional_int(raw.get("sample_rate")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BuildError("Persistent runtime synthesis result is invalid") from error


def _alignment_to_dict(result: AlignmentResult) -> dict[str, object]:
    return {
        "tokens": [asdict(item) for item in result.tokens],
        "speech_regions": [asdict(item) for item in result.speech_regions],
        "backend": result.backend,
        "model": result.model,
        "fingerprint": result.fingerprint,
        "segments": [asdict(item) for item in result.segments],
        "issues": list(result.issues),
        "language": result.language,
        "timing_source": result.timing_source,
        "transcript": result.transcript,
        "decode_passes": [asdict(item) for item in result.decode_passes],
        "consensus_score": result.consensus_score,
        "maximum_timing_delta_ms": result.maximum_timing_delta_ms,
        "consensus_reason_codes": list(result.consensus_reason_codes),
        "prompt_sensitivity": result.prompt_sensitivity,
        "clip_start_seconds": result.clip_start_seconds,
        "clip_end_seconds": result.clip_end_seconds,
    }


def _alignment_from_dict(raw: dict[str, object]) -> AlignmentResult:
    try:
        token_values = _object_list(raw["tokens"])
        region_values = _object_list(raw["speech_regions"])
        segment_values = _object_list(raw["segments"])
        pass_values = _object_list(raw["decode_passes"])
        return AlignmentResult(
            tokens=tuple(_alignment_token(item) for item in token_values),
            speech_regions=tuple(_speech_region(item) for item in region_values),
            backend=str(raw["backend"]),
            model=str(raw["model"]),
            fingerprint=str(raw["fingerprint"]),
            segments=tuple(_alignment_segment(item) for item in segment_values),
            issues=tuple(cast(list[str], raw["issues"])),
            language=_optional_string(raw.get("language")),
            timing_source=str(raw["timing_source"]),
            transcript=str(raw.get("transcript", "")),
            decode_passes=tuple(_decode_pass(item) for item in pass_values),
            consensus_score=_optional_float(raw.get("consensus_score")),
            maximum_timing_delta_ms=_optional_float(raw.get("maximum_timing_delta_ms")),
            consensus_reason_codes=tuple(
                cast(list[str], raw["consensus_reason_codes"])
            ),
            prompt_sensitivity=str(raw["prompt_sensitivity"]),
            clip_start_seconds=_optional_float(raw.get("clip_start_seconds")),
            clip_end_seconds=_optional_float(raw.get("clip_end_seconds")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BuildError("Persistent runtime alignment result is invalid") from error


def _speech_region(value: object) -> SpeechRegion:
    if not isinstance(value, dict):
        raise TypeError
    raw = cast(dict[str, object], value)
    return SpeechRegion(
        start_seconds=float(cast(float, raw["start_seconds"])),
        end_seconds=float(cast(float, raw["end_seconds"])),
    )


def _alignment_token(value: object) -> AlignmentToken:
    if not isinstance(value, dict):
        raise TypeError
    raw = cast(dict[str, object], value)
    return AlignmentToken(
        text=str(raw["text"]),
        start_seconds=float(cast(float, raw["start_seconds"])),
        end_seconds=float(cast(float, raw["end_seconds"])),
        confidence=_optional_float(raw.get("confidence")),
    )


def _alignment_segment(value: object) -> AlignmentSegment:
    if not isinstance(value, dict):
        raise TypeError
    raw = cast(dict[str, object], value)
    return AlignmentSegment(
        start_seconds=float(cast(float, raw["start_seconds"])),
        end_seconds=float(cast(float, raw["end_seconds"])),
        average_log_probability=_optional_float(raw.get("average_log_probability")),
        compression_ratio=_optional_float(raw.get("compression_ratio")),
        no_speech_probability=_optional_float(raw.get("no_speech_probability")),
        temperature=_optional_float(raw.get("temperature")),
    )


def _decode_pass(value: object) -> DecodePassEvidence:
    if not isinstance(value, dict):
        raise TypeError
    raw = cast(dict[str, object], value)
    return DecodePassEvidence(
        name=str(raw["name"]),
        transcript=str(raw["transcript"]),
        tokens=tuple(str(item) for item in _object_list(raw["tokens"])),
        issues=tuple(str(item) for item in _object_list(raw["issues"])),
        minimum_confidence=_optional_float(raw.get("minimum_confidence")),
        matches_expected=_optional_bool(raw.get("matches_expected")),
    )


def _object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError
    return cast(list[object], value)


def _status_from_response(response: dict[str, object]) -> LocalRuntimeStatus:
    return LocalRuntimeStatus(
        running=bool(response.get("running")),
        pid=_optional_int(response.get("pid")),
        port=_optional_int(response.get("port")),
        idle_timeout_seconds=_optional_float(response.get("idle_timeout_seconds")),
        idle_seconds=_optional_float(response.get("idle_seconds")),
        tts_model_loaded=bool(response.get("tts_model_loaded")),
        whisper_models=int(cast(int, response.get("whisper_models", 0))),
        conditioning_cache_entries=int(
            cast(int, response.get("conditioning_cache_entries", 0))
        ),
        resident_memory_bytes=_optional_int(response.get("resident_memory_bytes")),
        maximum_memory_bytes=_optional_int(response.get("maximum_memory_bytes")),
    )


def _resident_memory_bytes() -> int | None:
    try:
        import resource  # noqa: PLC0415 - unavailable on Windows

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError, OSError:
        return None
    return int(value * (1 if sys.platform == "darwin" else 1_024))


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError
    return float(value)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError
    return value


async def _serve_from_arguments(arguments: argparse.Namespace) -> None:
    server = _RuntimeServer(
        Path(arguments.workspace),
        LocalRuntimeOptions(
            idle_timeout_seconds=arguments.idle_timeout,
            conditioning_cache_size=arguments.conditioning_cache_size,
            maximum_memory_bytes=arguments.maximum_memory_bytes,
        ),
    )
    loop = asyncio.get_running_loop()
    for name in ("SIGTERM", "SIGINT"):
        event = getattr(signal, name, None)
        if event is not None:
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(event, server.stop_event.set)
    await server.serve()


def main() -> None:
    """Run the private persistent-runtime server entry point."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("operation", choices=("serve",))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--idle-timeout", type=float, default=900.0)
    parser.add_argument("--conditioning-cache-size", type=int, default=8)
    parser.add_argument("--maximum-memory-bytes", type=int)
    arguments = parser.parse_args()
    asyncio.run(_serve_from_arguments(arguments))


if __name__ == "__main__":
    main()


__all__ = [
    "LocalRuntimeOptions",
    "LocalRuntimeStatus",
    "PersistentLocalSpeechService",
    "PersistentMlxWhisperAligner",
    "ensure_local_runtime",
    "local_runtime_status",
    "stop_local_runtime",
]
