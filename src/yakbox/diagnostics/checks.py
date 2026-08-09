from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import shutil
import sys
import tempfile
import time
import wave
from pathlib import Path

import httpx

from yakbox import __version__
from yakbox.audiobook.manifest import load_manifest
from yakbox.config import YakboxConfig, load_config
from yakbox.credentials import resolve_resemble_credential
from yakbox.diagnostics.models import (
    Diagnostic,
    DiagnosticSeverity,
    DiagnosticStatus,
    DoctorReport,
)
from yakbox.errors import ConfigurationError, ValidationError
from yakbox.local_alignment import MlxWhisperAligner
from yakbox.speech.short_utterances import ShortUtteranceStrategy
from yakbox.whisper_models import (
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WHISPER_REVISION,
    model_status,
)

MIN_FREE_BYTES = 1_000_000_000


async def run_doctor(
    manifest: Path | None = None,
    *,
    backend: str | None = None,
    target: str | None = None,
    network: bool = False,
    deep: bool = False,
    whisper: bool = False,
    api_key: str | None = None,
) -> DoctorReport:
    """Inspect installation, workspace, and optional backend readiness."""
    checks: list[Diagnostic] = [
        Diagnostic(
            id="python.version",
            status=DiagnosticStatus.PASS,
            severity=DiagnosticSeverity.INFO,
            summary=(
                f"Python {sys.version_info.major}.{sys.version_info.minor} supported"
            ),
            detail=f"yakbox {__version__}",
        ),
        _tool("tool.ffmpeg", "ffmpeg"),
        _tool("tool.ffprobe", "ffprobe"),
    ]
    config, config_check = _load_config_check()
    checks.append(config_check)
    if manifest is not None:
        checks.extend(_workspace_checks(manifest))
    selected_backend = _infer_backend(manifest, target, backend)
    checks.extend(
        await _backend_checks(
            selected_backend,
            config,
            api_key=api_key,
            network=network,
            deep=deep,
        )
    )
    alignment = _infer_alignment(manifest)
    if whisper or deep or alignment is not None:
        model, revision, required = alignment or (
            DEFAULT_WHISPER_MODEL,
            DEFAULT_WHISPER_REVISION,
            False,
        )
        checks.extend(
            await _whisper_checks(
                model,
                revision,
                required=required,
                deep=deep,
            )
        )
    return DoctorReport(schema_version=1, diagnostics=tuple(checks))


def _infer_alignment(path: Path | None) -> tuple[str, str | None, bool] | None:
    if path is None:
        return None
    try:
        policy = load_manifest(path).short_utterances
    except ValidationError:
        return None
    if policy.strategy is not ShortUtteranceStrategy.CONTEXT_EXTRACT:
        return None
    return policy.alignment_model, policy.alignment_revision, True


async def _whisper_checks(
    model: str,
    revision: str | None,
    *,
    required: bool,
    deep: bool,
) -> list[Diagnostic]:
    status = model_status(model, revision)
    failing_status = DiagnosticStatus.FAIL if required else DiagnosticStatus.WARN
    failing_severity = (
        DiagnosticSeverity.ERROR if required else DiagnosticSeverity.WARNING
    )
    checks = [
        Diagnostic(
            id="alignment.whisper.platform",
            status=(
                DiagnosticStatus.PASS
                if sys.platform == "darwin" and platform.machine() == "arm64"
                else failing_status
            ),
            severity=(
                DiagnosticSeverity.INFO
                if sys.platform == "darwin" and platform.machine() == "arm64"
                else failing_severity
            ),
            summary=f"Whisper platform is {sys.platform}/{platform.machine()}",
            action=(
                None
                if sys.platform == "darwin" and platform.machine() == "arm64"
                else "Use Apple Silicon for the supported MLX Whisper runtime"
            ),
        ),
        Diagnostic(
            id="alignment.whisper.package",
            status=(
                DiagnosticStatus.PASS
                if status.mlx_whisper_available
                else failing_status
            ),
            severity=(
                DiagnosticSeverity.INFO
                if status.mlx_whisper_available
                else failing_severity
            ),
            summary=(
                "MLX Whisper package is available"
                if status.mlx_whisper_available
                else "MLX Whisper package is not installed"
            ),
            action=(
                None
                if status.mlx_whisper_available
                else "Install alignment support with: uv sync --extra alignment"
            ),
        ),
        Diagnostic(
            id="alignment.whisper.model",
            status=DiagnosticStatus.PASS if status.verified else failing_status,
            severity=DiagnosticSeverity.INFO if status.verified else failing_severity,
            summary=(
                "Pinned Whisper model verified "
                f"({status.size_bytes / (1024**3):.2f} GiB)"
                if status.verified
                else "Pinned Whisper model is not ready"
            ),
            detail=", ".join(status.issues) or str(status.local_path),
            action=(None if status.verified else "Run: yakbox whisper models install"),
            evidence={
                "model": model,
                "revision": revision,
                "path": str(status.local_path) if status.local_path else None,
                "fingerprint": status.fingerprint,
                "file_count": status.file_count,
            },
        ),
    ]
    free_bytes = shutil.disk_usage(Path.cwd()).free
    checks.append(
        Diagnostic(
            id="alignment.whisper.storage",
            status=(
                DiagnosticStatus.PASS
                if free_bytes >= MIN_FREE_BYTES
                else DiagnosticStatus.WARN
            ),
            severity=(
                DiagnosticSeverity.INFO
                if free_bytes >= MIN_FREE_BYTES
                else DiagnosticSeverity.WARNING
            ),
            summary=f"{free_bytes / (1024**3):.1f} GiB workspace storage available",
            evidence={"free_bytes": free_bytes},
        )
    )
    physical_memory = _physical_memory_bytes()
    required_memory = max(4_000_000_000, status.size_bytes * 2)
    memory_ready = physical_memory is None or physical_memory >= required_memory
    checks.append(
        Diagnostic(
            id="alignment.whisper.memory",
            status=DiagnosticStatus.PASS if memory_ready else DiagnosticStatus.WARN,
            severity=(
                DiagnosticSeverity.INFO if memory_ready else DiagnosticSeverity.WARNING
            ),
            summary=(
                "Physical memory could not be measured"
                if physical_memory is None
                else f"{physical_memory / (1024**3):.1f} GiB physical memory"
            ),
            detail=(
                "The platform does not expose physical page counts"
                if physical_memory is None
                else None
            ),
            action=(
                "Close memory-heavy applications before loading Whisper"
                if not memory_ready
                else None
            ),
            evidence={
                "physical_bytes": physical_memory,
                "estimated_required_bytes": required_memory,
            },
        )
    )
    if deep and status.verified and status.mlx_whisper_available:
        checks.append(await _whisper_deep(model, revision))
    else:
        checks.append(
            Diagnostic(
                id="alignment.whisper.runtime",
                status=DiagnosticStatus.SKIP,
                severity=DiagnosticSeverity.INFO,
                summary="Deep Whisper model-load smoke test not requested",
                action="Run yakbox doctor --whisper --deep",
                skipped_by_policy=True,
            )
        )
    return checks


def _physical_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
    except AttributeError, OSError, ValueError:
        return None
    if not isinstance(page_size, int) or not isinstance(physical_pages, int):
        return None
    return page_size * physical_pages


async def _whisper_deep(model: str, revision: str | None) -> Diagnostic:
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="yakbox-whisper-doctor-") as raw:
            audio = Path(raw) / "silence.wav"
            with wave.open(str(audio), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16_000)
                writer.writeframes(bytes(16_000 // 4 * 2))
            result = await MlxWhisperAligner(
                model=model,
                revision=revision,
                timeout_seconds=180,
                prompted_timing=False,
            ).align(audio, "", language="en")
        return Diagnostic(
            id="alignment.whisper.runtime",
            status=DiagnosticStatus.PASS,
            severity=DiagnosticSeverity.INFO,
            summary="Whisper model loaded and accepted word-timestamp mode",
            elapsed_seconds=time.monotonic() - started,
            evidence={"backend": result.backend, "parser_issues": list(result.issues)},
        )
    except Exception as error:  # noqa: BLE001 - diagnostic must report optional runtime failures.
        return Diagnostic(
            id="alignment.whisper.runtime",
            status=DiagnosticStatus.FAIL,
            severity=DiagnosticSeverity.ERROR,
            summary="Whisper deep runtime smoke test failed",
            detail=_redact(str(error)),
            action="Run yakbox whisper models verify, then inspect the local runtime",
            elapsed_seconds=time.monotonic() - started,
        )


def _load_config_check() -> tuple[YakboxConfig | None, Diagnostic]:
    try:
        config = load_config()
    except ConfigurationError as error:
        return None, Diagnostic(
            id="config.load",
            status=DiagnosticStatus.FAIL,
            severity=DiagnosticSeverity.ERROR,
            summary="Configuration is invalid",
            detail=_redact(str(error)),
        )
    return config, Diagnostic(
        id="config.load",
        status=DiagnosticStatus.PASS,
        severity=DiagnosticSeverity.INFO,
        summary="Configuration loads",
    )


def _infer_backend(
    manifest: Path | None,
    target: str | None,
    backend: str | None,
) -> str | None:
    if backend is not None or manifest is None:
        return backend
    try:
        loaded = load_manifest(manifest)
        selected_target = loaded.target(target or "default")
        return loaded.profile(selected_target.profile).backend
    except ValidationError:
        return None


async def _backend_checks(
    selected_backend: str | None,
    config: YakboxConfig | None,
    *,
    api_key: str | None,
    network: bool,
    deep: bool,
) -> list[Diagnostic]:
    checks: list[Diagnostic] = []
    if selected_backend in {"local", "chatterbox", "chatterbox-local"} or deep:
        installed = importlib.util.find_spec("chatterbox") is not None
        checks.append(
            Diagnostic(
                id="backend.chatterbox.package",
                status=DiagnosticStatus.PASS if installed else DiagnosticStatus.WARN,
                severity=(
                    DiagnosticSeverity.INFO if installed else DiagnosticSeverity.WARNING
                ),
                summary=(
                    "Local Chatterbox package is available"
                    if installed
                    else "Local Chatterbox package is not installed"
                ),
                action=None
                if installed
                else 'Install local support with: uv tool install "yakbox[local]"',
            )
        )
        if deep and installed:
            checks.append(_local_deep())
        elif installed:
            checks.append(
                Diagnostic(
                    id="backend.chatterbox.runtime",
                    status=DiagnosticStatus.SKIP,
                    severity=DiagnosticSeverity.INFO,
                    summary="Deep local runtime check not requested",
                    action="Run yakbox doctor --deep to inspect devices",
                    skipped_by_policy=True,
                )
            )
    if selected_backend in {"resemble", "cloud"} or network:
        credential = resolve_resemble_credential(
            explicit=api_key,
            environment=os.environ.get("RESEMBLE_API_KEY"),
            keyring=None,
            legacy_config=config.legacy_resemble_api_key if config else None,
            profile="default",
        )
        resolved_key = credential.value if credential is not None else None
        has_key = bool(resolved_key)
        checks.append(
            Diagnostic(
                id="backend.resemble.credentials",
                status=DiagnosticStatus.PASS if has_key else DiagnosticStatus.WARN,
                severity=(
                    DiagnosticSeverity.INFO if has_key else DiagnosticSeverity.WARNING
                ),
                summary=(
                    "Resemble credential is present"
                    if has_key
                    else "Resemble credential is not configured"
                ),
                action=None if has_key else "Set RESEMBLE_API_KEY",
            )
        )
        if network and has_key:
            checks.append(await _resemble_network(resolved_key or ""))
        elif not network:
            checks.append(
                Diagnostic(
                    id="backend.resemble.network",
                    status=DiagnosticStatus.SKIP,
                    severity=DiagnosticSeverity.INFO,
                    summary="Network check not requested",
                    action="Run yakbox doctor --network to perform a read-only check",
                    skipped_by_policy=True,
                )
            )
    if selected_backend in {"remote", "chatterbox-remote"}:
        checks.append(
            Diagnostic(
                id="backend.chatterbox_remote.contract",
                status=DiagnosticStatus.SKIP,
                severity=DiagnosticSeverity.WARNING,
                summary="Remote Chatterbox has no verified service contract",
                skipped_by_policy=True,
            )
        )
    return checks


def _tool(identifier: str, name: str) -> Diagnostic:
    path = shutil.which(name)
    return Diagnostic(
        id=identifier,
        status=DiagnosticStatus.PASS if path else DiagnosticStatus.WARN,
        severity=DiagnosticSeverity.INFO if path else DiagnosticSeverity.WARNING,
        summary=f"{name} {'available' if path else 'not found'}",
        detail=path,
        action=None if path else f"Install {name} for mastering and inspection",
    )


def _workspace_checks(path: Path) -> list[Diagnostic]:
    checks: list[Diagnostic] = []
    try:
        manifest = load_manifest(path)
    except ValidationError as error:
        return [
            Diagnostic(
                id="workspace.manifest",
                status=DiagnosticStatus.FAIL,
                severity=DiagnosticSeverity.ERROR,
                summary="Audiobook manifest is invalid",
                detail=str(error),
            )
        ]
    checks.append(
        Diagnostic(
            id="workspace.manifest",
            status=DiagnosticStatus.PASS,
            severity=DiagnosticSeverity.INFO,
            summary="Audiobook manifest is valid",
        )
    )
    unreadable = [
        source for source in manifest.sources if not os.access(source, os.R_OK)
    ]
    checks.append(
        Diagnostic(
            id="workspace.sources",
            status=DiagnosticStatus.FAIL if unreadable else DiagnosticStatus.PASS,
            severity=(
                DiagnosticSeverity.ERROR if unreadable else DiagnosticSeverity.INFO
            ),
            summary=(
                f"{len(unreadable)} source file(s) are unreadable"
                if unreadable
                else f"{len(manifest.sources)} source file(s) are readable"
            ),
            evidence={"unreadable_count": len(unreadable)},
        )
    )
    free = shutil.disk_usage(manifest.root).free
    checks.append(
        Diagnostic(
            id="workspace.storage",
            status=(
                DiagnosticStatus.PASS
                if free >= MIN_FREE_BYTES
                else DiagnosticStatus.WARN
            ),
            severity=(
                DiagnosticSeverity.INFO
                if free >= MIN_FREE_BYTES
                else DiagnosticSeverity.WARNING
            ),
            summary=f"{free / 1_000_000_000:.1f} GB free",
            action="Free disk space before a full build"
            if free < MIN_FREE_BYTES
            else None,
        )
    )
    try:
        with tempfile.NamedTemporaryFile(dir=manifest.root, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(b"yakbox")
            stream.flush()
            os.fsync(stream.fileno())
        target = temporary.with_suffix(".atomic-check")
        temporary.replace(target)
        target.unlink()
    except OSError as error:
        checks.append(
            Diagnostic(
                id="workspace.atomic_write",
                status=DiagnosticStatus.FAIL,
                severity=DiagnosticSeverity.ERROR,
                summary="Workspace atomic writes are unavailable",
                detail=str(error),
            )
        )
    else:
        checks.append(
            Diagnostic(
                id="workspace.atomic_write",
                status=DiagnosticStatus.PASS,
                severity=DiagnosticSeverity.INFO,
                summary="Workspace supports atomic output replacement",
            )
        )
    locks = tuple((manifest.root / ".yakbox" / "locks").glob("*.lock"))
    checks.append(
        Diagnostic(
            id="workspace.locks",
            status=DiagnosticStatus.WARN if locks else DiagnosticStatus.PASS,
            severity=(DiagnosticSeverity.WARNING if locks else DiagnosticSeverity.INFO),
            summary=(
                f"{len(locks)} target lock(s) currently exist"
                if locks
                else "No target lock blocks a build"
            ),
            action="Inspect active builds before removing a stale lock"
            if locks
            else None,
            evidence={"lock_count": len(locks)},
        )
    )
    output_roots = tuple(target.output_root for target in manifest.targets)
    writable = all(
        os.access(_nearest_existing_parent(root), os.W_OK) for root in output_roots
    )
    checks.append(
        Diagnostic(
            id="workspace.outputs",
            status=DiagnosticStatus.PASS if writable else DiagnosticStatus.FAIL,
            severity=(
                DiagnosticSeverity.INFO if writable else DiagnosticSeverity.ERROR
            ),
            summary=(
                "Artifact output roots are writable"
                if writable
                else "An artifact output root is not writable"
            ),
            evidence={"target_count": len(output_roots)},
        )
    )
    return checks


async def _resemble_network(api_key: str) -> Diagnostic:
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        ) as client:
            response = await client.get(
                "https://app.resemble.ai/api/v2/voices",
                params={"page": 1, "page_size": 1},
            )
    except httpx.HTTPError as error:
        return Diagnostic(
            id="backend.resemble.network",
            status=DiagnosticStatus.FAIL,
            severity=DiagnosticSeverity.ERROR,
            summary="Resemble read-only connectivity check failed",
            detail=_redact(str(error), api_key),
            elapsed_seconds=time.monotonic() - started,
        )
    return Diagnostic(
        id="backend.resemble.network",
        status=(
            DiagnosticStatus.PASS if response.is_success else DiagnosticStatus.FAIL
        ),
        severity=(
            DiagnosticSeverity.INFO if response.is_success else DiagnosticSeverity.ERROR
        ),
        summary=(
            "Resemble read-only connectivity check passed"
            if response.is_success
            else f"Resemble returned HTTP {response.status_code}"
        ),
        elapsed_seconds=time.monotonic() - started,
        evidence={
            "method": "GET",
            "resource": "voices",
            "status_code": response.status_code,
            "mutating": False,
        },
    )


def _local_deep() -> Diagnostic:
    started = time.monotonic()
    try:
        torch = importlib.import_module("torch")
        cuda = getattr(torch, "cuda", None)
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None)
        cuda_available = bool(cuda and cuda.is_available())
        mps_available = bool(mps and mps.is_available())
        version = str(getattr(torch, "__version__", "unknown"))
    except (ImportError, AttributeError, RuntimeError) as error:
        return Diagnostic(
            id="backend.chatterbox.runtime",
            status=DiagnosticStatus.FAIL,
            severity=DiagnosticSeverity.ERROR,
            summary="Local runtime/device inspection failed",
            detail=_redact(str(error)),
            elapsed_seconds=time.monotonic() - started,
        )
    return Diagnostic(
        id="backend.chatterbox.runtime",
        status=DiagnosticStatus.PASS,
        severity=DiagnosticSeverity.INFO,
        summary="Local runtime is importable without loading a model",
        elapsed_seconds=time.monotonic() - started,
        evidence={
            "torch_version": version,
            "cuda_available": cuda_available,
            "mps_available": mps_available,
            "model_loaded": False,
        },
    )


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        candidate = candidate.parent
    return candidate


def _redact(value: str, secret: str | None = None) -> str:
    redacted = value.replace(secret, "[REDACTED]") if secret else value
    return redacted[:2_048]
