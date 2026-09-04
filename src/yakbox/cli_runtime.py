"""Lifecycle commands for the opt-in persistent local-model runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, Protocol

import click

from yakbox.audiobook import load_manifest
from yakbox.errors import YakboxError
from yakbox.local_runtime import (
    LocalRuntimeOptions,
    ensure_local_runtime,
    local_runtime_status,
    stop_local_runtime,
)
from yakbox.speech.analysis_runtime_install import (
    AnalysisRuntimeInstaller,
    default_analysis_runtime_root,
)


class _Emit(Protocol):
    def __call__(
        self,
        value: dict[str, object],
        message: str,
        *,
        status: str = "ok",
        exit_code: int = 0,
    ) -> None: ...


_emit_callback: _Emit | None = None
_fail_callback: Callable[[Exception], NoReturn] | None = None


def register_runtime_commands(
    main: click.Group,
    *,
    emit: _Emit,
    fail: Callable[[Exception], NoReturn],
) -> None:
    """Attach persistent-runtime lifecycle commands to the main CLI."""
    global _emit_callback, _fail_callback  # noqa: PLW0603 - CLI composition root
    _emit_callback = emit
    _fail_callback = fail
    main.add_command(runtimes_group)


def _emit(
    value: dict[str, object],
    message: str,
    *,
    status: str = "ok",
    exit_code: int = 0,
) -> None:
    if _emit_callback is None:
        raise RuntimeError("Runtime CLI commands were not registered")
    _emit_callback(value, message, status=status, exit_code=exit_code)


def _fail(error: Exception) -> NoReturn:
    if _fail_callback is None:
        raise RuntimeError("Runtime CLI commands were not registered")
    _fail_callback(error)


@click.group("runtimes")
def runtimes_group() -> None:
    """Install analysis runtimes and manage the local warm process."""


@runtimes_group.command("install")
@click.argument(
    "families",
    type=click.Choice(("whisper", "parakeet", "qwen")),
    nargs=-1,
    required=True,
)
@click.option(
    "--root",
    type=click.Path(path_type=Path),
    default=default_analysis_runtime_root,
    show_default="platform cache directory",
    help="Install into this managed analysis-runtime root.",
)
def runtimes_install_command(families: tuple[str, ...], root: Path) -> None:
    """Explicitly install frozen dependency-family runtimes."""
    try:
        report = AnalysisRuntimeInstaller(root).install(families)
    except (YakboxError, OSError, ValueError) as error:
        _fail(error)
    _emit(
        report.to_dict(),
        f"Installed and verified {len(report.runtimes)} analysis runtime(s)",
    )


@runtimes_group.command("verify")
@click.argument(
    "families",
    type=click.Choice(("whisper", "parakeet", "qwen")),
    nargs=-1,
)
@click.option(
    "--root",
    type=click.Path(path_type=Path),
    default=default_analysis_runtime_root,
    show_default="platform cache directory",
    help="Verify this managed analysis-runtime root.",
)
def runtimes_verify_command(families: tuple[str, ...], root: Path) -> None:
    """Verify immutable runtime identities without installing anything."""
    try:
        requested = families or ("whisper", "parakeet", "qwen")
        report = AnalysisRuntimeInstaller(root).verify(requested)
    except (YakboxError, OSError, ValueError) as error:
        _fail(error)
    _emit(
        report.to_dict(),
        (
            "All requested analysis runtimes are verified"
            if report.verified
            else "One or more analysis runtimes require installation"
        ),
        status="ok" if report.verified else "failed",
        exit_code=0 if report.verified else 1,
    )


@runtimes_group.command("start")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
def runtime_start_command(manifest: Path) -> None:
    """Start or inspect the configured workspace runtime."""
    try:
        loaded = load_manifest(manifest)
        policy = loaded.runtime
        status = asyncio.run(
            ensure_local_runtime(
                loaded.root,
                options=LocalRuntimeOptions(
                    idle_timeout_seconds=policy.idle_timeout_seconds,
                    conditioning_cache_size=policy.conditioning_cache_size,
                    maximum_memory_bytes=policy.maximum_memory_bytes,
                ),
            )
        )
    except (YakboxError, OSError, ValueError) as error:
        _fail(error)
    _emit(status.to_dict(), f"Local runtime is running as process {status.pid}")


@runtimes_group.command("status")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
def runtime_status_command(manifest: Path) -> None:
    """Report active model and bounded-cache health without starting a process."""
    try:
        loaded = load_manifest(manifest)
        status = asyncio.run(local_runtime_status(loaded.root))
    except (YakboxError, OSError, ValueError) as error:
        _fail(error)
    _emit(
        status.to_dict(),
        (
            f"Local runtime is running as process {status.pid}"
            if status.running
            else "Local runtime is stopped"
        ),
    )


@runtimes_group.command("stop")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
def runtime_stop_command(manifest: Path) -> None:
    """Gracefully stop the workspace runtime and release resident models."""
    try:
        loaded = load_manifest(manifest)
        stopped = asyncio.run(stop_local_runtime(loaded.root))
    except (YakboxError, OSError, ValueError) as error:
        _fail(error)
    _emit({"stopped": stopped}, "Local runtime stopped" if stopped else "No runtime")


__all__ = ["register_runtime_commands"]
