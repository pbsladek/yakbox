"""Bounded media subprocesses and cancellation-safe thread offloading."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from functools import partial
from typing import BinaryIO

from yakbox.errors import ArtifactError

_DIAGNOSTIC_BYTES = 64 * 1024
_STDOUT_BYTES = 2 * 1024 * 1024


def _kill(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


@dataclass
class _Operation:
    cancelled: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    processes: set[subprocess.Popen[bytes]] = field(default_factory=set)

    def register(self, process: subprocess.Popen[bytes]) -> None:
        with self.lock:
            self.processes.add(process)
            if self.cancelled.is_set():
                _kill(process)

    def cancel(self) -> None:
        self.cancelled.set()
        with self.lock:
            for process in self.processes:
                _kill(process)

    def discard(self, process: subprocess.Popen[bytes]) -> None:
        with self.lock:
            self.processes.discard(process)


_OPERATION: ContextVar[_Operation | None] = ContextVar("media_operation", default=None)


async def run_media[**P, R](
    function: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> R:
    """Stop subprocesses and drain the writer before propagating cancellation."""
    operation = _Operation()
    context = copy_context()
    context.run(_OPERATION.set, operation)
    # An executor Future is not a Task: asyncio.run's shutdown cannot cancel
    # this completion signal while its underlying thread is still writing.
    work = asyncio.get_running_loop().run_in_executor(
        None, partial(context.run, function, *args, **kwargs)
    )
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        operation.cancel()
        await _drain(work)
        raise


async def _drain(work: asyncio.Future[object]) -> None:
    while not work.done():
        try:
            await asyncio.shield(work)
        except asyncio.CancelledError:
            continue
        except Exception:  # noqa: BLE001 - cancellation owns the result; retrieve failures
            break
    if not work.cancelled():
        work.exception()


@dataclass
class _Capture:
    limit: int
    consume: Callable[[str], None] | None = None
    data: bytearray = field(default_factory=bytearray)
    overflow: bool = False
    error: Exception | None = None

    def read(self, stream: BinaryIO) -> None:
        try:
            with stream:
                while chunk := stream.readline(_DIAGNOSTIC_BYTES):
                    self.data.extend(chunk)
                    if len(self.data) > self.limit:
                        self.overflow = True
                        del self.data[: -self.limit]
                    if self.consume is not None:
                        self.consume(chunk.decode("utf-8", errors="replace"))
        except Exception as error:  # noqa: BLE001 - reader boundary forwards failures
            self.error = error

    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")


def run_command(
    command: list[str],
    *,
    timeout: float,
    stderr_line: Callable[[str], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run fixed media argv with bounded capture, timeout, and child cleanup."""
    operation = _OPERATION.get()
    if operation is not None and operation.cancelled.is_set():
        raise ArtifactError("Media operation cancelled")
    with subprocess.Popen(  # noqa: S603 - callers provide fixed tool argv; no shell
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name == "posix",
    ) as process:
        if operation is not None:
            operation.register(process)
        try:
            return _collect(process, command, timeout, stderr_line)
        finally:
            if operation is not None:
                operation.discard(process)


def _collect(
    process: subprocess.Popen[bytes],
    command: list[str],
    timeout: float,
    stderr_line: Callable[[str], None] | None,
) -> subprocess.CompletedProcess[str]:
    deadline = time.monotonic() + timeout
    output = _Capture(_STDOUT_BYTES)
    diagnostic = _Capture(_DIAGNOSTIC_BYTES, stderr_line)
    streams = ((process.stdout, output), (process.stderr, diagnostic))
    readers = [
        threading.Thread(target=capture.read, args=(stream,), daemon=True)
        for stream, capture in streams
        if stream is not None
    ]
    for reader in readers:
        reader.start()
    try:
        process.wait(timeout=timeout)
    except BaseException:
        _kill(process)
        process.wait()
        raise
    finally:
        for reader in readers:
            reader.join(max(0, deadline - time.monotonic()))
        if any(reader.is_alive() for reader in readers):
            _kill(process)
            for reader in readers:
                reader.join()
            raise subprocess.TimeoutExpired(command, timeout)
    for capture in (output, diagnostic):
        if capture.error is not None:
            raise ArtifactError("Cannot capture media tool output") from capture.error
    if output.overflow:
        raise ArtifactError("Media tool response exceeded the output limit")
    return subprocess.CompletedProcess(
        command, process.returncode, output.text(), diagnostic.text()
    )
