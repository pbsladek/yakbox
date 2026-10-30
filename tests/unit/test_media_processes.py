from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from yakbox.audio.measurements import QualityCapture
from yakbox.audio.processes import run_command, run_media
from yakbox.audiobook.journal import target_lock
from yakbox.errors import ArtifactError

_SLEEP_SCRIPT = (
    "import os,pathlib,sys,time;"
    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));"
    "print('ready',file=sys.stderr,flush=True);time.sleep(30)"
)


def test_tool_capture_is_bounded_and_callbacks_receive_the_entire_stream() -> None:
    received = 0

    def consume(line: str) -> None:
        nonlocal received
        received += len(line)

    size = 2_000_000
    value = run_command(
        [
            sys.executable,
            "-c",
            f'import sys;sys.stderr.write("x"*{size});print("done")',
        ],
        timeout=10,
        stderr_line=consume,
    )
    assert value.returncode == 0
    assert value.stdout == "done\n"
    assert len(value.stderr) <= 64 * 1024
    assert received == size


def test_oversized_tool_response_is_rejected() -> None:
    with pytest.raises(ArtifactError, match="output limit"):
        run_command(
            [
                sys.executable,
                "-c",
                'print("x"*3000000)',
            ],
            timeout=10,
        )


def test_timeout_kills_and_reaps_the_child(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(
            [
                sys.executable,
                "-c",
                _SLEEP_SCRIPT,
                str(pid_file),
            ],
            timeout=1,
        )
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


@pytest.mark.asyncio
async def test_cancelling_media_kills_the_child_before_returning(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "pid"
    ready = asyncio.Event()
    loop = asyncio.get_running_loop()

    def on_ready(_line: str) -> None:
        loop.call_soon_threadsafe(ready.set)

    task = asyncio.create_task(
        run_media(
            run_command,
            [
                sys.executable,
                "-c",
                _SLEEP_SCRIPT,
                str(pid_file),
            ],
            timeout=20,
            stderr_line=on_ready,
        )
    )
    async with asyncio.timeout(10):
        await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_a_thread_that_cannot_be_killed() -> None:
    started = asyncio.Event()
    loop = asyncio.get_running_loop()
    release, finished = threading.Event(), threading.Event()

    def work() -> None:
        loop.call_soon_threadsafe(started.set)
        if not release.wait(5):
            raise RuntimeError("Worker was not released")
        finished.set()

    task = asyncio.create_task(run_media(work))
    try:
        async with asyncio.timeout(5):
            await started.wait()
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    finally:
        release.set()


@pytest.mark.asyncio
async def test_cancelled_operation_does_not_start_a_later_subprocess(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    loop = asyncio.get_running_loop()
    release = threading.Event()
    destination = tmp_path / "must-not-exist"

    def work() -> None:
        loop.call_soon_threadsafe(started.set)
        if not release.wait(5):
            raise RuntimeError("Worker was not released")
        run_command(
            [
                sys.executable,
                "-c",
                "import pathlib,sys;pathlib.Path(sys.argv[1]).touch()",
                str(destination),
            ],
            timeout=5,
        )

    task = asyncio.create_task(run_media(work))
    try:
        async with asyncio.timeout(5):
            await started.wait()
        task.cancel()
        await asyncio.sleep(0.01)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not destination.exists()
    finally:
        release.set()


def test_event_loop_shutdown_drains_writers_before_releasing_the_lock(
    tmp_path: Path,
) -> None:
    release = threading.Event()
    observed: list[bool] = []

    async def application() -> None:
        started = asyncio.Event()
        loop = asyncio.get_running_loop()

        def writer() -> None:
            loop.call_soon_threadsafe(started.set)
            if not release.wait(5):
                raise RuntimeError("Writer was not released")
            observed.append((tmp_path / "locks/default.lock").exists())

        async def build() -> None:
            with target_lock(tmp_path, "default"):
                await run_media(writer)

        task = asyncio.create_task(build())
        await started.wait()
        loop.call_later(0.05, release.set)
        assert not task.done()
        # Returning triggers asyncio.run's cancellation of every pending Task.

    try:
        asyncio.run(application())
    finally:
        release.set()
    assert observed == [True]
    assert not (tmp_path / "locks/default.lock").exists()


def test_reader_failure_is_reported_to_the_caller() -> None:
    def reject(_line: str) -> None:
        raise ValueError("bad measurement")

    with pytest.raises(ArtifactError, match="capture media tool output"):
        run_command(
            [sys.executable, "-c", 'import sys;print("x",file=sys.stderr)'],
            timeout=5,
            stderr_line=reject,
        )


def test_inherited_pipes_cannot_outlive_the_command_timeout() -> None:
    script = (
        "import subprocess,sys;"
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        run_command([sys.executable, "-c", script], timeout=1)


def test_quality_edges_survive_truncated_diagnostic_history() -> None:
    capture = QualityCapture()
    prefix = "silence_start: 0\nsilence_end: 0.2\n"
    summary = "silence_start: 9.8\nI: -18 LUFS\nPeak: -3 dBFS\nLRA: 1 LU\n"
    result = run_command(
        [
            sys.executable,
            "-c",
            "import sys;sys.stderr.write(sys.argv[1]);"
            "sys.stderr.write('x'*200000+'\\n');sys.stderr.write(sys.argv[2])",
            prefix,
            summary,
        ],
        timeout=5,
        stderr_line=capture.consume,
    )
    assert "silence_end: 0.2" not in result.stderr
    assert capture.result(10) == pytest.approx((-18, -3, 1, 0.2, 0.2))


@pytest.mark.parametrize("missing", ["I: -18 LUFS", "Peak: -3 dBFS", "LRA: 1 LU"])
def test_quality_capture_rejects_incomplete_successful_output(missing: str) -> None:
    capture = QualityCapture()
    for line in ["I: -18 LUFS", "Peak: -3 dBFS", "LRA: 1 LU"]:
        if line != missing:
            capture.consume(line)
    with pytest.raises(ArtifactError, match="required metric"):
        capture.result(10)
