from __future__ import annotations

import asyncio
import importlib
import json
import shutil
import sys
import threading
from pathlib import Path

import pytest
from click.testing import CliRunner
from tests.schema_helpers import validate_contract

from yakbox.audio.processes import run_command
from yakbox.audiobook import load_manifest
from yakbox.audiobook.build import build_audiobook
from yakbox.audiobook.planner import PlanNode
from yakbox.cli import main
from yakbox.errors import BuildError

build_module = importlib.import_module("yakbox.audiobook.build")


@pytest.mark.parametrize(
    "corruption",
    ["absolute", "escape", "identity", "kind", "dependencies", "schema", "list"],
)
@pytest.mark.asyncio
async def test_build_rejects_redirected_or_mismatched_cached_artifacts(
    book_workspace: Path,
    corruption: str,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    first = await build_audiobook(manifest)
    assert first.status == "complete"
    record = next(item for item in first.artifacts if item.id.endswith(":inspect"))
    sidecar = record.path.with_suffix(record.path.suffix + ".artifact.json")
    outside = book_workspace / "outside-managed-root.json"
    shutil.copyfile(record.path, outside)
    value = json.loads(sidecar.read_text())
    changes = {
        "absolute": ("path", str(outside)),
        "escape": ("path", "../../../outside-managed-root.json"),
        "identity": ("id", "other:inspect"),
        "kind": ("kind", "master"),
        "dependencies": ("dependencies", []),
        "schema": ("schema_version", 99),
    }
    if corruption == "list":
        value = []
    else:
        key, replacement = changes[corruption]
        value[key] = replacement
    sidecar.write_text(json.dumps(value))
    record.path.unlink()
    second = await build_audiobook(manifest)
    assert second.status == "complete"
    assert record.path.is_file()
    assert record.id not in second.reused_nodes
    assert all(
        item.path.is_relative_to(manifest.target("default").output_root)
        for item in second.artifacts
    )


@pytest.mark.asyncio
async def test_cancelled_writer_keeps_target_lock_until_it_finishes(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    started = asyncio.Event()
    loop = asyncio.get_running_loop()
    release, finished = threading.Event(), threading.Event()
    outputs: list[Path] = []

    def blocked_copy(source: Path, destination: Path, **_options: object) -> None:
        outputs.append(destination)
        loop.call_soon_threadsafe(started.set)
        try:
            if not release.wait(10):
                raise RuntimeError("Writer was not released")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        finally:
            finished.set()

    task: asyncio.Task[object] | None = None
    with monkeypatch.context() as media:
        media.setattr(build_module, "master_wav", blocked_copy)
        try:
            task = asyncio.create_task(build_audiobook(manifest))
            async with asyncio.timeout(10):
                await started.wait()
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done()
            assert (book_workspace / ".yakbox/locks/default.lock").exists()
            with pytest.raises(BuildError, match="locked"):
                await build_audiobook(manifest)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
            assert outputs[0].is_file()
            assert not (book_workspace / ".yakbox/locks/default.lock").exists()
        finally:
            release.set()
            if task is not None and not task.done():
                task.cancel()
    resumed = await build_audiobook(manifest)
    assert resumed.resumed and resumed.status == "complete"


@pytest.mark.asyncio
async def test_normalization_runs_off_the_event_loop(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_thread = threading.get_ident()
    original = build_module._normalize_synthesis_chunks
    calls: list[int] = []

    def normalize(
        node: PlanNode,
        chunk_paths: list[Path],
        *,
        preferred_sample_rate: int,
    ) -> tuple[int, int, int]:
        calls.append(threading.get_ident())
        assert threading.get_ident() != main_thread
        return original(node, chunk_paths, preferred_sample_rate=preferred_sample_rate)

    monkeypatch.setattr(build_module, "_normalize_synthesis_chunks", normalize)
    result = await build_audiobook(load_manifest(book_workspace / "yakbox.toml"))
    assert result.status == "complete" and calls


def test_cli_rebuilds_a_report_with_a_redirected_sidecar(book_workspace: Path) -> None:
    runner = CliRunner()
    arguments = ["--json", "build", str(book_workspace / "yakbox.toml")]
    first = runner.invoke(main, arguments)
    assert first.exit_code == 0, first.output
    report = next((book_workspace / "build/yakbox/reports").glob("*.inspection.json"))
    sidecar = report.with_suffix(report.suffix + ".artifact.json")
    outside = book_workspace / "outside.json"
    shutil.copyfile(report, outside)
    value = json.loads(sidecar.read_text())
    value["path"] = str(outside)
    sidecar.write_text(json.dumps(value))
    report.unlink()
    rebuilt = runner.invoke(main, arguments)
    assert rebuilt.exit_code == 0, rebuilt.output
    payload = json.loads(rebuilt.output)
    validate_contract("cli-output", payload)
    assert payload["data"]["status"] == "complete"
    assert report.exists()
    assert json.loads(sidecar.read_text())["path"] != str(outside)


def test_cli_ctrl_c_waits_for_the_writer_and_returns_130(book_workspace: Path) -> None:
    script = """
import importlib, json, os, shutil, signal, sys, threading
from pathlib import Path
from click.testing import CliRunner
from yakbox.cli import main
root = Path(sys.argv[1])
observed = []
def writer(source, destination, **options):
    release = threading.Event()
    threading.Timer(.05, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    threading.Timer(.15, release.set).start()
    if not release.wait(5):
        raise RuntimeError("Writer was not released")
    observed.append((root / ".yakbox/locks/default.lock").exists())
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
importlib.import_module("yakbox.audiobook.build").master_wav = writer
result = CliRunner().invoke(main, ["build", str(root / "yakbox.toml")])
print(json.dumps({"exit_code": result.exit_code, "held_during_write": observed}))
"""
    completed = run_command(
        [sys.executable, "-c", script, str(book_workspace)], timeout=10
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(completed.stdout)
    assert value == {"exit_code": 130, "held_during_write": [True]}
    assert not (book_workspace / ".yakbox/locks/default.lock").exists()
