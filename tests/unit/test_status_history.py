from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner
from tests.schema_helpers import validate_contract

from yakbox._files import atomic_write_json
from yakbox.audiobook.status import read_run_history
from yakbox.cli import main
from yakbox.contracts import runtime_metadata
from yakbox.errors import ArtifactError, BuildError


def _summary(workspace: Path, run_id: str, target: str, order: int) -> Path:
    path = workspace / ".yakbox" / "runs" / run_id / "run.json"
    atomic_write_json(
        path,
        {
            **runtime_metadata("audiobook-run"),
            "run_id": run_id,
            "target": target,
            "status": "complete",
            "plan_fingerprint": "a" * 64,
        },
    )
    modified = 1_700_000_000_000_000_000 + order
    os.utime(path, ns=(modified, modified))
    return path


@pytest.fixture
def history(book_workspace: Path) -> Path:
    _summary(book_workspace, "old", "default", 0)
    _summary(book_workspace, "new", "default", 2)
    _summary(book_workspace, "other", "draft", 3)
    with (book_workspace / "yakbox.toml").open("a") as stream:
        stream.write(
            '\n[targets.draft]\nextends = "default"\noutput_root = "build/draft"\n'
        )
    return book_workspace


@pytest.mark.parametrize(
    ("limit", "expected"), [(None, ["new", "old"]), (1, ["new"]), (5, ["new", "old"])]
)
def test_history_filters_target_before_applying_limit(
    history: Path, limit: int | None, expected: list[str]
) -> None:
    original = {path: path.read_bytes() for path in history.rglob("run.json")}
    summaries = read_run_history(history, "default", limit=limit)
    assert [summary["run_id"] for summary in summaries] == expected
    for summary in summaries:
        validate_contract("audiobook-run", summary)
    assert {path: path.read_bytes() for path in original} == original
    assert [summary["run_id"] for summary in read_run_history(history, "draft")] == [
        "other"
    ]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ([], ["new", "old"]),
        (["--latest"], ["new"]),
        (["--limit", "2"], ["new", "old"]),
        (["--target", "draft", "--limit", "1"], ["other"]),
    ],
)
def test_cli_status_limits_target_history_and_preserves_inventory(
    history: Path, options: list[str], expected: list[str]
) -> None:
    result = CliRunner().invoke(
        main, ["--json", "status", str(history / "yakbox.toml"), *options]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    assert [summary["run_id"] for summary in payload["data"].pop("runs")] == expected
    assert payload["data"] == {
        "managed_artifacts": 0,
        "managed_bytes": 0,
        "unknown_files": [],
    }


def test_history_limit_does_not_parse_older_summaries(history: Path) -> None:
    old = history / ".yakbox/runs/old/run.json"
    modified = old.stat().st_mtime_ns
    old.write_bytes(b"private invalid summary")
    os.utime(old, ns=(modified, modified))
    assert [
        summary["run_id"] for summary in read_run_history(history, "default", limit=1)
    ] == ["new"]
    with pytest.raises(BuildError, match="Invalid saved run summary"):
        read_run_history(history, "default")


def test_history_uses_run_id_to_break_modification_time_ties(tmp_path: Path) -> None:
    _summary(tmp_path, "a", "default", 0)
    _summary(tmp_path, "b", "default", 0)
    assert [summary["run_id"] for summary in read_run_history(tmp_path, "default")] == [
        "b",
        "a",
    ]


def test_missing_history_is_read_only(tmp_path: Path) -> None:
    assert read_run_history(tmp_path, "default", limit=1) == ()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("limit", [0, -1])
def test_history_rejects_nonpositive_limits(tmp_path: Path, limit: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        read_run_history(tmp_path, "default", limit=limit)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("content", [b"[]", b"{}", b"\xff", b'{"target": "default"}'])
def test_invalid_history_returns_bounded_cli_error(
    history: Path, content: bytes
) -> None:
    path = history / ".yakbox/runs/new/run.json"
    path.write_bytes(content)
    result = CliRunner().invoke(
        main, ["--json", "status", str(history / "yakbox.toml")]
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    assert payload["error"] == {
        "code": "build_error",
        "message": "Invalid saved run summary",
    }
    assert path.read_bytes() == content


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("$schema", "wrong"),
        ("schema_version", 2),
        ("run_id", "other"),
        ("target", None),
    ],
)
def test_history_rejects_invalid_identity(
    tmp_path: Path, field: str, value: object
) -> None:
    path = _summary(tmp_path, "run", "default", 0)
    summary = json.loads(path.read_bytes())
    summary[field] = value
    atomic_write_json(path, summary)
    with pytest.raises(BuildError, match="Invalid saved run summary"):
        read_run_history(tmp_path, "default")


def test_history_does_not_follow_paths_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "book"
    outside = tmp_path / "outside"
    path = _summary(outside, "run", "default", 0)
    runs = workspace / ".yakbox/runs"
    runs.mkdir(parents=True)
    (runs / "run").symlink_to(path.parent, target_is_directory=True)
    with pytest.raises(ArtifactError, match="escapes managed root"):
        read_run_history(workspace, "default")


def test_history_ignores_summary_removed_before_stat(
    history: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    removed = history / ".yakbox/runs/new/run.json"
    original_stat = Path.stat

    def stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path == removed:
            path.unlink(missing_ok=True)
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", stat)
    summaries = read_run_history(history, "default", limit=1)
    assert [summary["run_id"] for summary in summaries] == ["old"]


def test_history_ignores_summary_removed_before_read(
    history: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    removed = history / ".yakbox/runs/new/run.json"
    original_read = Path.read_bytes

    def read(path: Path) -> bytes:
        if path == removed:
            path.unlink()
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    summaries = read_run_history(history, "default", limit=1)
    assert [summary["run_id"] for summary in summaries] == ["old"]
