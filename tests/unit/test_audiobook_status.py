from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from yakbox._files import atomic_write_json
from yakbox.audiobook.journal import RunJournal, target_lock
from yakbox.audiobook.status import read_run_progress
from yakbox.contracts import runtime_metadata
from yakbox.errors import ArtifactError, BuildError


def _run(
    root: Path,
    name: str = "run-1",
    *,
    target: str = "default",
    selected: list[str] | None = None,
) -> RunJournal:
    directory = root / ".yakbox/runs" / name
    directory.mkdir(parents=True)
    plan: dict[str, object] = {
        **runtime_metadata("audiobook-plan"),
        "target": target,
        "nodes": [
            {"id": "one:synthesize", "chapter_id": "one", "stage": "synthesize"},
            {"id": "one:master", "chapter_id": "one", "stage": "master"},
            {"id": "two:synthesize", "chapter_id": "two", "stage": "synthesize"},
        ],
    }
    if selected is not None:
        plan["execution"] = {"node_ids": selected}
    atomic_write_json(directory / "plan.json", plan)
    journal = RunJournal(directory / "journal.ndjson", name)
    journal.append("run_started")
    return journal


def test_status_reads_active_runs_before_a_summary_exists(tmp_path: Path) -> None:
    journal = _run(tmp_path)
    journal.append("node_completed", node_id="one:synthesize")
    journal.append("node_started", node_id="one:master")
    with target_lock(tmp_path / ".yakbox", "default"):
        before = {
            path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
        }
        progress = read_run_progress(tmp_path, "default")
        assert progress is not None
        assert progress.run_id == "run-1"
        assert progress.state == "running"
        assert progress.completed_nodes == 1
        assert progress.completed_chapters == 0
        assert progress.chapters == ("one", "two")
        assert [
            (node.chapter, node.stage)
            for node in progress.nodes
            if node.state == "running"
        ] == [("one", "master")]
        assert {
            path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
        } == before
    assert not journal.path.with_name("run.json").exists()


def test_watch_never_repairs_a_partial_append_and_reads_it_after_completion(
    tmp_path: Path,
) -> None:
    journal = _run(tmp_path)
    payload = (
        json.dumps(
            {
                **runtime_metadata("audiobook-journal"),
                "run_id": journal.run_id,
                "event": "node_completed",
                "node_id": "one:synthesize",
            }
        ).encode()
        + b"\n"
    )
    half = len(payload) // 2
    with journal.path.open("ab") as stream:
        stream.write(payload[:half])
    original = journal.path.read_bytes()
    progress = read_run_progress(tmp_path, "default")
    assert progress is not None and progress.completed_nodes == 0
    assert journal.path.read_bytes() == original
    assert not journal.path.with_suffix(".torn").exists()
    with journal.path.open("ab") as stream:
        stream.write(payload[half:])
    progress = read_run_progress(tmp_path, "default")
    assert progress is not None and progress.completed_nodes == 1


def test_status_uses_only_selected_stages_and_counts_reuse_once(tmp_path: Path) -> None:
    journal = _run(tmp_path, selected=["one:synthesize"])
    journal.append("node_completed", node_id="one:synthesize")
    journal.append("node_reused", node_id="one:synthesize")
    journal.append("run_complete")
    progress = read_run_progress(tmp_path, "default")
    assert progress is not None
    assert progress.state == "complete"
    assert (
        len(progress.nodes)
        == progress.completed_nodes
        == progress.completed_chapters
        == 1
    )
    assert progress.nodes[0].state == "reused"


def test_resume_resets_old_failed_and_skipped_attempts(tmp_path: Path) -> None:
    journal = _run(tmp_path)
    journal.append("node_failed", node_id="one:synthesize")
    journal.append("node_not_run", node_id="one:master")
    journal.append("run_failed")
    failed = read_run_progress(tmp_path, "default")
    assert failed is not None and failed.state == "failed"
    assert [node.state for node in failed.nodes] == ["failed", "skipped", "pending"]
    journal.append("run_resumed")
    journal.append("node_started", node_id="one:synthesize")
    with target_lock(tmp_path / ".yakbox", "default"):
        resumed = read_run_progress(tmp_path, "default")
    assert resumed is not None and resumed.state == "running"
    assert [node.state for node in resumed.nodes] == ["running", "pending", "pending"]


@pytest.mark.parametrize(
    "event,expected",
    [
        ("run_complete", "complete"),
        ("run_failed", "failed"),
        ("run_interrupted", "interrupted"),
        ("run_cancelled", "cancelled"),
        ("", "stopped"),
    ],
)
def test_status_distinguishes_terminal_and_unfinished_unlocked_runs(
    tmp_path: Path, event: str, expected: str
) -> None:
    journal = _run(tmp_path)
    if event:
        journal.append(event)
    progress = read_run_progress(tmp_path, "default")
    assert progress is not None and progress.state == expected


def test_status_ignores_other_targets_and_follows_recently_resumed_runs(
    tmp_path: Path,
) -> None:
    earlier = _run(tmp_path, "earlier")
    later = _run(tmp_path, "later")
    proof = _run(tmp_path, "proof", target="proof")
    for journal, timestamp in ((earlier, 30), (later, 20), (proof, 40)):
        os.utime(journal.path, ns=(timestamp, timestamp))
    progress = read_run_progress(tmp_path, "default")
    assert progress is not None and progress.run_id == "earlier"
    progress = read_run_progress(tmp_path, "proof")
    assert progress is not None and progress.run_id == "proof"


def test_status_without_runs_does_not_create_state(tmp_path: Path) -> None:
    assert read_run_progress(tmp_path, "default") is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "corrupt",
    [
        b"not-json\n",
        b"invalid\rrecord\n",
        json.dumps(
            {
                **runtime_metadata("audiobook-journal"),
                "run_id": "other",
                "event": "run_complete",
            }
        ).encode()
        + b"\n",
    ],
)
def test_status_rejects_complete_corrupt_records_without_changing_them(
    tmp_path: Path, corrupt: bytes
) -> None:
    journal = _run(tmp_path)
    with journal.path.open("ab") as stream:
        stream.write(corrupt)
    before = journal.path.read_bytes()
    with pytest.raises(BuildError):
        read_run_progress(tmp_path, "default")
    assert journal.path.read_bytes() == before
    assert not journal.path.with_suffix(".torn").exists()


@pytest.mark.parametrize(
    "selected", [["unknown"], ["one:synthesize", "one:synthesize"], [True]]
)
def test_status_rejects_invalid_saved_execution_selection(
    tmp_path: Path, selected: list[object]
) -> None:
    journal = _run(tmp_path)
    path = journal.path.with_name("plan.json")
    plan = json.loads(path.read_text())
    plan["execution"] = {"node_ids": selected}
    atomic_write_json(path, plan, overwrite=True)
    with pytest.raises(BuildError, match="execution selection"):
        read_run_progress(tmp_path, "default")


def test_status_rejects_unplanned_journal_nodes(tmp_path: Path) -> None:
    journal = _run(tmp_path)
    journal.append("node_completed", node_id="unknown")
    with pytest.raises(BuildError, match="unplanned node"):
        read_run_progress(tmp_path, "default")


def test_status_does_not_follow_run_state_outside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "book"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (workspace / ".yakbox").symlink_to(external, target_is_directory=True)
    with pytest.raises(ArtifactError):
        read_run_progress(workspace, "default")
