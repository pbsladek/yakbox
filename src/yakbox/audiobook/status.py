"""Read-only progress snapshots from build plans and append-only journals."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from yakbox._files import safe_child
from yakbox.audiobook.journal import _validate_event
from yakbox.audiobook.planner import BuildStage
from yakbox.contracts import schema_uri
from yakbox.errors import BuildError

_NODE_EVENTS = {
    "node_started": "running",
    "node_completed": "complete",
    "node_reused": "reused",
    "node_failed": "failed",
    "node_not_run": "skipped",
}
_RUN_EVENTS = {
    "run_started": "running",
    "run_resumed": "running",
    "run_complete": "complete",
    "run_failed": "failed",
    "run_interrupted": "interrupted",
    "run_cancelled": "cancelled",
}


@dataclass(frozen=True, slots=True)
class NodeStatus:
    """Latest recorded state of one planned chapter stage."""

    id: str
    chapter: str
    stage: str
    state: str = "pending"


@dataclass(frozen=True, slots=True)
class RunProgress:
    """An in-memory snapshot; reading it never recovers or modifies a journal."""

    run_id: str
    target: str
    state: str
    nodes: tuple[NodeStatus, ...]

    @property
    def completed_nodes(self) -> int:
        """Count completed or reused stages without counting retries twice."""
        return sum(node.state in {"complete", "reused"} for node in self.nodes)

    @property
    def chapters(self) -> tuple[str, ...]:
        """Return the selected chapter IDs in plan order."""
        return tuple(dict.fromkeys(node.chapter for node in self.nodes))

    @property
    def completed_chapters(self) -> int:
        """Count chapters whose selected stages have all completed."""
        unfinished = {
            node.chapter
            for node in self.nodes
            if node.state not in {"complete", "reused"}
        }
        return len(self.chapters) - len(unfinished)


def read_run_progress(workspace: Path, target: str) -> RunProgress | None:
    """Read the most recently active run for a target, including unfinished runs."""
    root = workspace.resolve()
    runs = safe_child(root, root / ".yakbox" / "runs")
    for plan_path in _recent_plans(root, runs):
        try:
            plan = _read_plan(plan_path)
        except FileNotFoundError:
            continue  # Cleanup can remove an old run between discovery and reading.
        if plan.get("target") != target:
            continue
        nodes = _plan_nodes(plan)
        run_id = plan_path.parent.name
        journal = safe_child(root, plan_path.with_name("journal.ndjson"))
        events = _read_events(journal, run_id)
        state, updated = _apply_events(nodes, events)
        lock = safe_child(root, root / ".yakbox" / "locks" / f"{target}.lock")
        if state in {"running", "waiting"} and not lock.exists():
            state = "stopped"
        return RunProgress(run_id, target, state, updated)
    return None


def _recent_plans(root: Path, runs: Path) -> tuple[Path, ...]:
    candidates: list[tuple[int, str, Path]] = []
    for candidate in runs.glob("*/plan.json"):
        path = safe_child(root, candidate)
        journal = safe_child(root, path.with_name("journal.ndjson"))
        try:
            activity = journal.stat() if journal.exists() else path.stat()
        except FileNotFoundError:
            continue
        candidates.append((activity.st_mtime_ns, path.parent.name, path))
    return tuple(path for _, _, path in sorted(candidates, reverse=True))


def _read_plan(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_bytes())
    except (ValueError, UnicodeDecodeError) as error:
        raise BuildError("Invalid saved build plan") from error
    if (
        not isinstance(raw, dict)
        or raw.get("$schema") != schema_uri("audiobook-plan")
        or raw.get("schema_version") != 1
    ):
        raise BuildError("Unsupported saved build plan")
    return cast(dict[str, object], raw)


def _plan_nodes(plan: dict[str, object]) -> tuple[NodeStatus, ...]:
    raw_nodes = plan.get("nodes")
    if not isinstance(raw_nodes, list):
        raise BuildError("Invalid saved build nodes")
    nodes = tuple(_node(value) for value in raw_nodes)
    ids = {node.id for node in nodes}
    if len(ids) != len(nodes):
        raise BuildError("Duplicate saved build node")
    execution = plan.get("execution")
    if execution is None:
        return nodes
    selected = execution.get("node_ids") if isinstance(execution, dict) else None
    if not isinstance(selected, list) or any(
        not isinstance(item, str) for item in selected
    ):
        raise BuildError("Invalid saved build execution selection")
    selected_ids = set(selected)
    if not selected_ids.issubset(ids) or len(selected_ids) != len(selected):
        raise BuildError("Invalid saved build execution selection")
    return tuple(node for node in nodes if node.id in selected_ids)


def _node(value: object) -> NodeStatus:
    if not isinstance(value, dict):
        raise BuildError("Invalid saved build node")
    identifier, chapter, stage = (
        value.get("id"),
        value.get("chapter_id"),
        value.get("stage"),
    )
    if (
        not isinstance(identifier, str)
        or not identifier
        or not isinstance(chapter, str)
        or not chapter
        or not isinstance(stage, str)
        or stage not in {item.value for item in BuildStage}
    ):
        raise BuildError("Invalid saved build node fields")
    return NodeStatus(identifier, chapter, cast(str, stage))


def _read_events(path: Path, run_id: str) -> tuple[dict[str, object], ...]:
    try:
        with path.open("rb") as stream:
            # Read only the bytes present at the start of this snapshot.
            content = stream.read(os.fstat(stream.fileno()).st_size)
    except FileNotFoundError:
        return ()
    result: list[dict[str, object]] = []
    # Only LF terminates NDJSON records. Ignore the incomplete final append.
    for index, line in enumerate(content.split(b"\n")[:-1], start=1):
        try:
            raw = json.loads(line)
        except (ValueError, UnicodeDecodeError) as error:
            raise BuildError(f"Corrupt journal record {index}") from error
        _validate_event(raw, run_id, index)
        result.append(cast(dict[str, object], raw))
    return tuple(result)


def _apply_events(
    nodes: tuple[NodeStatus, ...], events: tuple[dict[str, object], ...]
) -> tuple[str, tuple[NodeStatus, ...]]:
    state = "waiting"
    states: dict[str, str] = {}
    ids = {node.id for node in nodes}
    for event in events:
        name = cast(str, event["event"])
        if name in {"run_started", "run_resumed"}:
            states.clear()
        if name in _RUN_EVENTS:
            state = _RUN_EVENTS[name]
        if name in _NODE_EVENTS:
            identifier = event.get("node_id")
            if not isinstance(identifier, str) or identifier not in ids:
                raise BuildError("Run journal references an unplanned node")
            states[identifier] = _NODE_EVENTS[name]
    return state, tuple(
        replace(node, state=states.get(node.id, "pending")) for node in nodes
    )
