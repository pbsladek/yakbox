"""Human-readable live build status without changing JSON output contracts."""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.text import Text

from yakbox.audiobook.status import RunProgress, read_run_progress


def watch_status(workspace: Path, target: str, console: Console) -> None:
    """Refresh read-only status every two seconds until interrupted."""
    with Live(console=console, auto_refresh=False) as display:
        while True:
            snapshot = read_run_progress(workspace, target)
            rendered = Text(_progress_message(snapshot, target))
            if console.is_terminal:
                display.update(rendered, refresh=True)
            else:
                console.print(rendered)
            time.sleep(2)


def _progress_message(progress: RunProgress | None, target: str) -> str:
    if progress is None:
        return f"Waiting for a build for target {target}. Ctrl+C to stop."
    lines = [
        f"Target: {target} | Run: {progress.run_id} | State: {progress.state}",
        f"Chapters: {progress.completed_chapters}/{len(progress.chapters)} complete",
        f"Stages: {progress.completed_nodes}/{len(progress.nodes)} complete, "
        f"{sum(node.state == 'reused' for node in progress.nodes)} reused, "
        f"{sum(node.state == 'failed' for node in progress.nodes)} failed, "
        f"{sum(node.state == 'skipped' for node in progress.nodes)} skipped",
    ]
    lines.extend(
        f"{node.state.capitalize()}: {node.chapter} ({node.stage})"
        for node in progress.nodes
        if node.state == "failed"
        or (node.state == "running" and progress.state == "running")
    )
    lines.append("Refreshing every 2 seconds. Ctrl+C to stop.")
    return "\n".join(lines)
