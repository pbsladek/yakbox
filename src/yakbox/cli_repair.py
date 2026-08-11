"""Localized regeneration, audition, approval, and source-location commands."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, Protocol

import click

from yakbox.audiobook import build_audiobook, load_manifest
from yakbox.audiobook.assembly_manifest import locate_assembly_time
from yakbox.audiobook.repair import (
    RepairMode,
    approve_repair_session,
    generate_repair_session,
    plan_repair,
)
from yakbox.errors import YakboxError


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


def register_repair_commands(
    main: click.Group,
    *,
    emit: _Emit,
    fail: Callable[[Exception], NoReturn],
) -> None:
    """Attach localized-repair commands to the main CLI."""
    global _emit_callback, _fail_callback  # noqa: PLW0603 - CLI composition root
    _emit_callback = emit
    _fail_callback = fail
    main.add_command(repair_group)


def _emit(value: dict[str, object], message: str) -> None:
    if _emit_callback is None:
        raise RuntimeError("Repair CLI commands were not registered")
    _emit_callback(value, message)


def _fail(error: Exception) -> NoReturn:
    if _fail_callback is None:
        raise RuntimeError("Repair CLI commands were not registered")
    _fail_callback(error)


def _selector_options(function: Callable[..., object]) -> Callable[..., object]:
    options = (
        click.option("--chunk-id", help="Stable chunk ID from `yakbox plan`."),
        click.option(
            "--line",
            "source_line",
            type=click.IntRange(min=1),
            help="Select the unique speech chunk covering this source line.",
        ),
        click.option("--text", "text_match", help="Unique source-text substring."),
        click.option("--speaker", help="Optional speaker-route filter."),
        click.option(
            "--chapter",
            "chapter_selector",
            help="Limit matching to one chapter selector.",
        ),
        click.option(
            "--target",
            "target_name",
            default="default",
            show_default=True,
            help="Select the named build target and its repair decisions.",
        ),
        click.option(
            "--mode",
            type=click.Choice([item.value for item in RepairMode]),
            help="Override the [repairs] mode default.",
        ),
    )
    decorated = function
    for option in reversed(options):
        decorated = option(decorated)
    return decorated


@click.group("repair")
def repair_group() -> None:
    """Regenerate and approve only the passages that need correction."""


@repair_group.command("plan")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@_selector_options
def repair_plan_command(  # noqa: PLR0917 - Click option boundary.
    manifest: Path,
    chunk_id: str | None,
    source_line: int | None,
    text_match: str | None,
    speaker: str | None,
    chapter_selector: str | None,
    target_name: str,
    mode: str | None,
) -> None:
    """Resolve repair scope without loading a model or writing audio."""
    try:
        loaded = load_manifest(manifest)
        plan = plan_repair(
            loaded,
            target_name=target_name,
            chapter_selector=chapter_selector,
            chunk_id=chunk_id,
            source_line=source_line,
            text_match=text_match,
            speaker=speaker,
            mode=mode or loaded.repairs.mode,
        )
    except (YakboxError, OSError, ValueError) as error:
        _fail(error)
    value = plan.to_dict(workspace=loaded.root)
    _emit(
        value,
        f"{plan.chapter_id}: {len(plan.chunks)} synthesis chunk(s), "
        f"{len(plan.affected_join_indices)} affected join(s)",
    )


@repair_group.command("generate")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@_selector_options
@click.option(
    "--takes",
    type=click.IntRange(min=1, max=20),
    help="Override the [repairs] number of audition takes.",
)
@click.option(
    "--whisper/--no-whisper",
    "use_whisper",
    default=None,
    help="Override Whisper transcript and timing QA for generated takes.",
)
def repair_generate_command(  # noqa: PLR0917 - Click option boundary.
    manifest: Path,
    chunk_id: str | None,
    source_line: int | None,
    text_match: str | None,
    speaker: str | None,
    chapter_selector: str | None,
    target_name: str,
    mode: str | None,
    takes: int | None,
    use_whisper: bool | None,
) -> None:
    """Generate multiple deterministic takes in one warm backend session."""
    try:
        loaded = load_manifest(manifest)
        plan = plan_repair(
            loaded,
            target_name=target_name,
            chapter_selector=chapter_selector,
            chunk_id=chunk_id,
            source_line=source_line,
            text_match=text_match,
            speaker=speaker,
            mode=mode or loaded.repairs.mode,
        )
        session = asyncio.run(
            generate_repair_session(
                loaded,
                plan,
                takes=takes or loaded.repairs.takes,
                whisper_qa=(
                    loaded.repairs.whisper_qa if use_whisper is None else use_whisper
                ),
                api_key=os.environ.get("RESEMBLE_API_KEY"),
            )
        )
    except (YakboxError, OSError, ValueError) as error:
        _fail(error)
    value = session.to_dict(workspace=loaded.root)
    value["report_path"] = session.report_path.relative_to(loaded.root).as_posix()
    paths = "\n".join(str(take.audition_path) for take in session.takes)
    _emit(value, f"Repair session {session.id}\n{paths}")


@repair_group.command("approve")
@click.argument("repair_id")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option(
    "--take",
    type=click.IntRange(min=1),
    required=True,
    help="Approve this one-based take number from the repair session.",
)
@click.option(
    "--rebuild/--no-rebuild",
    default=None,
    help="Override whether approval immediately rebuilds the affected chapter.",
)
def repair_approve_command(
    repair_id: str,
    manifest: Path,
    take: int,
    rebuild: bool | None,
) -> None:
    """Approve one QA-passing take and rebuild only its chapter."""
    try:
        loaded = load_manifest(manifest)
        approval = approve_repair_session(
            loaded,
            repair_id=repair_id,
            take=take,
        )
        should_rebuild = (
            loaded.repairs.rebuild_on_approval if rebuild is None else rebuild
        )
        result = (
            asyncio.run(
                build_audiobook(
                    loaded,
                    target_name=approval.target,
                    chapter_selector=approval.chapter_id,
                )
            )
            if should_rebuild
            else None
        )
    except (YakboxError, OSError, ValueError) as error:
        _fail(error)
    _emit(
        {
            "repair_id": repair_id,
            "take": take,
            "target": approval.target,
            "chapter_id": approval.chapter_id,
            "approved_chunks": [item.chunk_id for item in approval.repairs],
            "rebuild_run_id": result.run_id if result is not None else None,
            "reused_nodes": list(result.reused_nodes) if result is not None else [],
        },
        f"Approved take {take} for {len(approval.repairs)} chunk(s)"
        + (f"; rebuilt in run {result.run_id}" if result is not None else ""),
    )


@repair_group.command("locate")
@click.argument("chapter_id")
@click.argument("at_seconds", type=click.FloatRange(min=0))
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--target", default="default", show_default=True)
def repair_locate_command(
    chapter_id: str,
    at_seconds: float,
    manifest: Path,
    target: str,
) -> None:
    """Map a heard timestamp to a source location and stable chunk ID."""
    try:
        loaded = load_manifest(manifest)
        location = locate_assembly_time(
            loaded.root,
            target=target,
            chapter_id=chapter_id,
            at_seconds=at_seconds,
        )
    except (YakboxError, OSError, ValueError) as error:
        _fail(error)
    chunk = location["chunk"]
    chunk_id = chunk.get("id") if isinstance(chunk, dict) else "unknown"
    _emit(location, f"{chapter_id} at {at_seconds:g}s -> chunk {chunk_id}")
