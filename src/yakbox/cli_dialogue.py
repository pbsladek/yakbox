"""Reviewed dialogue-routing and transformation Click commands."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, Protocol

import click

from yakbox._files import atomic_write_bytes
from yakbox.audiobook import load_manifest, normalize_sources, plan_audiobook
from yakbox.audiobook.dialogue import (
    dialogue_transformation_report,
    render_dialogue_routes,
    suggest_dialogue_routes,
)
from yakbox.errors import ValidationError, YakboxError


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


def register_dialogue_commands(
    main: click.Group,
    *,
    emit: _Emit,
    fail: Callable[[Exception], NoReturn],
) -> None:
    """Attach the predeclared reviewed-dialogue commands."""
    global _emit_callback, _fail_callback  # noqa: PLW0603 - CLI composition root
    _emit_callback = emit
    _fail_callback = fail
    main.add_command(dialogue_group)


def _emit(value: dict[str, object], message: str) -> None:
    if _emit_callback is None:
        raise RuntimeError("Dialogue CLI commands were not registered")
    _emit_callback(value, message)


def _fail(error: Exception) -> NoReturn:
    if _fail_callback is None:
        raise RuntimeError("Dialogue CLI commands were not registered")
    _fail_callback(error)


@click.group("dialogue")
def dialogue_group() -> None:
    """Review speaker routes and transformed dialogue before synthesis."""


@dialogue_group.group("routes")
def dialogue_routes_group() -> None:
    """Generate and validate reviewed speaker-route sidecars."""


@dialogue_routes_group.command("suggest")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="dialogue-routes.toml",
    show_default=True,
    help="Workspace-relative path for the review-required TOML sidecar.",
)
@click.option("--force", is_flag=True, help="Replace an existing suggestion file.")
def dialogue_routes_suggest_command(
    manifest: Path,
    output: Path,
    force: bool,
) -> None:
    """Write source-located route suggestions for explicit human review."""
    try:
        loaded = load_manifest(manifest)
        suggestions = suggest_dialogue_routes(loaded)
        destination = (
            output if output.is_absolute() else loaded.root / output
        ).resolve()
        if not destination.is_relative_to(loaded.root):
            raise ValidationError("Dialogue route output must stay in the workspace")
        if destination.exists() and not force:
            raise ValidationError(
                f"Dialogue route output already exists: {destination}; use --force"
            )
        if not suggestions.routes:
            raise ValidationError("No reviewable dialogue routes were found")
        atomic_write_bytes(
            destination,
            render_dialogue_routes(
                suggestions, relative_to=destination.parent
            ).encode(),
        )
    except (YakboxError, OSError, ValueError) as error:
        _fail(error)
    _emit(
        {
            "path": str(destination),
            "suggested_routes": len(suggestions.routes),
            "unresolved_dialogue_paragraphs": suggestions.unresolved,
            "review_required": True,
        },
        f"Wrote {len(suggestions.routes)} route suggestion(s) to {destination}; "
        "review is required",
    )


@dialogue_routes_group.command("check")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
def dialogue_routes_check_command(manifest: Path) -> None:
    """Validate that every enabled sidecar route is approved and applicable."""
    try:
        loaded = load_manifest(manifest)
        if loaded.dialogue.routes is None:
            raise ValidationError("dialogue.routes is not configured")
        document = normalize_sources(
            loaded.sources,
            pronunciations=loaded.pronunciations,
            max_pause_ms=loaded.max_pause_ms,
            strip_attribution_tags=loaded.dialogue.strip_attribution_tags,
            dialogue_routes=loaded.dialogue.routes,
            expressive_tag_handling=loaded.dialogue.expressive_tag_handling,
            retain_first_attribution_per_scene=(
                loaded.dialogue.retain_first_attribution_per_scene
            ),
        )
        plan_audiobook(loaded, document)
    except YakboxError as error:
        _fail(error)
    _emit(
        {"path": str(loaded.dialogue.routes), "ready": True},
        f"Dialogue routes ready: {loaded.dialogue.routes}",
    )


@dialogue_group.command("preview")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="dialogue-preview.json",
    show_default=True,
    help="Workspace-relative path for the explicit-text JSON report.",
)
@click.option("--force", is_flag=True, help="Replace an existing preview report.")
def dialogue_preview_command(manifest: Path, output: Path, force: bool) -> None:
    """Write an explicit-text local preview of routed dialogue transformations."""
    try:
        loaded = load_manifest(manifest)
        destination = (
            output if output.is_absolute() else loaded.root / output
        ).resolve()
        if not destination.is_relative_to(loaded.root):
            raise ValidationError("Dialogue preview output must stay in the workspace")
        if destination.exists() and not force:
            raise ValidationError(
                f"Dialogue preview output already exists: {destination}; use --force"
            )
        report = dialogue_transformation_report(loaded)
        atomic_write_bytes(
            destination,
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
            overwrite=force,
        )
    except (YakboxError, OSError, ValueError) as error:
        _fail(error)
    _emit(
        {
            "path": str(destination),
            "paragraph_count": report["paragraph_count"],
            "stripped_tag_count": report["stripped_tag_count"],
        },
        f"Dialogue preview written to {destination}",
    )
