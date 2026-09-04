"""Gated CLI commands for explicit speech-analysis review dispositions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, Protocol

import click

from yakbox.errors import ValidationError, YakboxError
from yakbox.speech.analysis_disposition import (
    HumanReviewDecision,
    HumanReviewState,
    HumanReviewStore,
    human_review_status_report,
)

_MAXIMUM_NOTES_BYTES = 4_096


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


def register_speech_review_commands(
    main: click.Group,
    *,
    emit: _Emit,
    fail: Callable[[Exception], NoReturn],
    expose: bool = False,
) -> None:
    """Configure review commands and expose them only after public cutover."""
    global _emit_callback, _fail_callback  # noqa: PLW0603 - CLI composition root
    _emit_callback = emit
    _fail_callback = fail
    if expose:
        main.add_command(speech_group)


def _emit(value: dict[str, object], message: str) -> None:
    if _emit_callback is None:
        raise RuntimeError("Speech review CLI commands were not registered")
    _emit_callback(value, message)


def _fail(error: Exception) -> NoReturn:
    if _fail_callback is None:
        raise RuntimeError("Speech review CLI commands were not registered")
    _fail_callback(error)


@click.group("speech")
def speech_group() -> None:
    """Inspect and resolve generic speech-analysis evidence."""


@speech_group.group("reviews")
def speech_reviews_group() -> None:
    """Manage explicit dispositions for review-eligible soft rejections."""


@speech_reviews_group.command("list")
@click.argument(
    "manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
def speech_reviews_list_command(manifest: Path) -> None:
    """List review state without printing manuscript or transcript text."""
    try:
        store = _store_for_manifest(manifest)
        statuses = store.list()
    except (YakboxError, OSError, ValueError) as error:
        _fail(error)
    _emit(
        human_review_status_report(statuses),
        f"Found {len(statuses)} speech review candidate(s)",
    )


@speech_reviews_group.command("show")
@click.argument(
    "manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
@click.argument("review_id")
def speech_reviews_show_command(manifest: Path, review_id: str) -> None:
    """Show one bounded review identity and its current evidence state."""
    try:
        status = _store_for_manifest(manifest).show(review_id)
    except (YakboxError, OSError, ValueError) as error:
        _fail(error)
    _emit(
        human_review_status_report((status,)),
        f"Speech review {review_id} is {status.state.value}",
    )


@speech_reviews_group.command("resolve")
@click.argument(
    "manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
@click.argument("review_id")
@click.option(
    "--decision",
    type=click.Choice(tuple(item.value for item in HumanReviewDecision)),
    required=True,
)
@click.option(
    "--notes-file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    required=True,
)
@click.option(
    "--reviewer",
    envvar="YAKBOX_REVIEWER_ID",
    required=True,
    help="Stable local reviewer identifier; stored only as a fingerprint.",
)
def speech_reviews_resolve_command(
    manifest: Path,
    review_id: str,
    decision: str,
    notes_file: Path,
    reviewer: str,
) -> None:
    """Resolve one current soft rejection after revalidating all bound files."""
    try:
        store = _store_for_manifest(manifest)
        shown = store.show(review_id)
        if shown.state is HumanReviewState.STALE:
            raise ValidationError("Speech review evidence is stale")
        disposition = store.resolve(
            review_id,
            expected_candidate_fingerprint=shown.candidate_fingerprint,
            decision=HumanReviewDecision(decision),
            reviewer_identifier=reviewer,
            notes=_read_notes(notes_file),
        )
    except (YakboxError, OSError, UnicodeError, ValueError) as error:
        _fail(error)
    _emit(
        disposition.to_dict(),
        f"Speech review {review_id} resolved as {disposition.decision.value}",
    )


def _store_for_manifest(manifest: Path) -> HumanReviewStore:
    workspace = manifest.expanduser().resolve().parent
    return HumanReviewStore(
        workspace / ".yakbox" / "speech-analysis" / "reviews",
        evidence_root=workspace,
    )


def _read_notes(path: Path) -> str:
    if path.stat().st_size > _MAXIMUM_NOTES_BYTES:
        raise ValidationError("Reviewer notes exceed 4096 UTF-8 bytes")
    notes = path.read_text(encoding="utf-8")
    if len(notes.encode("utf-8")) > _MAXIMUM_NOTES_BYTES:
        raise ValidationError("Reviewer notes exceed 4096 UTF-8 bytes")
    return notes


__all__ = ["register_speech_review_commands", "speech_group"]
