"""Whisper and short-utterance Click commands."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast

import click

from yakbox._files import atomic_write_json
from yakbox.audiobook.sources import SpeechSegment, normalize_sources
from yakbox.errors import BackendUnavailableError, ValidationError, YakboxError
from yakbox.phoneme_models import (
    DEFAULT_PHONEME_MODEL,
    DEFAULT_PHONEME_REVISION,
    install_phoneme_model,
    phoneme_model_status,
)
from yakbox.phoneme_qa import inspect_phonemes
from yakbox.short_testing import (
    list_short_reviews,
    play_short_review,
    run_short_test,
    write_short_review,
)
from yakbox.voice_quality import qualify_audition_voices
from yakbox.whisper_calibration import (
    DEFAULT_CALIBRATION_CORPUS,
    calibrate_frozen_corpus,
)
from yakbox.whisper_inspection import inspect_with_whisper
from yakbox.whisper_models import (
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WHISPER_REVISION,
    install_model,
    model_status,
)
from yakbox.whisper_qa import (
    JoinSpecification,
    WhisperClipType,
    inspect_joins,
    verify_manuscript,
)
from yakbox.yaml_config import load_yaml


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


def register_whisper_commands(
    main: click.Group,
    *,
    emit: _Emit,
    fail: Callable[[Exception], NoReturn],
) -> None:
    """Attach the predeclared Whisper, short-test, and review commands."""
    global _emit_callback, _fail_callback  # noqa: PLW0603 - one CLI composition root
    _emit_callback = emit
    _fail_callback = fail
    main.add_command(whisper_group)
    main.add_command(short_test_command)
    main.add_command(short_review_group)


def _emit(
    value: dict[str, object],
    message: str,
    *,
    status: str = "ok",
    exit_code: int = 0,
) -> None:
    if _emit_callback is None:
        raise RuntimeError("Whisper CLI commands were not registered")
    _emit_callback(value, message, status=status, exit_code=exit_code)


def _fail(error: Exception) -> NoReturn:
    if _fail_callback is None:
        raise RuntimeError("Whisper CLI commands were not registered")
    _fail_callback(error)


@click.group("whisper")
def whisper_group() -> None:
    """Inspect audio and manage pinned local Whisper models."""


@whisper_group.command("inspect")
@click.argument(
    "audio",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
@click.option("--expected", help="Compare against this exact transcript.")
@click.option(
    "--language",
    default="en",
    show_default=True,
    help="BCP 47 language code used for recognition.",
)
@click.option(
    "--model",
    default=DEFAULT_WHISPER_MODEL,
    show_default=True,
    help="Local path or Hugging Face model identifier.",
)
@click.option(
    "--revision",
    default=DEFAULT_WHISPER_REVISION,
    show_default=True,
    help="Immutable model revision to resolve from the local cache.",
)
@click.option(
    "--clip-type",
    type=click.Choice(tuple(item.value for item in WhisperClipType)),
    help="Override automatic confidence calibration for this clip.",
)
@click.option("--start", type=float, help="Start a targeted inspection at seconds.")
@click.option("--end", type=float, help="End a targeted inspection at seconds.")
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    help="Write the versioned JSON inspection report.",
)
def whisper_inspect_command(  # noqa: PLR0917 - Click injects CLI parameters.
    audio: Path,
    expected: str | None,
    language: str,
    model: str,
    revision: str | None,
    clip_type: str | None,
    start: float | None,
    end: float | None,
    out: Path | None,
) -> None:
    """Transcribe one audio file and explain timing and quality evidence."""
    try:
        report = asyncio.run(
            inspect_with_whisper(
                audio,
                expected_text=expected,
                language=language,
                model=model,
                revision=revision,
                clip_type=WhisperClipType(clip_type) if clip_type else None,
                start_seconds=start,
                end_seconds=end,
            )
        )
    except (YakboxError, OSError) as error:
        _fail(error)
    data = report.to_dict()
    if out is not None:
        atomic_write_json(out, data)
    summary = (
        f"{'PASS' if report.accepted else 'FAIL'}: "
        f"{data['recognized_text'] or '[no speech]'}\n"
        f"timing={report.result.timing_source}; "
        f"words={len(report.result.tokens)}; "
        f"reasons={', '.join(report.reason_codes) or 'none'}"
    )
    _emit(
        data,
        summary,
        status="ok" if report.accepted else "partial_failure",
        exit_code=0 if report.accepted else 1,
    )
    if not report.accepted:
        raise click.exceptions.Exit(1)


@whisper_group.command("reinspect")
@click.argument(
    "audio",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
@click.option("--expected", required=True, help="Exact transcript in the window.")
@click.option("--start", required=True, type=float, help="Window start in seconds.")
@click.option("--end", required=True, type=float, help="Window end in seconds.")
@click.option(
    "--language", default="en", show_default=True, help="Spoken language code."
)
@click.option(
    "--model",
    default=DEFAULT_WHISPER_MODEL,
    show_default=True,
    help="Pinned local Whisper model.",
)
@click.option(
    "--revision",
    default=DEFAULT_WHISPER_REVISION,
    show_default=True,
    help="Immutable model revision.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    help="Write the versioned targeted inspection report.",
)
def whisper_reinspect_command(  # noqa: PLR0917 - Click injects parameters.
    audio: Path,
    expected: str,
    start: float,
    end: float,
    language: str,
    model: str,
    revision: str | None,
    out: Path | None,
) -> None:
    """Reinspect one suspect time range with independent decodes."""
    try:
        report = asyncio.run(
            inspect_with_whisper(
                audio,
                expected_text=expected,
                language=language,
                model=model,
                revision=revision,
                start_seconds=start,
                end_seconds=end,
            )
        )
        data = report.to_dict()
        if out is not None:
            atomic_write_json(out, data)
    except (YakboxError, OSError, ValueError) as error:
        _fail(error)
    _emit(
        data,
        f"{'PASS' if report.accepted else 'FAIL'}: {start:g}-{end:g}s; "
        f"reasons={', '.join(report.reason_codes) or 'none'}",
        status="ok" if report.accepted else "partial_failure",
        exit_code=0 if report.accepted else 1,
    )
    if not report.accepted:
        raise click.exceptions.Exit(1)


@whisper_group.command("verify-manuscript")
@click.argument(
    "audio",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
@click.argument(
    "manuscript",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
@click.option("--chapter", help="Chapter id when the manuscript contains several.")
@click.option(
    "--pronunciations",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    help="Apply this pronunciation-rule file while normalizing the manuscript.",
)
@click.option(
    "--language", default="en", show_default=True, help="Spoken language code."
)
@click.option(
    "--model",
    default=DEFAULT_WHISPER_MODEL,
    show_default=True,
    help="Pinned local Whisper model.",
)
@click.option(
    "--revision",
    default=DEFAULT_WHISPER_REVISION,
    show_default=True,
    help="Immutable model revision.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    help="Write the versioned manuscript verification report.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=Path(".yakbox/cache/whisper"),
    show_default=True,
    help="Content-addressed local Whisper cache.",
)
@click.option("--no-cache", is_flag=True, help="Bypass the local Whisper cache.")
def whisper_verify_manuscript_command(  # noqa: PLR0917 - Click boundary.
    audio: Path,
    manuscript: Path,
    chapter: str | None,
    pronunciations: Path | None,
    language: str,
    model: str,
    revision: str | None,
    out: Path | None,
    cache_dir: Path,
    no_cache: bool,
) -> None:
    """Verify a complete chapter against normalized speakable manuscript text."""
    try:
        document = normalize_sources((manuscript,), pronunciations=pronunciations)
        selected = tuple(
            item for item in document.chapters if chapter is None or item.id == chapter
        )
        if len(selected) != 1:
            choices = ", ".join(item.id for item in document.chapters)
            raise ValidationError(
                "Select exactly one manuscript chapter with --chapter; "
                f"available: {choices}"
            )
        expected = " ".join(
            segment.text
            for segment in selected[0].segments
            if isinstance(segment, SpeechSegment)
        )
        report = asyncio.run(
            verify_manuscript(
                audio,
                manuscript,
                expected,
                language=language,
                model=model,
                revision=revision,
                cache_root=None if no_cache else cache_dir,
            )
        )
        data = report.to_dict()
        if out is not None:
            atomic_write_json(out, data)
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit(
        data,
        f"{'PASS' if report.accepted else 'FAIL'}: "
        f"{report.matched_token_count}/{report.expected_token_count} manuscript "
        f"tokens; mismatches={len(report.mismatches)}",
        status="ok" if report.accepted else "partial_failure",
        exit_code=0 if report.accepted else 1,
    )
    if not report.accepted:
        raise click.exceptions.Exit(1)


@whisper_group.command("inspect-joins")
@click.argument(
    "audio",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--spec",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    help="YAML file declaring every join timestamp and optional local context.",
)
@click.option(
    "--window",
    type=float,
    default=1.5,
    show_default=True,
    help="Seconds of audio to inspect on each side of a join.",
)
@click.option(
    "--language", default="en", show_default=True, help="Spoken language code."
)
@click.option(
    "--model",
    default=DEFAULT_WHISPER_MODEL,
    show_default=True,
    help="Pinned local Whisper model.",
)
@click.option(
    "--revision",
    default=DEFAULT_WHISPER_REVISION,
    show_default=True,
    help="Immutable model revision.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    help="Write the versioned join inspection report.",
)
@click.option(
    "--coalesce-gap-ms",
    type=click.IntRange(min=0),
    default=100,
    show_default=True,
    help="Merge nearby context-free joins into one Whisper window.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=Path(".yakbox/cache/whisper"),
    show_default=True,
    help="Content-addressed local Whisper cache.",
)
@click.option("--no-cache", is_flag=True, help="Bypass the local Whisper cache.")
def whisper_inspect_joins_command(  # noqa: PLR0917 - Click boundary.
    audio: Path,
    spec: Path,
    window: float,
    language: str,
    model: str,
    revision: str | None,
    out: Path | None,
    coalesce_gap_ms: int,
    cache_dir: Path,
    no_cache: bool,
) -> None:
    """Automatically inspect every physical splice declared in a join spec."""
    try:
        joins = _load_join_spec(spec)
        report = asyncio.run(
            inspect_joins(
                audio,
                joins,
                language=language,
                model=model,
                revision=revision,
                window_seconds=window,
                coalesce_gap_seconds=coalesce_gap_ms / 1_000,
                cache_root=None if no_cache else cache_dir,
            )
        )
        data = report.to_dict()
        if out is not None:
            atomic_write_json(out, data)
    except (YakboxError, OSError) as error:
        _fail(error)
    failed = sum(not item.accepted for item in report.joins)
    _emit(
        data,
        f"{'PASS' if report.accepted else 'FAIL'}: "
        f"{len(report.joins)} joins; failed={failed}",
        status="ok" if report.accepted else "partial_failure",
        exit_code=0 if report.accepted else 1,
    )
    if not report.accepted:
        raise click.exceptions.Exit(1)


def _load_join_spec(path: Path) -> tuple[JoinSpecification, ...]:
    raw = load_yaml(path, description="Join specification")
    values = raw.get("joins") if isinstance(raw, dict) else raw
    if not isinstance(values, list):
        raise ValidationError("Join spec must be an array or an object with joins")
    joins: list[JoinSpecification] = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise ValidationError(f"Join spec entry {index} must be an object")
        entry = cast(dict[str, object], value)
        unknown = set(entry) - {
            "at_seconds",
            "expected_before",
            "expected_after",
            "boundary",
        }
        if unknown:
            raise ValidationError(
                f"Join spec entry {index} has unknown keys: "
                f"{', '.join(sorted(unknown))}"
            )
        at = entry.get("at_seconds")
        if not isinstance(at, (int, float)) or isinstance(at, bool):
            raise ValidationError(f"Join spec entry {index}.at_seconds must be numeric")
        before = entry.get("expected_before", "")
        after = entry.get("expected_after", "")
        boundary = entry.get("boundary", "unknown")
        if not all(isinstance(item, str) for item in (before, after, boundary)):
            raise ValidationError(f"Join spec entry {index} context must be text")
        joins.append(
            JoinSpecification(
                float(at),
                cast(str, before),
                cast(str, after),
                cast(str, boundary),
            )
        )
    return tuple(joins)


@whisper_group.command("inspect-phonemes")
@click.argument(
    "audio",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
@click.option("--expected", required=True, help="Exact text to force-align.")
@click.option(
    "--language",
    default="en-us",
    show_default=True,
    help="eSpeak language or voice used to derive expected IPA phonemes.",
)
@click.option(
    "--model",
    default=DEFAULT_PHONEME_MODEL,
    show_default=True,
    help="Installed phoneme CTC model path or Hugging Face identifier.",
)
@click.option(
    "--revision",
    default=DEFAULT_PHONEME_REVISION,
    show_default=True,
    help="Immutable phoneme-model revision.",
)
@click.option(
    "--minimum-confidence",
    type=click.FloatRange(min=0, max=1),
    default=0.20,
    show_default=True,
    help="Reject audio when any aligned phoneme falls below this confidence.",
)
@click.option("--out", type=click.Path(path_type=Path))
def whisper_inspect_phonemes_command(  # noqa: PLR0917 - Click boundary.
    audio: Path,
    expected: str,
    language: str,
    model: str,
    revision: str | None,
    minimum_confidence: float,
    out: Path | None,
) -> None:
    """Force expected IPA phonemes onto audio and explain weak boundaries."""
    try:
        report = asyncio.run(
            inspect_phonemes(
                audio,
                expected,
                language=language,
                model=model,
                revision=revision,
                minimum_confidence=minimum_confidence,
            )
        )
        data = report.to_dict()
        if out is not None:
            atomic_write_json(out, data)
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit(
        data,
        f"{'PASS' if report.accepted else 'FAIL'}: "
        f"{len(report.result.phonemes)} phonemes; "
        f"minimum confidence={report.decision.confidence or 0:.3f}",
        status="ok" if report.accepted else "partial_failure",
        exit_code=0 if report.accepted else 1,
    )
    if not report.accepted:
        raise click.exceptions.Exit(1)


@whisper_group.command("calibrate")
@click.option(
    "--corpus",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    default=DEFAULT_CALIBRATION_CORPUS,
    show_default=True,
    help="Frozen TOML evidence corpus to evaluate.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    help="Write the versioned calibration report to this path.",
)
def whisper_calibrate_command(corpus: Path, out: Path | None) -> None:
    """Evaluate current gates against the frozen known-failure corpus."""
    try:
        report = calibrate_frozen_corpus(corpus)
        if out is not None:
            atomic_write_json(out, report)
    except (YakboxError, OSError) as error:
        _fail(error)
    failed = bool(report["false_accepts"] or report["false_rejects"])
    token_accuracy = report["token_accuracy"]
    if not isinstance(token_accuracy, (int, float)):
        _fail(ValidationError("Whisper calibration token accuracy is invalid"))
    _emit(
        report,
        f"{report['case_count']} cases; false accepts={report['false_accepts']}; "
        f"false rejects={report['false_rejects']}; "
        f"token accuracy={token_accuracy:.1%}",
        status="partial_failure" if failed else "ok",
        exit_code=1 if failed else 0,
    )
    if failed:
        raise click.exceptions.Exit(1)


@whisper_group.command("qualify-voices")
@click.argument(
    "audition_report",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--expected-file",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    help="Exact fixed text synthesized by every auditioned voice.",
)
@click.option(
    "--baseline",
    "baselines",
    multiple=True,
    required=True,
    help="Human-approved audition variant used to derive quality thresholds.",
)
@click.option(
    "--language", default="en", show_default=True, help="Spoken language code."
)
@click.option(
    "--model",
    default=DEFAULT_WHISPER_MODEL,
    show_default=True,
    help="Pinned local Whisper model.",
)
@click.option(
    "--revision",
    default=DEFAULT_WHISPER_REVISION,
    show_default=True,
    help="Immutable model revision.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    help="Write the versioned voice-quality report.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=Path(".yakbox/cache/whisper"),
    show_default=True,
    help="Content-addressed local Whisper cache.",
)
@click.option("--no-cache", is_flag=True, help="Bypass the local Whisper cache.")
def whisper_qualify_voices_command(  # noqa: PLR0917 - Click boundary.
    audition_report: Path,
    expected_file: Path,
    baselines: tuple[str, ...],
    language: str,
    model: str,
    revision: str | None,
    out: Path | None,
    cache_dir: Path,
    no_cache: bool,
) -> None:
    """Qualify fixed-text voice auditions against approved baseline voices."""
    try:
        expected = expected_file.read_text(encoding="utf-8")
        report = asyncio.run(
            qualify_audition_voices(
                audition_report,
                expected,
                baseline_voices=baselines,
                language=language,
                model=model,
                revision=revision,
                cache_root=None if no_cache else cache_dir,
            )
        )
        data = report.to_dict()
        if out is not None:
            atomic_write_json(out, data)
    except (YakboxError, OSError, UnicodeDecodeError) as error:
        _fail(error)
    suspect = tuple(item.voice for item in report.voices if not item.accepted)
    _emit(
        data,
        f"{'PASS' if report.accepted else 'SUSPECT'}: "
        f"{len(report.voices) - len(suspect)}/{len(report.voices)} voices qualify; "
        f"suspect={', '.join(suspect) or 'none'}",
        status="ok" if report.accepted else "partial_failure",
        exit_code=0 if report.accepted else 1,
    )
    if not report.accepted:
        raise click.exceptions.Exit(1)


@whisper_group.group("models")
def whisper_models_group() -> None:
    """Install and verify models without surprise build-time downloads."""


def _model_options(function: Any) -> Any:  # noqa: ANN401 - Click decorator
    function = click.option(
        "--revision",
        default=DEFAULT_WHISPER_REVISION,
        show_default=True,
        help="Immutable model revision.",
    )(function)
    return click.option(
        "--model",
        default=DEFAULT_WHISPER_MODEL,
        show_default=True,
        help="Local path or Hugging Face model identifier.",
    )(function)


@whisper_models_group.command("status")
@_model_options
def whisper_models_status_command(model: str, revision: str | None) -> None:
    """Show package, cache, size, and integrity state."""
    status = model_status(model, revision)
    _emit(
        status.to_dict(),
        f"{'verified' if status.verified else 'not ready'}: "
        f"{status.local_path or model} ({status.size_bytes / (1024**3):.2f} GiB; "
        f"{', '.join(status.issues) or 'no issues'})",
        status="ok" if status.verified else "partial_failure",
        exit_code=0 if status.verified else 1,
    )
    if not status.verified:
        raise click.exceptions.Exit(1)


@whisper_models_group.command("install")
@_model_options
def whisper_models_install_command(model: str, revision: str | None) -> None:
    """Explicitly download a pinned model and verify it."""
    try:
        status = install_model(model, revision)
    except YakboxError as error:
        _fail(error)
    _emit(status.to_dict(), f"Installed and verified {status.local_path}")


@whisper_models_group.command("verify")
@_model_options
def whisper_models_verify_command(model: str, revision: str | None) -> None:
    """Verify a cached model without network access."""
    status = model_status(model, revision)
    _emit(
        status.to_dict(),
        f"{'Verified' if status.verified else 'Invalid'} {status.local_path or model}",
        status="ok" if status.verified else "partial_failure",
        exit_code=0 if status.verified else 1,
    )
    if not status.verified:
        raise click.exceptions.Exit(1)


@whisper_models_group.command("path")
@_model_options
def whisper_models_path_command(model: str, revision: str | None) -> None:
    """Print the resolved local snapshot path without downloading."""
    status = model_status(model, revision)
    if status.local_path is None:
        _fail(BackendUnavailableError("Pinned Whisper model is not installed"))
    _emit(status.to_dict(), str(status.local_path))


@whisper_group.group("phoneme-models")
def phoneme_models_group() -> None:
    """Install and verify the pinned local phoneme acoustic model."""


def _phoneme_model_options(function: Any) -> Any:  # noqa: ANN401 - Click decorator
    function = click.option(
        "--revision",
        default=DEFAULT_PHONEME_REVISION,
        show_default=True,
        help="Immutable model revision.",
    )(function)
    return click.option(
        "--model",
        default=DEFAULT_PHONEME_MODEL,
        show_default=True,
        help="Local path or Hugging Face model identifier.",
    )(function)


@phoneme_models_group.command("status")
@_phoneme_model_options
def phoneme_models_status_command(model: str, revision: str | None) -> None:
    """Show local phoneme-model cache and integrity state."""
    status = phoneme_model_status(model, revision)
    _emit(
        status.to_dict(),
        f"{'verified' if status.verified else 'not ready'}: "
        f"{status.local_path or model}; {', '.join(status.issues) or 'no issues'}",
        status="ok" if status.verified else "partial_failure",
        exit_code=0 if status.verified else 1,
    )
    if not status.verified:
        raise click.exceptions.Exit(1)


@phoneme_models_group.command("install")
@_phoneme_model_options
def phoneme_models_install_command(model: str, revision: str | None) -> None:
    """Explicitly download and verify the pinned phoneme model."""
    try:
        status = install_phoneme_model(model, revision)
    except YakboxError as error:
        _fail(error)
    _emit(status.to_dict(), f"Installed and verified {status.local_path}")


@click.command("short-test")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--profile", required=True, help="Manifest backend profile to test.")
@click.option("--text", required=True, help="One- to three-word target text.")
@click.option(
    "--previous-context",
    help="Previous sentence used to build a natural carrier.",
)
@click.option(
    "--next-context",
    help="Following sentence used to build a natural carrier.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("build/short-tests"),
    show_default=True,
    help="Parent directory for the unique diagnostic run.",
)
def short_test_command(  # noqa: PLR0917 - Click injects CLI parameters.
    manifest: Path,
    profile: str,
    text: str,
    previous_context: str | None,
    next_context: str | None,
    out_dir: Path,
) -> None:
    """Generate direct, carrier, cropped, refined, and selected test takes."""
    try:
        result = asyncio.run(
            run_short_test(
                manifest,
                profile_name=profile,
                text=text,
                output_root=out_dir,
                previous_context=previous_context,
                next_context=next_context,
            )
        )
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit(
        {"run_directory": str(result), "report": str(result / "report.json")},
        f"Short-utterance test package: {result}",
    )


@click.group("short-review")
def short_review_group() -> None:
    """List, play, approve, or reject hash-bound listening reviews."""


@short_review_group.command("list")
@click.argument("root", type=click.Path(path_type=Path), default="build")
def short_review_list_command(root: Path) -> None:
    """List short-utterance QA reports beneath ROOT."""
    reviews = list_short_reviews(root)
    _emit(
        {"reviews": list(reviews)},
        "\n".join(
            f"{item['status']:10} {item['report']} -> {item['selected_audio']}"
            for item in reviews
        )
        or "No short-utterance reviews found",
    )


@short_review_group.command("play")
@click.argument(
    "report",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
def short_review_play_command(report: Path) -> None:
    """Play the integrity-checked selected candidate."""
    try:
        audio = play_short_review(report)
    except YakboxError as error:
        _fail(error)
    _emit({"report": str(report), "audio": str(audio)}, f"Playing {audio}")


def _review_decision(report: Path, notes: str, *, approved: bool) -> None:
    try:
        review = write_short_review(report, approved=approved, notes=notes)
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit(
        {"review": str(review), "approved": approved},
        f"{'Approved' if approved else 'Rejected'} {report}",
    )


def _review_report(function: Any) -> Any:  # noqa: ANN401 - Click decorator
    return click.argument(
        "report",
        type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    )(function)


def _review_notes(function: Any) -> Any:  # noqa: ANN401 - Click decorator
    return click.option(
        "--notes",
        default="",
        show_default=True,
        help="Listening notes stored with the hash-bound decision.",
    )(function)


@short_review_group.command("approve")
@_review_report
@_review_notes
def short_review_approve_command(report: Path, notes: str) -> None:
    """Approve the selected take, bound to report and audio hashes."""
    _review_decision(report, notes, approved=True)


@short_review_group.command("reject")
@_review_report
@_review_notes
def short_review_reject_command(report: Path, notes: str) -> None:
    """Reject the selected take, bound to report and audio hashes."""
    _review_decision(report, notes, approved=False)
