from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table

from yakbox import __version__
from yakbox._files import atomic_write_bytes
from yakbox.audio import inspect_audio
from yakbox.audiobook import (
    ArtifactKind,
    BuildProgress,
    BuildProgressCallback,
    BuildResult,
    apply_cache_cleanup,
    apply_cleanup,
    assemble_release,
    audit_pronunciations,
    audition_audiobook,
    build_audiobook,
    check_release,
    diff_releases,
    explain_synthesis_chunk,
    export_shard_manifests,
    inventory_artifacts,
    inventory_synthesis_cache,
    load_manifest,
    normalize_sources,
    plan_audiobook,
    plan_cache_cleanup,
    plan_cleanup,
    preflight_audiobook_build,
    preview_audiobook,
    purge_trash,
    repair_artifact_metadata,
    restore_trash,
    select_build_chapters,
    verify_shard_manifests,
)
from yakbox.audiobook.artifacts import verify_artifact
from yakbox.audiobook.manifest import AudiobookManifest, BuildTarget
from yakbox.cli_dialogue import register_dialogue_commands
from yakbox.cli_help import configure_cli_help
from yakbox.cli_migrate import register_migration_commands
from yakbox.cli_options import (
    audio_output_options,
    deprecated_api_key_option,
    explicit_text_options,
    hosted_budget_options,
    text_file_option,
)
from yakbox.cli_repair import register_repair_commands
from yakbox.cli_runtime import register_runtime_commands
from yakbox.cli_whisper import register_whisper_commands
from yakbox.cloud import (
    AudioFormat as CloudAudioFormat,
)
from yakbox.cloud import (
    ClientOptions,
    FileSynthesisResult,
    HostedUsageGate,
    Page,
    Precision,
    Project,
    Recording,
    ResembleClient,
    ResembleSpeechService,
    StreamRequest,
    SynthesisRequest,
    Voice,
)
from yakbox.cloud.batch import (
    MAX_SYNTHESIS_CHARACTERS,
    BatchReport,
    BatchResult,
    BatchStatus,
    ProgressCallback,
    run_cloud_batch,
)
from yakbox.config import YakboxConfig, load_config
from yakbox.contracts import runtime_metadata
from yakbox.credentials import CredentialSource, resolve_resemble_credential
from yakbox.diagnostics import run_doctor
from yakbox.errors import (
    BackendUnavailableError,
    ConfigurationError,
    ValidationError,
    YakboxError,
    stable_error_code,
)
from yakbox.speech import (
    AudioFormat,
    BackendCapabilities,
    CurrencyCode,
    HostedUsageBudget,
    HostedUsageReportingService,
    HostedUsageSnapshot,
    HostedWorkEstimate,
    PricingSourceId,
    SpeechArtifact,
    SpeechSynthesisRequest,
    SpeechTransformationRequest,
    estimate_hosted_work,
    hosted_confirmation_reasons,
    open_speech_backend,
    open_transformation_backend,
    validate_hosted_preflight,
)
from yakbox.textutils import BatchRow, iter_batch_rows, read_batch_rows

console = Console(stderr=False)
error_console = Console(stderr=True)


class Context:
    def __init__(self, *, json_output: bool, quiet: bool, verbose: bool) -> None:
        self.json_output = json_output
        self.quiet = quiet
        self.verbose = verbose
        self.credential_profile = "default"


class KeyringApi(Protocol):
    def set_password(self, service: str, username: str, password: str) -> None: ...

    def get_password(self, service: str, username: str) -> str | None: ...

    def delete_password(self, service: str, username: str) -> None: ...


class YakboxGroup(click.Group):
    """Preserve machine-readable output even when Click rejects arguments."""

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,  # noqa: ANN401 - matches Click's dynamic override
    ) -> Any:  # noqa: ANN401 - Click's main() return type is dynamic
        arguments = list(args) if args is not None else sys.argv[1:]
        if not standalone_mode:
            return super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        json_requested = "--json" in arguments
        try:
            result = super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        except click.ClickException as error:
            if json_requested:
                _emit_bootstrap_error(
                    code=_click_error_code(error),
                    message=error.format_message(),
                    exit_code=error.exit_code,
                    command=_command_hint(arguments),
                )
            error.show()
            raise SystemExit(error.exit_code) from error
        except click.Abort:
            if json_requested:
                _emit_bootstrap_error(
                    code="interrupted",
                    message="Operation interrupted",
                    exit_code=130,
                    status="aborted",
                    command=_command_hint(arguments),
                )
            click.echo("Aborted!", err=True)
            raise SystemExit(130) from None
        except Exception:
            if json_requested:
                _emit_bootstrap_error(
                    code="internal_error",
                    message="An unexpected internal error occurred",
                    exit_code=1,
                    command=_command_hint(arguments),
                )
            raise
        raise SystemExit(result if isinstance(result, int) else 0)


def _context() -> Context:
    context = click.get_current_context().find_root().obj
    if isinstance(context, Context):
        return context
    return Context(json_output=False, quiet=False, verbose=False)


@click.group(
    cls=YakboxGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=False,
)
@click.version_option(version=__version__, prog_name="yakbox")
@click.option("--json", "json_output", is_flag=True, help="Emit versioned JSON.")
@click.option("--no-color", is_flag=True, help="Disable color output.")
@click.option("-q", "--quiet", is_flag=True, help="Suppress non-error output.")
@click.option("-v", "--verbose", is_flag=True, help="Show safe diagnostic detail.")
@click.pass_context
def main(
    context: click.Context,
    json_output: bool,
    no_color: bool,
    quiet: bool,
    verbose: bool,
) -> None:
    """Build reproducible audiobooks and use speech backends directly."""
    if quiet and verbose:
        raise click.UsageError("--quiet and --verbose are mutually exclusive")
    if no_color or os.environ.get("NO_COLOR"):
        console.no_color = True
        error_console.no_color = True
    context.obj = Context(json_output=json_output, quiet=quiet, verbose=verbose)


@main.command("init")
@click.argument("directory", type=click.Path(path_type=Path), default=".")
def init_command(directory: Path) -> None:
    """Initialize a minimal audiobook workspace."""
    root = directory.resolve()
    manifest = root / "yakbox.toml"
    source = root / "source" / "book.md"
    if manifest.exists():
        raise click.ClickException(f"Manifest already exists: {manifest}")
    source.parent.mkdir(parents=True, exist_ok=True)
    template = """"$schema" = "https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"
schema_version = 1
sources = ["source/book.md"]
pronunciations = "pronunciations.toml"

[book]
title = "My Audiobook"
author = "Author Name"
narrator = "Narrator Name"
language = "en"

[voices.narrator]
display_name = "Narrator"
rights_basis = "not_applicable"

[profiles.default]
backend = "fake"
voice = "narrator"
sample_rate = 16000

[targets.default]
profile = "default"
output_root = "build/yakbox"
chunk_chars = 2800
mastering = true
wav_sample_rate = 44100
mp3_bitrate = "192k"
m4b = false
provider_concurrency = 5
media_concurrency = 2

[targets.draft]
extends = "default"
output_root = "build/yakbox-draft"
chunk_chars = 1200
mastering = false
through_stage = "synthesize"

[targets.proof]
extends = "default"
output_root = "build/yakbox-proof"
mastering = false

[targets.release]
extends = "default"
output_root = "build/yakbox-release"
m4b = true
quality_min_lufs = -23.0
quality_max_lufs = -16.0
quality_max_true_peak_dbfs = -1.0
quality_max_leading_silence_seconds = 2.0
quality_max_trailing_silence_seconds = 2.0

[repairs]
mode = "context"
takes = 4
whisper_qa = true
rebuild_on_approval = true

[retention]
keep_successful_runs = 3
audition_days = 30
preview_days = 7
raw_until_release = true
"""
    pronunciations = """schema_version = 1

# [[terms]]
# written = "Example"
# spoken = "Egg zample"
# language = "en"
# match = "whole_word"
# case = "sensitive"
# priority = 100
# status = "approved"
# enabled = true
"""
    atomic_write_bytes(manifest, template.encode())
    atomic_write_bytes(root / "pronunciations.toml", pronunciations.encode())
    if not source.exists():
        atomic_write_bytes(
            source,
            b"# Chapter One\n\nReplace this text with your manuscript.\n",
        )
    _emit({"manifest": str(manifest), "source": str(source)}, f"Created {manifest}")


@main.command("validate")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
def validate_command(manifest: Path) -> None:
    """Validate an audiobook manifest and normalized source."""
    try:
        loaded = load_manifest(manifest)
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
        attribution_findings: dict[str, dict[str, object]] = {}
        for target in loaded.targets:
            plan = plan_audiobook(loaded, document, target_name=target.name)
            for finding in plan.attribution_findings:
                value = finding.to_dict(root=loaded.root)
                attribution_findings[json.dumps(value, sort_keys=True)] = value
    except YakboxError as error:
        _fail(error)
    findings = list(attribution_findings.values())
    _emit(
        {
            "manifest": str(loaded.path),
            "chapters": len(document.chapters),
            "document_sha256": document.sha256,
            "attribution_finding_count": len(findings),
            "attribution_findings": findings,
        },
        f"Valid: {len(document.chapters)} chapter(s); "
        f"{len(findings)} attribution suggestion(s)",
    )


@main.command("plan")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--target", default="default", show_default=True)
@click.option("--chapter", "--chapters")
def plan_command(manifest: Path, target: str, chapter: str | None) -> None:
    """Resolve a deterministic build plan without generating audio."""
    try:
        loaded = load_manifest(manifest)
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
        plan = plan_audiobook(
            loaded, document, target_name=target, chapter_selector=chapter
        )
        preflight = preflight_audiobook_build(
            loaded,
            target_name=target,
            chapter_selector=chapter,
        )
    except (YakboxError, ValueError) as error:
        _fail(error)
    _emit(
        {
            **plan.to_dict(root=loaded.root),
            "change_summary": preflight.change_summary.to_dict(),
            "preflight": preflight.to_dict(),
        },
        f"{len(plan.nodes)} nodes; plan {plan.fingerprint[:12]}; "
        f"synthesize {preflight.pending_synthesis_chunks}/"
        f"{preflight.synthesis_chunks} chunk(s), "
        f"inspect {preflight.affected_join_count} affected join(s)",
    )


@main.group("pronunciations")
def pronunciations_group() -> None:
    """Inspect the pronunciation lexicon against speakable source text."""


@pronunciations_group.command("audit")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option(
    "--fail-unused",
    is_flag=True,
    help="Exit non-zero when an approved rule has no source matches.",
)
def pronunciations_audit_command(manifest: Path, fail_unused: bool) -> None:
    """Report applied, unused, and priority-shadowed pronunciation rules."""
    try:
        loaded = load_manifest(manifest)
        audit = audit_pronunciations(
            loaded.sources,
            loaded.pronunciations,
            max_pause_ms=loaded.max_pause_ms,
            strip_attribution_tags=loaded.dialogue.strip_attribution_tags,
            dialogue_routes=loaded.dialogue.routes,
            expressive_tag_handling=loaded.dialogue.expressive_tag_handling,
            retain_first_attribution_per_scene=(
                loaded.dialogue.retain_first_attribution_per_scene
            ),
        )
    except (YakboxError, OSError) as error:
        _fail(error)
    failed = fail_unused and audit.unused_rules > 0
    _emit(
        audit.to_dict(root=loaded.root),
        f"{len(audit.rules)} rule(s): {audit.unused_rules} unused, "
        f"{audit.shadowed_matches} shadowed match(es)",
        status="partial_failure" if failed else "ok",
        exit_code=1 if failed else 0,
    )
    if failed:
        raise click.exceptions.Exit(1)


@main.command("build")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--target", default="default", show_default=True)
@click.option(
    "--mode",
    type=click.Choice(["draft", "proof", "release"]),
    help="Select a target named draft, proof, or release.",
)
@click.option("--chapter", "--chapters")
@click.option(
    "--changed",
    is_flag=True,
    help="Build only chapters changed since the last success.",
)
@click.option(
    "--failed", is_flag=True, help="Build only chapters from the latest failed run."
)
@click.option(
    "--missing",
    is_flag=True,
    help="Build only chapters with missing or invalid artifacts.",
)
@click.option(
    "--since",
    metavar="RELEASE_ID_OR_PATH",
    help="With --changed, compare against a release manifest.",
)
@click.option("--profile")
@click.option("--resume/--no-resume", default=True, show_default=True)
@click.option("--dry-run", is_flag=True)
@click.option(
    "--from",
    "from_stage",
    type=click.Choice(["synthesize", "master", "encode_mp3", "inspect"]),
)
@click.option(
    "--through",
    "through_stage",
    type=click.Choice(["synthesize", "master", "encode_mp3", "inspect"]),
)
@click.option(
    "--stage",
    type=click.Choice(["synthesize", "master", "encode_mp3", "inspect"]),
    help="Run exactly one stage; cannot be combined with --from/--through.",
)
@hosted_budget_options
@click.option(
    "--no-progress", is_flag=True, help="Disable the interactive progress bar."
)
@deprecated_api_key_option
def build_command(  # noqa: PLR0913,PLR0917 - Click injects CLI parameters.
    manifest: Path,
    target: str,
    mode: str | None,
    chapter: str | None,
    changed: bool,
    failed: bool,
    missing: bool,
    since: str | None,
    profile: str | None,
    resume: bool,
    dry_run: bool,
    from_stage: str | None,
    through_stage: str | None,
    stage: str | None,
    max_submitted_characters: int | None,
    max_provider_requests: int | None,
    max_estimated_spend: Decimal | None,
    currency: str | None,
    pricing_source: str | None,
    price_per_character: Decimal | None,
    confirm_above_characters: int | None,
    confirm_above_requests: int | None,
    yes: bool,
    no_progress: bool,
    api_key: str | None,
) -> None:
    """Build raw, mastered WAV, MP3, and inspection artifacts."""
    options = _BuildCommandOptions(
        manifest=manifest,
        target=target,
        mode=mode,
        chapter=chapter,
        changed=changed,
        failed=failed,
        missing=missing,
        since=since,
        profile=profile,
        resume=resume,
        dry_run=dry_run,
        from_stage=from_stage,
        through_stage=through_stage,
        stage=stage,
        max_submitted_characters=max_submitted_characters,
        max_provider_requests=max_provider_requests,
        max_estimated_spend=max_estimated_spend,
        currency=currency,
        pricing_source=pricing_source,
        price_per_character=price_per_character,
        confirm_above_characters=confirm_above_characters,
        confirm_above_requests=confirm_above_requests,
        yes=yes,
        no_progress=no_progress,
        api_key=api_key,
    )
    try:
        result = _run_build_command(options)
    except (YakboxError, OSError) as error:
        _fail(error)
    if result is None:
        return
    _emit_build_result(result)


@dataclass(frozen=True, slots=True)
class _BuildCommandOptions:
    manifest: Path
    target: str
    mode: str | None
    chapter: str | None
    changed: bool
    failed: bool
    missing: bool
    since: str | None
    profile: str | None
    resume: bool
    dry_run: bool
    from_stage: str | None
    through_stage: str | None
    stage: str | None
    max_submitted_characters: int | None
    max_provider_requests: int | None
    max_estimated_spend: Decimal | None
    currency: str | None
    pricing_source: str | None
    price_per_character: Decimal | None
    confirm_above_characters: int | None
    confirm_above_requests: int | None
    yes: bool
    no_progress: bool
    api_key: str | None


@dataclass(frozen=True, slots=True)
class _ResolvedBuildRequest:
    target: str
    selection: str | None
    from_stage: str | None
    through_stage: str | None


def _run_build_command(options: _BuildCommandOptions) -> BuildResult | None:
    request = _resolve_build_request(options)
    config = _load_config()
    manifest = load_manifest(options.manifest)
    target = manifest.target(request.target)
    chapter = _selected_chapter_filter(options, request, manifest, target)
    if request.selection is not None and chapter is None:
        _emit_up_to_date(request)
        return None
    budget = _build_budget(options, target)
    preflight = preflight_audiobook_build(
        manifest,
        target_name=request.target,
        profile_override=options.profile,
        chapter_selector=chapter,
        price_per_character=options.price_per_character,
        from_stage=request.from_stage,
        through_stage=request.through_stage,
    )
    _validate_build_preflight(options, preflight.hosted_work, budget)
    with _build_progress(options, preflight.planned_nodes) as progress:
        return asyncio.run(
            build_audiobook(
                manifest,
                target_name=request.target,
                profile_override=options.profile,
                chapter_selector=chapter,
                dry_run=options.dry_run,
                resume=options.resume,
                api_key=_optional_api_key(options.api_key, config),
                max_submitted_characters=options.max_submitted_characters,
                max_provider_requests=options.max_provider_requests,
                max_estimated_spend=options.max_estimated_spend,
                currency=options.currency,
                pricing_source=options.pricing_source,
                price_per_character=options.price_per_character,
                confirm_above_characters=options.confirm_above_characters,
                confirm_above_requests=options.confirm_above_requests,
                from_stage=request.from_stage,
                through_stage=request.through_stage,
                progress=progress,
            )
        )


def _resolve_build_request(options: _BuildCommandOptions) -> _ResolvedBuildRequest:
    if options.mode is not None and options.target != "default":
        raise click.UsageError("--mode cannot be combined with a non-default --target")
    if options.stage is not None and (
        options.from_stage is not None or options.through_stage is not None
    ):
        raise click.UsageError("--stage cannot be combined with --from or --through")
    if sum((options.changed, options.failed, options.missing)) > 1:
        raise click.UsageError(
            "--changed, --failed, and --missing are mutually exclusive"
        )
    if options.since is not None and not options.changed:
        raise click.UsageError("--since requires --changed")
    selection = next(
        (
            name
            for name, selected in (
                ("changed", options.changed),
                ("failed", options.failed),
                ("missing", options.missing),
            )
            if selected
        ),
        None,
    )
    return _ResolvedBuildRequest(
        target=options.mode or options.target,
        selection=selection,
        from_stage=options.stage or options.from_stage,
        through_stage=options.stage or options.through_stage,
    )


def _selected_chapter_filter(
    options: _BuildCommandOptions,
    request: _ResolvedBuildRequest,
    manifest: AudiobookManifest,
    target: BuildTarget,
) -> str | None:
    if request.selection is None:
        return options.chapter
    since_manifest = (
        _release_manifest_path(options.since, target.output_root / "release")
        if options.since is not None
        else None
    )
    selected = select_build_chapters(
        manifest,
        selection=request.selection,
        target_name=request.target,
        profile_override=options.profile,
        chapter_selector=options.chapter,
        since_release=since_manifest,
        from_stage=request.from_stage,
        through_stage=request.through_stage,
    )
    return ",".join(selected) if selected else None


def _build_budget(
    options: _BuildCommandOptions,
    target: BuildTarget,
) -> HostedUsageBudget:
    return _resolved_hosted_budget(
        max_submitted_characters=options.max_submitted_characters,
        max_provider_requests=options.max_provider_requests,
        max_estimated_spend=options.max_estimated_spend,
        currency=options.currency,
        pricing_source=options.pricing_source,
        confirm_above_characters=options.confirm_above_characters,
        confirm_above_requests=options.confirm_above_requests,
        target=target,
    )


def _validate_build_preflight(
    options: _BuildCommandOptions,
    hosted_work: HostedWorkEstimate | None,
    budget: HostedUsageBudget,
) -> None:
    if hosted_work is None:
        return
    validate_hosted_preflight(budget, hosted_work)
    _confirm_hosted_work(
        hosted_work,
        budget,
        yes=options.yes,
        dry_run=options.dry_run,
        operation="audiobook build",
    )


@contextmanager
def _build_progress(
    options: _BuildCommandOptions,
    total: int,
) -> Iterator[BuildProgressCallback | None]:
    enabled = (
        not options.no_progress
        and not options.dry_run
        and not _context().json_output
        and not _context().quiet
        and sys.stderr.isatty()
    )
    if not enabled:
        yield None
        return
    display = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=error_console,
    )
    display.start()
    task = display.add_task("Building audiobook", total=total)

    def update(event: BuildProgress) -> None:
        if event.event == "started":
            display.update(task, description=f"{event.stage}: {event.node_id}")
        else:
            display.update(task, completed=event.completed)

    try:
        yield update
    finally:
        display.stop()


def _emit_up_to_date(request: _ResolvedBuildRequest) -> None:
    _emit(
        {
            "schema_version": 1,
            "status": "up_to_date",
            "target": request.target,
            "selection": request.selection,
            "chapters": [],
        },
        f"No {request.selection} chapters to build",
    )


def _emit_build_result(result: BuildResult) -> None:
    _emit(
        {
            "schema_version": 1,
            "run_id": result.run_id,
            "status": result.status,
            "target": result.target,
            "plan_fingerprint": result.plan_fingerprint,
            "artifacts": [str(item.path) for item in result.artifacts],
            "reused_nodes": list(result.reused_nodes),
            "failed_nodes": list(result.failed_nodes),
            "hosted_usage": _usage_value(result.hosted_usage),
            "preflight": result.preflight.to_dict(),
            "resumed": result.resumed,
        },
        f"Build {result.status}: {len(result.artifacts)} artifact(s), "
        f"{len(result.reused_nodes)} reused; synthesize "
        f"{result.preflight.pending_synthesis_chunks}/"
        f"{result.preflight.synthesis_chunks} chunk(s)",
    )


@main.command("audition")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--target", default="default", show_default=True)
@click.option("--profile", "--profiles", "profiles", multiple=True, required=True)
@click.option("--chapter", "--chapters")
@explicit_text_options
@click.option(
    "--matrix",
    multiple=True,
    metavar="KEY=VALUE[,VALUE...]",
    help="Render the deterministic Cartesian product of profile settings.",
)
@deprecated_api_key_option
def audition_command(  # noqa: PLR0917 - Click injects one parameter per CLI option.
    manifest: Path,
    target: str,
    profiles: tuple[str, ...],
    chapter: str | None,
    text: str | None,
    text_file: Path | None,
    matrix: tuple[str, ...],
    api_key: str | None,
) -> None:
    """Render a short profile comparison for local listening."""
    config = _load_config()
    try:
        sample = _optional_text(text, text_file)
        if sample is not None and chapter is not None:
            raise ValidationError(
                "Provide a chapter selector or explicit audition text, not both"
            )
        records = asyncio.run(
            audition_audiobook(
                load_manifest(manifest),
                profiles=profiles,
                target_name=target,
                text=sample,
                chapter_selector=chapter,
                api_key=_optional_api_key(api_key, config),
                matrix=matrix,
            )
        )
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit(
        {
            "artifacts": [
                record.to_dict(root=manifest.resolve().parent) for record in records
            ]
        },
        "\n".join(str(record.path) for record in records),
    )


@main.command("preview")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--target", default="default", show_default=True)
@click.option("--profile")
@click.option("--chapter", "--chapters")
@explicit_text_options
@deprecated_api_key_option
def preview_command(  # noqa: PLR0917 - Click injects one parameter per CLI option.
    manifest: Path,
    target: str,
    profile: str | None,
    chapter: str | None,
    text: str | None,
    text_file: Path | None,
    api_key: str | None,
) -> None:
    """Generate a bounded preview without mutating production artifacts."""

    config = _load_config()
    try:
        sample = _optional_text(text, text_file)
        if sample is not None and chapter is not None:
            raise ValidationError(
                "Provide a chapter selector or explicit preview text, not both"
            )
        record = asyncio.run(
            preview_audiobook(
                load_manifest(manifest),
                target_name=target,
                profile_override=profile,
                text=sample,
                chapter_selector=chapter,
                api_key=_optional_api_key(api_key, config),
            )
        )
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit(
        {"artifact": record.to_dict(root=manifest.resolve().parent)},
        f"Wrote preview {record.path}",
    )


@main.command("inspect")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path))
@click.option("--target", default="default", show_default=True)
def inspect_command(manifest: Path, paths: tuple[Path, ...], target: str) -> None:
    """Inspect generated audio with FFprobe."""
    try:
        loaded = load_manifest(manifest)
        selected = paths or tuple(
            record.path
            for record in inventory_artifacts(loaded.target(target).output_root).records
            if record.media_type and record.media_type.startswith("audio/")
        )
        inspections = [inspect_audio(path).to_dict() for path in selected]
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit({"inspections": inspections}, f"Inspected {len(inspections)} file(s)")


@main.command("assemble")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--target", default="default", show_default=True)
def assemble_command(manifest: Path, target: str) -> None:
    """Assemble the optional configured M4B release."""
    try:
        destination = assemble_release(load_manifest(manifest), target_name=target)
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit({"path": str(destination)}, f"Assembled {destination}")


@main.group("release")
def release_group() -> None:
    """Validate and publish immutable release evidence."""


@release_group.command("check")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--target", default="default", show_default=True)
@click.option(
    "--write-manifest",
    is_flag=True,
    help=(
        "Publish a new immutable release directory and release manifest after "
        "validation."
    ),
)
def release_check_command(manifest: Path, target: str, write_manifest: bool) -> None:
    """Validate release readiness; optionally publish immutable evidence."""
    try:
        result = check_release(
            load_manifest(manifest),
            target_name=target,
            write_manifest=write_manifest,
        )
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit(
        result.to_dict(),
        "Release complete" if result.complete else "\n".join(result.issues),
        status="ok" if result.complete else "partial_failure",
        exit_code=0 if result.complete else 1,
    )
    if not result.complete:
        raise click.exceptions.Exit(1)


@release_group.command("diff")
@click.argument("left")
@click.argument("right")
@click.option(
    "--manifest",
    type=click.Path(path_type=Path),
    default="yakbox.toml",
    show_default=True,
)
@click.option("--target", default="default", show_default=True)
def release_diff_command(
    left: str,
    right: str,
    manifest: Path,
    target: str,
) -> None:
    """Compare two release IDs or release.json paths."""
    try:
        loaded = load_manifest(manifest)
        release_root = loaded.target(target).output_root / "release"
        left_path = _release_manifest_path(left, release_root)
        right_path = _release_manifest_path(right, release_root)
        result = diff_releases(left_path, right_path)
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit(
        result.to_dict(),
        (
            "Releases are identical"
            if result.identical
            else f"{len(result.added_artifacts)} added, "
            f"{len(result.removed_artifacts)} removed, "
            f"{len(result.changed_artifacts)} changed artifact(s), "
            f"{len(result.metadata_changes)} metadata change(s)"
        ),
    )


@main.command("status")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--target", default="default", show_default=True)
def status_command(manifest: Path, target: str) -> None:
    """Show run and managed-artifact status."""
    try:
        loaded = load_manifest(manifest)
        run_files = sorted((loaded.root / ".yakbox" / "runs").glob("*/run.json"))
        runs = [json.loads(path.read_text(encoding="utf-8")) for path in run_files]
        inventory = inventory_artifacts(loaded.target(target).output_root)
    except (YakboxError, OSError, json.JSONDecodeError) as error:
        _fail(error)
    _emit(
        {
            "runs": runs,
            "managed_artifacts": len(inventory.records),
            "managed_bytes": inventory.managed_bytes,
            "unknown_files": [str(path) for path in inventory.unknown_files],
        },
        f"{len(runs)} run(s), {len(inventory.records)} managed artifact(s), "
        f"{inventory.managed_bytes} bytes",
    )


@main.command("explain")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--target", default="default", show_default=True)
@click.option("--chapter", "--chapters")
@click.option("--artifact")
def explain_command(
    manifest: Path, target: str, chapter: str | None, artifact: str | None
) -> None:
    """Explain planned fingerprints and artifact reuse."""
    try:
        loaded = load_manifest(manifest)
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
        plan = plan_audiobook(
            loaded, document, target_name=target, chapter_selector=chapter
        )
        preflight = preflight_audiobook_build(
            loaded,
            target_name=target,
            chapter_selector=chapter,
        )
        reasons = dict(preflight.change_summary.reasons)
        nodes = [
            node
            for node in plan.nodes
            if artifact is None or artifact in {node.id, str(node.output)}
        ]
        states = {
            node.id: (
                "reusable",
                reasons.get(node.id, "verified artifact matches"),
            )
            if node.id in preflight.reusable_node_ids
            else (
                "scheduled",
                reasons.get(node.id, "required artifact is missing or invalid"),
            )
            for node in nodes
        }
    except (YakboxError, ValueError) as error:
        _fail(error)
    _emit(
        {
            "plan_fingerprint": plan.fingerprint,
            "nodes": [
                {
                    "id": node.id,
                    "stage": node.stage.value,
                    "fingerprint": node.fingerprint,
                    "dependencies": list(node.dependencies),
                    "output": str(node.output),
                    "exists": node.output.exists(),
                    "state": states[node.id][0],
                    "reason": states[node.id][1],
                }
                for node in nodes
            ],
        },
        "\n".join(
            f"{node.id}: {node.fingerprint[:12]} "
            f"({states[node.id][0]}: {states[node.id][1]})"
            for node in nodes
        ),
    )


@main.command("doctor")
@click.argument("manifest", required=False, type=click.Path(path_type=Path))
@click.option("--target")
@click.option("--backend")
@click.option("--network", is_flag=True)
@click.option("--deep", is_flag=True)
@click.option(
    "--whisper",
    "check_whisper",
    is_flag=True,
    help="Check the local MLX Whisper package and pinned model.",
)
def doctor_command(  # noqa: PLR0917 - Click injects CLI parameters.
    manifest: Path | None,
    target: str | None,
    backend: str | None,
    network: bool,
    deep: bool,
    check_whisper: bool,
) -> None:
    """Run read-only installation and workspace diagnostics."""
    config = _load_config()
    try:
        report = asyncio.run(
            run_doctor(
                manifest,
                target=target,
                backend=backend,
                network=network,
                deep=deep,
                whisper=check_whisper,
                api_key=_optional_api_key(None, config),
            )
        )
    except (YakboxError, OSError) as error:
        _fail(error)
    if _context().json_output:
        _emit(
            report.to_dict(),
            "",
            status="ok" if report.healthy else "partial_failure",
            exit_code=0 if report.healthy else 1,
        )
    elif not _context().quiet:
        table = Table("Check", "Status", "Summary", "Action")
        for item in report.diagnostics:
            table.add_row(item.id, item.status.value, item.summary, item.action or "")
        console.print(table)
    if not report.healthy:
        raise click.exceptions.Exit(1)


@main.group("artifacts")
def artifacts_group() -> None:
    """Inspect and safely clean generated artifacts."""


@artifacts_group.command("list")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--target", default="default")
@click.option("--kind", type=click.Choice([item.value for item in ArtifactKind]))
def artifacts_list_command(manifest: Path, target: str, kind: str | None) -> None:
    loaded = _manifest(manifest)
    report = inventory_artifacts(loaded.target(target).output_root)
    records = [
        record
        for record in report.records
        if kind is None or record.kind is ArtifactKind(kind)
    ]
    _emit(
        {"artifacts": [record.to_dict(root=loaded.root) for record in records]},
        "\n".join(f"{record.kind.value:10} {record.path}" for record in records)
        or "No managed artifacts",
    )


@artifacts_group.command("usage")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--target", default="default")
def artifacts_usage_command(manifest: Path, target: str) -> None:
    loaded = _manifest(manifest)
    report = inventory_artifacts(loaded.target(target).output_root)
    _emit(
        {
            "managed_bytes": report.managed_bytes,
            "total_bytes": report.total_bytes,
            "unknown_files": [str(path) for path in report.unknown_files],
        },
        f"Managed {report.managed_bytes} of {report.total_bytes} bytes",
    )


@artifacts_group.group("cache")
def artifacts_cache_group() -> None:
    """Inspect and prune reusable synthesis chunks."""


@artifacts_cache_group.command("list")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
def artifacts_cache_list_command(manifest: Path) -> None:
    loaded = _manifest(manifest)
    inventory = inventory_synthesis_cache(loaded.root)
    _emit(
        inventory.to_dict(root=loaded.root),
        f"{len(inventory.entries)} cached chunk(s), "
        f"{inventory.total_bytes} bytes, {inventory.invalid_entries} invalid",
    )


@artifacts_cache_group.command("inspect")
@click.argument("fingerprint")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
def artifacts_cache_inspect_command(fingerprint: str, manifest: Path) -> None:
    """Show privacy-safe request metadata for one cache entry."""
    loaded = _manifest(manifest)
    entry = next(
        (
            item
            for item in inventory_synthesis_cache(loaded.root).entries
            if item.fingerprint == fingerprint
        ),
        None,
    )
    if entry is None:
        _fail(ValidationError(f"Unknown synthesis cache fingerprint: {fingerprint}"))
    value = entry.to_dict(root=loaded.root)
    _emit(value, f"{fingerprint}: {'valid' if entry.valid else 'invalid'}")


@artifacts_cache_group.command("why-miss")
@click.argument("chunk_id")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--target", default="default", show_default=True)
def artifacts_cache_why_miss_command(
    chunk_id: str,
    manifest: Path,
    target: str,
) -> None:
    """Explain the exact settings that caused a chunk cache miss."""
    try:
        value = explain_synthesis_chunk(
            load_manifest(manifest),
            chunk_id=chunk_id,
            target_name=target,
        )
    except (YakboxError, OSError, ValueError) as error:
        _fail(error)
    reasons = value.get("reasons")
    detail = (
        ", ".join(str(item) for item in reasons) if isinstance(reasons, list) else ""
    )
    _emit(value, f"{chunk_id}: {value['status']}" + (f" ({detail})" if detail else ""))


@artifacts_cache_group.command("clean")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--max-age-days", type=click.IntRange(min=0))
@click.option("--max-bytes", type=click.IntRange(min=0))
@click.option("--apply", "apply_changes", is_flag=True)
def artifacts_cache_clean_command(
    manifest: Path,
    max_age_days: int | None,
    max_bytes: int | None,
    apply_changes: bool,
) -> None:
    """Plan cache cleanup; delete only when --apply is explicit."""
    loaded = _manifest(manifest)
    try:
        plan = plan_cache_cleanup(
            loaded.root,
            max_age_days=max_age_days,
            max_bytes=max_bytes,
        )
        removed = apply_cache_cleanup(plan) if apply_changes else 0
    except (YakboxError, OSError) as error:
        _fail(error)
    value = plan.to_dict(workspace=loaded.root)
    value["applied"] = apply_changes
    value["removed"] = removed
    _emit(
        value,
        (
            f"Removed {removed} cached chunk(s), reclaimed up to "
            f"{plan.bytes_reclaimed} bytes"
            if apply_changes
            else f"Would remove {len(plan.candidates)} cached chunk(s), "
            f"reclaiming {plan.bytes_reclaimed} bytes; pass --apply"
        ),
    )


@artifacts_group.command("verify")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--target", default="default")
@click.option(
    "--repair-metadata",
    is_flag=True,
    help="Explicitly accept valid current bytes and refresh digest metadata.",
)
def artifacts_verify_command(
    manifest: Path,
    target: str,
    repair_metadata: bool,
) -> None:
    loaded = _manifest(manifest)
    artifact_root = loaded.target(target).output_root
    report = inventory_artifacts(artifact_root)
    failures: list[dict[str, str | None]] = []
    repaired: list[str] = []
    for record in report.records:
        valid, error = verify_artifact(record)
        if valid:
            continue
        if repair_metadata and error in {
            "size differs from manifest",
            "digest differs from manifest",
        }:
            try:
                repair_artifact_metadata(record, root=artifact_root)
            except YakboxError as repair_error:
                failures.append({"path": str(record.path), "error": str(repair_error)})
            else:
                repaired.append(str(record.path))
            continue
        failures.append({"path": str(record.path), "error": error})
    _emit(
        {
            "verified": len(report.records) - len(failures),
            "repaired": repaired,
            "failures": failures,
        },
        "All managed artifacts verified"
        if not failures
        else "\n".join(f"{item['path']}: {item['error']}" for item in failures),
        status="partial_failure" if failures else "ok",
        exit_code=1 if failures else 0,
    )
    if failures:
        raise click.exceptions.Exit(1)


@artifacts_group.command("clean")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--target", default="default")
@click.option("--kind", type=click.Choice([item.value for item in ArtifactKind]))
@click.option("--older-than", type=click.IntRange(min=0))
@click.option("--keep-runs", type=click.IntRange(min=0))
@click.option("--apply", "apply_plan", is_flag=True)
def artifacts_clean_command(  # noqa: PLR0917 - Click injects CLI parameters.
    manifest: Path,
    target: str,
    kind: str | None,
    older_than: int | None,
    keep_runs: int | None,
    apply_plan: bool,
) -> None:
    loaded = _manifest(manifest)
    try:
        current_plan = plan_audiobook(
            loaded,
            normalize_sources(
                loaded.sources,
                pronunciations=loaded.pronunciations,
                max_pause_ms=loaded.max_pause_ms,
                strip_attribution_tags=loaded.dialogue.strip_attribution_tags,
                dialogue_routes=loaded.dialogue.routes,
                expressive_tag_handling=loaded.dialogue.expressive_tag_handling,
                retain_first_attribution_per_scene=(
                    loaded.dialogue.retain_first_attribution_per_scene
                ),
            ),
            target_name=target,
        )
        plan = plan_cleanup(
            loaded.root,
            loaded.target(target).output_root,
            kind=ArtifactKind(kind) if kind else None,
            older_than_days=older_than,
            target=target,
            keep_successful_runs=(
                keep_runs
                if keep_runs is not None
                else loaded.retention.keep_successful_runs
            ),
            audition_days=loaded.retention.audition_days,
            preview_days=loaded.retention.preview_days,
            raw_until_release=loaded.retention.raw_until_release,
            current_paths=tuple(
                node.output
                for node in current_plan.nodes
                if node.stage.value in {"master", "encode_mp3", "inspect"}
            ),
        )
        trash = apply_cleanup(plan) if apply_plan else None
    except YakboxError as error:
        _fail(error)
    _emit(
        {
            **plan.to_dict(),
            "applied": apply_plan,
            "trash": str(trash) if trash else None,
        },
        f"{len(plan.candidates)} candidate(s), {plan.bytes_reclaimable} bytes"
        + (f"; quarantined in {trash}" if trash else "; dry run"),
    )


@artifacts_group.group("trash")
def artifacts_trash_group() -> None:
    """Restore or permanently purge quarantine entries."""


@artifacts_trash_group.command("list")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
def artifacts_trash_list_command(manifest: Path) -> None:
    loaded = _manifest(manifest)
    trash = loaded.root / ".yakbox" / "trash"
    entries = (
        sorted(path.name for path in trash.iterdir() if path.is_dir())
        if trash.exists()
        else []
    )
    _emit({"entries": entries}, "\n".join(entries) or "Trash is empty")


@artifacts_trash_group.command("restore")
@click.argument("cleanup_id")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option(
    "--path",
    "relative_path",
    type=click.Path(path_type=Path),
    help="Restore only this path relative to the artifact root.",
)
def artifacts_trash_restore_command(
    cleanup_id: str,
    manifest: Path,
    relative_path: Path | None,
) -> None:
    loaded = _manifest(manifest)
    try:
        count = restore_trash(
            loaded.root,
            cleanup_id,
            relative_path=relative_path,
        )
    except YakboxError as error:
        _fail(error)
    _emit(
        {
            "cleanup_id": cleanup_id,
            "path": str(relative_path) if relative_path else None,
            "restored": count,
        },
        f"Restored {count} artifact(s)",
    )


@artifacts_trash_group.command("purge")
@click.argument("cleanup_id", required=False)
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option(
    "--all",
    "purge_all",
    is_flag=True,
    help="Permanently remove every quarantined cleanup entry.",
)
@click.option(
    "--yes",
    is_flag=True,
    required=True,
    help="Confirm that permanent deletion is intentional.",
)
def artifacts_trash_purge_command(
    cleanup_id: str | None, manifest: Path, purge_all: bool, yes: bool
) -> None:
    del yes
    if (cleanup_id is None) == (not purge_all):
        raise click.UsageError("Provide CLEANUP_ID or --all, but not both")
    loaded = _manifest(manifest)
    try:
        count = purge_trash(loaded.root, cleanup_id)
    except YakboxError as error:
        _fail(error)
    _emit({"purged": count}, f"Permanently removed {count} quarantine entrie(s)")


@main.command("tts")
@click.argument("text", required=False)
@text_file_option
@click.option("--backend")
@click.option("--voice", default="narrator", show_default=True)
@click.option("--profile")
@audio_output_options("speech.wav")
@hosted_budget_options
def tts_command(  # noqa: PLR0917 - Click injects one parameter per CLI option.
    text: str | None,
    text_file: Path | None,
    backend: str | None,
    voice: str,
    profile: str | None,
    out: Path,
    output_format: str,
    overwrite: bool,
    max_submitted_characters: int | None,
    max_provider_requests: int | None,
    max_estimated_spend: Decimal | None,
    currency: str | None,
    pricing_source: str | None,
    price_per_character: Decimal | None,
    confirm_above_characters: int | None,
    confirm_above_requests: int | None,
    yes: bool,
) -> None:
    """Synthesize a single text using the shared speech service."""
    content = _read_text(text, text_file)
    config = _load_config()
    selected = backend or config.default_backend
    try:
        budget = _resolved_hosted_budget(
            max_submitted_characters=max_submitted_characters,
            max_provider_requests=max_provider_requests,
            max_estimated_spend=max_estimated_spend,
            currency=currency,
            pricing_source=pricing_source,
            confirm_above_characters=confirm_above_characters,
            confirm_above_requests=confirm_above_requests,
        )
        hosted = selected.casefold() in {"resemble", "cloud"}
        if hosted:
            estimate = estimate_hosted_work(
                (content,),
                price_per_character=price_per_character,
            )
            validate_hosted_preflight(budget, estimate)
            _confirm_hosted_work(
                estimate,
                budget,
                yes=yes,
                dry_run=False,
                operation="hosted TTS",
            )
        artifact, usage = asyncio.run(
            _direct_tts(
                content,
                backend=selected,
                voice=voice,
                profile=profile,
                out=out,
                output_format=AudioFormat(output_format),
                overwrite=overwrite,
                api_key=_api_key(None, config) if hosted else None,
                budget=budget,
                price_per_character=price_per_character,
            )
        )
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit(
        {**_artifact_value(artifact), "hosted_usage": _usage_value(usage)},
        f"Wrote {artifact.path}",
    )


@main.command("vc")
@click.argument("input_audio", type=click.Path(exists=True, path_type=Path))
@click.option("--backend", default="local", show_default=True)
@click.option("--voice", default="narrator", show_default=True)
@click.option("--profile")
@click.option("--reference-audio", type=click.Path(exists=True, path_type=Path))
@audio_output_options("converted.wav", include_format=False)
def vc_command(  # noqa: PLR0917 - Click injects one parameter per CLI option.
    input_audio: Path,
    backend: str,
    voice: str,
    profile: str | None,
    reference_audio: Path | None,
    out: Path,
    overwrite: bool,
) -> None:
    """Transform an audio file using a shared transformation service."""
    try:
        artifact = asyncio.run(
            _direct_vc(
                input_audio,
                backend=backend,
                voice=voice,
                profile=profile,
                reference_audio=reference_audio,
                out=out,
                overwrite=overwrite,
            )
        )
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit(_artifact_value(artifact), f"Wrote {artifact.path}")


@main.command("batch")
@click.argument("script_file", type=click.Path(exists=True, path_type=Path))
@click.option("--backend")
@click.option("--voice", default="narrator")
@click.option("--out-dir", type=click.Path(path_type=Path), default="batch_output")
def local_batch_command(
    script_file: Path, backend: str | None, voice: str, out_dir: Path
) -> None:
    """Run a sequential local batch using the shared input format."""
    config = _load_config()
    selected = backend or config.default_backend
    if selected.casefold() in {"resemble", "cloud"}:
        raise click.UsageError(
            "The batch command is local-only; use 'yakbox cloud batch' for "
            "hosted synthesis, budgets, durable journals, and resume support"
        )
    if selected.casefold() not in {"fake", "local", "chatterbox", "chatterbox-local"}:
        raise click.UsageError(
            f"The batch command does not support configured backend {selected!r}; "
            "choose a local backend or use 'yakbox cloud batch'"
        )
    rows = read_batch_rows(script_file)
    try:
        artifacts = asyncio.run(_direct_batch(rows, selected, voice, out_dir, None))
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit(
        {"artifacts": [_artifact_value(item) for item in artifacts]},
        f"Wrote {len(artifacts)} file(s)",
    )


@main.command("verify")
@click.argument(
    "audio", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path)
)
def verify_command(audio: tuple[Path, ...]) -> None:
    """Inspect local audio files."""
    try:
        reports = [inspect_audio(path).to_dict() for path in audio]
    except YakboxError as error:
        _fail(error)
    _emit({"inspections": reports}, f"Verified {len(reports)} file(s)")


@main.command("models")
def models_command() -> None:
    """Report local model-package availability without loading models."""
    available = importlib.util.find_spec("chatterbox") is not None
    _emit(
        {"chatterbox_installed": available},
        "Chatterbox installed" if available else "Chatterbox not installed",
    )


@main.group("backends")
def backends_group() -> None:
    """Inspect available speech backends."""


@backends_group.command("list")
def backends_list_command() -> None:
    config = _load_config()
    credential = _optional_api_key(None, config)
    _emit(
        {
            "backends": [
                {"name": "fake", "available": True},
                {"name": "chatterbox-local", "available": _module("chatterbox")},
                {
                    "name": "resemble",
                    "available": credential is not None,
                },
                {
                    "name": "chatterbox-remote",
                    "available": False,
                    "reason": "no verified service contract",
                },
            ]
        },
        "fake\nchatterbox-local\nresemble\nchatterbox-remote (contract unavailable)",
    )


@backends_group.command("capabilities")
@click.argument("name")
def backends_capabilities_command(name: str) -> None:
    config = _load_config()

    async def load() -> BackendCapabilities:
        credential = (
            _api_key(None, config) if name.casefold() in {"resemble", "cloud"} else None
        )
        async with open_speech_backend(name, api_key=credential) as service:
            return service.capabilities

    try:
        capabilities = asyncio.run(load())
    except YakboxError as error:
        _fail(error)
    _emit(asdict(capabilities), str(capabilities))


@main.group("shards")
def shards_group() -> None:
    """Export and verify deterministic build shards."""


@shards_group.command("export")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.option("--target", default="default", show_default=True)
@click.option("--count", type=click.IntRange(min=1), required=True)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=".yakbox/shards",
    show_default=True,
)
def shards_export_command(
    manifest: Path, target: str, count: int, out_dir: Path
) -> None:
    try:
        loaded = load_manifest(manifest)
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
        plan = plan_audiobook(loaded, document, target_name=target)
        directory = (
            out_dir if out_dir.is_absolute() else loaded.root / out_dir
        ).resolve()
        paths = export_shard_manifests(plan, directory, count=count, root=loaded.root)
    except (YakboxError, ValueError, OSError) as error:
        _fail(error)
    _emit(
        {"shards": [str(path) for path in paths]},
        f"Exported {len(paths)} shard manifest(s) to {directory}",
    )


@shards_group.command("verify")
@click.argument("manifest", type=click.Path(path_type=Path), default="yakbox.toml")
@click.argument(
    "shard_files", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path)
)
def shards_verify_command(manifest: Path, shard_files: tuple[Path, ...]) -> None:
    loaded = _manifest(manifest)
    try:
        shards = verify_shard_manifests(shard_files, root=loaded.root)
    except YakboxError as error:
        _fail(error)
    _emit(
        {
            "verified": len(shards),
            "plan_fingerprint": shards[0].plan_fingerprint,
        },
        f"Verified {len(shards)} complete shard(s)",
    )


@main.group("cloud")
@click.option(
    "--credential-profile",
    "--profile",
    "credential_profile",
    default="default",
    show_default=True,
    help=(
        "Select the OS-keyring credential profile; --profile is a compatibility alias."
    ),
)
def cloud_group(credential_profile: str) -> None:
    """Use Resemble.ai synthesis and provider management."""
    _context().credential_profile = credential_profile


@cloud_group.command("tts")
@click.argument("text", required=False)
@text_file_option
@click.option("--voice-uuid")
@click.option("--project-uuid")
@audio_output_options("cloud.wav")
@click.option(
    "--precision",
    type=click.Choice([item.value for item in Precision]),
    default="PCM_32",
)
@click.option("--sample-rate", type=click.IntRange(min=1))
@click.option("--title")
@click.option("--hd", is_flag=True)
@click.option("--custom-pronunciations", is_flag=True)
@hosted_budget_options
@deprecated_api_key_option
def cloud_tts_command(  # noqa: PLR0913,PLR0917 - Click injects CLI parameters.
    text: str | None,
    text_file: Path | None,
    voice_uuid: str | None,
    project_uuid: str | None,
    out: Path,
    output_format: str,
    precision: str,
    sample_rate: int | None,
    title: str | None,
    hd: bool,
    custom_pronunciations: bool,
    overwrite: bool,
    max_submitted_characters: int | None,
    max_provider_requests: int | None,
    max_estimated_spend: Decimal | None,
    currency: str | None,
    pricing_source: str | None,
    price_per_character: Decimal | None,
    confirm_above_characters: int | None,
    confirm_above_requests: int | None,
    yes: bool,
    api_key: str | None,
) -> None:
    content = _read_text(text, text_file)
    config = _load_config()
    voice = voice_uuid or config.resemble_voice_uuid
    if not voice:
        _fail(ValueError("voice UUID is required"))
    try:
        budget = _resolved_hosted_budget(
            max_submitted_characters=max_submitted_characters,
            max_provider_requests=max_provider_requests,
            max_estimated_spend=max_estimated_spend,
            currency=currency,
            pricing_source=pricing_source,
            confirm_above_characters=confirm_above_characters,
            confirm_above_requests=confirm_above_requests,
        )
        estimate = estimate_hosted_work(
            (content,),
            price_per_character=price_per_character,
        )
        validate_hosted_preflight(budget, estimate)
        _confirm_hosted_work(
            estimate,
            budget,
            yes=yes,
            dry_run=False,
            operation="cloud TTS",
        )
        result, usage = asyncio.run(
            _cloud_tts(
                _api_key(api_key, config),
                SynthesisRequest(
                    text=content,
                    voice_uuid=voice,
                    project_uuid=project_uuid or config.resemble_project_uuid,
                    title=title,
                    precision=Precision(precision),
                    output_format=CloudAudioFormat(output_format),
                    sample_rate=sample_rate,
                    use_hd=hd,
                    apply_custom_pronunciations=custom_pronunciations,
                ),
                out=out,
                overwrite=overwrite,
                budget=budget,
                price_per_character=price_per_character,
            )
        )
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit(
        {
            "path": str(result.path),
            "bytes_written": result.bytes_written,
            "attempts": result.attempts,
            "request_id": result.request_id,
            "hosted_usage": _usage_value(usage),
        },
        f"Wrote {result.path}",
    )


@cloud_group.command("stream")
@click.argument("text", required=False)
@text_file_option
@click.option("--voice-uuid")
@click.option("--project-uuid")
@audio_output_options("stream.wav", include_format=False)
@click.option(
    "--precision",
    type=click.Choice([item.value for item in Precision]),
    default="PCM_32",
)
@click.option("--sample-rate", type=click.IntRange(min=1))
@click.option("--hd", is_flag=True)
@click.option("--custom-pronunciations", is_flag=True)
@hosted_budget_options
@deprecated_api_key_option
def cloud_stream_command(  # noqa: PLR0917 - Click injects CLI parameters.
    text: str | None,
    text_file: Path | None,
    voice_uuid: str | None,
    project_uuid: str | None,
    out: Path,
    precision: str,
    sample_rate: int | None,
    hd: bool,
    custom_pronunciations: bool,
    overwrite: bool,
    max_submitted_characters: int | None,
    max_provider_requests: int | None,
    max_estimated_spend: Decimal | None,
    currency: str | None,
    pricing_source: str | None,
    price_per_character: Decimal | None,
    confirm_above_characters: int | None,
    confirm_above_requests: int | None,
    yes: bool,
    api_key: str | None,
) -> None:
    content = _read_text(text, text_file)
    config = _load_config()
    voice = voice_uuid or config.resemble_voice_uuid
    if not voice:
        _fail(ValueError("voice UUID is required"))
    try:
        budget = _resolved_hosted_budget(
            max_submitted_characters=max_submitted_characters,
            max_provider_requests=max_provider_requests,
            max_estimated_spend=max_estimated_spend,
            currency=currency,
            pricing_source=pricing_source,
            confirm_above_characters=confirm_above_characters,
            confirm_above_requests=confirm_above_requests,
        )
        estimate = estimate_hosted_work(
            (content,),
            price_per_character=price_per_character,
        )
        validate_hosted_preflight(budget, estimate)
        _confirm_hosted_work(
            estimate,
            budget,
            yes=yes,
            dry_run=False,
            operation="cloud stream",
        )
        result, usage = asyncio.run(
            _cloud_stream(
                _api_key(api_key, config),
                StreamRequest(
                    text=content,
                    voice_uuid=voice,
                    project_uuid=project_uuid or config.resemble_project_uuid,
                    precision=Precision(precision),
                    sample_rate=sample_rate,
                    use_hd=hd,
                    apply_custom_pronunciations=custom_pronunciations,
                ),
                out=out,
                overwrite=overwrite,
                budget=budget,
                price_per_character=price_per_character,
            )
        )
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit(
        {
            "path": str(result.path),
            "bytes_written": result.bytes_written,
            "hosted_usage": _usage_value(usage),
        },
        f"Wrote {result.path}",
    )


@cloud_group.command("batch")
@click.argument("script_file", type=click.Path(exists=True, path_type=Path))
@click.option("--voice-uuid")
@click.option("--project-uuid")
@click.option(
    "--out-dir", type=click.Path(path_type=Path), default="cloud_batch_output"
)
@click.option("--concurrency", type=click.IntRange(1, 100))
@hosted_budget_options
@click.option(
    "--format", "output_format", type=click.Choice(["wav", "mp3"]), default="wav"
)
@click.option(
    "--precision",
    type=click.Choice([item.value for item in Precision]),
    default="PCM_32",
)
@click.option("--sample-rate", type=click.IntRange(min=1))
@click.option("--hd", is_flag=True)
@click.option("--custom-pronunciations", is_flag=True)
@click.option("--overwrite", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--journal", type=click.Path(path_type=Path))
@click.option(
    "--resume-from",
    "--resume",
    "resume",
    type=click.Path(exists=True, path_type=Path),
    help="Resume from a compatible cloud batch journal; --resume is an alias.",
)
@click.option("--report", type=click.Path(path_type=Path))
@click.option("--no-report", is_flag=True)
@click.option("--ignore-errors", is_flag=True)
@click.option("--no-progress", is_flag=True)
@deprecated_api_key_option
def cloud_batch_command(  # noqa: PLR0913,PLR0917 - Click injects CLI parameters.
    script_file: Path,
    voice_uuid: str | None,
    project_uuid: str | None,
    out_dir: Path,
    concurrency: int | None,
    max_submitted_characters: int | None,
    max_provider_requests: int | None,
    max_estimated_spend: Decimal | None,
    currency: str | None,
    pricing_source: str | None,
    price_per_character: Decimal | None,
    confirm_above_characters: int | None,
    confirm_above_requests: int | None,
    output_format: str,
    precision: str,
    sample_rate: int | None,
    hd: bool,
    custom_pronunciations: bool,
    overwrite: bool,
    dry_run: bool,
    journal: Path | None,
    resume: Path | None,
    report: Path | None,
    no_report: bool,
    ignore_errors: bool,
    no_progress: bool,
    yes: bool,
    api_key: str | None,
) -> None:
    config = _load_config()
    total_rows = sum(1 for _ in iter_batch_rows(script_file))
    selected_concurrency = concurrency or config.cloud_concurrency
    budget = _resolved_hosted_budget(
        max_submitted_characters=max_submitted_characters,
        max_provider_requests=max_provider_requests,
        max_estimated_spend=max_estimated_spend,
        currency=currency,
        pricing_source=pricing_source,
        confirm_above_characters=confirm_above_characters,
        confirm_above_requests=confirm_above_requests,
    )
    estimate = estimate_hosted_work(
        (
            row.text
            for row in iter_batch_rows(script_file)
            if row.validation_error is None
            and 0 < len(row.text) <= MAX_SYNTHESIS_CHARACTERS
        ),
        price_per_character=price_per_character,
    )
    try:
        validate_hosted_preflight(budget, estimate)
        _confirm_hosted_work(
            estimate,
            budget,
            yes=yes,
            dry_run=dry_run,
            operation="cloud batch",
        )
    except YakboxError as error:
        _fail(error)
    progress_enabled = (
        not no_progress
        and not _context().json_output
        and not _context().quiet
        and sys.stderr.isatty()
    )
    progress_display: Progress | None = None
    progress_task: int | None = None
    if progress_enabled:
        progress_display = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=error_console,
        )
        progress_display.start()
        progress_task = progress_display.add_task(
            "Synthesizing",
            total=total_rows,
        )

    def progress(_result: BatchResult) -> None:
        if progress_display is not None and progress_task is not None:
            progress_display.advance(progress_task)

    try:
        try:
            resolved_key = _optional_api_key(api_key, config)
            if resolved_key is None:
                resolved_key = "dry-run" if dry_run else _api_key(api_key, config)
            batch_report = asyncio.run(
                _cloud_batch(
                    resolved_key,
                    iter_batch_rows(script_file),
                    _CloudBatchRunOptions(
                        budget=budget,
                        price_per_character=price_per_character,
                        voice=voice_uuid or config.resemble_voice_uuid,
                        project=project_uuid or config.resemble_project_uuid,
                        out_dir=out_dir,
                        concurrency=selected_concurrency,
                        output_format=AudioFormat(output_format),
                        hd=hd,
                        precision=Precision(precision),
                        sample_rate=sample_rate,
                        custom_pronunciations=custom_pronunciations,
                        overwrite=overwrite,
                        dry_run=dry_run,
                        progress=progress,
                        journal=journal,
                        resume=resume,
                        report=report,
                        write_report=not no_report,
                        preflight=estimate,
                    ),
                )
            )
        except (YakboxError, OSError) as error:
            _fail(error)
    finally:
        if progress_display is not None:
            progress_display.stop()
    batch_status = (
        "aborted"
        if batch_report.aborted
        else "partial_failure"
        if batch_report.failed
        else "ok"
    )
    batch_exit_code = (
        1 if batch_report.aborted or (batch_report.failed and not ignore_errors) else 0
    )
    _emit(
        batch_report.to_dict(),
        f"{batch_report.ok} ok, {batch_report.failed} failed, "
        f"{sum(item.status is BatchStatus.NOT_RUN for item in batch_report.results)} "
        "not run"
        + (f" — aborted: {batch_report.abort_reason}" if batch_report.aborted else ""),
        status=batch_status,
        exit_code=batch_exit_code,
    )
    if batch_exit_code:
        raise click.exceptions.Exit(1)


@cloud_group.group("voices")
def cloud_voices_group() -> None:
    """Manage Resemble voices."""


@cloud_voices_group.command("list")
@click.option("--page", type=click.IntRange(min=1), default=1)
@click.option("--page-size", type=click.IntRange(1, 100), default=10)
def cloud_voices_list_command(page: int, page_size: int) -> None:
    config = _load_config()
    try:
        result = asyncio.run(_list_voices(_api_key(None, config), page, page_size))
    except YakboxError as error:
        _fail(error)
    _emit(
        {"items": [asdict(item) for item in result.items]},
        "\n".join(f"{item.uuid}  {item.name}" for item in result.items),
    )


@cloud_voices_group.group("recordings")
def cloud_recordings_group() -> None:
    """Manage voice recordings."""


@cloud_recordings_group.command("create")
@click.argument("voice_uuid")
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option("--name", required=True)
@click.option("--text", required=True)
@click.option("--emotion")
@click.option("--active/--inactive", default=True)
@click.option("--fill/--no-fill", default=False)
def cloud_recording_create_command(  # noqa: PLR0917 - Click injects CLI parameters.
    voice_uuid: str,
    audio_file: Path,
    name: str,
    text: str,
    emotion: str | None,
    active: bool,
    fill: bool,
) -> None:
    _recording_create_impl(
        voice_uuid,
        audio_file,
        name=name,
        text=text,
        emotion=emotion,
        active=active,
        fill=fill,
        deprecated_alias=False,
    )


@cloud_voices_group.command("recording", hidden=True)
@click.argument("voice_uuid")
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option("--name", required=True)
@click.option("--text", required=True)
@click.option("--emotion")
@click.option("--active/--inactive", default=True)
@click.option("--fill/--no-fill", default=False)
def cloud_recording_compat_command(  # noqa: PLR0917 - Click injects CLI parameters.
    voice_uuid: str,
    audio_file: Path,
    name: str,
    text: str,
    emotion: str | None,
    active: bool,
    fill: bool,
) -> None:
    _recording_create_impl(
        voice_uuid,
        audio_file,
        name=name,
        text=text,
        emotion=emotion,
        active=active,
        fill=fill,
        deprecated_alias=True,
    )


def _recording_create_impl(
    voice_uuid: str,
    audio_file: Path,
    *,
    name: str,
    text: str,
    emotion: str | None,
    active: bool,
    fill: bool,
    deprecated_alias: bool,
) -> None:
    if deprecated_alias:
        error_console.print(
            "Warning: cloud voices recording is deprecated; use "
            "cloud voices recordings create",
            style="yellow",
        )
    config = _load_config()
    try:
        result = asyncio.run(
            _create_recording(
                _api_key(None, config),
                voice_uuid,
                audio_file,
                name=name,
                text=text,
                emotion=emotion,
                active=active,
                fill=fill,
            )
        )
    except YakboxError as error:
        _fail(error)
    _emit(asdict(result), f"Created recording {result.uuid}")


@cloud_group.group("projects")
def cloud_projects_group() -> None:
    """Manage Resemble projects."""


@cloud_projects_group.command("list")
@click.option("--page", type=click.IntRange(min=1), default=1)
@click.option("--page-size", type=click.IntRange(1, 100), default=10)
def cloud_projects_list_command(page: int, page_size: int) -> None:
    config = _load_config()
    try:
        result = asyncio.run(_list_projects(_api_key(None, config), page, page_size))
    except YakboxError as error:
        _fail(error)
    _emit(
        {"items": [asdict(item) for item in result.items]},
        "\n".join(f"{item.uuid}  {item.name}" for item in result.items),
    )


@cloud_projects_group.command("create")
@click.argument("name")
@click.option("--description")
@click.option("--collaborative/--not-collaborative", default=False)
@click.option("--archived/--not-archived", default=False)
def cloud_project_create_command(
    name: str, description: str | None, collaborative: bool, archived: bool
) -> None:
    config = _load_config()
    try:
        result = asyncio.run(
            _create_project(
                _api_key(None, config),
                name,
                description,
                collaborative,
                archived,
            )
        )
    except YakboxError as error:
        _fail(error)
    _emit(asdict(result), f"Created project {result.uuid}")


@main.group("config")
def config_group() -> None:
    """Manage optional secure credentials."""


@config_group.group("auth")
def config_auth_group() -> None:
    """Store Resemble credentials in the operating-system keyring."""


@config_auth_group.command("login")
@click.option(
    "--credential-profile",
    "--profile",
    "profile",
    default="default",
    help="Name the OS-keyring credential profile; --profile is an alias.",
)
@click.password_option("--token", prompt="Resemble API token")
def auth_login_command(profile: str, token: str) -> None:
    keyring = _keyring()
    try:
        keyring.set_password("yakbox/resemble", profile, token)
    except Exception as error:  # noqa: BLE001 - third-party keyring boundary
        _fail(ConfigurationError(f"Secure keyring unavailable: {error}"))
    _emit({"profile": profile, "stored": True}, f"Stored profile {profile}")


@config_auth_group.command("logout")
@click.option(
    "--credential-profile",
    "--profile",
    "profile",
    default="default",
    help="Name the OS-keyring credential profile; --profile is an alias.",
)
def auth_logout_command(profile: str) -> None:
    keyring = _keyring()
    try:
        keyring.delete_password("yakbox/resemble", profile)
    except Exception as error:  # noqa: BLE001 - third-party keyring boundary
        _fail(ConfigurationError(f"Cannot remove keyring profile: {error}"))
    _emit({"profile": profile, "stored": False}, f"Removed profile {profile}")


@config_auth_group.command("status")
@click.option(
    "--credential-profile",
    "--profile",
    "profile",
    default="default",
    help="Name the OS-keyring credential profile; --profile is an alias.",
)
def auth_status_command(profile: str) -> None:
    keyring = _keyring()
    try:
        present = keyring.get_password("yakbox/resemble", profile) is not None
    except Exception as error:  # noqa: BLE001 - third-party keyring boundary
        _fail(ConfigurationError(f"Cannot access keyring: {error}"))
    _emit(
        {"profile": profile, "stored": present},
        "configured" if present else "not configured",
    )


async def _direct_tts(
    text: str,
    *,
    backend: str,
    voice: str,
    profile: str | None,
    out: Path,
    output_format: AudioFormat,
    overwrite: bool,
    api_key: str | None,
    budget: HostedUsageBudget,
    price_per_character: Decimal | None,
) -> tuple[SpeechArtifact, HostedUsageSnapshot | None]:
    async with open_speech_backend(
        backend,
        api_key=api_key,
        hosted_budget=budget,
        price_per_character=price_per_character,
    ) as service:
        artifact = await service.synthesize_to_file(
            SpeechSynthesisRequest(
                text=text,
                voice=voice,
                backend=backend,
                profile=profile,
                output_format=output_format,
            ),
            out,
            overwrite=overwrite,
        )
        usage = (
            await service.usage_snapshot()
            if isinstance(service, HostedUsageReportingService)
            else None
        )
        return artifact, usage


async def _direct_vc(
    input_audio: Path,
    *,
    backend: str,
    voice: str,
    profile: str | None,
    reference_audio: Path | None,
    out: Path,
    overwrite: bool,
) -> SpeechArtifact:
    async with open_transformation_backend(backend) as service:
        return await service.transform_to_file(
            SpeechTransformationRequest(
                input_path=input_audio,
                voice=voice,
                backend=backend,
                profile=profile,
                reference_audio=reference_audio,
            ),
            out,
            overwrite=overwrite,
        )


async def _direct_batch(
    rows: tuple[BatchRow, ...],
    backend: str,
    voice: str,
    out_dir: Path,
    api_key: str | None,
) -> tuple[SpeechArtifact, ...]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[SpeechArtifact] = []
    async with open_speech_backend(backend, api_key=api_key) as service:
        for row in rows:
            if row.validation_error or not row.text:
                continue
            result = await service.synthesize_to_file(
                SpeechSynthesisRequest(
                    text=row.text,
                    voice=row.voice_uuid or voice,
                    backend=backend,
                ),
                out_dir / f"{row.index:06d}.wav",
            )
            results.append(result)
    return tuple(results)


async def _cloud_tts(
    api_key: str,
    request: SynthesisRequest,
    *,
    out: Path,
    overwrite: bool,
    budget: HostedUsageBudget,
    price_per_character: Decimal | None,
) -> tuple[FileSynthesisResult, HostedUsageSnapshot]:
    usage = HostedUsageGate(budget, price_per_character=price_per_character)
    async with ResembleClient(api_key, usage_gate=usage) as client:
        result = await client.synthesize_to_file(request, out, overwrite=overwrite)
        return result, await usage.snapshot()


async def _cloud_stream(
    api_key: str,
    request: StreamRequest,
    *,
    out: Path,
    overwrite: bool,
    budget: HostedUsageBudget,
    price_per_character: Decimal | None,
) -> tuple[FileSynthesisResult, HostedUsageSnapshot]:
    usage = HostedUsageGate(budget, price_per_character=price_per_character)
    async with ResembleClient(api_key, usage_gate=usage) as client:
        result = await client.stream_to_file(request, out, overwrite=overwrite)
        return result, await usage.snapshot()


@dataclass(frozen=True, slots=True)
class _CloudBatchRunOptions:
    budget: HostedUsageBudget
    price_per_character: Decimal | None
    voice: str | None
    project: str | None
    out_dir: Path
    concurrency: int
    output_format: AudioFormat
    hd: bool
    precision: Precision
    sample_rate: int | None
    custom_pronunciations: bool
    overwrite: bool
    dry_run: bool
    progress: ProgressCallback
    journal: Path | None
    resume: Path | None
    report: Path | None
    write_report: bool
    preflight: HostedWorkEstimate


async def _cloud_batch(
    api_key: str,
    rows: Iterable[BatchRow],
    settings: _CloudBatchRunOptions,
) -> BatchReport:
    usage = HostedUsageGate(
        settings.budget,
        price_per_character=settings.price_per_character,
    )
    options = ClientOptions(
        max_connections=max(20, settings.concurrency),
        max_keepalive_connections=max(20, settings.concurrency),
    )
    async with ResembleClient(api_key, options=options, usage_gate=usage) as client:
        service = ResembleSpeechService(client, concurrency=settings.concurrency)
        try:
            return await run_cloud_batch(
                rows,
                service,
                default_voice=settings.voice,
                project_uuid=settings.project,
                out_dir=settings.out_dir,
                concurrency=settings.concurrency,
                output_format=settings.output_format,
                use_hd=settings.hd,
                precision=settings.precision,
                sample_rate=settings.sample_rate,
                apply_custom_pronunciations=settings.custom_pronunciations,
                overwrite=settings.overwrite,
                dry_run=settings.dry_run,
                progress=settings.progress,
                journal_path=settings.journal,
                resume_path=settings.resume,
                report_path=settings.report,
                write_report=settings.write_report,
                usage_gate=usage,
                preflight=settings.preflight,
            )
        finally:
            await service.aclose()


async def _list_voices(api_key: str, page: int, page_size: int) -> Page[Voice]:
    async with ResembleClient(api_key) as client:
        return await client.list_voices(page=page, page_size=page_size)


async def _list_projects(api_key: str, page: int, page_size: int) -> Page[Project]:
    async with ResembleClient(api_key) as client:
        return await client.list_projects(page=page, page_size=page_size)


async def _create_recording(
    api_key: str,
    voice_uuid: str,
    audio_file: Path,
    *,
    name: str,
    text: str,
    emotion: str | None,
    active: bool,
    fill: bool,
) -> Recording:
    async with ResembleClient(api_key) as client:
        return await client.create_recording(
            voice_uuid,
            audio_file,
            name=name,
            text=text,
            emotion=emotion,
            is_active=active,
            fill=fill,
        )


async def _create_project(
    api_key: str,
    name: str,
    description: str | None,
    collaborative: bool,
    archived: bool,
) -> Project:
    async with ResembleClient(api_key) as client:
        return await client.create_project(
            name,
            description=description,
            is_collaborative=collaborative,
            is_archived=archived,
        )


def _read_text(text: str | None, text_file: Path | None) -> str:
    if text is not None and text_file is not None:
        raise click.UsageError("Provide either TEXT or --text-file, not both")
    if text_file is not None:
        if str(text_file) == "-":
            return sys.stdin.read()
        return text_file.read_text(encoding="utf-8")
    if text is None:
        raise click.UsageError("Provide TEXT or --text-file")
    return text


def _optional_text(text: str | None, text_file: Path | None) -> str | None:
    if text is None and text_file is None:
        return None
    return _read_text(text, text_file)


def _load_config() -> YakboxConfig:
    try:
        return load_config()
    except YakboxError as error:
        _fail(error)


def _resolved_api_key(
    explicit: str | None,
    config: YakboxConfig,
) -> str | None:
    environment_key = os.environ.get("RESEMBLE_API_KEY") or None
    profile = _context().credential_profile
    keyring_key = (
        _keyring_password(profile)
        if explicit is None and environment_key is None
        else None
    )
    credential = resolve_resemble_credential(
        explicit=explicit,
        environment=environment_key,
        keyring=keyring_key,
        legacy_config=config.legacy_resemble_api_key,
        profile=profile,
    )
    if credential is None:
        return None
    if credential.source is CredentialSource.EXPLICIT:
        error_console.print(
            "Warning: --api-key is deprecated; use RESEMBLE_API_KEY or keyring",
            style="yellow",
        )
    elif credential.source is CredentialSource.LEGACY_CONFIG:
        error_console.print(
            "Warning: plaintext config API keys are deprecated; migrate with "
            "yakbox config auth login",
            style="yellow",
        )
    return credential.value


def _optional_api_key(explicit: str | None, config: YakboxConfig) -> str | None:
    return _resolved_api_key(explicit, config)


def _api_key(explicit: str | None, config: YakboxConfig) -> str:
    key = _resolved_api_key(explicit, config)
    if key is None:
        _fail(
            BackendUnavailableError(
                "No Resemble API key found; set RESEMBLE_API_KEY or configure "
                f"keyring profile {_context().credential_profile!r}"
            )
        )
    return key


def _manifest(path: Path) -> AudiobookManifest:
    try:
        return load_manifest(path)
    except YakboxError as error:
        _fail(error)


def _release_manifest_path(value: str, release_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    return (release_root / value / "release.json").resolve()


def _emit(
    value: dict[str, object],
    message: str,
    *,
    status: str = "ok",
    exit_code: int = 0,
) -> None:
    context = _context()
    if context.json_output:
        payload = {
            **runtime_metadata("cli-output"),
            "command": _command_identifier(),
            "status": status,
            "exit_code": exit_code,
            "data": value,
        }
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    elif not context.quiet and message:
        console.print(message)


def _emit_bootstrap_error(
    *,
    code: str,
    message: str,
    exit_code: int,
    status: str = "error",
    command: str | None = None,
) -> NoReturn:
    click.echo(
        json.dumps(
            {
                **runtime_metadata("cli-output"),
                "command": command or _command_identifier(),
                "status": status,
                "exit_code": exit_code,
                "error": {
                    "code": code,
                    "message": message.replace("\n", " ")[:2_048],
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    raise SystemExit(exit_code)


def _click_error_code(error: click.ClickException) -> str:
    if isinstance(error, click.NoSuchOption):
        return "unknown_option"
    if isinstance(error, click.MissingParameter):
        return "missing_parameter"
    if isinstance(error, click.BadParameter):
        return "invalid_argument"
    if isinstance(error, click.UsageError):
        return "usage_error"
    return "cli_error"


def _fail(error: Exception) -> NoReturn:
    context = _context()
    message = str(error).replace("\n", " ")[:2048]
    if context.json_output:
        click.echo(
            json.dumps(
                {
                    **runtime_metadata("cli-output"),
                    "command": _command_identifier(),
                    "status": "error",
                    "exit_code": 1,
                    "error": {
                        "code": stable_error_code(error),
                        "message": message,
                    },
                },
                sort_keys=True,
            )
        )
        raise click.exceptions.Exit(1)
    raise click.ClickException(message)


def _command_identifier() -> str:
    context = click.get_current_context(silent=True)
    names: list[str] = []
    while context is not None and context.parent is not None:
        if context.info_name:
            names.append(context.info_name)
        context = context.parent
    return " ".join(reversed(names)) or "unknown"


def _command_hint(arguments: Sequence[str]) -> str:
    command: click.Command = main
    names: list[str] = []
    for argument in arguments:
        if not isinstance(command, click.Group):
            break
        child = command.commands.get(argument)
        if child is not None:
            names.append(argument)
            command = child
    return " ".join(names) or "unknown"


def _resolved_hosted_budget(
    *,
    max_submitted_characters: int | None,
    max_provider_requests: int | None,
    max_estimated_spend: Decimal | None,
    currency: str | None,
    pricing_source: str | None,
    confirm_above_characters: int | None,
    confirm_above_requests: int | None,
    target: BuildTarget | None = None,
) -> HostedUsageBudget:
    resolved_currency = currency or (target.currency if target else None)
    resolved_source = pricing_source or (target.pricing_source if target else None)
    return HostedUsageBudget(
        max_submitted_characters=(
            max_submitted_characters
            if max_submitted_characters is not None
            else target.max_submitted_characters
            if target
            else None
        ),
        max_provider_requests=(
            max_provider_requests
            if max_provider_requests is not None
            else target.max_provider_requests
            if target
            else None
        ),
        max_estimated_spend=(
            max_estimated_spend
            if max_estimated_spend is not None
            else target.max_estimated_spend
            if target
            else None
        ),
        currency=CurrencyCode(resolved_currency) if resolved_currency else None,
        pricing_source=PricingSourceId(resolved_source) if resolved_source else None,
        confirm_above_characters=(
            confirm_above_characters
            if confirm_above_characters is not None
            else target.confirm_above_characters
            if target
            else None
        ),
        confirm_above_requests=(
            confirm_above_requests
            if confirm_above_requests is not None
            else target.confirm_above_requests
            if target
            else None
        ),
    )


def _confirm_hosted_work(
    estimate: HostedWorkEstimate,
    budget: HostedUsageBudget,
    *,
    yes: bool,
    dry_run: bool,
    operation: str,
) -> None:
    reasons = hosted_confirmation_reasons(budget, estimate)
    if not reasons or yes or dry_run:
        return
    summary = "; ".join(reasons)
    if not sys.stdin.isatty():
        raise ValidationError(
            f"{operation} requires confirmation: {summary}. "
            "Re-run with --yes after reviewing the preflight estimate"
        )
    click.confirm(
        f"{operation.capitalize()} requires confirmation ({summary}). Continue?",
        abort=True,
    )


def _artifact_value(artifact: SpeechArtifact) -> dict[str, object]:
    return {
        "path": str(artifact.path),
        "backend": artifact.backend,
        "voice": artifact.voice,
        "format": artifact.output_format.value,
        "bytes_written": artifact.bytes_written,
        "sha256": artifact.sha256,
    }


def _usage_value(value: HostedUsageSnapshot | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "logical_items": value.logical_items,
        "provider_attempts": value.provider_attempts,
        "submitted_characters": value.submitted_characters,
        "estimated_spend": (
            str(value.estimated_spend) if value.estimated_spend is not None else None
        ),
        "currency": str(value.currency) if value.currency is not None else None,
        "ambiguous_attempts": value.ambiguous_attempts,
        "basis": "estimate" if value.estimated_spend is not None else "usage_only",
    }


def _module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _keyring() -> KeyringApi:
    try:
        module = importlib.import_module("keyring")
    except ImportError:
        _fail(
            ConfigurationError(
                'Keyring support is not installed; install with "yakbox[credentials]"'
            )
        )
    return cast(KeyringApi, module)


def _keyring_password(profile: str) -> str | None:
    if not _module("keyring"):
        return None
    keyring = cast(KeyringApi, importlib.import_module("keyring"))
    try:
        return keyring.get_password("yakbox/resemble", profile)
    except Exception as error:  # noqa: BLE001 - optional keyring lookup is best-effort
        if _context().verbose:
            error_console.print(
                f"Warning: keyring profile {profile!r} is unavailable: {error}",
                style="yellow",
            )
        return None


register_dialogue_commands(main, emit=_emit, fail=_fail)
register_migration_commands(main, emit=_emit, fail=_fail)
register_repair_commands(main, emit=_emit, fail=_fail)
register_runtime_commands(main, emit=_emit, fail=_fail)
register_whisper_commands(main, emit=_emit, fail=_fail)
configure_cli_help(main)
