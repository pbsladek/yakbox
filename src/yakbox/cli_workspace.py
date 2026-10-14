"""Workspace initialization and shell completion commands."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import click
from click.shell_completion import ShellComplete, get_completion_class

from yakbox._files import atomic_write_bytes


def register_workspace_commands(
    main: click.Group, *, emit: Callable[[dict[str, object], str], None]
) -> None:
    """Register workspace helpers without importing the CLI composition root."""

    @main.command("init")
    @click.argument("directory", type=click.Path(path_type=Path), default=".")
    @click.option("--title", default="My Audiobook", help="Set the new book title.")
    @click.option("--author", default="Author Name", help="Set the new book author.")
    @click.option(
        "--backend",
        type=click.Choice(["fake", "chatterbox-local", "resemble"]),
        default="fake",
        help="Configure the starter profile for this speech backend.",
    )
    @click.option(
        "--voice-uuid",
        help="Set the Resemble voice ID; required with --backend resemble.",
    )
    def init_command(
        directory: Path, title: str, author: str, backend: str, voice_uuid: str | None
    ) -> None:
        """Initialize a minimal audiobook workspace."""
        for name, value in (("title", title), ("author", author)):
            if not value.strip():
                raise click.BadParameter("must not be blank", param_hint=f"--{name}")
        if backend == "resemble" and (voice_uuid is None or not voice_uuid.strip()):
            raise click.BadParameter(
                "is required with --backend resemble", param_hint="--voice-uuid"
            )
        if backend != "resemble" and voice_uuid is not None:
            raise click.BadParameter(
                "requires --backend resemble", param_hint="--voice-uuid"
            )
        manifest, source = _initialize_workspace(
            directory, title, author, _starter_profile(backend, voice_uuid)
        )
        emit({"manifest": str(manifest), "source": str(source)}, f"Created {manifest}")

    @main.command("completion")
    @click.option(
        "--shell",
        type=click.Choice(["bash", "zsh", "fish"]),
        required=True,
        help="Select the shell for the completion script.",
    )
    def completion_command(shell: str) -> None:
        """Print a shell completion script for the yakbox executable."""
        completion = get_completion_class(shell)
        if completion is None:
            raise click.BadParameter("unsupported shell", param_hint="--shell")
        # The script may target another machine; do not probe the local shell.
        script = ShellComplete.source(
            completion(main, {}, "yakbox", "_YAKBOX_COMPLETE")
        )
        context = click.get_current_context().find_root()
        if context.params.get("json_output"):
            emit({"shell": shell, "script": script}, "")
        elif not context.params.get("quiet"):
            click.echo(script)


def _initialize_workspace(
    directory: Path, title: str, author: str, profile: str
) -> tuple[Path, Path]:
    root = directory.resolve()
    manifest = root / "yakbox.toml"
    source = root / "source" / "book.md"
    if manifest.exists():
        raise click.ClickException(f"Manifest already exists: {manifest}")
    source.parent.mkdir(parents=True, exist_ok=True)
    template = f""""$schema" = "https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"
schema_version = 1
sources = ["source/book.md"]
pronunciations = "pronunciations.toml"

[book]
title = {_toml_string(title)}
author = {_toml_string(author)}
narrator = "Narrator Name"
language = "en"

[voices.narrator]
display_name = "Narrator"
rights_basis = "not_applicable"

[profiles.default]
{profile}

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
    return manifest, source


def _starter_profile(backend: str, voice_uuid: str | None) -> str:
    settings = [f"backend = {_toml_string(backend)}", 'voice = "narrator"']
    if backend == "fake":
        settings.append("sample_rate = 16000")
    elif backend == "chatterbox-local":
        settings.extend(
            ('device = "cpu"', "max_processes = 1", "threads_per_process = 1")
        )
    else:
        settings.append(f"voice_uuid = {_toml_string(voice_uuid or '')}")
    return "\n".join(settings)


def _toml_string(value: str) -> str:
    # TOML requires Unicode scalar values and escapes for DEL as well as C0.
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
