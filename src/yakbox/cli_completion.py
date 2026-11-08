"""Offline completion of names from the current command's audiobook manifest."""

from __future__ import annotations

from pathlib import Path

import click
from click.core import ParameterSource
from click.shell_completion import CompletionItem

from yakbox.audiobook.manifest import load_manifest
from yakbox.errors import YakboxError


class _ManifestName(click.types.StringParamType):
    def shell_complete(
        self, ctx: click.Context, param: click.Parameter, incomplete: str
    ) -> list[CompletionItem]:
        manifest = _manifest_path(ctx)
        if manifest is None:
            return []
        try:
            loaded = load_manifest(manifest)
        except YakboxError, OSError, ValueError:
            return []  # Incomplete or invalid projects must not break completion.
        if not isinstance(param, click.Option):
            return []
        names = loaded.targets if "--target" in param.opts else loaded.profiles
        return [
            CompletionItem(name)
            for name in sorted(item.name for item in names)
            if name.startswith(incomplete)
        ]


def _manifest_path(ctx: click.Context) -> Path | None:
    value = ctx.params.get("manifest", Path("yakbox.toml"))
    if ctx.get_parameter_source("manifest") is not ParameterSource.COMMANDLINE:
        # Click can leave positional arguments unprocessed when the final option
        # is waiting for a value. Recover the manifest from those arguments.
        position = 0
        for parameter in ctx.command.params:
            if not isinstance(parameter, click.Argument):
                continue
            if parameter.name == "manifest":
                if position < len(ctx.args):
                    value = ctx.args[position]
                break
            if parameter.nargs < 0:
                break
            position += parameter.nargs
    return Path(value) if isinstance(value, (str, Path)) else None


def configure_cli_completion(root: click.Command) -> None:
    """Complete manifest targets and profiles on commands with a manifest input."""
    has_manifest = any(parameter.name == "manifest" for parameter in root.params)
    if has_manifest:
        for parameter in root.params:
            if (
                isinstance(parameter, click.Option)
                and parameter.type is click.STRING
                and {"--target", "--profile", "--profiles"}.intersection(parameter.opts)
            ):
                parameter.type = _ManifestName()
    if isinstance(root, click.Group):
        for command in root.commands.values():
            configure_cli_completion(command)
