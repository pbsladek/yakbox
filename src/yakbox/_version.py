"""Version output reads immutable package metadata without consulting Git."""

from __future__ import annotations

import click

from yakbox import __version__, _build_commit


def version_output(version: str, commit: str | None) -> str:
    return f"yakbox, version {version} (commit {commit or 'unknown'})"


def show_version(
    context: click.Context, _parameter: click.Parameter, value: bool
) -> None:
    if not value or context.resilient_parsing:
        return
    click.echo(version_output(__version__, _build_commit.COMMIT))
    context.exit()
