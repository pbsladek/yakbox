"""Reusable Click option families for cross-command policy consistency."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import click


def text_file_option[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Add the shared UTF-8 file-or-stdin text source option."""
    return click.option(
        "--text-file",
        type=click.Path(path_type=Path),
        help="Read UTF-8 text from this file, or use '-' for stdin.",
    )(function)


def explicit_text_options[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Add explicit text and file options used by preview-style commands."""
    decorated = click.option(
        "--text",
        help="Use this explicit text instead of selecting manuscript content.",
    )(function)
    return text_file_option(decorated)


def audio_output_options[**P, R](
    default: str,
    *,
    include_format: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Create a shared atomic destination, format, and overwrite decorator."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        decorated = click.option(
            "--out",
            type=click.Path(path_type=Path),
            default=default,
            help="Write the generated audio to this path.",
        )(function)
        if include_format:
            decorated = click.option(
                "--format",
                "output_format",
                type=click.Choice(["wav", "mp3"]),
                default="wav",
                help="Select the generated audio container format.",
            )(decorated)
        return click.option(
            "--overwrite",
            is_flag=True,
            help="Replace an existing destination after successful generation.",
        )(decorated)

    return decorate


def hosted_budget_options[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Add the complete hosted budget and confirmation option family."""
    decorators = (
        click.option(
            "--max-submitted-characters",
            type=click.IntRange(min=0),
            help="Reject work submitting more billable characters.",
        ),
        click.option(
            "--max-provider-requests",
            type=click.IntRange(min=0),
            help="Reject work requiring more provider attempts.",
        ),
        click.option(
            "--max-estimated-spend",
            type=Decimal,
            help="Reject work estimated above this monetary amount.",
        ),
        click.option(
            "--currency",
            help="Set the three-letter currency code for spending estimates.",
        ),
        click.option(
            "--pricing-source",
            help="Identify the account pricing source used for the estimate.",
        ),
        click.option(
            "--price-per-character",
            type=Decimal,
            help="Estimate spend using this currency amount per character.",
        ),
        click.option(
            "--confirm-above-characters",
            type=click.IntRange(min=0),
            help="Require --yes above this estimated submitted-character count.",
        ),
        click.option(
            "--confirm-above-requests",
            type=click.IntRange(min=0),
            help="Require --yes above this estimated provider-request count.",
        ),
        click.option(
            "--yes",
            is_flag=True,
            help="Confirm reviewed hosted work without an interactive prompt.",
        ),
    )
    decorated = function
    for decorator in reversed(decorators):
        decorated = decorator(decorated)
    return decorated


def deprecated_api_key_option[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Keep the hidden compatibility credential option on hosted commands."""
    return click.option("--api-key", hidden=True)(function)
