"""Manifest migration commands for the schema-version-2 cutover."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, Protocol

import click

from yakbox.contracts import runtime_metadata
from yakbox.errors import YakboxError
from yakbox.speech.analysis_migration import (
    ManifestMigrationPreview,
    preview_manifest_migration,
    write_manifest_migration,
)


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


def register_migration_commands(
    main: click.Group,
    *,
    emit: _Emit,
    fail: Callable[[Exception], NoReturn],
) -> None:
    """Attach guarded migration commands to the CLI composition root."""
    global _emit_callback, _fail_callback  # noqa: PLW0603 - CLI composition root
    _emit_callback = emit
    _fail_callback = fail
    main.add_command(migrate_group)


def _emit(value: dict[str, object], message: str) -> None:
    if _emit_callback is None:
        raise RuntimeError("Migration CLI commands were not registered")
    _emit_callback(value, message)


def _fail(error: Exception) -> NoReturn:
    if _fail_callback is None:
        raise RuntimeError("Migration CLI commands were not registered")
    _fail_callback(error)


@click.group("migrate")
def migrate_group() -> None:
    """Preview or explicitly write versioned project migrations."""


@migrate_group.command("manifest")
@click.argument(
    "manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    default="yakbox.toml",
)
@click.option("--check", is_flag=True, help="Preview without modifying any file.")
@click.option("--write", is_flag=True, help="Atomically write the migration.")
@click.option(
    "--destination",
    type=click.Path(path_type=Path),
    help="Write to a clean path instead of replacing MANIFEST with a backup.",
)
@click.option(
    "--resolve",
    "resolved_findings",
    multiple=True,
    metavar="CODE",
    help="Acknowledge one review-required finding by its stable code.",
)
def migrate_manifest_command(
    manifest: Path,
    check: bool,
    write: bool,
    destination: Path | None,
    resolved_findings: tuple[str, ...],
) -> None:
    """Migrate a version-1 audiobook manifest to the strict version-2 shape."""
    if check == write:
        raise click.UsageError("Choose exactly one of --check or --write")
    if check and (destination is not None or resolved_findings):
        raise click.UsageError("--destination and --resolve require --write")
    try:
        preview = preview_manifest_migration(manifest)
        if check:
            _emit(
                _preview_report(preview),
                _preview_message(preview),
            )
            return
        result = write_manifest_migration(
            preview,
            destination=destination,
            resolved_finding_codes=resolved_findings,
        )
    except (YakboxError, OSError) as error:
        _fail(error)
    _emit(
        result.to_dict(),
        f"Migrated manifest version 1 to version 2: {result.manifest_path}",
    )


def _preview_report(preview: ManifestMigrationPreview) -> dict[str, object]:
    """Avoid echoing arbitrary manifest values, credentials, or full text."""
    return {
        **runtime_metadata("audiobook-manifest-migration-preview"),
        "preview_version": 1,
        "source_manifest_digest": preview.source_manifest_digest,
        "preview_fingerprint": preview.fingerprint,
        "target_schema_version": 2,
        "review_required": preview.review_required,
        "findings": [
            {
                "code": item.code,
                "path": item.path,
                "detail": item.detail,
                "lossy": item.lossy,
                "review_required": item.review_required,
            }
            for item in preview.findings
        ],
        "preserved_repair_count": len(preview.preserved_repairs),
        "pronunciation_count": len(preview.pronunciations),
    }


def _preview_message(preview: ManifestMigrationPreview) -> str:
    review = sum(item.review_required for item in preview.findings)
    return (
        f"Manifest migration preview: {len(preview.findings)} findings; "
        f"{review} require explicit resolution"
    )


__all__ = ["register_migration_commands"]
