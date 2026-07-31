from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from yakbox._files import commit_temporary_file
from yakbox.audio.inspect import inspect_audio
from yakbox.errors import ArtifactError, BackendUnavailableError


def assemble_m4b(
    chapters: tuple[Path, ...],
    destination: Path,
    *,
    title: str,
    author: str | None = None,
    narrator: str | None = None,
    subtitle: str | None = None,
    genre: str | None = None,
    publisher: str | None = None,
    copyright: str | None = None,
    language: str | None = None,
    date: str | None = None,
    series: str | None = None,
    series_position: str | None = None,
    cover: Path | None = None,
    chapter_titles: tuple[str, ...] | None = None,
    bitrate: str = "192k",
    overwrite: bool = False,
) -> None:
    """Assemble ordered chapter audio into an atomic, metadata-rich M4B file."""
    _validate_assembly_inputs(chapters, destination, overwrite=overwrite)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, list_name = tempfile.mkstemp(
        prefix=".yakbox-concat-", suffix=".txt", dir=destination.parent
    )
    list_path = Path(list_name)
    metadata_path = list_path.with_suffix(".ffmetadata")
    output_descriptor, output_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".m4b", dir=destination.parent
    )
    os.close(output_descriptor)
    output = Path(output_name)
    output.unlink()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for chapter in chapters:
                escaped = str(chapter.resolve()).replace("'", "'\\''")
                stream.write(f"file '{escaped}'\n")
        if chapter_titles is not None and len(chapter_titles) != len(chapters):
            raise ArtifactError("M4B chapter_titles must match the chapter count")
        _write_chapter_metadata(
            metadata_path,
            chapters,
            title=title,
            author=author,
            chapter_titles=chapter_titles,
        )
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-i",
            str(metadata_path),
        ]
        if cover is not None:
            if not cover.is_file():
                raise ArtifactError(f"Missing M4B cover image: {cover}")
            command.extend(["-i", str(cover)])
        command.extend(
            [
                "-map",
                "0:a:0",
                "-map_metadata",
                "1",
                "-map_chapters",
                "1",
                "-c:a",
                "aac",
                "-b:a",
                bitrate,
            ]
        )
        if cover is not None:
            command.extend(
                [
                    "-map",
                    "2:v:0",
                    "-c:v",
                    "copy",
                    "-disposition:v:0",
                    "attached_pic",
                ]
            )
        metadata = {
            "title": title,
            "artist": author,
            "composer": narrator,
            "subtitle": subtitle,
            "genre": genre,
            "publisher": publisher,
            "copyright": copyright,
            "language": language,
            "date": date,
            "show": series,
            "episode_sort": series_position,
        }
        for key, value in metadata.items():
            if value:
                command.extend(["-metadata", f"{key}={value}"])
        command.append(str(output))
        _run_ffmpeg_assembly(command)
        inspection = inspect_audio(output)
        if not inspection.valid:
            raise ArtifactError(
                "Assembled M4B is invalid: " + "; ".join(inspection.issues)
            )
        commit_temporary_file(output, destination, overwrite=overwrite)
    finally:
        list_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        output.unlink(missing_ok=True)


def _validate_assembly_inputs(
    chapters: tuple[Path, ...],
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    if not chapters:
        raise ArtifactError("At least one chapter is required for assembly")
    if shutil.which("ffmpeg") is None:
        raise BackendUnavailableError("FFmpeg is required for M4B assembly")
    if destination.exists() and not overwrite:
        raise ArtifactError(f"Output already exists: {destination}")
    missing = next((chapter for chapter in chapters if not chapter.is_file()), None)
    if missing is not None:
        raise ArtifactError(f"Missing chapter for assembly: {missing}")
    if len({chapter.resolve() for chapter in chapters}) != len(chapters):
        raise ArtifactError("Assembly chapters must not contain duplicates")


def _run_ffmpeg_assembly(command: list[str]) -> None:
    try:
        result = subprocess.run(  # noqa: S603 - validated argv; shell is disabled
            command, check=False, capture_output=True, text=True, timeout=1800
        )
    except subprocess.TimeoutExpired as error:
        raise ArtifactError("FFmpeg assembly timed out") from error
    except OSError as error:
        raise ArtifactError(f"Cannot start FFmpeg assembly: {error}") from error
    if result.returncode:
        raise ArtifactError(f"FFmpeg assembly failed: {result.stderr[-2048:]}")


def _write_chapter_metadata(
    destination: Path,
    chapters: tuple[Path, ...],
    *,
    title: str,
    author: str | None,
    chapter_titles: tuple[str, ...] | None,
) -> None:
    lines = [";FFMETADATA1", f"title={_escape_metadata(title)}"]
    if author:
        lines.append(f"artist={_escape_metadata(author)}")
    start = 0
    for index, chapter in enumerate(chapters):
        inspection = inspect_audio(chapter)
        if not inspection.valid:
            raise ArtifactError(
                f"Cannot assemble invalid chapter {chapter}: "
                + "; ".join(inspection.issues)
            )
        duration = max(1, round(inspection.duration_seconds * 1_000))
        end = start + duration
        chapter_title = chapter_titles[index] if chapter_titles else chapter.stem
        lines.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start}",
                f"END={end}",
                f"title={_escape_metadata(chapter_title)}",
            ]
        )
        start = end
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape_metadata(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
        .replace("\n", " ")
    )
