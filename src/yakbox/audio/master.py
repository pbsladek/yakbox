from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from yakbox._files import commit_temporary_file
from yakbox.errors import ArtifactError, BackendUnavailableError


def master_wav(
    source: Path,
    destination: Path,
    *,
    sample_rate: int = 44_100,
    normalize: bool = True,
    overwrite: bool = False,
) -> None:
    _require_ffmpeg()
    _require_source(source)
    _prepare(destination, overwrite)
    temporary = _temporary_for(destination)
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
    ]
    if normalize:
        command.extend(["-af", "loudnorm=I=-18:TP=-3:LRA=11"])
    command.extend(["-c:a", "pcm_s24le", str(temporary)])
    try:
        _run(command, temporary)
        commit_temporary_file(temporary, destination, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def encode_mp3(
    source: Path,
    destination: Path,
    *,
    bitrate: str = "192k",
    title: str | None = None,
    album: str | None = None,
    artist: str | None = None,
    track: int | None = None,
    overwrite: bool = False,
) -> None:
    _require_ffmpeg()
    _require_source(source)
    _prepare(destination, overwrite)
    temporary = _temporary_for(destination)
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
    ]
    metadata = {
        "title": title,
        "album": album,
        "artist": artist,
        "track": str(track) if track is not None else None,
    }
    for key, value in metadata.items():
        if value:
            command.extend(["-metadata", f"{key}={value}"])
    command.append(str(temporary))
    try:
        _run(command, temporary)
        commit_temporary_file(temporary, destination, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def copy_audio(source: Path, destination: Path, *, overwrite: bool = False) -> None:
    _require_source(source)
    _prepare(destination, overwrite)
    temporary = _temporary_for(destination)
    try:
        shutil.copyfile(source, temporary)
        commit_temporary_file(temporary, destination, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise BackendUnavailableError(
            "FFmpeg is required for mastering and encoding; "
            "install ffmpeg or disable mastering"
        )


def _prepare(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ArtifactError(f"Output already exists: {path}")


def _require_source(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ArtifactError(f"Input audio is missing or empty: {path}")


def _temporary_for(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary


def _run(command: list[str], temporary: Path) -> None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as error:
        temporary.unlink(missing_ok=True)
        raise ArtifactError("FFmpeg timed out") from error
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise ArtifactError(f"Cannot start FFmpeg: {error}") from error
    if result.returncode:
        temporary.unlink(missing_ok=True)
        detail = result.stderr.strip()[-2048:]
        raise ArtifactError(f"FFmpeg failed: {detail}")
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise ArtifactError("FFmpeg produced no output")
