from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from yakbox.contracts import runtime_metadata
from yakbox.errors import ArtifactError, BackendUnavailableError


@dataclass(frozen=True, slots=True)
class AudioInspection:
    schema_version: int
    path: Path
    format_name: str
    codec: str
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_rate: int | None
    size: int
    valid: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.update(runtime_metadata("audio-inspection"))
        value["path"] = str(self.path)
        value["issues"] = list(self.issues)
        return value


def inspect_audio(path: Path) -> AudioInspection:
    if shutil.which("ffprobe") is None:
        raise BackendUnavailableError("FFprobe is required for audio inspection")
    if not path.is_file() or path.stat().st_size == 0:
        raise ArtifactError(f"Audio is missing or empty: {path}")
    raw = _run_ffprobe(path)
    stream, media_format, duration, sample_rate, channels = _parse_ffprobe(raw, path)
    format_name = str(media_format.get("format_name", ""))
    codec = str(stream.get("codec_name", ""))
    actual_size = path.stat().st_size
    reported_size = _integer_or_none(media_format.get("size"))
    issues = _inspection_issues(
        duration=duration,
        format_name=format_name,
        codec=codec,
        sample_rate=sample_rate,
        channels=channels,
        actual_size=actual_size,
        reported_size=reported_size,
        path=path,
    )
    return AudioInspection(
        schema_version=1,
        path=path.resolve(),
        format_name=format_name,
        codec=codec,
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        bit_rate=_integer_or_none(media_format.get("bit_rate")),
        size=actual_size,
        valid=not issues,
        issues=issues,
    )


def _run_ffprobe(path: Path) -> object:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=format_name,duration,bit_rate,size:"
                "stream=codec_name,sample_rate,channels",
                "-select_streams",
                "a:0",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as error:
        raise ArtifactError(f"FFprobe timed out for {path}") from error
    except OSError as error:
        raise ArtifactError(f"Cannot start FFprobe: {error}") from error
    if result.returncode:
        raise ArtifactError(f"FFprobe failed: {result.stderr.strip()[-2048:]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ArtifactError(f"Invalid FFprobe response for {path}") from error


def _parse_ffprobe(
    raw: object,
    path: Path,
) -> tuple[dict[str, object], dict[str, object], float, int, int]:
    try:
        if not isinstance(raw, dict):
            raise TypeError
        document = cast(dict[str, object], raw)
        streams_value = document.get("streams")
        format_value = document.get("format")
        if (
            not isinstance(streams_value, list)
            or not streams_value
            or not isinstance(streams_value[0], dict)
            or not isinstance(format_value, dict)
        ):
            raise TypeError
        stream = cast(dict[str, object], streams_value[0])
        media_format = cast(dict[str, object], format_value)
        duration = float(str(media_format["duration"]))
        sample_rate = int(str(stream["sample_rate"]))
        channels = int(str(stream["channels"]))
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ArtifactError(f"Invalid FFprobe response for {path}") from error
    return stream, media_format, duration, sample_rate, channels


def _inspection_issues(
    *,
    duration: float,
    format_name: str,
    codec: str,
    sample_rate: int,
    channels: int,
    actual_size: int,
    reported_size: int | None,
    path: Path,
) -> tuple[str, ...]:
    issues: list[str] = []
    if not math.isfinite(duration):
        raise ArtifactError(f"FFprobe returned a non-finite duration for {path}")
    if duration <= 0:
        issues.append("duration is not positive")
    if not format_name:
        issues.append("container format is missing")
    if not codec:
        issues.append("audio codec is missing")
    if sample_rate <= 0:
        issues.append("sample rate is not positive")
    if channels not in {1, 2}:
        issues.append(f"unexpected channel count: {channels}")
    if reported_size is not None and reported_size != actual_size:
        issues.append("reported size differs from the audio file")
    return tuple(issues)


def _integer_or_none(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        result = int(str(value))
    except ValueError as error:
        raise ArtifactError(
            f"FFprobe returned an invalid integer: {value!r}"
        ) from error
    return result
