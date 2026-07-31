from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

from yakbox.contracts import runtime_metadata
from yakbox.errors import ArtifactError, BackendUnavailableError

_SILENCE_EDGE_TOLERANCE_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class AudioInspection:
    """Measured media properties and quality findings for one audio file."""

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
    integrated_loudness_lufs: float | None = None
    true_peak_dbfs: float | None = None
    loudness_range_lu: float | None = None
    leading_silence_seconds: float | None = None
    trailing_silence_seconds: float | None = None

    def to_dict(self, *, root: Path | None = None) -> dict[str, object]:
        """Serialize the inspection using its versioned JSON contract."""
        value = asdict(self)
        value.update(runtime_metadata("audio-inspection"))
        value["path"] = (
            self.path.relative_to(root).as_posix()
            if root is not None and self.path.is_relative_to(root)
            else self.path.as_posix()
        )
        value["issues"] = list(self.issues)
        return value


@dataclass(frozen=True, slots=True)
class AudioQualityPolicy:
    """Optional loudness, peak, and edge-silence limits for audio inspection."""

    minimum_loudness_lufs: float | None = None
    maximum_loudness_lufs: float | None = None
    maximum_true_peak_dbfs: float | None = None
    maximum_leading_silence_seconds: float | None = None
    maximum_trailing_silence_seconds: float | None = None


def inspect_audio(
    path: Path,
    *,
    quality: AudioQualityPolicy | None = None,
) -> AudioInspection:
    """Inspect an audio file with FFprobe and evaluate an optional quality policy."""
    if shutil.which("ffprobe") is None:
        raise BackendUnavailableError("FFprobe is required for audio inspection")
    if not path.is_file() or path.stat().st_size == 0:
        raise ArtifactError(f"Audio is missing or empty: {path}")
    resolved = path.resolve()
    before = _file_signature(resolved)
    inspection = _inspect_snapshot(str(resolved), quality, *before)
    after = _file_signature(resolved)
    if after != before:
        inspection = _inspect_snapshot(str(resolved), quality, *after)
    return inspection


@lru_cache(maxsize=2_048)
def _inspect_snapshot(
    path_value: str,
    quality: AudioQualityPolicy | None,
    _size: int,
    _mtime_ns: int,
    _ctime_ns: int,
    _inode: int,
) -> AudioInspection:
    path = Path(path_value)
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
    loudness: tuple[float | None, float | None, float, float, float] | None = None
    if quality is not None:
        loudness = _run_quality_analysis(path, duration)
        issues = (
            *issues,
            *_quality_issues(
                integrated_loudness_lufs=loudness[0],
                true_peak_dbfs=loudness[1],
                leading_silence_seconds=loudness[3],
                trailing_silence_seconds=loudness[4],
                policy=quality,
            ),
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
        integrated_loudness_lufs=loudness[0] if loudness else None,
        true_peak_dbfs=loudness[1] if loudness else None,
        loudness_range_lu=loudness[2] if loudness else None,
        leading_silence_seconds=loudness[3] if loudness else None,
        trailing_silence_seconds=loudness[4] if loudness else None,
        valid=not issues,
        issues=tuple(issues),
    )


def _file_signature(path: Path) -> tuple[int, int, int, int]:
    status = path.stat()
    return (
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
        status.st_ino,
    )


def _run_ffprobe(path: Path) -> object:
    try:
        result = subprocess.run(  # noqa: S603 - fixed ffprobe argv
            [  # noqa: S607 - executable availability is checked before use
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


def _run_quality_analysis(
    path: Path,
    duration: float,
) -> tuple[float | None, float | None, float, float, float]:
    try:
        result = subprocess.run(  # noqa: S603 - fixed ffmpeg argv
            [  # noqa: S607 - executable availability is checked before use
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-i",
                str(path),
                "-af",
                "silencedetect=noise=-50dB:d=0.25,ebur128=peak=true",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as error:
        raise ArtifactError(f"FFmpeg quality analysis timed out for {path}") from error
    except OSError as error:
        raise ArtifactError(f"Cannot start FFmpeg quality analysis: {error}") from error
    if result.returncode:
        raise ArtifactError(
            f"FFmpeg quality analysis failed: {result.stderr.strip()[-2048:]}"
        )
    stderr = result.stderr
    integrated = _last_metric(stderr, r"\bI:\s*(-?(?:inf|\d+(?:\.\d+)?))\s+LUFS")
    true_peak = _last_metric(
        stderr,
        r"\bPeak:\s*(-?(?:inf|\d+(?:\.\d+)?))\s+dBFS",
    )
    loudness_range = _last_metric(stderr, r"\bLRA:\s*(\d+(?:\.\d+)?)\s+LU")
    if loudness_range is None:
        raise ArtifactError("FFmpeg quality analysis returned no loudness range")
    leading, trailing = _silence_edges(stderr, duration)
    return integrated, true_peak, loudness_range, leading, trailing


def _last_metric(value: str, pattern: str) -> float | None:
    matches = re.findall(pattern, value, flags=re.IGNORECASE)
    if not matches:
        raise ArtifactError("FFmpeg quality analysis omitted a required metric")
    raw = matches[-1].casefold()
    if raw in {"-inf", "inf"}:
        return None
    return float(raw)


def _silence_edges(value: str, duration: float) -> tuple[float, float]:
    events = [
        (match.group(1), float(match.group(2)))
        for match in re.finditer(
            r"silence_(start|end):\s*(\d+(?:\.\d+)?)",
            value,
        )
    ]
    leading = 0.0
    trailing = 0.0
    if (
        events
        and events[0][0] == "start"
        and events[0][1] <= _SILENCE_EDGE_TOLERANCE_SECONDS
    ):
        end = next((time for kind, time in events[1:] if kind == "end"), duration)
        leading = min(duration, max(0.0, end))
    starts = [time for kind, time in events if kind == "start"]
    ends = [time for kind, time in events if kind == "end"]
    if starts:
        last_start = starts[-1]
        later_end = next((time for time in ends if time >= last_start), None)
        if later_end is None or later_end >= duration - _SILENCE_EDGE_TOLERANCE_SECONDS:
            trailing = max(0.0, duration - last_start)
    return leading, trailing


def _quality_issues(
    *,
    integrated_loudness_lufs: float | None,
    true_peak_dbfs: float | None,
    leading_silence_seconds: float,
    trailing_silence_seconds: float,
    policy: AudioQualityPolicy,
) -> tuple[str, ...]:
    issues: list[str] = []
    if integrated_loudness_lufs is None and (
        policy.minimum_loudness_lufs is not None
        or policy.maximum_loudness_lufs is not None
    ):
        issues.append("integrated loudness is not measurable (audio is silent)")
    if (
        integrated_loudness_lufs is not None
        and policy.minimum_loudness_lufs is not None
        and integrated_loudness_lufs < policy.minimum_loudness_lufs
    ):
        issues.append(
            f"integrated loudness {integrated_loudness_lufs:.1f} LUFS is below "
            f"{policy.minimum_loudness_lufs:.1f} LUFS"
        )
    if (
        integrated_loudness_lufs is not None
        and policy.maximum_loudness_lufs is not None
        and integrated_loudness_lufs > policy.maximum_loudness_lufs
    ):
        issues.append(
            f"integrated loudness {integrated_loudness_lufs:.1f} LUFS exceeds "
            f"{policy.maximum_loudness_lufs:.1f} LUFS"
        )
    if (
        true_peak_dbfs is not None
        and policy.maximum_true_peak_dbfs is not None
        and true_peak_dbfs > policy.maximum_true_peak_dbfs
    ):
        issues.append(
            f"true peak {true_peak_dbfs:.1f} dBFS exceeds "
            f"{policy.maximum_true_peak_dbfs:.1f} dBFS"
        )
    if (
        policy.maximum_leading_silence_seconds is not None
        and leading_silence_seconds > policy.maximum_leading_silence_seconds
    ):
        issues.append(
            f"leading silence {leading_silence_seconds:.2f}s exceeds "
            f"{policy.maximum_leading_silence_seconds:.2f}s"
        )
    if (
        policy.maximum_trailing_silence_seconds is not None
        and trailing_silence_seconds > policy.maximum_trailing_silence_seconds
    ):
        issues.append(
            f"trailing silence {trailing_silence_seconds:.2f}s exceeds "
            f"{policy.maximum_trailing_silence_seconds:.2f}s"
        )
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
