"""Deterministic canonical audio and shared analysis-window materialization."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from yakbox._files import (
    atomic_output_path,
    atomic_write_json,
    safe_child,
    sha256_file,
)
from yakbox.errors import ArtifactError, BackendUnavailableError, ValidationError
from yakbox.fingerprints import media_tool_versions
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_models import (
    AudioSpan,
    CanonicalAudioIdentity,
    FrameCoordinateMap,
)

CANONICAL_SAMPLE_RATE = 16_000
CANONICAL_SAMPLE_WIDTH = 2
CANONICAL_CHANNELS = 1
CANONICAL_AUDIO_VERSION = 1


@dataclass(frozen=True, slots=True)
class PreparedCanonicalAudio:
    """Managed canonical WAV and its immutable coordinate identity."""

    path: Path
    identity: CanonicalAudioIdentity


@dataclass(frozen=True, slots=True)
class PreparedAnalysisWindow:
    """One exact materialized PCM window shared by all recognizers."""

    path: Path
    source_span: AudioSpan
    window_span: AudioSpan


class CanonicalAudioPreparer:
    """Decode each source digest once into deterministic mono 16 kHz PCM."""

    def __init__(self, managed_root: Path) -> None:
        self.managed_root = managed_root.resolve()

    def prepare(self, source: Path) -> PreparedCanonicalAudio:
        if source.is_symlink():
            raise ArtifactError("Analysis audio must be a non-empty regular file")
        source = source.resolve()
        _require_regular_file(source)
        _require_ffmpeg_tools()
        source_digest = sha256_file(source)
        source_rate, source_frames, source_delay_frames = _probe_audio_shape(source)
        preprocessing = _preprocessing_fingerprint()
        destination = safe_child(
            self.managed_root,
            self.managed_root
            / "canonical"
            / source_digest
            / preprocessing
            / "audio.wav",
        )
        metadata = safe_child(
            self.managed_root,
            destination.with_name("identity.json"),
        )
        if not _canonical_cache_matches(
            destination,
            metadata,
            source_digest=source_digest,
            preprocessing_fingerprint=preprocessing,
        ):
            self._render(source, destination, overwrite=destination.exists())
        canonical_rate, canonical_frames = _read_pcm_shape(destination)
        if canonical_rate != CANONICAL_SAMPLE_RATE:
            raise ArtifactError("Canonical analysis audio has an invalid sample rate")
        canonical_digest = sha256_file(destination)
        _write_canonical_metadata(
            metadata,
            source_digest=source_digest,
            preprocessing_fingerprint=preprocessing,
            canonical_digest=canonical_digest,
            sample_rate=canonical_rate,
            frame_count=canonical_frames,
        )
        identity = CanonicalAudioIdentity(
            source_digest=source_digest,
            source_format=source.suffix.lower().removeprefix(".") or "unknown",
            canonical_digest=canonical_digest,
            canonical_format="wav-pcm-s16le-mono",
            preprocessing_fingerprint=preprocessing,
            frame_map=FrameCoordinateMap(
                source_rate=source_rate,
                analysis_rate=canonical_rate,
                source_frame_count=source_frames,
                analysis_frame_count=canonical_frames,
                source_delay_frames=source_delay_frames,
            ),
        )
        return PreparedCanonicalAudio(destination, identity)

    def materialize_window(
        self,
        prepared: PreparedCanonicalAudio,
        span: AudioSpan,
    ) -> PreparedAnalysisWindow:
        identity = prepared.identity
        if span.audio_digest != identity.canonical_digest:
            raise ValidationError("Analysis window span uses a different audio digest")
        if span.sample_rate != identity.frame_map.analysis_rate:
            raise ValidationError("Analysis window span uses a different sample rate")
        if span.end_frame > identity.frame_map.analysis_frame_count:
            raise ValidationError("Analysis window exceeds canonical audio")
        fingerprint = semantic_fingerprint("canonical-window-v1", span)
        destination = safe_child(
            self.managed_root,
            self.managed_root / "windows" / fingerprint / "audio.wav",
        )
        if not _window_cache_matches(prepared.path, destination, span):
            _write_window(
                prepared.path,
                destination,
                span,
                overwrite=destination.exists(),
            )
        window_digest = sha256_file(destination)
        rate, frames = _read_pcm_shape(destination)
        window_span = AudioSpan(window_digest, 0, frames, rate)
        return PreparedAnalysisWindow(destination, span, window_span)

    def _render(
        self, source: Path, destination: Path, *, overwrite: bool = False
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with atomic_output_path(destination, overwrite=overwrite) as temporary:
            command = [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-i",
                str(source),
                "-map_metadata",
                "-1",
                "-vn",
                "-ac",
                str(CANONICAL_CHANNELS),
                "-ar",
                str(CANONICAL_SAMPLE_RATE),
                "-sample_fmt",
                "s16",
                "-c:a",
                "pcm_s16le",
                "-fflags",
                "+bitexact",
                "-flags:a",
                "+bitexact",
                str(temporary),
            ]
            _run_ffmpeg(command)


def canonical_to_source_frames(
    mapping: FrameCoordinateMap,
    start_frame: int,
    end_frame: int,
) -> tuple[int, int]:
    """Map canonical bounds outward to safe source-frame boundaries."""
    if start_frame < 0 or end_frame <= start_frame:
        raise ValidationError("Canonical frame range is invalid")
    start = math.floor(start_frame * mapping.source_rate / mapping.analysis_rate)
    end = math.ceil(end_frame * mapping.source_rate / mapping.analysis_rate)
    start = max(0, start + mapping.source_delay_frames)
    end = min(mapping.source_frame_count, end + mapping.source_delay_frames)
    if end <= start:
        raise ValidationError("Mapped source frame range is empty")
    return start, end


def _write_window(
    source: Path,
    destination: Path,
    span: AudioSpan,
    *,
    overwrite: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as reader:
        if (
            reader.getnchannels() != CANONICAL_CHANNELS
            or reader.getsampwidth() != CANONICAL_SAMPLE_WIDTH
            or reader.getframerate() != span.sample_rate
        ):
            raise ArtifactError("Canonical window source has an invalid PCM shape")
        reader.setpos(span.start_frame)
        frames = reader.readframes(span.end_frame - span.start_frame)
    with (
        atomic_output_path(destination, overwrite=overwrite) as temporary,
        wave.open(str(temporary), "wb") as writer,
    ):
        writer.setnchannels(CANONICAL_CHANNELS)
        writer.setsampwidth(CANONICAL_SAMPLE_WIDTH)
        writer.setframerate(span.sample_rate)
        writer.writeframes(frames)


def _probe_audio_shape(path: Path) -> tuple[int, int, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,duration_ts,time_base,start_pts,start_time:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ArtifactError("FFprobe could not inspect analysis audio") from error
    if completed.returncode:
        raise ArtifactError("FFprobe rejected analysis audio")
    try:
        raw = json.loads(completed.stdout)
        stream = raw["streams"][0]
        sample_rate = int(stream["sample_rate"])
        duration = _stream_duration(stream, raw)
        start_time = _stream_start_time(stream)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactError(
            "FFprobe returned invalid analysis audio metadata"
        ) from error
    frames = round(duration * sample_rate)
    if sample_rate <= 0 or frames <= 0:
        raise ArtifactError("Analysis source has no usable audio frames")
    delay_frames = max(0, round(start_time * sample_rate))
    return sample_rate, frames, delay_frames


def _stream_duration(stream: dict[str, object], raw: dict[str, object]) -> float:
    duration_ts = stream.get("duration_ts")
    time_base = stream.get("time_base")
    if isinstance(duration_ts, int | str) and isinstance(time_base, str):
        try:
            return int(duration_ts) * float(Fraction(time_base))
        except ValueError, ZeroDivisionError:
            pass
    format_value = raw.get("format")
    if not isinstance(format_value, dict):
        return 0
    duration = format_value.get("duration", 0)
    if not isinstance(duration, int | float | str) or isinstance(duration, bool):
        return 0
    return float(duration)


def _stream_start_time(stream: dict[str, object]) -> float:
    value = stream.get("start_time")
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return 0
    return 0


def _canonical_cache_matches(
    audio: Path,
    metadata: Path,
    *,
    source_digest: str,
    preprocessing_fingerprint: str,
) -> bool:
    if audio.is_symlink() or metadata.is_symlink():
        return False
    if not audio.is_file() or not metadata.is_file():
        return False
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
        rate, frames = _read_pcm_shape(audio)
    except ArtifactError, OSError, UnicodeDecodeError, json.JSONDecodeError:
        return False
    return bool(
        isinstance(value, dict)
        and value.get("source_digest") == source_digest
        and value.get("preprocessing_fingerprint") == preprocessing_fingerprint
        and value.get("canonical_digest") == sha256_file(audio)
        and value.get("sample_rate") == rate
        and value.get("frame_count") == frames
    )


def _write_canonical_metadata(
    path: Path,
    *,
    source_digest: str,
    preprocessing_fingerprint: str,
    canonical_digest: str,
    sample_rate: int,
    frame_count: int,
) -> None:
    atomic_write_json(
        path,
        {
            "source_digest": source_digest,
            "preprocessing_fingerprint": preprocessing_fingerprint,
            "canonical_digest": canonical_digest,
            "sample_rate": sample_rate,
            "frame_count": frame_count,
        },
    )


def _window_cache_matches(source: Path, window: Path, span: AudioSpan) -> bool:
    if window.is_symlink() or not window.is_file():
        return False
    try:
        with wave.open(str(source), "rb") as source_reader:
            source_reader.setpos(span.start_frame)
            expected = source_reader.readframes(span.end_frame - span.start_frame)
        with wave.open(str(window), "rb") as window_reader:
            actual = window_reader.readframes(window_reader.getnframes())
            shape_matches = (
                window_reader.getnchannels() == CANONICAL_CHANNELS
                and window_reader.getsampwidth() == CANONICAL_SAMPLE_WIDTH
                and window_reader.getframerate() == span.sample_rate
                and window_reader.getnframes() == span.end_frame - span.start_frame
            )
    except OSError, EOFError, wave.Error:
        return False
    return shape_matches and actual == expected


def _read_pcm_shape(path: Path) -> tuple[int, int]:
    try:
        with wave.open(str(path), "rb") as reader:
            if (
                reader.getnchannels() != CANONICAL_CHANNELS
                or reader.getsampwidth() != CANONICAL_SAMPLE_WIDTH
            ):
                raise ArtifactError("Analysis WAV is not mono 16-bit PCM")
            return reader.getframerate(), reader.getnframes()
    except (OSError, EOFError, wave.Error) as error:
        raise ArtifactError("Analysis WAV is invalid") from error


def _preprocessing_fingerprint() -> str:
    return semantic_fingerprint(
        "canonical-audio-preprocessing-v1",
        {
            "version": CANONICAL_AUDIO_VERSION,
            "sample_rate": CANONICAL_SAMPLE_RATE,
            "sample_width": CANONICAL_SAMPLE_WIDTH,
            "channels": CANONICAL_CHANNELS,
            "channel_mix": "ffmpeg-default-mono",
            "sample_format": "s16",
            "dither": "ffmpeg-default",
            "media_tools": media_tool_versions(),
        },
    )


def _run_ffmpeg(command: list[str]) -> None:
    try:
        completed = subprocess.run(  # noqa: S603 - validated argv, no shell
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ArtifactError(
            "FFmpeg could not prepare canonical analysis audio"
        ) from error
    if completed.returncode:
        detail = completed.stderr.strip()[-2048:]
        raise ArtifactError(f"FFmpeg failed to prepare analysis audio: {detail}")


def _require_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ArtifactError("Analysis audio must be a non-empty regular file")


def _require_ffmpeg_tools() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise BackendUnavailableError(
            "FFmpeg and FFprobe are required for canonical speech analysis"
        )


__all__ = [
    "CANONICAL_AUDIO_VERSION",
    "CANONICAL_CHANNELS",
    "CANONICAL_SAMPLE_RATE",
    "CANONICAL_SAMPLE_WIDTH",
    "CanonicalAudioPreparer",
    "PreparedAnalysisWindow",
    "PreparedCanonicalAudio",
    "canonical_to_source_frames",
]
