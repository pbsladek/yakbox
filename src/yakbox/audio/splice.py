"""Conservative PCM matching and adaptive crossfades for localized repairs."""

from __future__ import annotations

import math
import wave
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from yakbox._files import atomic_output_path
from yakbox.audio.splice_stream import SpliceStream, read_region
from yakbox.errors import ArtifactError

_CONTEXT_MS = 80
_MINIMUM_CROSSFADE_MS = 3
_MAXIMUM_CROSSFADE_MS = 16
_MAXIMUM_GAIN_DB = 6.0
_MINIMUM_OVERLAP_FRAMES = 2


@dataclass(frozen=True, slots=True)
class AdaptiveSpliceEvidence:
    """Measurements and bounded transformations used for one replacement."""

    gain_db: float
    replacement_high_frequency_ratio_before: float
    replacement_high_frequency_ratio_after: float
    context_high_frequency_ratio: float
    spectral_smoothing: float
    leading_trim_ms: float
    trailing_trim_ms: float
    leading_crossfade_ms: float
    trailing_crossfade_ms: float
    output_duration_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FrameReplacement:
    """One replacement resolved against immutable source frame coordinates."""

    repair_id: str
    start_frame: int
    end_frame: int
    audio: Path

    def __post_init__(self) -> None:
        if (
            not self.repair_id
            or self.start_frame < 0
            or self.end_frame <= self.start_frame
        ):
            raise ArtifactError("Frame replacement identity and interval are required")


@dataclass(frozen=True, slots=True)
class _Wave:
    channels: int
    sample_width: int
    sample_rate: int
    samples: tuple[int, ...]
    source_frames: int = 0

    @property
    def frame_count(self) -> int:
        return self.source_frames or len(self.samples) // self.channels


def splice_wav_region(
    original: Path,
    replacement: Path,
    destination: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    overwrite: bool = False,
) -> AdaptiveSpliceEvidence:
    """Replace one aligned region using matched level, tone, and adaptive overlaps."""
    if not math.isfinite(start_seconds) or not math.isfinite(end_seconds):
        raise ArtifactError("Repair splice timing must be finite")
    with _source_wave(original) as (reader, source):
        item = FrameReplacement(
            "single",
            round(start_seconds * source.sample_rate),
            round(end_seconds * source.sample_rate),
            replacement,
        )
        prepared = _prepare_replacements(reader, source, (item,))
        frames = _render_replacements(
            reader, source, prepared, destination, overwrite=overwrite, single=True
        )
        return replace(
            prepared[0][2], output_duration_seconds=frames / source.sample_rate
        )


def splice_wav_regions(
    original: Path,
    replacements: tuple[FrameReplacement, ...],
    destination: Path,
    *,
    overwrite: bool = False,
) -> tuple[AdaptiveSpliceEvidence, ...]:
    """Reconstruct once from immutable base slices and canonical replacements."""
    if not replacements:
        raise ArtifactError("Multi-repair splice requires at least one replacement")
    ordered = tuple(
        sorted(
            replacements,
            key=lambda item: (item.start_frame, item.end_frame, item.repair_id),
        )
    )
    if len({item.repair_id for item in ordered}) != len(ordered):
        raise ArtifactError("Multi-repair splice IDs must be unique")
    with _source_wave(original) as (reader, source):
        prepared = _prepare_replacements(reader, source, ordered)
        _render_replacements(reader, source, prepared, destination, overwrite=overwrite)
        return tuple(item[2] for item in prepared)


@contextmanager
def _source_wave(path: Path) -> Generator[tuple[wave.Wave_read, _Wave]]:
    try:
        with wave.open(str(path), "rb") as reader:
            if reader.getcomptype() != "NONE" or reader.getsampwidth() not in {
                1,
                2,
                3,
                4,
            }:
                raise ArtifactError("Adaptive repair splice requires PCM WAV audio")
            source = _Wave(
                reader.getnchannels(),
                reader.getsampwidth(),
                reader.getframerate(),
                (),
                reader.getnframes(),
            )
            yield reader, source
    except (OSError, EOFError, wave.Error) as error:
        raise ArtifactError(f"Cannot process PCM WAV: {path}") from error


def _prepare_replacements(
    reader: wave.Wave_read,
    source: _Wave,
    replacements: tuple[FrameReplacement, ...],
) -> list[tuple[FrameReplacement, tuple[int, ...], AdaptiveSpliceEvidence]]:
    prepared: list[
        tuple[FrameReplacement, tuple[int, ...], AdaptiveSpliceEvidence]
    ] = []
    previous_end = 0
    for item in replacements:
        if item.start_frame < previous_end:
            raise ArtifactError("Multi-repair frame intervals overlap")
        if item.end_frame > source.frame_count:
            raise ArtifactError("Multi-repair interval falls outside the base WAV")
        prepared.append(_prepare_one(reader, source, item, previous_end))
        previous_end = item.end_frame
    return prepared


def _prepare_one(
    reader: wave.Wave_read,
    source: _Wave,
    item: FrameReplacement,
    previous_end: int,
) -> tuple[FrameReplacement, tuple[int, ...], AdaptiveSpliceEvidence]:
    insert = _read_wave(item.audio)
    if (source.channels, source.sample_width, source.sample_rate) != (
        insert.channels,
        insert.sample_width,
        insert.sample_rate,
    ):
        raise ArtifactError("Repair splice WAV formats are incompatible")
    context_frames = max(1, round(source.sample_rate * _CONTEXT_MS / 1_000))
    left = read_region(
        reader, max(previous_end, item.start_frame - context_frames), item.start_frame
    )
    right = read_region(
        reader, item.end_frame, min(source.frame_count, item.end_frame + context_frames)
    )
    context_left = read_region(
        reader, max(0, item.start_frame - context_frames), item.start_frame
    )
    matched, gain, smoothing, before, after, context_ratio = _match_replacement(
        insert.samples, context_left + right, source=source
    )
    trimmed, leading_trim, trailing_trim = _trim_to_quiet_edges(
        matched, channels=source.channels, sample_rate=source.sample_rate
    )
    leading = _crossfade_frames(
        left, trimmed, channels=source.channels, sample_rate=source.sample_rate
    )
    trailing = _crossfade_frames(
        trimmed, right, channels=source.channels, sample_rate=source.sample_rate
    )
    evidence = AdaptiveSpliceEvidence(
        gain,
        before,
        after,
        context_ratio,
        smoothing,
        leading_trim * 1_000 / source.sample_rate,
        trailing_trim * 1_000 / source.sample_rate,
        leading * 1_000 / source.sample_rate,
        trailing * 1_000 / source.sample_rate,
        len(trimmed) / source.channels / source.sample_rate,
    )
    return item, trimmed, evidence


def _render_replacements(
    reader: wave.Wave_read,
    source: _Wave,
    prepared: list[tuple[FrameReplacement, tuple[int, ...], AdaptiveSpliceEvidence]],
    destination: Path,
    *,
    overwrite: bool,
    single: bool = False,
) -> int:
    def overlap(left: tuple[int, ...], right: tuple[int, ...]) -> int:
        return _crossfade_frames(
            left, right, channels=source.channels, sample_rate=source.sample_rate
        )

    def mix(
        left: tuple[int, ...], right: tuple[int, ...], frames: int
    ) -> tuple[int, ...]:
        return _crossfade(left, right, channels=source.channels, frames=frames)

    with (
        atomic_output_path(destination, overwrite=overwrite) as temporary,
        wave.open(str(temporary), "wb") as writer,
    ):
        writer.setparams(
            (
                source.channels,
                source.sample_width,
                source.sample_rate,
                0,
                "NONE",
                "not compressed",
            )
        )
        stream = SpliceStream(
            writer,
            channels=source.channels,
            width=source.sample_width,
            rate=source.sample_rate,
            overlap=overlap,
            mix=mix,
        )
        cursor = 0
        for item, samples, _ in prepared:
            stream.append_region(reader, cursor, item.start_frame)
            stream.append_samples(samples)
            cursor = item.end_frame
        trailing = (
            round(prepared[0][2].trailing_crossfade_ms * source.sample_rate / 1_000)
            if single
            else None
        )
        stream.append_region(
            reader, cursor, source.frame_count, overlap_frames=trailing
        )
        return stream.finish()


def _match_replacement(
    samples: tuple[int, ...],
    context: tuple[int, ...],
    *,
    source: _Wave,
) -> tuple[tuple[int, ...], float, float, float, float, float]:
    context_rms = _rms(context)
    replacement_rms = _rms(samples)
    ratio = context_rms / replacement_rms if replacement_rms else 1.0
    maximum = 10 ** (_MAXIMUM_GAIN_DB / 20)
    ratio = max(1 / maximum, min(maximum, ratio))
    gain_db = 20 * math.log10(ratio) if ratio > 0 else 0.0
    context_ratio = _high_frequency_ratio(context, source.channels)
    before_ratio = _high_frequency_ratio(samples, source.channels)
    smoothing = 0.0
    transformed = samples
    if before_ratio > context_ratio * 1.25 and before_ratio > 0:
        smoothing = min(0.35, (before_ratio - context_ratio) / before_ratio * 0.5)
        transformed = _smooth(transformed, channels=source.channels, amount=smoothing)
    peak_limit = (1 << (source.sample_width * 8 - 1)) - 1
    peak = max((abs(value) for value in transformed), default=0)
    if peak and peak * ratio > peak_limit * 0.98:
        ratio = peak_limit * 0.98 / peak
        gain_db = 20 * math.log10(ratio)
    matched = tuple(round(value * ratio) for value in transformed)
    return (
        matched,
        gain_db,
        smoothing,
        before_ratio,
        _high_frequency_ratio(matched, source.channels),
        context_ratio,
    )


def _trim_to_quiet_edges(
    samples: tuple[int, ...],
    *,
    channels: int,
    sample_rate: int,
) -> tuple[tuple[int, ...], int, int]:
    frames = len(samples) // channels
    search = min(max(1, round(sample_rate * 0.012)), max(1, frames // 8))
    leading = _quietest_frame(samples, channels=channels, start=0, end=search)
    trailing = _quietest_frame(
        samples,
        channels=channels,
        start=max(leading + 1, frames - search),
        end=frames,
    )
    if trailing <= leading:
        return samples, 0, 0
    return (
        samples[leading * channels : trailing * channels],
        leading,
        frames - trailing,
    )


def _quietest_frame(
    samples: tuple[int, ...],
    *,
    channels: int,
    start: int,
    end: int,
) -> int:
    if end <= start:
        return start
    return min(
        range(start, end),
        key=lambda frame: sum(
            abs(samples[frame * channels + channel]) for channel in range(channels)
        ),
    )


def _crossfade_frames(
    left: tuple[int, ...],
    right: tuple[int, ...],
    *,
    channels: int,
    sample_rate: int,
) -> int:
    available = min(len(left), len(right)) // channels
    if available < _MINIMUM_OVERLAP_FRAMES:
        return 0
    left_peak = _edge_peak(left, channels=channels, leading=False)
    right_peak = _edge_peak(right, channels=channels, leading=True)
    peak = max(left_peak, right_peak)
    normalized = min(1.0, peak / 8_192)
    milliseconds = round(
        _MAXIMUM_CROSSFADE_MS
        - normalized * (_MAXIMUM_CROSSFADE_MS - _MINIMUM_CROSSFADE_MS)
    )
    return min(available // 2, max(1, round(sample_rate * milliseconds / 1_000)))


def _edge_peak(
    samples: tuple[int, ...],
    *,
    channels: int,
    leading: bool,
) -> int:
    count = min(len(samples), channels * 32)
    edge = samples[:count] if leading else samples[-count:]
    return max((abs(value) for value in edge), default=0)


def _crossfade(
    left: tuple[int, ...],
    right: tuple[int, ...],
    *,
    channels: int,
    frames: int,
) -> tuple[int, ...]:
    count = frames * channels
    if count == 0:
        return (*left, *right)
    left_edge = left[-count:]
    right_edge = right[:count]
    mixed: list[int] = []
    for frame in range(frames):
        right_weight = (frame + 1) / (frames + 1)
        left_weight = 1.0 - right_weight
        for channel in range(channels):
            index = frame * channels + channel
            mixed.append(
                round(left_edge[index] * left_weight + right_edge[index] * right_weight)
            )
    return (*left[:-count], *mixed, *right[count:])


def _smooth(
    samples: tuple[int, ...],
    *,
    channels: int,
    amount: float,
) -> tuple[int, ...]:
    previous = [0.0] * channels
    output: list[int] = []
    for index, sample in enumerate(samples):
        channel = index % channels
        value = sample * (1.0 - amount) + previous[channel] * amount
        previous[channel] = value
        output.append(round(value))
    return tuple(output)


def _rms(samples: tuple[int, ...]) -> float:
    return math.sqrt(sum(value * value for value in samples) / max(1, len(samples)))


def _high_frequency_ratio(samples: tuple[int, ...], channels: int) -> float:
    if len(samples) <= channels:
        return 0.0
    energy = sum(value * value for value in samples)
    delta = sum(
        (samples[index] - samples[index - channels]) ** 2
        for index in range(channels, len(samples))
    )
    return delta / max(1.0, 4.0 * energy)


def _read_wave(path: Path) -> _Wave:
    try:
        with wave.open(str(path), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            rate = reader.getframerate()
            if reader.getcomptype() != "NONE" or width not in {1, 2, 3, 4}:
                raise ArtifactError("Adaptive repair splice requires PCM WAV audio")
            frames = reader.readframes(reader.getnframes())
    except (OSError, EOFError, wave.Error) as error:
        raise ArtifactError(f"Cannot read PCM WAV: {path}") from error
    samples = tuple(
        _decode_sample(frames[index : index + width], width)
        for index in range(0, len(frames), width)
    )
    return _Wave(channels, width, rate, samples)


def _decode_sample(value: bytes, width: int) -> int:
    return (
        value[0] - 128 if width == 1 else int.from_bytes(value, "little", signed=True)
    )


__all__ = [
    "AdaptiveSpliceEvidence",
    "FrameReplacement",
    "splice_wav_region",
    "splice_wav_regions",
]
