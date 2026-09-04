"""Conservative PCM matching and adaptive crossfades for localized repairs."""

from __future__ import annotations

import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from yakbox._files import atomic_output_path
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

    @property
    def frame_count(self) -> int:
        return len(self.samples) // self.channels


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
    source = _read_wave(original)
    insert = _read_wave(replacement)
    if (
        source.channels,
        source.sample_width,
        source.sample_rate,
    ) != (insert.channels, insert.sample_width, insert.sample_rate):
        raise ArtifactError("Repair splice WAV formats are incompatible")
    start = round(start_seconds * source.sample_rate)
    end = round(end_seconds * source.sample_rate)
    if not 0 <= start < end <= source.frame_count:
        raise ArtifactError("Repair splice timing falls outside the original WAV")
    prefix = source.samples[: start * source.channels]
    suffix = source.samples[end * source.channels :]
    context_frames = max(1, round(source.sample_rate * _CONTEXT_MS / 1_000))
    context = (
        prefix[-context_frames * source.channels :]
        + suffix[: context_frames * source.channels]
    )
    matched, gain_db, smoothing, before_ratio, after_ratio, context_ratio = (
        _match_replacement(insert.samples, context, source=source)
    )
    trimmed, leading_trim, trailing_trim = _trim_to_quiet_edges(
        matched,
        channels=source.channels,
        sample_rate=source.sample_rate,
    )
    leading_frames = _crossfade_frames(
        prefix,
        trimmed,
        channels=source.channels,
        sample_rate=source.sample_rate,
    )
    trailing_frames = _crossfade_frames(
        trimmed,
        suffix,
        channels=source.channels,
        sample_rate=source.sample_rate,
    )
    joined = _crossfade(
        prefix,
        trimmed,
        channels=source.channels,
        frames=leading_frames,
    )
    joined = _crossfade(
        joined,
        suffix,
        channels=source.channels,
        frames=trailing_frames,
    )
    _write_wave(destination, source, joined, overwrite=overwrite)
    return AdaptiveSpliceEvidence(
        gain_db=gain_db,
        replacement_high_frequency_ratio_before=before_ratio,
        replacement_high_frequency_ratio_after=after_ratio,
        context_high_frequency_ratio=context_ratio,
        spectral_smoothing=smoothing,
        leading_trim_ms=leading_trim * 1_000 / source.sample_rate,
        trailing_trim_ms=trailing_trim * 1_000 / source.sample_rate,
        leading_crossfade_ms=leading_frames * 1_000 / source.sample_rate,
        trailing_crossfade_ms=trailing_frames * 1_000 / source.sample_rate,
        output_duration_seconds=(len(joined) / source.channels) / source.sample_rate,
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
    source = _read_wave(original)
    ordered = tuple(
        sorted(
            replacements,
            key=lambda item: (item.start_frame, item.end_frame, item.repair_id),
        )
    )
    if len({item.repair_id for item in ordered}) != len(ordered):
        raise ArtifactError("Multi-repair splice IDs must be unique")
    previous_end = 0
    prepared: list[
        tuple[FrameReplacement, tuple[int, ...], AdaptiveSpliceEvidence]
    ] = []
    context_frames = max(1, round(source.sample_rate * _CONTEXT_MS / 1_000))
    for replacement in ordered:
        if replacement.start_frame < previous_end:
            raise ArtifactError("Multi-repair frame intervals overlap")
        if replacement.end_frame > source.frame_count:
            raise ArtifactError("Multi-repair interval falls outside the base WAV")
        insert = _read_wave(replacement.audio)
        if (
            source.channels,
            source.sample_width,
            source.sample_rate,
        ) != (insert.channels, insert.sample_width, insert.sample_rate):
            raise ArtifactError("Multi-repair WAV formats are incompatible")
        prefix_start = max(0, replacement.start_frame - context_frames)
        suffix_end = min(source.frame_count, replacement.end_frame + context_frames)
        context = (
            source.samples[
                prefix_start * source.channels : replacement.start_frame
                * source.channels
            ]
            + source.samples[
                replacement.end_frame * source.channels : suffix_end * source.channels
            ]
        )
        matched, gain, smoothing, before, after, context_ratio = _match_replacement(
            insert.samples,
            context,
            source=source,
        )
        trimmed, leading_trim, trailing_trim = _trim_to_quiet_edges(
            matched,
            channels=source.channels,
            sample_rate=source.sample_rate,
        )
        left = source.samples[
            previous_end * source.channels : replacement.start_frame * source.channels
        ]
        right_context = source.samples[
            replacement.end_frame * source.channels : min(
                source.frame_count, replacement.end_frame + context_frames
            )
            * source.channels
        ]
        leading_frames = _crossfade_frames(
            left,
            trimmed,
            channels=source.channels,
            sample_rate=source.sample_rate,
        )
        trailing_frames = _crossfade_frames(
            trimmed,
            right_context,
            channels=source.channels,
            sample_rate=source.sample_rate,
        )
        evidence = AdaptiveSpliceEvidence(
            gain,
            before,
            after,
            context_ratio,
            smoothing,
            leading_trim * 1_000 / source.sample_rate,
            trailing_trim * 1_000 / source.sample_rate,
            leading_frames * 1_000 / source.sample_rate,
            trailing_frames * 1_000 / source.sample_rate,
            len(trimmed) / source.channels / source.sample_rate,
        )
        prepared.append((replacement, trimmed, evidence))
        previous_end = replacement.end_frame
    output: tuple[int, ...] = ()
    cursor = 0
    for replacement, insert, _evidence in prepared:
        base = source.samples[
            cursor * source.channels : replacement.start_frame * source.channels
        ]
        base_overlap = _crossfade_frames(
            output,
            base,
            channels=source.channels,
            sample_rate=source.sample_rate,
        )
        output = _crossfade(
            output,
            base,
            channels=source.channels,
            frames=base_overlap,
        )
        leading = _crossfade_frames(
            output,
            insert,
            channels=source.channels,
            sample_rate=source.sample_rate,
        )
        output = _crossfade(
            output,
            insert,
            channels=source.channels,
            frames=leading,
        )
        cursor = replacement.end_frame
    suffix = source.samples[cursor * source.channels :]
    trailing = _crossfade_frames(
        output,
        suffix,
        channels=source.channels,
        sample_rate=source.sample_rate,
    )
    output = _crossfade(
        output,
        suffix,
        channels=source.channels,
        frames=trailing,
    )
    _write_wave(destination, source, output, overwrite=overwrite)
    return tuple(item[2] for item in prepared)


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


def _write_wave(
    path: Path,
    template: _Wave,
    samples: tuple[int, ...],
    *,
    overwrite: bool,
) -> None:
    payload = b"".join(
        _encode_sample(value, template.sample_width) for value in samples
    )
    with (
        atomic_output_path(path, overwrite=overwrite) as temporary,
        wave.open(str(temporary), "wb") as writer,
    ):
        writer.setnchannels(template.channels)
        writer.setsampwidth(template.sample_width)
        writer.setframerate(template.sample_rate)
        writer.writeframes(payload)


def _decode_sample(value: bytes, width: int) -> int:
    return (
        value[0] - 128 if width == 1 else int.from_bytes(value, "little", signed=True)
    )


def _encode_sample(value: int, width: int) -> bytes:
    if width == 1:
        return bytes((max(0, min(255, value + 128)),))
    limit = 1 << (width * 8 - 1)
    bounded = max(-limit, min(limit - 1, value))
    return bounded.to_bytes(width, "little", signed=True)


__all__ = [
    "AdaptiveSpliceEvidence",
    "FrameReplacement",
    "splice_wav_region",
    "splice_wav_regions",
]
