"""Safe PCM WAV cropping and independent energy-based speech evidence."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from yakbox._files import atomic_output_path
from yakbox.errors import ArtifactError

_MAXIMUM_BOUNDARY_PROTECTION_MS = 130
_ALIGNMENT_BOUNDARY_TOLERANCE_MS = 25
_STATIONARY_WINDOW_DB_DELTA = 1.5
_MINIMUM_HUMAN_PITCH_HZ = 60
_MAXIMUM_HUMAN_PITCH_HZ = 500


@dataclass(frozen=True, slots=True)
class SpeechRegion:
    """One acoustically detected region containing likely speech."""

    start_seconds: float
    end_seconds: float


@dataclass(frozen=True, slots=True)
class CropEvidence:
    """Exact frame boundaries and acoustic refinements applied to a crop."""

    source_start_seconds: float
    source_end_seconds: float
    crop_start_seconds: float
    crop_end_seconds: float
    pre_roll_ms: float
    post_roll_ms: float
    fade_ms: int
    zero_crossing_start_shift_ms: float
    zero_crossing_end_shift_ms: float


@dataclass(frozen=True, slots=True)
class SpeechIslandEvidence:
    """Acoustic speech clusters used to reject or refine contaminated edges."""

    duration_seconds: float
    regions: tuple[SpeechRegion, ...]
    islands: tuple[SpeechRegion, ...]
    primary_island: SpeechRegion | None
    detached_prefix: tuple[SpeechRegion, ...]
    detached_suffix: tuple[SpeechRegion, ...]
    leading_silence_ms: float
    trailing_silence_ms: float

    @property
    def detached_prefix_ms(self) -> float:
        """Return the total span of detached speech before the primary island."""
        return _region_duration_ms(self.detached_prefix)

    @property
    def detached_suffix_ms(self) -> float:
        """Return the total span of detached speech after the primary island."""
        return _region_duration_ms(self.detached_suffix)


@dataclass(frozen=True, slots=True)
class SignalQualityEvidence:
    """Independent adaptive-VAD and waveform artifact measurements."""

    peak_dbfs: float
    clipped_sample_ratio: float
    leading_boundary_jump_ratio: float
    trailing_boundary_jump_ratio: float
    mean_sample_delta_ratio: float
    adaptive_threshold_dbfs: float
    adaptive_speech_regions: tuple[SpeechRegion, ...]
    vad_disagreement_ms: float
    longest_stationary_voiced_ms: float
    estimated_pitch_hz: float | None
    pitch_variation_ratio: float | None
    high_frequency_energy_ratio: float


@dataclass(frozen=True, slots=True)
class _PcmWave:
    channels: int
    sample_width: int
    sample_rate: int
    frames: tuple[tuple[int, ...], ...]


def detect_speech_regions(
    path: Path,
    *,
    threshold_dbfs: float = -42.0,
    window_ms: int = 10,
    minimum_speech_ms: int = 20,
    merge_gap_ms: int = 40,
) -> tuple[SpeechRegion, ...]:
    """Detect audible PCM regions as an ASR-independent guard signal."""
    audio = _read_pcm_wave(path)
    window_frames = max(1, round(audio.sample_rate * window_ms / 1_000))
    full_scale = float(1 << (audio.sample_width * 8 - 1))
    threshold = full_scale * (10 ** (threshold_dbfs / 20))
    audible: list[tuple[int, int]] = []
    for start in range(0, len(audio.frames), window_frames):
        end = min(len(audio.frames), start + window_frames)
        values = (sample for frame in audio.frames[start:end] for sample in frame)
        count = (end - start) * audio.channels
        energy = sum(sample * sample for sample in values) / max(1, count)
        if math.sqrt(energy) >= threshold:
            audible.append((start, end))
    merged = _merge_regions(
        audible,
        maximum_gap=round(audio.sample_rate * merge_gap_ms / 1_000),
    )
    minimum_frames = round(audio.sample_rate * minimum_speech_ms / 1_000)
    return tuple(
        SpeechRegion(start / audio.sample_rate, end / audio.sample_rate)
        for start, end in merged
        if end - start >= minimum_frames
    )


def inspect_speech_islands(
    path: Path,
    *,
    threshold_dbfs: float = -48.0,
    island_gap_ms: int = 300,
) -> SpeechIslandEvidence:
    """Identify the dominant utterance and acoustically detached edge speech."""
    duration = wav_duration_seconds(path)
    regions = detect_speech_regions(
        path,
        threshold_dbfs=threshold_dbfs,
        window_ms=5,
        minimum_speech_ms=10,
        merge_gap_ms=10,
    )
    islands = _merge_speech_islands(regions, maximum_gap_ms=island_gap_ms)
    if not islands:
        return SpeechIslandEvidence(duration, regions, (), None, (), (), 0.0, 0.0)
    primary_index = max(
        range(len(islands)),
        key=lambda index: (
            islands[index].end_seconds - islands[index].start_seconds,
            -index,
        ),
    )
    primary = islands[primary_index]
    return SpeechIslandEvidence(
        duration_seconds=duration,
        regions=regions,
        islands=islands,
        primary_island=primary,
        detached_prefix=islands[:primary_index],
        detached_suffix=islands[primary_index + 1 :],
        leading_silence_ms=primary.start_seconds * 1_000,
        trailing_silence_ms=max(0.0, duration - primary.end_seconds) * 1_000,
    )


def inspect_signal_quality(path: Path) -> SignalQualityEvidence:
    """Measure clipping, clicks, prolonged tones, and adaptive VAD agreement."""
    audio = _read_pcm_wave(path)
    full_scale = float((1 << (audio.sample_width * 8 - 1)) - 1)
    mono = tuple(sum(frame) / audio.channels for frame in audio.frames)
    absolute = tuple(abs(value) for value in mono)
    peak = max(absolute, default=0.0)
    peak_dbfs = 20 * math.log10(max(1.0, peak) / full_scale)
    clipped = sum(value >= full_scale * 0.995 for value in absolute) / len(absolute)
    deltas = tuple(abs(following - previous) for previous, following in pairwise(mono))
    edge_frames = max(2, round(audio.sample_rate * 0.02))
    leading_jump = max(deltas[:edge_frames], default=0.0) / full_scale
    trailing_jump = max(deltas[-edge_frames:], default=0.0) / full_scale
    mean_delta = sum(deltas) / max(1, len(deltas)) / full_scale
    window_frames = max(1, round(audio.sample_rate * 0.01))
    window_dbfs = tuple(
        _window_dbfs(mono[start : start + window_frames], full_scale)
        for start in range(0, len(mono), window_frames)
    )
    ordered = sorted(window_dbfs)
    noise_floor = ordered[max(0, round((len(ordered) - 1) * 0.2))]
    threshold = min(
        -24.0,
        max(-54.0, min(noise_floor + 12.0, max(window_dbfs) - 6.0)),
    )
    adaptive_frames = [
        (
            index * window_frames,
            min(len(mono), (index + 1) * window_frames),
        )
        for index, value in enumerate(window_dbfs)
        if value >= threshold
    ]
    adaptive = tuple(
        SpeechRegion(start / audio.sample_rate, end / audio.sample_rate)
        for start, end in _merge_regions(
            adaptive_frames,
            maximum_gap=round(audio.sample_rate * 0.03),
        )
        if end - start >= round(audio.sample_rate * 0.02)
    )
    fixed = detect_speech_regions(path)
    disagreement = _region_xor_duration_ms(fixed, adaptive)
    stationary = _longest_stationary_voiced_ms(window_dbfs, threshold)
    pitch, pitch_variation = _pitch_statistics(
        mono,
        sample_rate=audio.sample_rate,
        full_scale=full_scale,
        threshold_dbfs=threshold,
    )
    second_deltas = tuple(
        following - previous for previous, following in pairwise(deltas)
    )
    high_frequency_ratio = sum(value * value for value in second_deltas) / max(
        1.0,
        sum(value * value for value in deltas),
    )
    return SignalQualityEvidence(
        peak_dbfs=peak_dbfs,
        clipped_sample_ratio=clipped,
        leading_boundary_jump_ratio=leading_jump,
        trailing_boundary_jump_ratio=trailing_jump,
        mean_sample_delta_ratio=mean_delta,
        adaptive_threshold_dbfs=threshold,
        adaptive_speech_regions=adaptive,
        vad_disagreement_ms=disagreement,
        longest_stationary_voiced_ms=stationary,
        estimated_pitch_hz=pitch,
        pitch_variation_ratio=pitch_variation,
        high_frequency_energy_ratio=high_frequency_ratio,
    )


def crop_aligned_wav(
    source: Path,
    destination: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    pre_roll_ms: int,
    post_roll_ms: int,
    fade_ms: int,
    speech_regions: tuple[SpeechRegion, ...] = (),
    zero_crossing_window_ms: int = 8,
    overwrite: bool = False,
) -> CropEvidence:
    """Crop aligned speech with bounded padding, zero crossings, and fades."""
    audio = _read_pcm_wave(source)
    total = len(audio.frames)
    aligned_start = round(start_seconds * audio.sample_rate)
    aligned_end = round(end_seconds * audio.sample_rate)
    if aligned_start < 0 or aligned_end <= aligned_start or aligned_end > total:
        raise ArtifactError("Aligned crop timing falls outside the source WAV")
    desired_start = max(
        0,
        aligned_start - round(audio.sample_rate * pre_roll_ms / 1_000),
    )
    desired_end = min(
        total,
        aligned_end + round(audio.sample_rate * post_roll_ms / 1_000),
    )
    desired_start, desired_end = _protect_overlapping_speech(
        speech_regions,
        aligned_start=aligned_start,
        aligned_end=aligned_end,
        desired_start=desired_start,
        desired_end=desired_end,
        sample_rate=audio.sample_rate,
        total_frames=total,
        pre_roll_ms=pre_roll_ms,
        post_roll_ms=post_roll_ms,
    )
    _validate_padding_speech(
        speech_regions,
        desired_start / audio.sample_rate,
        start_seconds,
        end_seconds,
        desired_end / audio.sample_rate,
    )
    search = max(1, round(audio.sample_rate * zero_crossing_window_ms / 1_000))
    crop_start = _quietest_crossing(
        audio.frames,
        desired_start,
        min(aligned_start, desired_start + search),
    )
    crop_end = _quietest_crossing(
        audio.frames,
        max(aligned_end, desired_end - search),
        desired_end,
    )
    if crop_end <= crop_start:
        raise ArtifactError("No safe positive-length crop boundary was found")
    frames = [list(frame) for frame in audio.frames[crop_start:crop_end]]
    _fade_frames(frames, audio.sample_rate, fade_ms)
    _write_pcm_wave(
        destination,
        audio,
        tuple(tuple(frame) for frame in frames),
        overwrite=overwrite,
    )
    return CropEvidence(
        source_start_seconds=start_seconds,
        source_end_seconds=end_seconds,
        crop_start_seconds=crop_start / audio.sample_rate,
        crop_end_seconds=crop_end / audio.sample_rate,
        pre_roll_ms=(aligned_start - crop_start) * 1_000 / audio.sample_rate,
        post_roll_ms=(crop_end - aligned_end) * 1_000 / audio.sample_rate,
        fade_ms=fade_ms,
        zero_crossing_start_shift_ms=(crop_start - desired_start)
        * 1_000
        / audio.sample_rate,
        zero_crossing_end_shift_ms=(crop_end - desired_end) * 1_000 / audio.sample_rate,
    )


def wav_duration_seconds(path: Path) -> float:
    """Return the duration of a readable PCM WAV."""
    audio = _read_pcm_wave(path)
    return len(audio.frames) / audio.sample_rate


def _read_pcm_wave(path: Path) -> _PcmWave:
    try:
        with wave.open(str(path), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            rate = reader.getframerate()
            count = reader.getnframes()
            compression = reader.getcomptype()
            content = reader.readframes(count)
    except (OSError, EOFError, wave.Error) as error:
        raise ArtifactError(f"Cannot read PCM WAV: {path}") from error
    if compression != "NONE" or channels < 1 or width not in {1, 2, 3, 4} or rate < 1:
        raise ArtifactError(f"Unsupported PCM WAV format: {path}")
    frame_size = channels * width
    if not content or len(content) % frame_size:
        raise ArtifactError(f"PCM WAV has invalid or empty frame data: {path}")
    frames = [
        tuple(
            _decode_sample(
                content[
                    frame_start + channel * width : frame_start + (channel + 1) * width
                ],
                width,
            )
            for channel in range(channels)
        )
        for frame_start in range(0, len(content), frame_size)
    ]
    return _PcmWave(channels, width, rate, tuple(frames))


def _write_pcm_wave(
    path: Path,
    source: _PcmWave,
    frames: tuple[tuple[int, ...], ...],
    *,
    overwrite: bool,
) -> None:
    content = b"".join(
        _encode_sample(sample, source.sample_width)
        for frame in frames
        for sample in frame
    )
    with (
        atomic_output_path(path, overwrite=overwrite) as temporary,
        wave.open(str(temporary), "wb") as writer,
    ):
        writer.setnchannels(source.channels)
        writer.setsampwidth(source.sample_width)
        writer.setframerate(source.sample_rate)
        writer.writeframes(content)


def _decode_sample(content: bytes, width: int) -> int:
    if width == 1:
        return content[0] - 128
    return int.from_bytes(content, "little", signed=True)


def _encode_sample(sample: int, width: int) -> bytes:
    minimum = -(1 << (width * 8 - 1))
    maximum = (1 << (width * 8 - 1)) - 1
    bounded = min(maximum, max(minimum, sample))
    if width == 1:
        return bytes((bounded + 128,))
    return bounded.to_bytes(width, "little", signed=True)


def _merge_regions(
    regions: list[tuple[int, int]], *, maximum_gap: int
) -> tuple[tuple[int, int], ...]:
    if not regions:
        return ()
    merged: list[tuple[int, int]] = [regions[0]]
    for start, end in regions[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end <= maximum_gap:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return tuple(merged)


def _merge_speech_islands(
    regions: tuple[SpeechRegion, ...], *, maximum_gap_ms: int
) -> tuple[SpeechRegion, ...]:
    if not regions:
        return ()
    maximum_gap = maximum_gap_ms / 1_000
    merged: list[SpeechRegion] = [regions[0]]
    for region in regions[1:]:
        previous = merged[-1]
        if region.start_seconds - previous.end_seconds <= maximum_gap:
            merged[-1] = SpeechRegion(previous.start_seconds, region.end_seconds)
        else:
            merged.append(region)
    return tuple(merged)


def _region_duration_ms(regions: tuple[SpeechRegion, ...]) -> float:
    return sum(
        max(0.0, region.end_seconds - region.start_seconds) * 1_000
        for region in regions
    )


def _window_dbfs(values: tuple[float, ...], full_scale: float) -> float:
    rms = math.sqrt(sum(value * value for value in values) / max(1, len(values)))
    return 20 * math.log10(max(1.0, rms) / full_scale)


def _region_xor_duration_ms(
    left: tuple[SpeechRegion, ...], right: tuple[SpeechRegion, ...]
) -> float:
    boundaries = sorted(
        {
            value
            for region in (*left, *right)
            for value in (region.start_seconds, region.end_seconds)
        }
    )
    disagreement = 0.0
    for start, end in pairwise(boundaries):
        midpoint = (start + end) / 2
        in_left = any(
            item.start_seconds <= midpoint <= item.end_seconds for item in left
        )
        in_right = any(
            item.start_seconds <= midpoint <= item.end_seconds for item in right
        )
        if in_left != in_right:
            disagreement += end - start
    return disagreement * 1_000


def _longest_stationary_voiced_ms(
    window_dbfs: tuple[float, ...], threshold: float
) -> float:
    longest = 0
    current = 0
    previous: float | None = None
    for value in window_dbfs:
        stationary = value >= threshold and (
            previous is None or abs(value - previous) <= _STATIONARY_WINDOW_DB_DELTA
        )
        current = current + 1 if stationary else 0
        longest = max(longest, current)
        previous = value
    return longest * 10.0


def _pitch_statistics(
    mono: tuple[float, ...],
    *,
    sample_rate: int,
    full_scale: float,
    threshold_dbfs: float,
) -> tuple[float | None, float | None]:
    window_frames = max(1, round(sample_rate * 0.04))
    pitches: list[float] = []
    for start in range(0, len(mono), window_frames):
        window = mono[start : start + window_frames]
        if _window_dbfs(window, full_scale) < threshold_dbfs:
            continue
        crossings = sum(
            (previous < 0 <= following) or (previous >= 0 > following)
            for previous, following in pairwise(window)
        )
        pitch = crossings * sample_rate / (2 * max(1, len(window)))
        if _MINIMUM_HUMAN_PITCH_HZ <= pitch <= _MAXIMUM_HUMAN_PITCH_HZ:
            pitches.append(pitch)
    if not pitches:
        return None, None
    ordered = sorted(pitches)
    median = ordered[len(ordered) // 2]
    mean = sum(pitches) / len(pitches)
    variance = sum((value - mean) ** 2 for value in pitches) / len(pitches)
    return median, math.sqrt(variance) / max(1.0, mean)


def _validate_padding_speech(
    regions: tuple[SpeechRegion, ...],
    crop_start: float,
    target_start: float,
    target_end: float,
    crop_end: float,
) -> None:
    tolerance = _ALIGNMENT_BOUNDARY_TOLERANCE_MS / 1_000
    maximum_protection = _MAXIMUM_BOUNDARY_PROTECTION_MS / 1_000
    prefix = any(
        region.end_seconds <= target_start
        and region.end_seconds > crop_start
        and region.start_seconds < target_start
        for region in regions
    )
    suffix = any(
        region.start_seconds >= target_end
        and region.start_seconds < crop_end
        and region.end_seconds > target_end
        and not (
            region.start_seconds - target_end <= tolerance
            and region.end_seconds - target_end <= maximum_protection
        )
        for region in regions
    )
    if prefix or suffix:
        raise ArtifactError("Configured crop padding overlaps adjacent speech")


def _protect_overlapping_speech(
    regions: tuple[SpeechRegion, ...],
    *,
    aligned_start: int,
    aligned_end: int,
    desired_start: int,
    desired_end: int,
    sample_rate: int,
    total_frames: int,
    pre_roll_ms: int,
    post_roll_ms: int,
) -> tuple[int, int]:
    start_seconds = aligned_start / sample_rate
    end_seconds = aligned_end / sample_rate
    tolerance = _ALIGNMENT_BOUNDARY_TOLERANCE_MS / 1_000
    leading = tuple(
        region
        for region in regions
        if region.start_seconds < start_seconds < region.end_seconds
    )
    trailing = tuple(
        region
        for region in regions
        if region.start_seconds <= end_seconds + tolerance
        and region.end_seconds > end_seconds
    )
    if leading:
        speech_start = round(
            min(region.start_seconds for region in leading) * sample_rate
        )
        maximum_extension = round(sample_rate * _MAXIMUM_BOUNDARY_PROTECTION_MS / 1_000)
        if aligned_start - speech_start <= maximum_extension:
            desired_start = min(
                desired_start,
                max(0, speech_start - round(sample_rate * pre_roll_ms / 1_000)),
            )
    if trailing:
        speech_end = round(max(region.end_seconds for region in trailing) * sample_rate)
        maximum_extension = round(sample_rate * _MAXIMUM_BOUNDARY_PROTECTION_MS / 1_000)
        if speech_end - aligned_end <= maximum_extension:
            desired_end = max(
                desired_end,
                min(
                    total_frames,
                    speech_end + round(sample_rate * post_roll_ms / 1_000),
                ),
            )
    return desired_start, desired_end


def _quietest_crossing(
    frames: tuple[tuple[int, ...], ...], start: int, end: int
) -> int:
    if start >= end:
        return start
    return min(
        range(start, end + 1),
        key=lambda index: _boundary_energy(frames, index),
    )


def _boundary_energy(frames: tuple[tuple[int, ...], ...], index: int) -> int:
    before = frames[max(0, min(len(frames) - 1, index - 1))]
    after = frames[max(0, min(len(frames) - 1, index))]
    return sum(abs(value) for value in before) + sum(abs(value) for value in after)


def _fade_frames(frames: list[list[int]], sample_rate: int, fade_ms: int) -> None:
    count = min(round(sample_rate * fade_ms / 1_000), len(frames) // 2)
    if count < 1:
        return
    for index in range(count):
        gain = (index + 1) / count
        frames[index] = [round(sample * gain) for sample in frames[index]]
        frames[-index - 1] = [round(sample * gain) for sample in frames[-index - 1]]
