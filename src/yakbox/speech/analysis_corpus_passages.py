"""Select independent, quiet-bounded passages from licensed source archives."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import wave
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

from yakbox._files import atomic_output_path, sha256_file
from yakbox.errors import ArtifactError, BackendUnavailableError, ValidationError
from yakbox.speech.analysis_corpus_archives import (
    CorpusSourceArchive,
    CorpusSourceArchiveInventory,
    load_corpus_source_archive_inventory,
)
from yakbox.speech.analysis_corpus_sources import (
    CorpusSourceInventory,
    CorpusSourceWindow,
    LicensedVoiceSource,
    load_licensed_voice_sources,
    write_corpus_source_inventory,
)
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_models import AudioSpan
from yakbox.speech.canonical_audio import CanonicalAudioPreparer

_PASSAGE_START_SECONDS = (120, 240, 360)
_EXTRACTION_SECONDS = 30
_MINIMUM_PASSAGE_COUNT = 3
_MINIMUM_EXTRACTION_SECONDS = 10
_SAMPLE_RATE = 16_000
_PCM16_SAMPLE_WIDTH = 2
_ENERGY_WINDOW_MILLISECONDS = 80
_MINIMUM_PASSAGE_RATIO = 0.5


def prepare_expanded_corpus_source_inventory(
    voice_registry: Path,
    archive_inventory: CorpusSourceArchiveInventory,
    *,
    repository_root: Path,
    archive_root: Path,
    output_root: Path,
    passage_starts_seconds: tuple[int, ...] = _PASSAGE_START_SECONDS,
    extraction_seconds: int = _EXTRACTION_SECONDS,
) -> CorpusSourceInventory:
    """Extract separate source passages and materialize quiet-bounded WAVs."""
    _validate_passage_plan(passage_starts_seconds, extraction_seconds)
    sources = load_licensed_voice_sources(
        voice_registry,
        repository_root=repository_root,
    )
    archives = {item.voice: item for item in archive_inventory.archives}
    if set(archives) != {item.voice for item in sources}:
        raise ValidationError("Corpus source archives do not match the voice registry")
    output_root = output_root.resolve()
    archive_root = archive_root.resolve()
    preparer = CanonicalAudioPreparer(output_root / "cache")
    windows: list[CorpusSourceWindow] = []
    for source in sources:
        archive = archives[source.voice]
        _validate_archive_source(source, archive)
        source_audio = archive_root / archive.relative_path
        for passage_index, start_seconds in enumerate(
            passage_starts_seconds,
            start=1,
        ):
            extracted = _extract_archive_passage(
                source_audio,
                output_root=output_root,
                voice=source.voice,
                start_seconds=start_seconds,
                extraction_seconds=extraction_seconds,
            )
            prepared = preparer.prepare(extracted)
            start_frame, end_frame = _quiet_passage_bounds(prepared.path)
            span = AudioSpan(
                prepared.identity.canonical_digest,
                start_frame,
                end_frame,
                prepared.identity.frame_map.analysis_rate,
            )
            materialized = preparer.materialize_window(prepared, span)
            group = f"{source.voice}-passage-{passage_index:02d}"
            windows.append(
                CorpusSourceWindow(
                    f"{group}-window-01",
                    group,
                    source.voice,
                    source.reader,
                    materialized.path.relative_to(output_root).as_posix(),
                    materialized.window_span.audio_digest,
                    materialized.window_span.sample_rate,
                    materialized.window_span.end_frame,
                    prepared.identity.canonical_digest,
                    start_frame,
                    end_frame,
                    start_seconds * 1_000,
                    (start_seconds + extraction_seconds) * 1_000,
                    source.source_url,
                    source.source_digest,
                    source.rights_id,
                    source.rights_url,
                )
            )
    registry_digest = semantic_fingerprint(
        "speech-corpus-expanded-source-input-v1",
        {
            "voice_registry_digest": sha256_file(voice_registry),
            "archive_inventory_fingerprint": archive_inventory.fingerprint,
            "passage_starts_seconds": passage_starts_seconds,
            "extraction_seconds": extraction_seconds,
        },
    )
    return CorpusSourceInventory(
        registry_digest,
        1,
        tuple(sorted(windows, key=lambda item: item.source_window_id)),
    )


def _validate_passage_plan(starts: tuple[int, ...], duration: int) -> None:
    if (
        len(starts) < _MINIMUM_PASSAGE_COUNT
        or starts != tuple(sorted(set(starts)))
        or duration < _MINIMUM_EXTRACTION_SECONDS
        or any(start < 0 for start in starts)
        or any(second - first < duration for first, second in pairwise(starts))
    ):
        raise ValidationError("Corpus passage extraction plan is invalid")


def _validate_archive_source(
    source: LicensedVoiceSource,
    archive: CorpusSourceArchive,
) -> None:
    if (
        archive.reader != source.reader
        or archive.source_url != source.source_url
        or archive.source_digest != source.source_digest
        or archive.rights_id != source.rights_id
        or archive.rights_url != source.rights_url
    ):
        raise ValidationError("Corpus source archive provenance differs")


def _extract_archive_passage(
    source: Path,
    *,
    output_root: Path,
    voice: str,
    start_seconds: int,
    extraction_seconds: int,
) -> Path:
    if shutil.which("ffmpeg") is None:
        raise BackendUnavailableError(
            "FFmpeg is required for corpus passage extraction"
        )
    destination = (
        output_root
        / "passage-clips"
        / voice
        / f"{start_seconds:06d}-{extraction_seconds:03d}.wav"
    )
    if _valid_extracted_audio(destination, extraction_seconds=extraction_seconds):
        return destination
    with atomic_output_path(destination, overwrite=destination.exists()) as temporary:
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-ss",
            str(start_seconds),
            "-t",
            str(extraction_seconds),
            "-i",
            str(source),
            "-map_metadata",
            "-1",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
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
        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                command,
                check=False,
                capture_output=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ArtifactError("FFmpeg could not extract corpus passage") from error
        if completed.returncode or not _valid_extracted_audio(
            temporary,
            extraction_seconds=extraction_seconds,
        ):
            raise ArtifactError("FFmpeg rejected corpus passage extraction")
    return destination


def _valid_extracted_audio(path: Path, *, extraction_seconds: int) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        with wave.open(str(path), "rb") as reader:
            expected_frames = extraction_seconds * _SAMPLE_RATE
            return (
                reader.getnchannels() == 1
                and reader.getsampwidth() == _PCM16_SAMPLE_WIDTH
                and reader.getframerate() == _SAMPLE_RATE
                and abs(reader.getnframes() - expected_frames) <= 1
            )
    except EOFError, OSError, wave.Error:
        return False


def _quiet_passage_bounds(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as reader:
        rate = reader.getframerate()
        frame_count = reader.getnframes()
        samples = memoryview(reader.readframes(frame_count)).cast("h")
    start = _quietest_frame(
        samples,
        round(frame_count * 0.05),
        round(frame_count * 0.2),
        rate,
    )
    end = _quietest_frame(
        samples,
        round(frame_count * 0.8),
        round(frame_count * 0.95),
        rate,
    )
    if end - start < round(frame_count * _MINIMUM_PASSAGE_RATIO):
        raise ArtifactError("Corpus passage has no safe quiet-bounded interval")
    return start, end


def _quietest_frame(
    samples: memoryview,
    lower: int,
    upper: int,
    sample_rate: int,
) -> int:
    width = max(1, round(sample_rate * _ENERGY_WINDOW_MILLISECONDS / 1_000))
    step = max(1, width // 4)
    return min(
        range(lower, upper + 1, step),
        key=lambda frame: (
            sum(abs(value) for value in samples[frame : frame + width]),
            abs(frame - (lower + upper) // 2),
            frame,
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select independent passages from licensed source archives"
    )
    parser.add_argument("--voice-registry", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--archive-inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    archives = load_corpus_source_archive_inventory(
        arguments.archive_inventory,
        archive_root=arguments.archive_root,
    )
    inventory = prepare_expanded_corpus_source_inventory(
        arguments.voice_registry,
        archives,
        repository_root=arguments.repository_root,
        archive_root=arguments.archive_root,
        output_root=arguments.output_root,
    )
    write_corpus_source_inventory(arguments.report_output, inventory)
    sys.stdout.write(
        json.dumps(
            {
                "fingerprint": inventory.fingerprint,
                "source_cluster_count": inventory.source_cluster_count,
                "window_count": len(inventory.windows),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "prepare_expanded_corpus_source_inventory"]
