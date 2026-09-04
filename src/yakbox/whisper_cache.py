"""Content-addressed, privacy-safe cache for local Whisper alignments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_json, safe_child, sha256_file
from yakbox.audio.crop import SpeechRegion
from yakbox.speech.alignment import (
    AlignmentResult,
    AlignmentSegment,
    AlignmentToken,
    DecodePassEvidence,
    WindowSpeechAligner,
)

_CACHE_VERSION = 1


class CachedWhisperAligner:
    """Cache exact aligner requests by audio, model, language, text hash, and range."""

    def __init__(self, delegate: WindowSpeechAligner, root: Path) -> None:
        self.delegate = delegate
        self.root = root.resolve()
        self.hits = 0
        self.misses = 0

    @property
    def fingerprint(self) -> str:
        """Return the wrapped aligner's stable runtime fingerprint."""
        return self.delegate.fingerprint

    async def align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
    ) -> AlignmentResult:
        """Return a cached complete-file alignment or compute it once."""
        return await self._resolve(
            audio,
            expected_text,
            language=language,
            start_seconds=None,
            end_seconds=None,
        )

    async def align_window(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
        start_seconds: float,
        end_seconds: float,
    ) -> AlignmentResult:
        """Return a cached bounded alignment or compute it once."""
        return await self._resolve(
            audio,
            expected_text,
            language=language,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )

    async def _resolve(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
        start_seconds: float | None,
        end_seconds: float | None,
    ) -> AlignmentResult:
        key = _cache_key(
            audio,
            aligner_fingerprint=self.fingerprint,
            expected_text=expected_text,
            language=language,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        path = safe_child(self.root, self.root / key[:2] / f"{key}.json")
        cached = _read_cache(path, expected_key=key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        if start_seconds is None or end_seconds is None:
            result = await self.delegate.align(
                audio,
                expected_text,
                language=language,
            )
        else:
            result = await self.delegate.align_window(
                audio,
                expected_text,
                language=language,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        atomic_write_json(path, _cache_document(key, result))
        return result


def _cache_key(
    audio: Path,
    *,
    aligner_fingerprint: str,
    expected_text: str,
    language: str,
    start_seconds: float | None,
    end_seconds: float | None,
) -> str:
    payload = {
        "version": _CACHE_VERSION,
        "audio_sha256": sha256_file(audio),
        "aligner_fingerprint": aligner_fingerprint,
        "expected_sha256": hashlib.sha256(expected_text.encode()).hexdigest(),
        "language": language.casefold(),
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cache_document(key: str, result: AlignmentResult) -> dict[str, object]:
    return {
        "cache_version": _CACHE_VERSION,
        "key": key,
        "alignment": {
            "tokens": [asdict(item) for item in result.tokens],
            "speech_regions": [asdict(item) for item in result.speech_regions],
            "backend": result.backend,
            "model": result.model,
            "fingerprint": result.fingerprint,
            "segments": [asdict(item) for item in result.segments],
            "issues": list(result.issues),
            "language": result.language,
            "timing_source": result.timing_source,
            "decode_passes": [
                {**asdict(item), "transcript": ""} for item in result.decode_passes
            ],
            "consensus_score": result.consensus_score,
            "maximum_timing_delta_ms": result.maximum_timing_delta_ms,
            "consensus_reason_codes": list(result.consensus_reason_codes),
            "prompt_sensitivity": result.prompt_sensitivity,
            "clip_start_seconds": result.clip_start_seconds,
            "clip_end_seconds": result.clip_end_seconds,
        },
    }


def _read_cache(path: Path, *, expected_key: str) -> AlignmentResult | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or raw.get("cache_version") != _CACHE_VERSION
            or raw.get("key") != expected_key
            or not isinstance(raw.get("alignment"), dict)
        ):
            return None
        value = cast(dict[str, object], raw["alignment"])
        return AlignmentResult(
            tokens=tuple(
                _alignment_token(cast(dict[str, object], item))
                for item in cast(list[object], value["tokens"])
            ),
            speech_regions=tuple(
                _speech_region(cast(dict[str, object], item))
                for item in cast(list[object], value["speech_regions"])
            ),
            backend=cast(str, value["backend"]),
            model=cast(str, value["model"]),
            fingerprint=cast(str, value["fingerprint"]),
            segments=tuple(
                _alignment_segment(cast(dict[str, object], item))
                for item in cast(list[object], value.get("segments", []))
            ),
            issues=tuple(cast(list[str], value.get("issues", []))),
            language=cast(str | None, value.get("language")),
            timing_source=cast(str, value.get("timing_source", "unprompted")),
            transcript=cast(str, value.get("transcript", "")),
            decode_passes=tuple(
                DecodePassEvidence(
                    name=cast(str, item["name"]),
                    transcript=cast(str, item["transcript"]),
                    tokens=tuple(cast(list[str], item["tokens"])),
                    issues=tuple(cast(list[str], item["issues"])),
                    minimum_confidence=cast(float | None, item["minimum_confidence"]),
                    matches_expected=cast(bool | None, item["matches_expected"]),
                )
                for item in cast(
                    list[dict[str, object]], value.get("decode_passes", [])
                )
            ),
            consensus_score=cast(float | None, value.get("consensus_score")),
            maximum_timing_delta_ms=cast(
                float | None, value.get("maximum_timing_delta_ms")
            ),
            consensus_reason_codes=tuple(
                cast(list[str], value.get("consensus_reason_codes", []))
            ),
            prompt_sensitivity=cast(str, value.get("prompt_sensitivity", "not_tested")),
            clip_start_seconds=cast(float | None, value.get("clip_start_seconds")),
            clip_end_seconds=cast(float | None, value.get("clip_end_seconds")),
        )
    except KeyError, OSError, TypeError, ValueError, json.JSONDecodeError:
        return None


def _alignment_token(value: dict[str, object]) -> AlignmentToken:
    text = value["text"]
    if not isinstance(text, str):
        raise TypeError
    return AlignmentToken(
        text=text,
        start_seconds=_number(value["start_seconds"]),
        end_seconds=_number(value["end_seconds"]),
        confidence=_optional_number(value.get("confidence")),
    )


def _speech_region(value: dict[str, object]) -> SpeechRegion:
    return SpeechRegion(
        start_seconds=_number(value["start_seconds"]),
        end_seconds=_number(value["end_seconds"]),
    )


def _alignment_segment(value: dict[str, object]) -> AlignmentSegment:
    return AlignmentSegment(
        start_seconds=_number(value["start_seconds"]),
        end_seconds=_number(value["end_seconds"]),
        average_log_probability=_optional_number(value.get("average_log_probability")),
        compression_ratio=_optional_number(value.get("compression_ratio")),
        no_speech_probability=_optional_number(value.get("no_speech_probability")),
        temperature=_optional_number(value.get("temperature")),
    )


def _number(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError
    return float(value)


def _optional_number(value: object) -> float | None:
    return None if value is None else _number(value)
