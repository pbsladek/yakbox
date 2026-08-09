"""Lazy Apple-Silicon Whisper alignment adapter."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol, cast

from yakbox.audio.crop import detect_speech_regions, wav_duration_seconds
from yakbox.errors import BackendUnavailableError, BuildError
from yakbox.speech.alignment import (
    AlignmentResult,
    AlignmentSegment,
    AlignmentToken,
    DecodePassEvidence,
    alignment_fingerprint,
    lexical_tokens,
)
from yakbox.whisper_models import require_model_path


class _MlxWhisper(Protocol):
    def transcribe(self, audio: str, **options: object) -> Mapping[str, object]: ...


class _MlxRandom(Protocol):
    def seed(self, seed: int) -> None: ...


class _MlxCore(Protocol):
    random: _MlxRandom


class MlxWhisperAligner:
    """Run MLX Whisper locally and expose backend-neutral word timestamps."""

    def __init__(
        self,
        *,
        model: str,
        revision: str | None = None,
        timeout_seconds: float = 180.0,
        prompted_timing: bool = True,
        decode_consensus: bool = False,
        prompt_sensitivity: bool = False,
        maximum_consensus_timing_delta_ms: int = 180,
        hallucination_silence_threshold: float = 0.8,
    ) -> None:
        self.model = model
        self.revision = revision
        self._module: _MlxWhisper | None = None
        self._model_path: str | None = None
        self.timeout_seconds = timeout_seconds
        self.prompted_timing = prompted_timing
        self.decode_consensus = decode_consensus
        self.prompt_sensitivity = prompt_sensitivity
        self.maximum_consensus_timing_delta_ms = maximum_consensus_timing_delta_ms
        self.hallucination_silence_threshold = hallucination_silence_threshold
        try:
            version = importlib.metadata.version("mlx-whisper")
        except importlib.metadata.PackageNotFoundError:
            version = "unavailable"
        identity = f"{model}@{revision or 'local'}"
        self._fingerprint = alignment_fingerprint(
            "mlx-whisper",
            identity,
            version,
            settings={
                "condition_on_previous_text": False,
                "word_timestamps": True,
                "authority_temperature": [0.0, 0.2, 0.4],
                "consensus": decode_consensus,
                "consensus_temperature": 0.2,
                "consensus_best_of": 5,
                "consensus_seed": 42,
                "prompt_sensitivity": prompt_sensitivity,
                "prompted_timing": prompted_timing,
                "maximum_consensus_timing_delta_ms": (
                    maximum_consensus_timing_delta_ms
                ),
                "hallucination_silence_threshold": (hallucination_silence_threshold),
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            },
        )

    @property
    def fingerprint(self) -> str:
        """Return the package/model fingerprint used by caches and reports."""
        return self._fingerprint

    async def align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
    ) -> AlignmentResult:
        """Transcribe without bias, then optionally refine already-matching timing."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._align_sync,
                    audio,
                    expected_text,
                    language,
                    None,
                ),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as error:
            raise BuildError(
                f"MLX Whisper exceeded the {self.timeout_seconds:g}s alignment timeout"
            ) from error

    async def align_window(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
        start_seconds: float,
        end_seconds: float,
    ) -> AlignmentResult:
        """Inspect one bounded audio range while preserving absolute timestamps."""
        duration = wav_duration_seconds(audio)
        if (
            not math.isfinite(start_seconds)
            or not math.isfinite(end_seconds)
            or start_seconds < 0
            or end_seconds <= start_seconds
            or end_seconds > duration + 0.001
        ):
            raise BuildError("Whisper inspection window falls outside the audio")
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._align_sync,
                    audio,
                    expected_text,
                    language,
                    (start_seconds, min(duration, end_seconds)),
                ),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as error:
            raise BuildError(
                f"MLX Whisper exceeded the {self.timeout_seconds:g}s alignment timeout"
            ) from error

    def _align_sync(
        self,
        audio: Path,
        expected_text: str,
        language: str,
        clip: tuple[float, float] | None,
    ) -> AlignmentResult:
        module = self._load_module()
        options: dict[str, object] = {
            "path_or_hf_repo": self._resolve_model_path(),
            "word_timestamps": True,
            "condition_on_previous_text": False,
            "temperature": (0.0, 0.2, 0.4),
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
        }
        if clip is not None:
            options["clip_timestamps"] = [clip[0], clip[1]]
        normalized_language = language.split("-", maxsplit=1)[0].strip().casefold()
        if normalized_language:
            options["language"] = normalized_language
        raw = module.transcribe(str(audio), **options)
        duration = wav_duration_seconds(audio)
        parsed = _parse_transcription(raw, audio_duration_seconds=duration)
        tokens = parsed.tokens
        timing_source = "unprompted"
        expected_tokens = lexical_tokens(expected_text)
        recognized_tokens = tuple(token.text for token in parsed.tokens)
        decode_passes = [
            _decode_pass_evidence(
                "authority",
                raw,
                parsed,
                expected_tokens=expected_tokens,
            )
        ]
        consensus_score: float | None = None
        maximum_timing_delta_ms: float | None = None
        consensus_reasons: tuple[str, ...] = ()
        if self.decode_consensus:
            _seed_mlx_sampling(module, seed=42)
            consensus_options = {
                **options,
                "temperature": 0.2,
                "best_of": 5,
            }
            consensus_raw = module.transcribe(str(audio), **consensus_options)
            consensus = _parse_transcription(
                consensus_raw,
                audio_duration_seconds=duration,
            )
            decode_passes.append(
                _decode_pass_evidence(
                    "sampled_consensus",
                    consensus_raw,
                    consensus,
                    expected_tokens=expected_tokens,
                )
            )
            (
                consensus_score,
                maximum_timing_delta_ms,
                consensus_reasons,
            ) = _compare_decodes(
                parsed,
                consensus,
                maximum_timing_delta_ms=self.maximum_consensus_timing_delta_ms,
            )
        prompt_state = "not_tested"
        should_prompt = bool(expected_tokens) and (
            self.prompt_sensitivity
            or (
                self.prompted_timing
                and not parsed.issues
                and recognized_tokens == expected_tokens
            )
        )
        if should_prompt:
            prompted_options = {
                **options,
                "temperature": 0.0,
                "initial_prompt": expected_text,
            }
            prompted_raw = module.transcribe(str(audio), **prompted_options)
            prompted = _parse_transcription(
                prompted_raw,
                audio_duration_seconds=duration,
            )
            prompted_tokens = tuple(token.text for token in prompted.tokens)
            decode_passes.append(
                _decode_pass_evidence(
                    "expected_prompt",
                    prompted_raw,
                    prompted,
                    expected_tokens=expected_tokens,
                )
            )
            prompt_state = _prompt_sensitivity(
                recognized_tokens,
                prompted_tokens,
                expected_tokens,
            )
            if (
                self.prompted_timing
                and not prompted.issues
                and recognized_tokens == expected_tokens
                and prompted_tokens == recognized_tokens
            ):
                tokens = prompted.tokens
                timing_source = "prompted_exact_match"
        speech_regions = detect_speech_regions(audio)
        if clip is not None:
            speech_regions = tuple(
                region
                for region in speech_regions
                if region.end_seconds > clip[0] and region.start_seconds < clip[1]
            )
        return AlignmentResult(
            tokens=tokens,
            speech_regions=speech_regions,
            backend="mlx-whisper",
            model=self.model,
            fingerprint=self.fingerprint,
            segments=parsed.segments,
            issues=parsed.issues,
            language=_optional_text(raw.get("language")),
            timing_source=timing_source,
            transcript=_optional_text(raw.get("text")) or "",
            decode_passes=tuple(decode_passes),
            consensus_score=consensus_score,
            maximum_timing_delta_ms=maximum_timing_delta_ms,
            consensus_reason_codes=consensus_reasons,
            prompt_sensitivity=prompt_state,
            clip_start_seconds=clip[0] if clip else None,
            clip_end_seconds=clip[1] if clip else None,
        )

    def _load_module(self) -> _MlxWhisper:
        if self._module is not None:
            return self._module
        try:
            self._module = cast(_MlxWhisper, importlib.import_module("mlx_whisper"))
        except ImportError as error:
            raise BackendUnavailableError(
                "MLX Whisper alignment is not installed; install Yakbox with "
                "the alignment extra: uv sync --extra alignment"
            ) from error
        return self._module

    def _resolve_model_path(self) -> str:
        if self._model_path is not None:
            return self._model_path
        self._model_path = str(require_model_path(self.model, self.revision))
        return self._model_path


def _seed_mlx_sampling(module: _MlxWhisper, *, seed: int) -> None:
    """Seed the real MLX sampler while leaving injected test adapters alone."""
    if getattr(module, "__name__", None) != "mlx_whisper":
        return
    core = cast(_MlxCore, importlib.import_module("mlx.core"))
    core.random.seed(seed)


def open_local_aligner(
    backend: str,
    *,
    model: str,
    revision: str | None,
    timeout_seconds: float = 180.0,
    prompted_timing: bool = True,
    decode_consensus: bool = False,
    prompt_sensitivity: bool = False,
    maximum_consensus_timing_delta_ms: int = 180,
    hallucination_silence_threshold: float = 0.8,
) -> MlxWhisperAligner:
    """Open a supported lazy local alignment backend."""
    if backend.casefold() == "mlx-whisper":
        return MlxWhisperAligner(
            model=model,
            revision=revision,
            timeout_seconds=timeout_seconds,
            prompted_timing=prompted_timing,
            decode_consensus=decode_consensus,
            prompt_sensitivity=prompt_sensitivity,
            maximum_consensus_timing_delta_ms=maximum_consensus_timing_delta_ms,
            hallucination_silence_threshold=hallucination_silence_threshold,
        )
    raise BackendUnavailableError(f"Unsupported local alignment backend: {backend}")


def _segment_tokens(segment: Mapping[str, object]) -> tuple[AlignmentToken, ...]:
    tokens, _ = _parse_segment_tokens(segment, segment_index=0)
    return tokens


def _decode_pass_evidence(
    name: str,
    raw: Mapping[str, object],
    parsed: _ParsedTranscription,
    *,
    expected_tokens: tuple[str, ...],
) -> DecodePassEvidence:
    tokens = tuple(token.text for token in parsed.tokens)
    confidences = tuple(
        token.confidence for token in parsed.tokens if token.confidence is not None
    )
    minimum_confidence = (
        min(confidences)
        if len(confidences) == len(parsed.tokens) and confidences
        else None
    )
    return DecodePassEvidence(
        name=name,
        transcript=_optional_text(raw.get("text")) or "",
        tokens=tokens,
        issues=parsed.issues,
        minimum_confidence=minimum_confidence,
        matches_expected=(tokens == expected_tokens if expected_tokens else None),
    )


def _compare_decodes(
    authority: _ParsedTranscription,
    comparison: _ParsedTranscription,
    *,
    maximum_timing_delta_ms: int,
) -> tuple[float, float | None, tuple[str, ...]]:
    authority_tokens = tuple(token.text for token in authority.tokens)
    comparison_tokens = tuple(token.text for token in comparison.tokens)
    reasons: list[str] = []
    if authority.issues or comparison.issues:
        reasons.append("decode_consensus_evidence_invalid")
    score = _sequence_agreement(authority_tokens, comparison_tokens)
    if authority_tokens != comparison_tokens:
        reasons.append("decode_consensus_mismatch")
        return score, None, tuple(reasons)
    deltas = tuple(
        abs(left - right) * 1_000
        for authority_token, comparison_token in zip(
            authority.tokens,
            comparison.tokens,
            strict=True,
        )
        for left, right in (
            (authority_token.start_seconds, comparison_token.start_seconds),
            (authority_token.end_seconds, comparison_token.end_seconds),
        )
    )
    maximum_delta = max(deltas, default=0.0)
    if maximum_delta > maximum_timing_delta_ms:
        reasons.append("decode_timing_instability")
    return score, maximum_delta, tuple(reasons)


def _sequence_agreement(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left and not right:
        return 1.0
    return SequenceMatcher(a=left, b=right, autojunk=False).ratio()


def _prompt_sensitivity(
    authority: tuple[str, ...],
    prompted: tuple[str, ...],
    expected: tuple[str, ...],
) -> str:
    if authority == prompted == expected:
        return "stable_match"
    if authority != expected and prompted == expected:
        return "prompt_rescued"
    if authority == expected and prompted != expected:
        return "prompt_destabilized"
    if authority == prompted:
        return "stable_mismatch"
    return "prompt_changed"


@dataclass(frozen=True, slots=True)
class _ParsedTranscription:
    tokens: tuple[AlignmentToken, ...]
    segments: tuple[AlignmentSegment, ...]
    issues: tuple[str, ...]


def _parse_transcription(
    raw: Mapping[str, object],
    *,
    audio_duration_seconds: float,
) -> _ParsedTranscription:
    raw_segments = raw.get("segments")
    if not _is_sequence(raw_segments):
        return _ParsedTranscription((), (), ("segments_missing",))
    tokens: list[AlignmentToken] = []
    segments: list[AlignmentSegment] = []
    issues: list[str] = []
    for index, raw_segment in enumerate(cast(Sequence[object], raw_segments)):
        if not isinstance(raw_segment, Mapping):
            issues.append(f"segment_{index}_malformed")
            continue
        segment = cast(Mapping[str, object], raw_segment)
        parsed_segment, segment_issues = _parse_segment(
            segment,
            segment_index=index,
            audio_duration_seconds=audio_duration_seconds,
        )
        if parsed_segment is not None:
            segments.append(parsed_segment)
        segment_tokens, token_issues = _parse_segment_tokens(
            segment,
            segment_index=index,
            audio_duration_seconds=audio_duration_seconds,
        )
        tokens.extend(segment_tokens)
        issues.extend((*segment_issues, *token_issues))
    if not raw_segments:
        issues.append("segments_empty")
    return _ParsedTranscription(
        tuple(tokens),
        tuple(segments),
        tuple(dict.fromkeys(issues)),
    )


def _parse_segment(
    segment: Mapping[str, object],
    *,
    segment_index: int,
    audio_duration_seconds: float,
) -> tuple[AlignmentSegment | None, tuple[str, ...]]:
    prefix = f"segment_{segment_index}"
    start = _finite_number(segment.get("start"))
    end = _finite_number(segment.get("end"))
    issues: list[str] = []
    if start is None or end is None:
        return None, (f"{prefix}_timing_missing",)
    if start < 0 or end <= start or end > audio_duration_seconds + 0.1:
        issues.append(f"{prefix}_timing_invalid")
    average_log_probability = _finite_number(segment.get("avg_logprob"))
    compression_ratio = _finite_number(segment.get("compression_ratio"))
    no_speech_probability = _finite_number(segment.get("no_speech_prob"))
    temperature = _finite_number(segment.get("temperature"))
    for value, name in (
        (average_log_probability, "avg_logprob"),
        (compression_ratio, "compression_ratio"),
        (no_speech_probability, "no_speech_prob"),
        (temperature, "temperature"),
    ):
        if value is None:
            issues.append(f"{prefix}_{name}_missing")
    return (
        AlignmentSegment(
            start_seconds=start,
            end_seconds=end,
            average_log_probability=average_log_probability,
            compression_ratio=compression_ratio,
            no_speech_probability=no_speech_probability,
            temperature=temperature,
        ),
        tuple(issues),
    )


def _parse_segment_tokens(
    segment: Mapping[str, object],
    *,
    segment_index: int,
    audio_duration_seconds: float = math.inf,
) -> tuple[tuple[AlignmentToken, ...], tuple[str, ...]]:
    words = segment.get("words")
    if not _is_sequence(words):
        return (), (f"segment_{segment_index}_words_missing",)
    result: list[AlignmentToken] = []
    issues: list[str] = []
    for word_index, raw in enumerate(cast(Sequence[object], words)):
        prefix = f"segment_{segment_index}_word_{word_index}"
        token, word_issues = _parse_word(
            raw,
            prefix=prefix,
            audio_duration_seconds=audio_duration_seconds,
        )
        issues.extend(word_issues)
        if token is not None:
            result.append(token)
    if not words:
        issues.append(f"segment_{segment_index}_words_empty")
    return tuple(result), tuple(issues)


def _parse_word(
    raw: object,
    *,
    prefix: str,
    audio_duration_seconds: float,
) -> tuple[AlignmentToken | None, tuple[str, ...]]:
    if not isinstance(raw, Mapping):
        return None, (f"{prefix}_malformed",)
    text = raw.get("word")
    start = _finite_number(raw.get("start"))
    end = _finite_number(raw.get("end"))
    if not isinstance(text, str):
        return None, (f"{prefix}_text_missing",)
    if start is None or end is None:
        return None, (f"{prefix}_timing_missing",)
    normalized = lexical_tokens(text)
    if len(normalized) != 1:
        return None, (f"{prefix}_lexical_invalid",)
    issues: list[str] = []
    if start < 0 or end <= start or end > audio_duration_seconds + 0.1:
        issues.append(f"{prefix}_timing_invalid")
    confidence = _finite_number(raw.get("probability"))
    if confidence is None:
        issues.append(f"{prefix}_probability_missing")
    elif not 0 <= confidence <= 1:
        issues.append(f"{prefix}_probability_invalid")
    return (
        AlignmentToken(normalized[0], start, end, confidence),
        tuple(issues),
    )


def _finite_number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
