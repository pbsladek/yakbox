"""Lazy local-only adapters for qualified speech-analysis model families."""

from __future__ import annotations

import asyncio
import importlib
import math
import sys
import wave
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

from yakbox._files import safe_child, sha256_file
from yakbox.errors import BackendUnavailableError, SpeechAnalysisError, ValidationError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint, text_fingerprint
from yakbox.speech.analysis_models import (
    AlignmentPurpose,
    AudioSpan,
    ConversionIdentity,
    ExecutionIdentity,
    ForcedAlignmentResult,
    ForcedAlignmentUnit,
    ModelArtifactIdentity,
    ParakeetEvidence,
    QwenEvidence,
    RecognitionResult,
    RecognitionToken,
    ScoreKind,
    VerifiedTextSpan,
    WhisperEvidence,
)
from yakbox.speech.analysis_protocol import ANALYSIS_WORKER_PROTOCOL_VERSION
from yakbox.speech.model_registry import ModelRecord, ModelRegistry
from yakbox.speech.normalization import normalize_english

ADAPTER_VERSION = 3
_MAXIMUM_TOKENS_PER_SECOND = 50
_MINIMUM_TOKEN_BUDGET = 100
_QWEN_MAXIMUM_TOKEN_BUDGET = 8_192
_PCM_S16_SAMPLE_WIDTH = 2
_MAXIMUM_ENGINE_ITEMS = 8_192
_MAXIMUM_TRANSCRIPT_BYTES = 256 * 1_024
_MAXIMUM_REPETITION_PHRASE_TOKENS = 5
_MINIMUM_REPETITIONS = 3
_MINIMUM_REPEATED_TOKENS = 6
_REPETITION_DOMINANCE_FACTOR = 2


class _WhisperModule(Protocol):
    def transcribe(self, audio: str, **options: object) -> Mapping[str, object]: ...


class _WhisperModelHolder(Protocol):
    model: object | None
    model_path: str | None


class _ParakeetModel(Protocol):
    def transcribe(self, audio: str, **options: object) -> object: ...


class _QwenModel(Protocol):
    def generate(self, audio: str, **options: object) -> object: ...


class _QwenForcedModel(Protocol):
    def generate(
        self,
        audio: str,
        text: str,
        language: str,
        **options: object,
    ) -> object: ...


class _BaseAdapter:
    def __init__(
        self,
        *,
        engine: str,
        registry: ModelRegistry,
        model_key: str | None = None,
        audio_root: Path,
        execution: ExecutionIdentity,
        timeout_seconds: float,
        decode_settings: Mapping[str, object],
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValidationError("Analysis adapter timeout must be positive")
        self.engine = engine
        self.registry = registry
        self.audio_root = audio_root.resolve()
        self.execution = execution
        self.timeout_seconds = timeout_seconds
        self.decode_settings = dict(decode_settings)
        self.model_key = model_key or engine
        self.record = registry.record(self.model_key)
        self._fingerprint = semantic_fingerprint(
            "speech-analysis-adapter-v1",
            {
                "adapter_version": ADAPTER_VERSION,
                "engine": engine,
                "model_record": self.record.fingerprint,
                "decode_settings": self.decode_settings,
            },
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _model_identity(self) -> tuple[Path, ModelArtifactIdentity]:
        path = self.registry.require_path(self.model_key)
        status = self.registry.status(self.model_key)
        if status.directory_fingerprint is None or status.package_version is None:
            raise BackendUnavailableError(
                f"Speech-analysis model {self.engine!r} has no verified identity"
            )
        conversion = ConversionIdentity(
            source=self.record.conversion_source,
            tool=self.record.conversion_tool,
            tool_version=self.record.conversion_tool_version,
            recipe_fingerprint=self.record.conversion_recipe_fingerprint,
            precision_policy=self.record.precision,
            verified=self.record.conversion_verified,
        )
        return path, ModelArtifactIdentity(
            engine=self.engine,
            backend_package=self.record.backend_package,
            backend_version=status.package_version,
            adapter_version=ADAPTER_VERSION,
            worker_protocol_version=ANALYSIS_WORKER_PROTOCOL_VERSION,
            converted_repository=self.record.converted_repository,
            converted_revision=self.record.converted_revision,
            converted_directory_fingerprint=status.directory_fingerprint,
            upstream_repository=self.record.upstream_repository,
            upstream_revision=self.record.upstream_revision,
            conversion=conversion,
            precision=self.record.precision,
            decode_fingerprint=semantic_fingerprint(
                f"{self.engine}-decode-v1", self.decode_settings
            ),
        )


class MlxWhisperRecognizer(_BaseAdapter):
    """Independent MLX Whisper recognizer with no expected-text prompt."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        audio_root: Path,
        execution: ExecutionIdentity,
        calibration_fingerprint: str,
        timeout_seconds: float = 180,
        module_loader: Callable[[], _WhisperModule] | None = None,
    ) -> None:
        super().__init__(
            engine="whisper",
            registry=registry,
            audio_root=audio_root,
            execution=execution,
            timeout_seconds=timeout_seconds,
            decode_settings={
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "word_timestamps": True,
            },
        )
        self.calibration_fingerprint = calibration_fingerprint
        self._module_loader = module_loader or _load_mlx_whisper
        self._module: _WhisperModule | None = None

    async def recognize(
        self,
        audio: Path,
        *,
        language: str,
        span: AudioSpan | None = None,
    ) -> RecognitionResult:
        return await _run_with_timeout(
            lambda: self._recognize_sync(audio, language, span),
            timeout_seconds=self.timeout_seconds,
            engine=self.engine,
        )

    def unload(self) -> None:
        """Release both the adapter and mlx-whisper process-global model cache."""
        self._module = None
        module = sys.modules.get("mlx_whisper.transcribe")
        holder_value = getattr(module, "ModelHolder", None)
        if holder_value is not None:
            holder = cast(_WhisperModelHolder, holder_value)
            holder.model = None
            holder.model_path = None

    def _recognize_sync(
        self, audio: Path, language: str, span: AudioSpan | None
    ) -> RecognitionResult:
        analyzed_span = _validated_audio_span(audio, self.audio_root, span)
        model_path, identity = self._model_identity()
        if self._module is None:
            self._module = self._module_loader()
        raw = self._module.transcribe(
            str(audio.resolve()),
            path_or_hf_repo=str(model_path),
            word_timestamps=True,
            condition_on_previous_text=False,
            temperature=0.0,
            language=_upstream_language(self.record, language, whisper=True),
        )
        transcript = _mapping_text(raw, "text")
        segments = _mapping_sequence(raw, "segments")
        tokens = _whisper_tokens(
            segments,
            span=analyzed_span,
            calibration_fingerprint=self.calibration_fingerprint,
        )
        evidence = WhisperEvidence(
            average_log_probability=_minimum_mapping_float(segments, "avg_logprob"),
            compression_ratio=_maximum_mapping_float(segments, "compression_ratio"),
            no_speech_probability=_maximum_mapping_float(segments, "no_speech_prob"),
            temperature=_maximum_mapping_float(segments, "temperature"),
        )
        return _recognition_result(
            engine=self.engine,
            identity=identity,
            execution=self.execution,
            span=analyzed_span,
            language=language,
            detected_language=_optional_mapping_text(raw, "language"),
            transcript=transcript,
            score_calibration_fingerprint=self.calibration_fingerprint,
            tokens=tokens,
            evidence=evidence,
        )


class ParakeetMlxRecognizer(_BaseAdapter):
    """Independent Parakeet TDT recognizer over canonical Yakbox windows."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        audio_root: Path,
        execution: ExecutionIdentity,
        calibration_fingerprint: str,
        timeout_seconds: float = 180,
        chunk_seconds: int = 120,
        overlap_seconds: int = 15,
        model_loader: Callable[[Path], _ParakeetModel] | None = None,
    ) -> None:
        super().__init__(
            engine="parakeet",
            registry=registry,
            audio_root=audio_root,
            execution=execution,
            timeout_seconds=timeout_seconds,
            decode_settings={
                "decoding": "greedy",
                "runtime_dtype": "bfloat16",
                "chunk_seconds": chunk_seconds,
                "overlap_seconds": overlap_seconds,
            },
        )
        if overlap_seconds < 0 or chunk_seconds <= overlap_seconds:
            raise ValidationError("Parakeet chunk and overlap settings are invalid")
        self.calibration_fingerprint = calibration_fingerprint
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self._model_loader = model_loader or _load_parakeet
        self._model: _ParakeetModel | None = None

    async def recognize(
        self,
        audio: Path,
        *,
        language: str,
        span: AudioSpan | None = None,
    ) -> RecognitionResult:
        return await _run_with_timeout(
            lambda: self._recognize_sync(audio, language, span),
            timeout_seconds=self.timeout_seconds,
            engine=self.engine,
        )

    def unload(self) -> None:
        self._model = None

    def _recognize_sync(
        self, audio: Path, language: str, span: AudioSpan | None
    ) -> RecognitionResult:
        _require_english(language)
        analyzed_span = _validated_audio_span(audio, self.audio_root, span)
        model_path, identity = self._model_identity()
        if self._model is None:
            self._model = self._model_loader(model_path)
        raw = self._model.transcribe(
            str(audio.resolve()),
            chunk_duration=float(self.chunk_seconds),
            overlap_duration=float(self.overlap_seconds),
        )
        transcript = _object_text(raw, "text")
        raw_tokens = _object_sequence(raw, "tokens")
        tokens = _parakeet_tokens(
            raw_tokens,
            span=analyzed_span,
            calibration_fingerprint=self.calibration_fingerprint,
        )
        confidences = tuple(
            value
            for item in _object_sequence(raw, "sentences")
            if (value := _optional_object_float(item, "confidence")) is not None
        )
        evidence = ParakeetEvidence(
            sentence_confidence=min(confidences) if confidences else None,
            decoding="greedy",
            chunk_duration_frames=self.chunk_seconds * analyzed_span.sample_rate,
            overlap_frames=self.overlap_seconds * analyzed_span.sample_rate,
        )
        return _recognition_result(
            engine=self.engine,
            identity=identity,
            execution=self.execution,
            span=analyzed_span,
            language=language,
            detected_language=None,
            transcript=transcript,
            score_calibration_fingerprint=self.calibration_fingerprint,
            tokens=tokens,
            evidence=evidence,
        )


class MlxAudioQwenRecognizer(_BaseAdapter):
    """Qwen3-ASR escalation recognizer loaded only from a verified local path."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        audio_root: Path,
        execution: ExecutionIdentity,
        calibration_fingerprint: str,
        model_key: str = "qwen",
        timeout_seconds: float = 300,
        model_loader: Callable[[Path], _QwenModel] | None = None,
    ) -> None:
        super().__init__(
            engine="qwen",
            registry=registry,
            model_key=model_key,
            audio_root=audio_root,
            execution=execution,
            timeout_seconds=timeout_seconds,
            decode_settings={
                "temperature": 0.0,
                "batch_size": 1,
                "stream": False,
                "maximum_tokens_per_second": _MAXIMUM_TOKENS_PER_SECOND,
                "minimum_token_budget": _MINIMUM_TOKEN_BUDGET,
                "maximum_token_budget": _QWEN_MAXIMUM_TOKEN_BUDGET,
            },
        )
        self.calibration_fingerprint = calibration_fingerprint
        self._model_loader = model_loader or _load_qwen
        self._model: _QwenModel | None = None

    async def recognize(
        self,
        audio: Path,
        *,
        language: str,
        span: AudioSpan | None = None,
    ) -> RecognitionResult:
        return await _run_with_timeout(
            lambda: self._recognize_sync(audio, language, span),
            timeout_seconds=self.timeout_seconds,
            engine=self.engine,
        )

    def unload(self) -> None:
        self._model = None

    def _recognize_sync(
        self, audio: Path, language: str, span: AudioSpan | None
    ) -> RecognitionResult:
        analyzed_span = _validated_audio_span(audio, self.audio_root, span)
        model_path, identity = self._model_identity()
        if self._model is None:
            self._model = self._model_loader(model_path)
        token_budget = _qwen_token_budget(analyzed_span)
        raw = self._model.generate(
            str(audio.resolve()),
            language=_upstream_language(self.record, language),
            temperature=0.0,
            batch_size=1,
            stream=False,
            verbose=False,
            max_tokens=token_budget,
        )
        transcript = _object_text(raw, "text")
        tokens = tuple(
            RecognitionToken(
                token.text,
                None,
                None,
                None,
                ScoreKind.UNAVAILABLE,
                self.calibration_fingerprint,
            )
            for token in normalize_english(transcript).tokens
        )
        generation_tokens = _object_integer(raw, "generation_tokens", default=0)
        evidence = QwenEvidence(
            finish_reason=(
                "length" if generation_tokens >= token_budget else "complete"
            ),
            prompt_tokens=_object_integer(raw, "prompt_tokens", default=0),
            generation_tokens=generation_tokens,
        )
        return _recognition_result(
            engine=self.engine,
            identity=identity,
            execution=self.execution,
            span=analyzed_span,
            language=language,
            detected_language=_optional_object_language(raw, "language"),
            transcript=transcript,
            score_calibration_fingerprint=self.calibration_fingerprint,
            tokens=tokens,
            evidence=evidence,
        )


class MlxAudioQwenForcedAligner(_BaseAdapter):
    """Timing-only Qwen3 forced aligner over consensus-authorized text."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        audio_root: Path,
        execution: ExecutionIdentity,
        model_key: str = "qwen-forced",
        timeout_seconds: float = 300,
        model_loader: Callable[[Path], _QwenForcedModel] | None = None,
    ) -> None:
        super().__init__(
            engine="qwen-forced",
            registry=registry,
            model_key=model_key,
            audio_root=audio_root,
            execution=execution,
            timeout_seconds=timeout_seconds,
            decode_settings={"mode": "argmax", "maximum_window_seconds": 300},
        )
        self._model_loader = model_loader or _load_qwen_forced
        self._model: _QwenForcedModel | None = None

    async def force_align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
        purpose: AlignmentPurpose,
        verified_span: VerifiedTextSpan | None = None,
        span: AudioSpan | None = None,
    ) -> ForcedAlignmentResult:
        if purpose is not AlignmentPurpose.NON_AUTHORITATIVE and verified_span is None:
            raise ValidationError(
                "Authoritative forced alignment requires a verified text span"
            )
        return await _run_with_timeout(
            lambda: self._align_sync(
                audio,
                expected_text=expected_text,
                language=language,
                purpose=purpose,
                verified_span=verified_span,
                span=span,
            ),
            timeout_seconds=self.timeout_seconds,
            engine=self.engine,
        )

    def unload(self) -> None:
        self._model = None

    def _align_sync(
        self,
        audio: Path,
        *,
        expected_text: str,
        language: str,
        purpose: AlignmentPurpose,
        verified_span: VerifiedTextSpan | None,
        span: AudioSpan | None,
    ) -> ForcedAlignmentResult:
        analyzed_span = _validated_audio_span(audio, self.audio_root, span)
        if (analyzed_span.end_frame - analyzed_span.start_frame) > (
            300 * analyzed_span.sample_rate
        ):
            raise ValidationError("Qwen forced-alignment window exceeds five minutes")
        model_path, identity = self._model_identity()
        if self._model is None:
            self._model = self._model_loader(model_path)
        raw = self._model.generate(
            str(audio.resolve()),
            text=expected_text,
            language=_upstream_language(self.record, language),
        )
        items = _object_sequence(raw, "items")
        units = tuple(
            ForcedAlignmentUnit(
                text_fingerprint(_object_text(item, "text")),
                _seconds_to_frame(
                    _object_float(item, "start_time"), analyzed_span.sample_rate
                ),
                _seconds_to_frame(
                    _object_float(item, "end_time"), analyzed_span.sample_rate
                ),
            )
            for item in items
        )
        expected_tokens = normalize_english(expected_text).tokens
        issues: tuple[str, ...] = ()
        if len(units) != len(expected_tokens):
            issues = ("forced_alignment_incomplete",)
        lexical_hash = (
            verified_span.lexical_span_hash
            if verified_span is not None
            else text_fingerprint(
                "\u001f".join(token.text for token in expected_tokens)
            )
        )
        coverage = len(units) / len(expected_tokens) if expected_tokens else 0
        return ForcedAlignmentResult(
            engine=self.engine,
            model=identity,
            execution=self.execution,
            span=analyzed_span,
            purpose=purpose,
            aligner_text_hash=text_fingerprint(expected_text),
            expected_lexical_span_hash=lexical_hash,
            units=units,
            coverage_ratio=coverage,
            issues=issues,
        )


def _recognition_result(
    *,
    engine: str,
    identity: ModelArtifactIdentity,
    execution: ExecutionIdentity,
    span: AudioSpan,
    language: str,
    detected_language: str | None,
    transcript: str,
    score_calibration_fingerprint: str,
    tokens: tuple[RecognitionToken, ...],
    evidence: WhisperEvidence | ParakeetEvidence | QwenEvidence,
) -> RecognitionResult:
    _require_english(language)
    normalized = "\u001f".join(token.text for token in tokens)
    issues = _recognition_structure_issues(tokens, span)
    return RecognitionResult(
        engine=engine,
        model=identity,
        execution=execution,
        span=span,
        requested_language=language,
        detected_language=detected_language,
        normalized_transcript_hash=text_fingerprint(normalized),
        raw_transcript_hash=text_fingerprint(transcript),
        score_calibration_fingerprint=score_calibration_fingerprint,
        tokens=tokens,
        evidence=evidence,
        issues=issues,
    )


def _whisper_tokens(
    segments: Sequence[Mapping[str, object]],
    *,
    span: AudioSpan,
    calibration_fingerprint: str,
) -> tuple[RecognitionToken, ...]:
    output: list[RecognitionToken] = []
    for segment in segments:
        for word in _mapping_sequence(segment, "words", required=False):
            text = _mapping_text(word, "word")
            start = _seconds_to_frame(_mapping_float(word, "start"), span.sample_rate)
            end = _seconds_to_frame(_mapping_float(word, "end"), span.sample_rate)
            start, end = _recognition_token_frames(start, end)
            score = _optional_mapping_float(word, "probability")
            output.extend(
                RecognitionToken(
                    token.text,
                    start,
                    end,
                    score,
                    ScoreKind.PROBABILITY,
                    calibration_fingerprint,
                )
                for token in normalize_english(text).tokens
            )
    return tuple(output)


def _parakeet_tokens(
    raw_tokens: Sequence[object],
    *,
    span: AudioSpan,
    calibration_fingerprint: str,
) -> tuple[RecognitionToken, ...]:
    output: list[RecognitionToken] = []
    for text, start_seconds, end_seconds, confidence in _parakeet_lexical_groups(
        raw_tokens
    ):
        start = _seconds_to_frame(start_seconds, span.sample_rate)
        end = _seconds_to_frame(end_seconds, span.sample_rate)
        start, end = _recognition_token_frames(start, end)
        output.extend(
            RecognitionToken(
                token.text,
                start,
                end,
                confidence,
                ScoreKind.TDT_CONFIDENCE,
                calibration_fingerprint,
            )
            for token in normalize_english(text).tokens
        )
    return tuple(output)


def _parakeet_lexical_groups(
    raw_tokens: Sequence[object],
) -> tuple[tuple[str, float, float, float | None], ...]:
    groups: list[tuple[str, float, float, float | None]] = []
    pieces: list[str] = []
    start = 0.0
    end = 0.0
    confidences: list[float] = []
    confidence_complete = True

    def flush() -> None:
        nonlocal pieces, start, end, confidences, confidence_complete
        if pieces:
            groups.append(
                (
                    "".join(pieces),
                    start,
                    end,
                    min(confidences) if confidence_complete and confidences else None,
                )
            )
        pieces = []
        start = 0.0
        end = 0.0
        confidences = []
        confidence_complete = True

    for raw in raw_tokens:
        text = _object_text(raw, "text")
        raw_start = _object_float(raw, "start")
        raw_end = _object_float(raw, "end")
        confidence = _optional_object_float(raw, "confidence")
        if text[:1].isspace() and normalize_english("".join(pieces)).tokens:
            flush()
        if not pieces:
            start = raw_start
        pieces.append(text)
        end = raw_end
        if confidence is None:
            confidence_complete = False
        else:
            confidences.append(confidence)
    flush()
    return tuple(groups)


def _recognition_structure_issues(
    tokens: tuple[RecognitionToken, ...], span: AudioSpan
) -> tuple[str, ...]:
    duration_seconds = (span.end_frame - span.start_frame) / span.sample_rate
    maximum_tokens = max(
        _MINIMUM_TOKEN_BUDGET,
        math.ceil(duration_seconds * _MAXIMUM_TOKENS_PER_SECOND),
    )
    if len(tokens) > maximum_tokens:
        return ("invalid_engine_result",)
    words = tuple(token.text for token in tokens)
    if _has_repetition_loop(words):
        return ("invalid_engine_result",)
    previous_end = 0
    for token in tokens:
        if token.start_frame is None:
            continue
        if (
            token.start_frame < previous_end
            or token.end_frame is None
            or token.end_frame > span.end_frame - span.start_frame
        ):
            return ("invalid_engine_result",)
        previous_end = token.end_frame
    return ()


def _has_repetition_loop(words: tuple[str, ...]) -> bool:
    maximum_width = _MAXIMUM_REPETITION_PHRASE_TOKENS + 1
    for width in range(1, min(maximum_width, len(words) // _MINIMUM_REPETITIONS + 1)):
        for start in range(len(words) - width * _MINIMUM_REPETITIONS + 1):
            phrase = words[start : start + width]
            repetitions = 1
            while (
                start + (repetitions + 1) * width <= len(words)
                and words[
                    start + repetitions * width : start + (repetitions + 1) * width
                ]
                == phrase
            ):
                repetitions += 1
            repeated_tokens = repetitions * width
            if (
                repetitions >= _MINIMUM_REPETITIONS
                and repeated_tokens >= _MINIMUM_REPEATED_TOKENS
                and repeated_tokens * _REPETITION_DOMINANCE_FACTOR >= len(words)
            ):
                return True
    return False


def _qwen_token_budget(span: AudioSpan) -> int:
    duration_seconds = (span.end_frame - span.start_frame) / span.sample_rate
    return min(
        _QWEN_MAXIMUM_TOKEN_BUDGET,
        max(
            _MINIMUM_TOKEN_BUDGET,
            math.ceil(duration_seconds * _MAXIMUM_TOKENS_PER_SECOND),
        ),
    )


async def _run_with_timeout[ResultT](
    operation: Callable[[], ResultT],
    *,
    timeout_seconds: float,
    engine: str,
) -> ResultT:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(operation),
            timeout=timeout_seconds,
        )
    except TimeoutError as error:
        raise SpeechAnalysisError(
            f"Speech-analysis engine {engine!r} exceeded its timeout"
        ) from error


def _validated_audio_span(
    audio: Path, audio_root: Path, span: AudioSpan | None
) -> AudioSpan:
    candidate = audio if audio.is_absolute() else audio_root / audio
    if candidate.is_symlink():
        raise ValidationError("Analysis adapter rejects symlink audio inputs")
    path = safe_child(audio_root, candidate)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValidationError("Analysis adapter requires a regular non-empty file")
    if path.suffix.casefold() != ".wav":
        raise ValidationError("Analysis adapters accept canonical WAV files only")
    try:
        with wave.open(str(path), "rb") as reader:
            if (
                reader.getnchannels() != 1
                or reader.getsampwidth() != _PCM_S16_SAMPLE_WIDTH
            ):
                raise ValidationError("Analysis adapter WAV must be mono 16-bit PCM")
            sample_rate = reader.getframerate()
            frames = reader.getnframes()
    except (OSError, EOFError, wave.Error) as error:
        raise ValidationError("Analysis adapter WAV is invalid") from error
    digest = sha256_file(path)
    full = AudioSpan(digest, 0, frames, sample_rate)
    if span is not None and span != full:
        raise ValidationError(
            "Adapters require a materialized window matching the complete WAV"
        )
    return full


def _load_mlx_whisper() -> _WhisperModule:
    try:
        return cast(_WhisperModule, importlib.import_module("mlx_whisper"))
    except ImportError as error:
        raise BackendUnavailableError(
            "MLX Whisper is unavailable; install the alignment runtime"
        ) from error


def _load_parakeet(path: Path) -> _ParakeetModel:
    try:
        module = importlib.import_module("parakeet_mlx")
        loader = cast(Callable[[str], _ParakeetModel], module.from_pretrained)
        return loader(str(path))
    except ImportError as error:
        raise BackendUnavailableError(
            "Parakeet MLX is unavailable; install the analysis-parakeet runtime"
        ) from error


def _load_qwen(path: Path) -> _QwenModel:
    return cast(_QwenModel, _mlx_audio_loader()(path))


def _load_qwen_forced(path: Path) -> _QwenForcedModel:
    return cast(_QwenForcedModel, _mlx_audio_loader()(path))


def _mlx_audio_loader() -> Callable[[Path], object]:
    try:
        module = importlib.import_module("mlx_audio.stt")
        return cast(Callable[[Path], object], module.load)
    except ImportError as error:
        raise BackendUnavailableError(
            "MLX Audio STT is unavailable; install the analysis-qwen runtime"
        ) from error


def _upstream_language(
    _record: ModelRecord, language: str, *, whisper: bool = False
) -> str:
    _require_english(language)
    return "en" if whisper else "English"


def _require_english(language: str) -> None:
    if language != "en":
        raise ValidationError("The initial speech-analysis runtime supports en only")


def _seconds_to_frame(seconds: float, sample_rate: int) -> int:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValidationError("Engine returned invalid timestamp data")
    return round(seconds * sample_rate)


def _recognition_token_frames(
    start_frame: int,
    end_frame: int,
) -> tuple[int | None, int | None]:
    """Preserve lexical evidence when an ASR quantizes one word to zero frames."""
    if end_frame < start_frame:
        raise ValidationError("Engine returned reversed timestamp data")
    if end_frame == start_frame:
        return None, None
    return start_frame, end_frame


def _mapping_sequence(
    raw: Mapping[str, object], key: str, *, required: bool = True
) -> tuple[Mapping[str, object], ...]:
    value = raw.get(key)
    if value is None and not required:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValidationError(f"Engine result {key!r} must be an array")
    if len(value) > _MAXIMUM_ENGINE_ITEMS:
        raise ValidationError(f"Engine result {key!r} exceeds the item limit")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValidationError(f"Engine result {key!r} entries must be objects")
    return tuple(cast(Mapping[str, object], item) for item in value)


def _object_sequence(raw: object, key: str) -> tuple[object, ...]:
    value = getattr(raw, key, None)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValidationError(f"Engine result {key!r} must be an array")
    if len(value) > _MAXIMUM_ENGINE_ITEMS:
        raise ValidationError(f"Engine result {key!r} exceeds the item limit")
    return tuple(value)


def _mapping_text(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValidationError(f"Engine result {key!r} must be text")
    _validate_unicode(value)
    return value


def _optional_mapping_text(raw: Mapping[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"Engine result {key!r} must be text")
    _validate_unicode(value)
    return value


def _object_text(raw: object, key: str) -> str:
    value = getattr(raw, key, None)
    if not isinstance(value, str):
        raise ValidationError(f"Engine result {key!r} must be text")
    _validate_unicode(value)
    return value


def _optional_object_text(raw: object, key: str) -> str | None:
    value = getattr(raw, key, None)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"Engine result {key!r} must be text")
    _validate_unicode(value)
    return value


def _optional_object_language(raw: object, key: str) -> str | None:
    value = getattr(raw, key, None)
    if value is None or isinstance(value, str):
        result = value
    elif isinstance(value, Sequence) and len(value) == 1 and isinstance(value[0], str):
        result = value[0]
    else:
        raise ValidationError(
            f"Engine result {key!r} must be text or a single-item text array"
        )
    if result is not None:
        _validate_unicode(result)
    return result


def _mapping_float(raw: Mapping[str, object], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"Engine result {key!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"Engine result {key!r} must be finite")
    return result


def _optional_mapping_float(raw: Mapping[str, object], key: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    return _mapping_float(raw, key)


def _object_float(raw: object, key: str) -> float:
    value = getattr(raw, key, None)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"Engine result {key!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"Engine result {key!r} must be finite")
    return result


def _optional_object_float(raw: object, key: str) -> float | None:
    value = getattr(raw, key, None)
    return None if value is None else _object_float(raw, key)


def _minimum_mapping_float(
    values: Sequence[Mapping[str, object]], key: str
) -> float | None:
    found = tuple(
        value
        for item in values
        if (value := _optional_mapping_float(item, key)) is not None
    )
    return min(found) if found else None


def _maximum_mapping_float(
    values: Sequence[Mapping[str, object]], key: str
) -> float | None:
    found = tuple(
        value
        for item in values
        if (value := _optional_mapping_float(item, key)) is not None
    )
    return max(found) if found else None


def _object_integer(raw: object, key: str, *, default: int) -> int:
    value = getattr(raw, key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"Engine result {key!r} must be a non-negative integer")
    return value


def _validate_unicode(value: str) -> None:
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValidationError("Engine result contains invalid Unicode") from error
    if len(encoded) > _MAXIMUM_TRANSCRIPT_BYTES:
        raise ValidationError("Engine result text exceeds the output limit")


__all__ = [
    "ADAPTER_VERSION",
    "MlxAudioQwenForcedAligner",
    "MlxAudioQwenRecognizer",
    "MlxWhisperRecognizer",
    "ParakeetMlxRecognizer",
]
