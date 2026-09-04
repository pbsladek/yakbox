from __future__ import annotations

import hashlib
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from yakbox._files import sha256_file
from yakbox.errors import ArtifactError, ValidationError, WorkerProtocolError
from yakbox.speech.analysis_adapters import (
    MlxAudioQwenForcedAligner,
    MlxAudioQwenRecognizer,
    MlxWhisperRecognizer,
    ParakeetMlxRecognizer,
)
from yakbox.speech.analysis_models import (
    AlignmentPurpose,
    AudioSpan,
    ClipClass,
    ExecutionIdentity,
    ForcedAlignmentResult,
    QwenEvidence,
    RecognitionResult,
)
from yakbox.speech.analysis_policy import (
    CalibrationThreshold,
    recognition_quality_issues,
)
from yakbox.speech.analysis_protocol import (
    AnalysisWorkerRequest,
    AnalysisWorkerResponse,
    RecognitionWorkerRequest,
    encode_worker_response,
    parse_worker_response,
)
from yakbox.speech.analysis_runtime import (
    BUILT_IN_WORKERS,
    IsolatedAnalysisWorker,
    WorkerBackedSpeechRecognizer,
)
from yakbox.speech.analysis_services import ForcedAligner, SpeechRecognizer
from yakbox.speech.analysis_worker import AnalysisWorkerApplication, EngineFactory
from yakbox.speech.model_registry import (
    ModelFileRecord,
    ModelRecord,
    ModelRegistry,
    ModelRegistryData,
    PackageRecord,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
REVISION_A = "a" * 40
REVISION_B = "b" * 40
ENGINES = ("whisper", "parakeet", "qwen", "qwen-forced")


def _execution() -> ExecutionIdentity:
    return ExecutionIdentity(
        SHA_A,
        SHA_B,
        "3.14.0",
        "Darwin",
        "26.0",
        "arm64",
        "1.0",
        None,
        "m5-64gb",
        "greedy",
        (),
    )


def _package(name: str) -> PackageRecord:
    return PackageRecord(
        name,
        "1.0",
        "MIT",
        f"https://example.invalid/{name}",
        REVISION_A,
        SHA_A,
        "",
        True,
        True,
        True,
        False,
        True,
        False,
    )


def _record(engine: str, package: str, data: bytes) -> ModelRecord:
    return ModelRecord(
        engine,
        package,
        f"example/{engine}",
        REVISION_A,
        f"upstream/{engine}",
        REVISION_B,
        "MIT",
        "https://example.invalid/upstream",
        "https://example.invalid/converted",
        "https://example.invalid/converter",
        "converter",
        "1",
        SHA_A,
        "bf16",
        True,
        len(data),
        (
            ModelFileRecord(
                "config.json",
                len(data),
                "sha256",
                hashlib.sha256(data).hexdigest(),
            ),
        ),
    )


def _registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModelRegistry:
    package_by_engine = {
        "whisper": "mlx-whisper",
        "parakeet": "parakeet-mlx",
        "qwen": "mlx-audio",
        "qwen-forced": "mlx-audio",
    }
    data = b'{"model_type":"test"}\n'
    packages = tuple(
        _package(name) for name in ("mlx-whisper", "parakeet-mlx", "mlx-audio")
    )
    records = tuple(
        _record(engine, package_by_engine[engine], data) for engine in ENGINES
    )
    registry = ModelRegistry(
        tmp_path / "models",
        data=ModelRegistryData(1, "en", "2026-08-13", packages, records),
    )
    for record in records:
        destination = registry.root / record.engine / record.converted_revision
        destination.mkdir(parents=True)
        (destination / "config.json").write_bytes(data)
    monkeypatch.setattr(
        "yakbox.speech.model_registry._package_version", lambda _name: "1.0"
    )
    return registry


def _write_wav(path: Path) -> AudioSpan:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\0\0" * 16_000)
    return AudioSpan(sha256_file(path), 0, 16_000, 16_000)


class _Whisper:
    def transcribe(self, audio: str, **_options: object) -> dict[str, object]:
        del audio
        return {
            "text": "Wren asked.",
            "language": "en",
            "segments": [
                {
                    "avg_logprob": -0.1,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.01,
                    "temperature": 0.0,
                    "words": [
                        {
                            "word": " Wren",
                            "start": 0.0,
                            "end": 0.4,
                            "probability": 0.99,
                        },
                        {
                            "word": " asked.",
                            "start": 0.4,
                            "end": 0.8,
                            "probability": 0.98,
                        },
                    ],
                }
            ],
        }


class _Parakeet:
    def transcribe(self, audio: str, **_options: object) -> object:
        del audio
        tokens = [
            SimpleNamespace(text=" Wr", start=0.0, end=0.2, confidence=0.99),
            SimpleNamespace(text="en", start=0.2, end=0.4, confidence=0.98),
            SimpleNamespace(text=" asked", start=0.4, end=0.7, confidence=0.97),
            SimpleNamespace(text=".", start=0.7, end=0.8, confidence=0.96),
        ]
        return SimpleNamespace(
            text="Wren asked.",
            tokens=tokens,
            sentences=[SimpleNamespace(confidence=0.98)],
        )


class _Qwen:
    def generate(self, audio: str, **_options: object) -> object:
        del audio
        return SimpleNamespace(
            text="Wren asked.",
            language=["English"],
            prompt_tokens=10,
            generation_tokens=2,
        )


class _TruncatedQwen:
    def __init__(self) -> None:
        self.options: dict[str, object] = {}

    def generate(self, audio: str, **options: object) -> object:
        del audio
        self.options = options
        return SimpleNamespace(
            text="Wren asked.",
            language="English",
            prompt_tokens=10,
            generation_tokens=100,
        )


class _RepeatedQwen:
    def generate(self, audio: str, **_options: object) -> object:
        del audio
        return SimpleNamespace(
            text="go now go now go now",
            language="English",
            prompt_tokens=10,
            generation_tokens=6,
        )


class _LegitimateRepetitionQwen:
    def generate(self, audio: str, **_options: object) -> object:
        del audio
        return SimpleNamespace(
            text=(
                "it was a great great great great great day and everyone "
                "remembered the celebration"
            ),
            language="English",
            prompt_tokens=10,
            generation_tokens=14,
        )


class _WhisperResult:
    def __init__(self, raw: dict[str, object]) -> None:
        self.raw = raw

    def transcribe(self, audio: str, **_options: object) -> dict[str, object]:
        del audio
        return self.raw


class _QwenForced:
    def generate(
        self,
        audio: str,
        text: str,
        language: str,
        **_options: object,
    ) -> object:
        del audio, text, language
        return SimpleNamespace(
            items=[
                SimpleNamespace(text="Wren", start_time=0.0, end_time=0.4),
                SimpleNamespace(text="asked", start_time=0.4, end_time=0.8),
            ]
        )


@pytest.mark.asyncio
async def test_all_recognizer_adapters_share_the_contract_and_load_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    audio = tmp_path / "audio.wav"
    span = _write_wav(audio)
    loads: list[str] = []
    adapters: tuple[SpeechRecognizer, ...] = (
        MlxWhisperRecognizer(
            registry=registry,
            audio_root=tmp_path,
            execution=_execution(),
            calibration_fingerprint=SHA_A,
            module_loader=lambda: loads.append("whisper") or _Whisper(),
        ),
        ParakeetMlxRecognizer(
            registry=registry,
            audio_root=tmp_path,
            execution=_execution(),
            calibration_fingerprint=SHA_A,
            model_loader=lambda _path: loads.append("parakeet") or _Parakeet(),
        ),
        MlxAudioQwenRecognizer(
            registry=registry,
            audio_root=tmp_path,
            execution=_execution(),
            calibration_fingerprint=SHA_A,
            model_loader=lambda _path: loads.append("qwen") or _Qwen(),
        ),
    )

    assert loads == []
    results = tuple(
        [
            await adapter.recognize(audio, language="en", span=span)
            for adapter in adapters
        ]
    )

    assert loads == ["whisper", "parakeet", "qwen"]
    assert all(isinstance(result, RecognitionResult) for result in results)
    assert {result.engine for result in results} == {"whisper", "parakeet", "qwen"}
    assert {tuple(token.text for token in result.tokens) for result in results} == {
        ("wren", "asked")
    }
    assert len({result.model.fingerprint for result in results}) == 3

    for adapter in adapters:
        await adapter.recognize(audio, language="en", span=span)
    assert loads == ["whisper", "parakeet", "qwen"]


@pytest.mark.asyncio
async def test_qwen_forced_adapter_is_timing_only_and_requires_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    audio = tmp_path / "audio.wav"
    span = _write_wav(audio)
    adapter = MlxAudioQwenForcedAligner(
        registry=registry,
        audio_root=tmp_path,
        execution=_execution(),
        model_loader=lambda _path: _QwenForced(),
    )

    result = await adapter.force_align(
        audio,
        "Wren asked.",
        language="en",
        purpose=AlignmentPurpose.NON_AUTHORITATIVE,
        span=span,
    )

    assert isinstance(result, ForcedAlignmentResult)
    assert result.engine == "qwen-forced"
    assert result.coverage_ratio == 1
    assert len(result.units) == 2
    with pytest.raises(ValidationError, match="verified text span"):
        await adapter.force_align(
            audio,
            "Wren asked.",
            language="en",
            purpose=AlignmentPurpose.VERIFIED_TARGET,
            span=span,
        )


@pytest.mark.asyncio
async def test_qwen_adapter_marks_token_budget_exhaustion_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    audio = tmp_path / "audio.wav"
    span = _write_wav(audio)
    model = _TruncatedQwen()
    adapter = MlxAudioQwenRecognizer(
        registry=registry,
        audio_root=tmp_path,
        execution=_execution(),
        calibration_fingerprint=SHA_A,
        model_loader=lambda _path: model,
    )

    result = await adapter.recognize(audio, language="en", span=span)

    assert model.options["max_tokens"] == 100
    assert isinstance(result.evidence, QwenEvidence)
    assert result.evidence.finish_reason == "length"
    threshold = CalibrationThreshold("qwen", ClipClass.SENTENCE, SHA_A)
    assert recognition_quality_issues(result, threshold) == ("engine_decode_invalid",)


@pytest.mark.asyncio
async def test_adapters_reject_invalid_timing_score_unicode_and_excess_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    audio = tmp_path / "audio.wav"
    span = _write_wav(audio)

    invalid_values = (
        (_whisper_raw(start=float("nan")), "must be finite"),
        (_whisper_raw(probability=1.1), "probability is out of range"),
        (_whisper_raw(text="\ud800"), "invalid Unicode"),
        (_whisper_raw(text="word " * 60_000), "output limit"),
    )
    for raw, message in invalid_values:
        adapter = MlxWhisperRecognizer(
            registry=registry,
            audio_root=tmp_path,
            execution=_execution(),
            calibration_fingerprint=SHA_A,
            module_loader=lambda raw=raw: _WhisperResult(raw),
        )
        with pytest.raises(ValidationError, match=message):
            await adapter.recognize(audio, language="en", span=span)


@pytest.mark.asyncio
async def test_whisper_preserves_word_when_timestamp_quantizes_to_zero_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    audio = tmp_path / "audio.wav"
    span = _write_wav(audio)
    adapter = MlxWhisperRecognizer(
        registry=registry,
        audio_root=tmp_path,
        execution=_execution(),
        calibration_fingerprint=SHA_A,
        module_loader=lambda: _WhisperResult(_whisper_raw(start=0.4, end=0.4)),
    )

    result = await adapter.recognize(audio, language="en", span=span)

    assert tuple(token.text for token in result.tokens) == ("wren",)
    assert result.tokens[0].start_frame is None
    assert result.tokens[0].end_frame is None


@pytest.mark.asyncio
async def test_adapters_mark_out_of_span_timing_and_repetition_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    audio = tmp_path / "audio.wav"
    span = _write_wav(audio)
    whisper = MlxWhisperRecognizer(
        registry=registry,
        audio_root=tmp_path,
        execution=_execution(),
        calibration_fingerprint=SHA_A,
        module_loader=lambda: _WhisperResult(_whisper_raw(end=2.0)),
    )
    qwen = MlxAudioQwenRecognizer(
        registry=registry,
        audio_root=tmp_path,
        execution=_execution(),
        calibration_fingerprint=SHA_A,
        model_loader=lambda _path: _RepeatedQwen(),
    )
    legitimate = MlxAudioQwenRecognizer(
        registry=registry,
        audio_root=tmp_path,
        execution=_execution(),
        calibration_fingerprint=SHA_A,
        model_loader=lambda _path: _LegitimateRepetitionQwen(),
    )

    timed = await whisper.recognize(audio, language="en", span=span)
    repeated = await qwen.recognize(audio, language="en", span=span)
    literary = await legitimate.recognize(audio, language="en", span=span)

    assert timed.issues == ("invalid_engine_result",)
    assert repeated.issues == ("invalid_engine_result",)
    assert literary.issues == ()


def _whisper_raw(
    *,
    text: str = "Wren asked.",
    start: float = 0.0,
    end: float = 0.4,
    probability: float = 0.99,
) -> dict[str, object]:
    return {
        "text": text,
        "language": "en",
        "segments": [
            {
                "avg_logprob": -0.1,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.01,
                "temperature": 0.0,
                "words": [
                    {
                        "word": " Wren",
                        "start": start,
                        "end": end,
                        "probability": probability,
                    }
                ],
            }
        ],
    }


class _Factory(EngineFactory):
    def __init__(self, recognizer: SpeechRecognizer) -> None:
        self.recognizer_adapter = recognizer

    def recognizer(self, engine: str) -> SpeechRecognizer:
        del engine
        return self.recognizer_adapter

    def forced_aligner(self, engine: str) -> ForcedAligner:
        del engine
        raise WorkerProtocolError("not configured")


class _LoopbackWorker(IsolatedAnalysisWorker):
    def __init__(self, application: AnalysisWorkerApplication) -> None:
        self.definition = BUILT_IN_WORKERS["whisper"]
        self.application = application
        self.worker_artifact_digest = SHA_A
        self.lock_digest = SHA_B

    async def request(
        self,
        request: AnalysisWorkerRequest,
        *,
        timeout_seconds: float,
        restart_once: bool = True,
    ) -> AnalysisWorkerResponse:
        del timeout_seconds, restart_once
        response = await self.application.handle(request)
        return parse_worker_response(encode_worker_response(response))


@pytest.mark.asyncio
async def test_worker_backed_and_direct_adapters_emit_equivalent_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    audio = tmp_path / "audio.wav"
    span = _write_wav(audio)
    direct = MlxWhisperRecognizer(
        registry=registry,
        audio_root=tmp_path,
        execution=_execution(),
        calibration_fingerprint=SHA_A,
        module_loader=_Whisper,
    )
    application = AnalysisWorkerApplication(
        family="whisper",
        audio_root=tmp_path,
        factory=_Factory(direct),
    )
    worker = _LoopbackWorker(application)
    worker_backed = WorkerBackedSpeechRecognizer(
        engine="whisper",
        worker=worker,
        audio_root=tmp_path,
        adapter_fingerprint=direct.fingerprint,
        timeout_seconds=10,
    )
    original_fingerprint = worker_backed.fingerprint
    worker.worker_artifact_digest = SHA_B
    assert worker_backed.fingerprint != original_fingerprint
    worker.worker_artifact_digest = SHA_A

    direct_result = await direct.recognize(audio, language="en", span=span)
    worker_result = await worker_backed.recognize(audio, language="en", span=span)

    assert worker_result == direct_result


@pytest.mark.asyncio
async def test_adapters_reject_root_escapes_and_non_english(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    inside = tmp_path / "audio.wav"
    span = _write_wav(inside)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-analysis.wav"
    outside_span = _write_wav(outside)
    adapter = MlxWhisperRecognizer(
        registry=registry,
        audio_root=tmp_path,
        execution=_execution(),
        calibration_fingerprint=SHA_A,
        module_loader=_Whisper,
    )

    with pytest.raises(ArtifactError, match="escapes managed root"):
        await adapter.recognize(outside, language="en", span=outside_span)
    with pytest.raises(ValidationError, match="supports en only"):
        await adapter.recognize(inside, language="fr", span=span)


def test_engine_fingerprint_changes_with_model_and_decode_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    first = ParakeetMlxRecognizer(
        registry=registry,
        audio_root=tmp_path,
        execution=_execution(),
        calibration_fingerprint=SHA_A,
        chunk_seconds=120,
        overlap_seconds=15,
        model_loader=lambda _path: _Parakeet(),
    )
    second = ParakeetMlxRecognizer(
        registry=registry,
        audio_root=tmp_path,
        execution=_execution(),
        calibration_fingerprint=SHA_A,
        chunk_seconds=60,
        overlap_seconds=10,
        model_loader=lambda _path: _Parakeet(),
    )

    assert first.fingerprint != second.fingerprint
    request = RecognitionWorkerRequest(
        "recognize-1",
        "parakeet",
        first.fingerprint,
        "audio.wav",
        SHA_A,
        "en",
        AudioSpan(SHA_A, 0, 1, 16_000),
    )
    assert request.engine_fingerprint == first.fingerprint
