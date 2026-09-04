from __future__ import annotations

import asyncio
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self, cast

import pytest

from yakbox._files import sha256_file
from yakbox.errors import ValidationError
from yakbox.speech.analysis_adapters import (
    MlxAudioQwenForcedAligner,
    MlxAudioQwenRecognizer,
    MlxWhisperRecognizer,
    ParakeetMlxRecognizer,
)
from yakbox.speech.analysis_models import (
    AlignmentPurpose,
    AudioSpan,
    ExecutionIdentity,
    RecognitionResult,
)
from yakbox.speech.analysis_protocol import (
    RecognitionWorkerRequest,
    StatusWorkerRequest,
    UnloadWorkerRequest,
    WorkerFailure,
    WorkerStatus,
    WorkerSuccess,
)
from yakbox.speech.analysis_runtime import (
    BUILT_IN_WORKERS,
    IsolatedAnalysisWorker,
    WorkerBackedForcedAligner,
    WorkerBackedSpeechRecognizer,
    worker_artifact_digest,
)
from yakbox.speech.analysis_runtime_identity import execution_identity_from_digests
from yakbox.speech.canonical_audio import CanonicalAudioPreparer, PreparedCanonicalAudio
from yakbox.speech.model_registry import (
    ModelRegistry,
    default_model_root,
    default_qualification_model_root,
    load_qualification_model_registry,
)
from yakbox.speech.normalization import normalize_english

pytestmark = pytest.mark.live

ROOT = Path(__file__).parents[2]
SOURCE_AUDIO = ROOT / "examples/local-chatterbox/voices/karen-savage.wav"
CALIBRATION_FINGERPRINT = "0" * 64


@dataclass(frozen=True, slots=True)
class QualificationAudio:
    root: Path
    full: Path
    short: Path
    expected_text: str


class _TextResult(Protocol):
    text: str


class _QualificationAudioArray(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, key: slice) -> object: ...


class _StreamingTranscriber(Protocol):
    @property
    def result(self) -> _TextResult: ...

    def add_audio(self, audio: object) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None: ...


class _Encoder(Protocol):
    layers: list[object]


class _Preprocessor(Protocol):
    sample_rate: int


class _ParakeetQualificationModel(Protocol):
    encoder: _Encoder
    preprocessor_config: _Preprocessor

    def transcribe(self, path: str, **options: object) -> _TextResult: ...

    def transcribe_stream(self, **options: object) -> _StreamingTranscriber: ...


class _ParakeetModule(Protocol):
    def from_pretrained(self, path: str) -> _ParakeetQualificationModel: ...


class _ParakeetAudioModule(Protocol):
    def load_audio(self, path: str, sample_rate: int) -> _QualificationAudioArray: ...


@pytest.fixture(scope="module")
def qualification_audio(tmp_path_factory: pytest.TempPathFactory) -> QualificationAudio:
    _require_live_opt_in()
    root = tmp_path_factory.mktemp("speech-analysis-live")
    preparer = CanonicalAudioPreparer(root)
    prepared = preparer.prepare(SOURCE_AUDIO)
    short = _short_window(preparer, prepared)
    expected_text = (
        "was sitting at her window keeping a sharp eye on everything that passed "
        "from brooks and children up and that if she noticed anything odd or out "
        "of place she would never rest until she had ferreted out the whys and "
        "wherefores thereof there are plenty of people in avonlea and out of it "
        "who can attend closely to their neighbors business by dint of neglecting "
        "their own"
    )
    return QualificationAudio(root, prepared.path, short, expected_text)


@pytest.mark.asyncio
async def test_real_adapters_share_lexical_contract_and_unload(
    qualification_audio: QualificationAudio,
) -> None:
    registry = _verified_registry()
    execution = _execution()
    recognizers = (
        MlxWhisperRecognizer(
            registry=registry,
            audio_root=qualification_audio.root,
            execution=execution,
            calibration_fingerprint=CALIBRATION_FINGERPRINT,
        ),
        ParakeetMlxRecognizer(
            registry=registry,
            audio_root=qualification_audio.root,
            execution=execution,
            calibration_fingerprint=CALIBRATION_FINGERPRINT,
        ),
        MlxAudioQwenRecognizer(
            registry=registry,
            audio_root=qualification_audio.root,
            execution=execution,
            calibration_fingerprint=CALIBRATION_FINGERPRINT,
        ),
    )
    results: list[RecognitionResult] = []
    for recognizer in recognizers:
        result = await recognizer.recognize(
            qualification_audio.full,
            language="en",
        )
        results.append(result)
        recognizer.unload()
        _clear_mlx_cache()

    assert all(not result.issues for result in results)
    assert {len(result.tokens) for result in results} == {68}
    expected = tuple(
        token.text
        for token in normalize_english(qualification_audio.expected_text).tokens
    )
    substitutions = {
        result.engine: sum(
            actual.text != wanted
            for actual, wanted in zip(result.tokens, expected, strict=True)
        )
        for result in results
    }
    assert substitutions == {"whisper": 1, "parakeet": 1, "qwen": 2}


@pytest.mark.asyncio
async def test_real_qwen_forced_aligner_is_timing_only(
    qualification_audio: QualificationAudio,
) -> None:
    aligner = MlxAudioQwenForcedAligner(
        registry=_verified_registry(),
        audio_root=qualification_audio.root,
        execution=_execution(),
    )

    result = await aligner.force_align(
        qualification_audio.full,
        qualification_audio.expected_text,
        language="en",
        purpose=AlignmentPurpose.NON_AUTHORITATIVE,
    )
    aligner.unload()
    _clear_mlx_cache()

    assert result.coverage_ratio == 1
    assert len(result.units) == 68
    assert not result.issues


@pytest.mark.asyncio
async def test_worker_backed_adapters_match_direct_normalized_results(
    qualification_audio: QualificationAudio,
) -> None:
    registry = _verified_registry()
    recognizers = (
        MlxWhisperRecognizer(
            registry=registry,
            audio_root=qualification_audio.root,
            execution=_execution_for_engine("whisper"),
            calibration_fingerprint=CALIBRATION_FINGERPRINT,
        ),
        ParakeetMlxRecognizer(
            registry=registry,
            audio_root=qualification_audio.root,
            execution=_execution_for_engine("parakeet"),
            calibration_fingerprint=CALIBRATION_FINGERPRINT,
        ),
        MlxAudioQwenRecognizer(
            registry=registry,
            audio_root=qualification_audio.root,
            execution=_execution_for_engine("qwen"),
            calibration_fingerprint=CALIBRATION_FINGERPRINT,
        ),
    )
    for direct in recognizers:
        direct_result = await direct.recognize(
            qualification_audio.short,
            language="en",
        )
        direct.unload()
        _clear_mlx_cache()
        worker = _worker(direct.engine, qualification_audio.root)
        backed = WorkerBackedSpeechRecognizer(
            engine=direct.engine,
            worker=worker,
            audio_root=qualification_audio.root,
            adapter_fingerprint=direct.fingerprint,
            timeout_seconds=120,
        )
        try:
            worker_result = await backed.recognize(
                qualification_audio.short,
                language="en",
            )
        finally:
            await worker.close()
        assert worker_result == direct_result

    direct_aligner = MlxAudioQwenForcedAligner(
        registry=registry,
        audio_root=qualification_audio.root,
        execution=_execution_for_engine("qwen"),
    )
    direct_alignment = await direct_aligner.force_align(
        qualification_audio.short,
        "was sitting at her window keeping a sharp eye on everything that passed",
        language="en",
        purpose=AlignmentPurpose.NON_AUTHORITATIVE,
    )
    direct_aligner.unload()
    _clear_mlx_cache()
    worker = _worker("qwen", qualification_audio.root)
    backed_aligner = WorkerBackedForcedAligner(
        engine="qwen-forced",
        worker=worker,
        audio_root=qualification_audio.root,
        adapter_fingerprint=direct_aligner.fingerprint,
        timeout_seconds=120,
    )
    try:
        worker_alignment = await backed_aligner.force_align(
            qualification_audio.short,
            "was sitting at her window keeping a sharp eye on everything that passed",
            language="en",
            purpose=AlignmentPurpose.NON_AUTHORITATIVE,
        )
    finally:
        await worker.close()
    assert worker_alignment == direct_alignment


@pytest.mark.asyncio
async def test_qwen_8bit_candidates_match_bf16_smoke_evidence(
    qualification_audio: QualificationAudio,
) -> None:
    if os.environ.get("YAKBOX_RUN_SPEECH_ANALYSIS_CANDIDATES") != "1":
        pytest.skip("set YAKBOX_RUN_SPEECH_ANALYSIS_CANDIDATES=1")
    candidate_registry = ModelRegistry(
        default_qualification_model_root(),
        data=load_qualification_model_registry(),
    )
    if not all(
        candidate_registry.status(engine).verified
        for engine in candidate_registry.engines()
    ):
        pytest.skip("install and verify the pinned Qwen 8-bit candidates")
    execution = _execution()
    bf16_asr = MlxAudioQwenRecognizer(
        registry=_verified_registry(),
        audio_root=qualification_audio.root,
        execution=execution,
        calibration_fingerprint=CALIBRATION_FINGERPRINT,
    )
    candidate_asr = MlxAudioQwenRecognizer(
        registry=candidate_registry,
        model_key="qwen-8bit",
        audio_root=qualification_audio.root,
        execution=execution,
        calibration_fingerprint=CALIBRATION_FINGERPRINT,
    )
    bf16_result = await bf16_asr.recognize(qualification_audio.full, language="en")
    bf16_asr.unload()
    _clear_mlx_cache()
    candidate_result = await candidate_asr.recognize(
        qualification_audio.full,
        language="en",
    )
    candidate_asr.unload()
    _clear_mlx_cache()

    assert not bf16_result.issues
    assert not candidate_result.issues
    assert tuple(token.text for token in candidate_result.tokens) == tuple(
        token.text for token in bf16_result.tokens
    )
    assert candidate_result.model.precision == "8bit-affine-group64"

    expected_text = qualification_audio.expected_text
    bf16_aligner = MlxAudioQwenForcedAligner(
        registry=_verified_registry(),
        audio_root=qualification_audio.root,
        execution=execution,
    )
    candidate_aligner = MlxAudioQwenForcedAligner(
        registry=candidate_registry,
        model_key="qwen-forced-8bit",
        audio_root=qualification_audio.root,
        execution=execution,
    )
    bf16_alignment = await bf16_aligner.force_align(
        qualification_audio.full,
        expected_text,
        language="en",
        purpose=AlignmentPurpose.NON_AUTHORITATIVE,
    )
    bf16_aligner.unload()
    _clear_mlx_cache()
    candidate_alignment = await candidate_aligner.force_align(
        qualification_audio.full,
        expected_text,
        language="en",
        purpose=AlignmentPurpose.NON_AUTHORITATIVE,
    )
    candidate_aligner.unload()
    _clear_mlx_cache()

    assert not bf16_alignment.issues
    assert not candidate_alignment.issues
    assert tuple(unit.text_hash for unit in candidate_alignment.units) == tuple(
        unit.text_hash for unit in bf16_alignment.units
    )
    boundary_deltas = tuple(
        abs(candidate.start_frame - baseline.start_frame)
        for candidate, baseline in zip(
            candidate_alignment.units,
            bf16_alignment.units,
            strict=True,
        )
    ) + tuple(
        abs(candidate.end_frame - baseline.end_frame)
        for candidate, baseline in zip(
            candidate_alignment.units,
            bf16_alignment.units,
            strict=True,
        )
    )
    assert max(boundary_deltas, default=0) <= 20 * 16


def test_clean_qwen_bf16_rebuilds_match_pinned_snapshots() -> None:
    paths = {
        "qwen": os.environ.get("YAKBOX_QWEN_ASR_REBUILD"),
        "qwen-forced": os.environ.get("YAKBOX_QWEN_FORCED_REBUILD"),
    }
    if not all(paths.values()):
        pytest.skip("set YAKBOX_QWEN_ASR_REBUILD and YAKBOX_QWEN_FORCED_REBUILD")
    registry = ModelRegistry(default_model_root())

    for engine, raw_path in paths.items():
        if raw_path is None:
            raise AssertionError("Qwen rebuild path unexpectedly missing")
        assert registry.verify_snapshot_directory(engine, Path(raw_path)) == ()


def test_parakeet_streaming_candidate_matches_offline_smoke_transcript(
    qualification_audio: QualificationAudio,
) -> None:
    if os.environ.get("YAKBOX_RUN_SPEECH_ANALYSIS_CANDIDATES") != "1":
        pytest.skip("set YAKBOX_RUN_SPEECH_ANALYSIS_CANDIDATES=1")
    registry = _verified_registry()
    module = cast(_ParakeetModule, importlib.import_module("parakeet_mlx"))
    audio_module = cast(
        _ParakeetAudioModule,
        importlib.import_module("parakeet_mlx.audio"),
    )
    model = module.from_pretrained(str(registry.require_path("parakeet")))
    offline = model.transcribe(
        str(qualification_audio.full),
        chunk_duration=120.0,
        overlap_duration=15.0,
    )
    audio = audio_module.load_audio(
        str(qualification_audio.full),
        model.preprocessor_config.sample_rate,
    )
    chunk_size = model.preprocessor_config.sample_rate
    with model.transcribe_stream(
        context_size=(256, 256),
        depth=1,
        keep_original_attention=False,
    ) as transcriber:
        for start in range(0, len(audio), chunk_size):
            transcriber.add_audio(audio[start : start + chunk_size])
        streamed = transcriber.result

    offline_tokens = tuple(
        token.text for token in normalize_english(offline.text).tokens
    )
    streaming_tokens = tuple(
        token.text for token in normalize_english(streamed.text).tokens
    )
    assert streaming_tokens == offline_tokens


@pytest.mark.asyncio
async def test_isolated_worker_reports_load_memory_unload_and_restart(
    qualification_audio: QualificationAudio,
) -> None:
    worker = _worker("whisper", qualification_audio.root)
    direct = MlxWhisperRecognizer(
        registry=_verified_registry(),
        audio_root=qualification_audio.root,
        execution=_execution(),
        calibration_fingerprint=CALIBRATION_FINGERPRINT,
    )
    recognizer = WorkerBackedSpeechRecognizer(
        engine="whisper",
        worker=worker,
        audio_root=qualification_audio.root,
        adapter_fingerprint=direct.fingerprint,
        timeout_seconds=60,
    )
    try:
        result = await recognizer.recognize(qualification_audio.full, language="en")
        loaded = await _status(worker, "loaded")
        await worker.request(
            UnloadWorkerRequest("unload", "whisper"),
            timeout_seconds=10,
        )
        unloaded = await _status(worker, "unloaded")
        cancelled = await worker.cancel("cancel-idle-worker")
        restarted = await _status(worker, "restarted")
    finally:
        await worker.close()

    assert len(result.tokens) == 68
    assert loaded.loaded_engines == ("whisper",)
    assert tuple((item.engine, item.count) for item in loaded.model_loads) == (
        ("whisper", 1),
    )
    assert loaded.peak_resident_memory_bytes > 0
    assert (
        loaded.peak_resident_memory_bytes
        <= BUILT_IN_WORKERS["whisper"].maximum_peak_memory_bytes
    )
    assert loaded.metal_peak_memory_bytes is not None
    assert (
        loaded.metal_peak_memory_bytes
        <= BUILT_IN_WORKERS["whisper"].maximum_peak_memory_bytes
    )
    assert unloaded.metal_active_memory_bytes is not None
    assert unloaded.metal_active_memory_bytes < 16 * 1024 * 1024
    assert cancelled.code.value == "cancelled"
    assert restarted.pid != loaded.pid
    assert worker.generation == 2


@pytest.mark.asyncio
async def test_two_family_memory_pressure_preserves_isolation_and_recovery(
    qualification_audio: QualificationAudio,
) -> None:
    if os.environ.get("YAKBOX_RUN_SPEECH_ANALYSIS_MEMORY_PRESSURE") != "1":
        pytest.skip("set YAKBOX_RUN_SPEECH_ANALYSIS_MEMORY_PRESSURE=1")
    registry = _verified_registry()
    execution = _execution()
    workers = {
        engine: _worker(engine, qualification_audio.root)
        for engine in ("parakeet", "qwen")
    }
    direct = {
        "parakeet": ParakeetMlxRecognizer(
            registry=registry,
            audio_root=qualification_audio.root,
            execution=execution,
            calibration_fingerprint=CALIBRATION_FINGERPRINT,
        ),
        "qwen": MlxAudioQwenRecognizer(
            registry=registry,
            audio_root=qualification_audio.root,
            execution=execution,
            calibration_fingerprint=CALIBRATION_FINGERPRINT,
        ),
    }
    recognizers = {
        engine: WorkerBackedSpeechRecognizer(
            engine=engine,
            worker=workers[engine],
            audio_root=qualification_audio.root,
            adapter_fingerprint=direct[engine].fingerprint,
            timeout_seconds=120,
        )
        for engine in workers
    }
    try:
        qwen_result = await recognizers["qwen"].recognize(
            qualification_audio.full,
            language="en",
        )
        parakeet_result = await recognizers["parakeet"].recognize(
            qualification_audio.full,
            language="en",
        )
        loaded = {
            engine: await _status(worker, f"pressure-loaded-{engine}")
            for engine, worker in workers.items()
        }
        parakeet_pid = loaded["parakeet"].pid
        assert loaded["parakeet"].loaded_engines == ("parakeet",)
        assert loaded["qwen"].loaded_engines == ("qwen",)
        _assert_worker_memory_within_ceiling("parakeet", loaded["parakeet"])
        _assert_worker_memory_within_ceiling("qwen", loaded["qwen"])

        timeout_response = await workers["qwen"].request(
            _recognition_request(
                "pressure-timeout-qwen",
                "qwen",
                qualification_audio.full,
                qualification_audio.root,
                direct["qwen"].fingerprint,
            ),
            timeout_seconds=0.001,
            restart_once=False,
        )
        assert isinstance(timeout_response, WorkerFailure)
        assert timeout_response.code.value == "timeout"

        cancellation_id = "pressure-cancel-qwen"
        pending = asyncio.create_task(
            workers["qwen"].request(
                _recognition_request(
                    cancellation_id,
                    "qwen",
                    qualification_audio.full,
                    qualification_audio.root,
                    direct["qwen"].fingerprint,
                ),
                timeout_seconds=120,
            )
        )
        await asyncio.sleep(0.05)
        cancellation = await workers["qwen"].cancel(cancellation_id)
        cancelled_response = await pending
        qwen_restarted = await _status(workers["qwen"], "pressure-restarted-qwen")

        parakeet_again = await recognizers["parakeet"].recognize(
            qualification_audio.full,
            language="en",
        )
        parakeet_still_loaded = await _status(
            workers["parakeet"],
            "pressure-stable-parakeet",
        )
        await workers["parakeet"].request(
            UnloadWorkerRequest("pressure-unload-parakeet", "parakeet"),
            timeout_seconds=10,
        )
        parakeet_unloaded = await _status(
            workers["parakeet"],
            "pressure-unloaded-parakeet",
        )
    finally:
        await asyncio.gather(*(worker.close() for worker in workers.values()))

    assert parakeet_result.tokens and qwen_result.tokens
    assert parakeet_again.fingerprint == parakeet_result.fingerprint
    assert parakeet_still_loaded.pid == parakeet_pid
    assert cancellation.code.value == "cancelled"
    assert isinstance(cancelled_response, WorkerFailure)
    assert cancelled_response.code.value == "cancelled"
    assert qwen_restarted.pid != loaded["qwen"].pid
    assert qwen_restarted.loaded_engines == ()
    assert parakeet_unloaded.metal_active_memory_bytes is not None
    assert parakeet_unloaded.metal_active_memory_bytes < 16 * 1024 * 1024


@pytest.mark.asyncio
async def test_frozen_dependency_family_environments_are_distinct_and_runnable(
    qualification_audio: QualificationAudio,
) -> None:
    executables = {
        "whisper": os.environ.get("YAKBOX_WHISPER_WORKER_PYTHON"),
        "parakeet": os.environ.get("YAKBOX_PARAKEET_WORKER_PYTHON"),
        "qwen": os.environ.get("YAKBOX_QWEN_WORKER_PYTHON"),
    }
    if not all(executables.values()):
        pytest.skip("set all three YAKBOX_*_WORKER_PYTHON variables")
    required_package = {
        "whisper": "mlx-whisper",
        "parakeet": "parakeet-mlx",
        "qwen": "mlx-audio",
    }
    prohibited = set(required_package.values())
    environment_fingerprints: set[str] = set()
    for engine, executable_value in executables.items():
        if executable_value is None:
            raise AssertionError("worker Python path unexpectedly missing")
        executable = Path(executable_value)
        packages = _installed_package_versions(executable)
        assert (
            packages[required_package[engine]]
            == BUILT_IN_WORKERS[engine].packages[0].version
        )
        assert not (prohibited - {required_package[engine]}) & packages.keys()
        worker = _worker(
            engine,
            qualification_audio.root,
            python_executable=executable,
        )
        direct_type = {
            "whisper": MlxWhisperRecognizer,
            "parakeet": ParakeetMlxRecognizer,
            "qwen": MlxAudioQwenRecognizer,
        }[engine]
        adapter = direct_type(
            registry=_verified_registry(),
            audio_root=qualification_audio.root,
            execution=_execution(),
            calibration_fingerprint=CALIBRATION_FINGERPRINT,
        )
        recognizer = WorkerBackedSpeechRecognizer(
            engine=engine,
            worker=worker,
            audio_root=qualification_audio.root,
            adapter_fingerprint=adapter.fingerprint,
            timeout_seconds=120,
        )
        try:
            result = await recognizer.recognize(
                qualification_audio.short,
                language="en",
            )
            status = await _status(worker, f"isolated-{engine}")
        finally:
            await worker.close()
        assert result.tokens
        assert status.family == engine
        assert status.python_version.startswith("3.14.")
        environment_fingerprints.add(status.environment_fingerprint)

    assert len(environment_fingerprints) == len(executables)
    combined = _worker(
        "whisper",
        qualification_audio.root,
        use_configured_runtime=False,
    )
    try:
        combined_status = await _status(combined, "combined-environment")
    finally:
        await combined.close()
    assert combined_status.environment_fingerprint not in environment_fingerprints


@pytest.mark.asyncio
async def test_long_lived_workers_endure_stable_repeated_inference(
    qualification_audio: QualificationAudio,
) -> None:
    if os.environ.get("YAKBOX_RUN_SPEECH_ANALYSIS_ENDURANCE") != "1":
        pytest.skip("set YAKBOX_RUN_SPEECH_ANALYSIS_ENDURANCE=1")
    calls = int(os.environ.get("YAKBOX_SPEECH_ANALYSIS_ENDURANCE_CALLS", "150"))
    if calls < 150:
        raise ValidationError("Speech-analysis endurance requires at least 150 calls")

    for engine in ("whisper", "parakeet", "qwen"):
        await _exercise_worker(engine, qualification_audio, calls)


async def _exercise_worker(
    engine: str,
    audio: QualificationAudio,
    calls: int,
) -> None:
    worker = _worker(engine, audio.root)
    direct_type = {
        "whisper": MlxWhisperRecognizer,
        "parakeet": ParakeetMlxRecognizer,
        "qwen": MlxAudioQwenRecognizer,
    }[engine]
    direct = direct_type(
        registry=_verified_registry(),
        audio_root=audio.root,
        execution=_execution(),
        calibration_fingerprint=CALIBRATION_FINGERPRINT,
    )
    recognizer = WorkerBackedSpeechRecognizer(
        engine=engine,
        worker=worker,
        audio_root=audio.root,
        adapter_fingerprint=direct.fingerprint,
        timeout_seconds=120,
    )
    timings: list[float] = []
    fingerprints: dict[Path, str] = {}
    active_samples: list[int] = []
    try:
        for index in range(calls):
            path = audio.full if index % 10 == 0 else audio.short
            started = time.perf_counter()
            result = await recognizer.recognize(path, language="en")
            timings.append(time.perf_counter() - started)
            previous = fingerprints.setdefault(path, result.fingerprint)
            assert result.fingerprint == previous
            if index % 10 == 0:
                status = await _status(worker, f"sample-{index}")
                if status.metal_active_memory_bytes is not None:
                    active_samples.append(status.metal_active_memory_bytes)
            if index and index % 50 == 0:
                await worker.request(
                    UnloadWorkerRequest(f"unload-{index}", engine),
                    timeout_seconds=10,
                )
            if index % 37 == 0:
                invalid = await worker.request(
                    RecognitionWorkerRequest(
                        request_id=f"invalid-{index}",
                        engine=engine,
                        engine_fingerprint=direct.fingerprint,
                        relative_audio_path=path.relative_to(audio.root).as_posix(),
                        audio_digest="f" * 64,
                        language="en",
                        span=_wav_span(path, audio_digest="f" * 64),
                    ),
                    timeout_seconds=10,
                )
                assert isinstance(invalid, WorkerFailure)
        status = await _status(worker, "endurance-complete")
        cancel_id = f"cancel-active-{engine}"
        cancel_request = RecognitionWorkerRequest(
            request_id=cancel_id,
            engine=engine,
            engine_fingerprint=direct.fingerprint,
            relative_audio_path=audio.full.relative_to(audio.root).as_posix(),
            audio_digest=sha256_file(audio.full),
            language="en",
            span=_wav_span(audio.full),
        )
        pending = asyncio.create_task(
            worker.request(cancel_request, timeout_seconds=120)
        )
        await asyncio.sleep(0.05)
        cancellation = await worker.cancel(cancel_id)
        cancelled_response = await pending
        restarted = await _status(worker, f"restarted-{engine}")
    finally:
        await worker.close()

    width = min(20, len(timings) // 2)
    assert statistics.median(timings[-width:]) <= statistics.median(timings[:width]) * 3
    if active_samples:
        assert max(active_samples) - min(active_samples) < 256 * 1024 * 1024
    expected_loads = 1 + (calls - 1) // 50
    loads = {item.engine: item.count for item in status.model_loads}
    assert loads[engine] == expected_loads
    _assert_worker_memory_within_ceiling(engine, status)
    assert cancellation.code.value == "cancelled"
    assert isinstance(cancelled_response, WorkerFailure)
    assert cancelled_response.code.value == "cancelled"
    assert restarted.pid != status.pid


def _assert_worker_memory_within_ceiling(
    engine: str,
    status: WorkerStatus,
) -> None:
    ceiling = BUILT_IN_WORKERS[engine].maximum_peak_memory_bytes
    assert status.peak_resident_memory_bytes <= ceiling
    if status.metal_peak_memory_bytes is not None:
        assert status.metal_peak_memory_bytes <= ceiling


def _recognition_request(
    request_id: str,
    engine: str,
    audio: Path,
    audio_root: Path,
    engine_fingerprint: str,
) -> RecognitionWorkerRequest:
    span = _wav_span(audio)
    return RecognitionWorkerRequest(
        request_id=request_id,
        engine=engine,
        engine_fingerprint=engine_fingerprint,
        relative_audio_path=audio.relative_to(audio_root).as_posix(),
        audio_digest=span.audio_digest,
        language="en",
        span=span,
    )


def _wav_span(audio: Path, *, audio_digest: str | None = None) -> AudioSpan:
    with wave.open(str(audio), "rb") as reader:
        return AudioSpan(
            audio_digest or sha256_file(audio),
            0,
            reader.getnframes(),
            reader.getframerate(),
        )


def _short_window(
    preparer: CanonicalAudioPreparer,
    prepared: PreparedCanonicalAudio,
) -> Path:
    span = AudioSpan(
        prepared.identity.canonical_digest,
        0,
        5 * prepared.identity.frame_map.analysis_rate,
        prepared.identity.frame_map.analysis_rate,
    )
    return preparer.materialize_window(prepared, span).path


def _verified_registry() -> ModelRegistry:
    registry = ModelRegistry(default_model_root())
    statuses = tuple(registry.status(engine) for engine in registry.engines())
    if not all(status.verified for status in statuses):
        pytest.skip("install and verify every registered speech-analysis model")
    return registry


def _execution():  # noqa: ANN202 - inferred concrete domain type is stable
    return execution_identity_from_digests(
        worker_artifact_digest=worker_artifact_digest(),
        lock_digest=sha256_file(ROOT / "uv.lock"),
    )


def _execution_for_engine(engine: str) -> ExecutionIdentity:
    executable = _configured_worker_python(engine)
    lock = executable.parents[2] / "uv.lock" if executable else ROOT / "uv.lock"
    return execution_identity_from_digests(
        worker_artifact_digest=worker_artifact_digest(),
        lock_digest=sha256_file(lock),
    )


def _configured_worker_python(engine: str) -> Path | None:
    family = "qwen" if engine in {"qwen", "qwen-forced"} else engine
    variable = {
        "whisper": "YAKBOX_WHISPER_WORKER_PYTHON",
        "parakeet": "YAKBOX_PARAKEET_WORKER_PYTHON",
        "qwen": "YAKBOX_QWEN_WORKER_PYTHON",
    }[family]
    value = os.environ.get(variable)
    return Path(value).absolute() if value else None


def _worker(
    engine: str,
    audio_root: Path,
    *,
    python_executable: Path | None = None,
    use_configured_runtime: bool = True,
) -> IsolatedAnalysisWorker:
    family = "qwen" if engine in {"qwen", "qwen-forced"} else engine
    if python_executable is None and use_configured_runtime:
        python_executable = _configured_worker_python(engine)
    worker_artifact_path = None
    lock_path = ROOT / "uv.lock"
    if python_executable is not None:
        worker_artifact_path = python_executable.parents[2] / "analysis-worker.pyz"
        lock_path = worker_artifact_path.parent / "uv.lock"
    return IsolatedAnalysisWorker(
        BUILT_IN_WORKERS[family],
        audio_root=audio_root,
        model_root=default_model_root(),
        calibration_fingerprint=CALIBRATION_FINGERPRINT,
        worker_artifact_digest=worker_artifact_digest(),
        lock_digest=sha256_file(lock_path),
        python_executable=python_executable,
        worker_artifact_path=worker_artifact_path,
    )


def _installed_package_versions(executable: Path) -> dict[str, str]:
    script = """
import json
from importlib.metadata import distributions
print(json.dumps({
    distribution.metadata['Name'].casefold().replace('_', '-'): distribution.version
    for distribution in distributions()
    if distribution.metadata.get('Name')
}, sort_keys=True))
"""
    completed = subprocess.run(  # noqa: S603 - reviewed qualification interpreter
        (str(executable), "-I", "-c", script),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise AssertionError("worker environment package inventory is malformed")
    return value


async def _status(worker: IsolatedAnalysisWorker, suffix: str) -> WorkerStatus:
    response = await worker.request(
        StatusWorkerRequest(f"status-{suffix}"),
        timeout_seconds=10,
    )
    if not isinstance(response, WorkerSuccess) or not isinstance(
        response.result, WorkerStatus
    ):
        raise AssertionError("Worker did not return status")
    return response.result


def _clear_mlx_cache() -> None:
    module = sys.modules.get("mlx.core")
    clear_cache = getattr(module, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()


def _require_live_opt_in() -> None:
    if os.environ.get("YAKBOX_RUN_SPEECH_ANALYSIS_LIVE") != "1":
        pytest.skip("set YAKBOX_RUN_SPEECH_ANALYSIS_LIVE=1")
    if sys.version_info[:2] != (3, 14):
        pytest.skip("speech-analysis qualification requires Python 3.14")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("speech-analysis qualification currently requires Apple Silicon")
