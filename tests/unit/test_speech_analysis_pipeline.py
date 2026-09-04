from __future__ import annotations

import wave
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema import ValidationError as SchemaError

from yakbox.errors import SpeechAnalysisError, ValidationError
from yakbox.schemas import load_schema
from yakbox.speech.analysis_cache import LayeredEvidenceCache
from yakbox.speech.analysis_fingerprints import text_fingerprint
from yakbox.speech.analysis_models import (
    AlignmentPurpose,
    AudioSpan,
    ClipClass,
    ConversionIdentity,
    ExecutionIdentity,
    ForcedAlignmentResult,
    ForcedAlignmentUnit,
    LexicalSpan,
    ModelArtifactIdentity,
    ParakeetEvidence,
    QwenEvidence,
    RecognitionResult,
    RecognitionToken,
    ScoreKind,
    VerificationScope,
    WhisperEvidence,
)
from yakbox.speech.analysis_pipeline import (
    BoundaryObservation,
    SpeechAnalysisEnsemble,
    boundary_agreement,
)
from yakbox.speech.analysis_policy import (
    CalibrationTable,
    CalibrationThreshold,
    EnginePolicy,
    SpeechAnalysisPolicy,
)
from yakbox.speech.analysis_release import (
    ChapterWindowInput,
    DeliveryTechnicalEvidence,
    ExpectedChunkSpan,
    JoinEvidenceStore,
    JoinPcmEvidence,
    ReleaseEvidenceStore,
    TimingMarkerObservation,
    chapter_report,
    delivery_identity,
    invalidation_for_repair,
    join_report,
    mastering_identity,
    qualify_frame_timing_map,
    release_report,
    release_status_message,
    release_verification_path,
    require_release_verified,
    selector_for_span,
    verify_chapter_hierarchical,
)
from yakbox.speech.analysis_repair import (
    AnalyzedArtifactState,
    RepairAnchorInput,
    bracket_sentence_repair,
)
from yakbox.speech.analysis_services import FakeForcedAligner, FakeSpeechRecognizer
from yakbox.speech.normalization import EquivalenceSet

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _write_wav(path: Path, frames: int = 1_000) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(1_000)
        writer.writeframes(b"\0\0" * frames)


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


def _model(engine: str) -> ModelArtifactIdentity:
    return ModelArtifactIdentity(
        engine,
        "test",
        "1",
        1,
        1,
        f"example/{engine}",
        "1" * 40,
        SHA_A,
        f"upstream/{engine}",
        "2" * 40,
        ConversionIdentity("source", "tool", "1", SHA_B, "bf16", True),
        "bf16",
        SHA_C,
    )


def _recognition(
    engine: str,
    tokens: tuple[str, ...],
    span: AudioSpan,
) -> RecognitionResult:
    evidence = (
        WhisperEvidence(-0.1, 1.0, 0.01, 0.0)
        if engine == "whisper"
        else (
            ParakeetEvidence(0.99, "greedy", 1_000, 100)
            if engine == "parakeet"
            else QwenEvidence("stop", 1, len(tokens))
        )
    )
    return RecognitionResult(
        engine,
        _model(engine),
        _execution(),
        span,
        "en",
        "en",
        text_fingerprint("\u001f".join(tokens)),
        text_fingerprint(" ".join(tokens)),
        SHA_C,
        tuple(
            RecognitionToken(
                token,
                100 + index * 100,
                200 + index * 100,
                0.99,
                ScoreKind.PROBABILITY,
                SHA_C,
            )
            for index, token in enumerate(tokens)
        ),
        evidence,
        (),
    )


def _policy() -> SpeechAnalysisPolicy:
    engines = ("whisper", "parakeet", "qwen", "qwen-forced")
    return SpeechAnalysisPolicy(
        1,
        "strict",
        "en",
        ("whisper", "parakeet"),
        "qwen",
        "qwen-forced",
        (
            ClipClass.ONE_WORD,
            ClipClass.SHORT_PHRASE,
            ClipClass.JOIN,
            ClipClass.REPAIRED_REGION,
        ),
        True,
        True,
        True,
        "error",
        "retry_then_reject",
        tuple(
            EnginePolicy(name, "test", f"example/{name}", "1" * 40, 10, "greedy")
            for name in engines
        ),
    )


def _calibration() -> CalibrationTable:
    return CalibrationTable(
        1,
        "en",
        SHA_A,
        True,
        SHA_B,
        (_execution().fingerprint,),
        tuple(
            CalibrationThreshold(engine, clip_class, SHA_C)
            for engine in ("whisper", "parakeet", "qwen")
            for clip_class in ClipClass
        ),
    )


def _ensemble(
    transcripts: Mapping[tuple[str, str], tuple[str, ...]],
    calls: list[str],
    *,
    evidence_cache: LayeredEvidenceCache | None = None,
) -> SpeechAnalysisEnsemble:
    def recognizer(engine: str) -> FakeSpeechRecognizer:
        def result(
            path: Path, _language: str, span: AudioSpan | None
        ) -> RecognitionResult:
            calls.append(f"recognition:{engine}:{path.name}")
            if span is None:
                raise AssertionError("Pipeline must provide the canonical span")
            return _recognition(engine, transcripts[(engine, path.name)], span)

        return FakeSpeechRecognizer(f"{engine}-fingerprint", result)

    def alignment(
        path: Path,
        expected: str,
        _language: str,
        purpose: AlignmentPurpose,
        _verified: object,
        span: AudioSpan | None,
    ) -> ForcedAlignmentResult:
        calls.append(f"forced:{path.name}")
        if span is None:
            raise AssertionError("Pipeline must provide the canonical span")
        tokens = tuple(expected.split())
        return ForcedAlignmentResult(
            "qwen-forced",
            _model("qwen-forced"),
            _execution(),
            span,
            purpose,
            text_fingerprint(expected),
            text_fingerprint("\u001f".join(tokens)),
            tuple(
                ForcedAlignmentUnit(
                    text_fingerprint(token),
                    100 + index * 100,
                    200 + index * 100,
                )
                for index, token in enumerate(tokens)
            ),
            1.0,
            (),
        )

    return SpeechAnalysisEnsemble(
        recognizers={
            name: recognizer(name) for name in ("whisper", "parakeet", "qwen")
        },
        forced_aligner=FakeForcedAligner("qwen-forced-fingerprint", alignment),
        policy=_policy(),
        calibration=_calibration(),
        equivalences=EquivalenceSet(1, ()),
        evidence_cache=evidence_cache,
    )


@pytest.mark.asyncio
async def test_new_ensemble_reuses_every_durable_evidence_layer(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "sentence.wav"
    _write_wav(audio)
    expected = ("wren", "asked")
    transcripts = {
        (engine, audio.name): expected for engine in ("whisper", "parakeet", "qwen")
    }
    calls: list[str] = []
    cache = LayeredEvidenceCache(tmp_path / "cache")

    first = await _ensemble(transcripts, calls, evidence_cache=cache).analyze(
        audio,
        expected_tokens=expected,
        clip_class=ClipClass.SENTENCE,
        scope=VerificationScope.CANDIDATE,
        high_risk=True,
    )
    first_calls = tuple(calls)
    repeated = await _ensemble(transcripts, calls, evidence_cache=cache).analyze(
        audio,
        expected_tokens=expected,
        clip_class=ClipClass.SENTENCE,
        scope=VerificationScope.CANDIDATE,
        high_risk=True,
    )

    assert tuple(calls) == first_calls
    assert repeated.verification.fingerprint == first.verification.fingerprint
    assert repeated.cache.miss_stages == ()
    assert set(repeated.cache.hit_stages) == {
        "recognition:whisper",
        "recognition:parakeet",
        "recognition:qwen",
        "consensus:escalation",
        "consensus",
        "forced_alignment",
        "verification",
    }


@pytest.mark.asyncio
async def test_carrier_crop_requires_consensus_for_carrier_and_final_clip(
    tmp_path: Path,
) -> None:
    carrier = tmp_path / "carrier.wav"
    extracted = tmp_path / "extracted.wav"
    before = tmp_path / "before-anchor.wav"
    after = tmp_path / "after-anchor.wav"
    _write_wav(carrier)
    _write_wav(extracted, 300)
    _write_wav(before, 310)
    _write_wav(after, 320)
    carrier_tokens = ("the", "room", "no", "settled")
    transcripts = {
        **{
            (engine, carrier.name): carrier_tokens
            for engine in ("whisper", "parakeet", "qwen")
        },
        **{
            (engine, extracted.name): ("no",)
            for engine in ("whisper", "parakeet", "qwen")
        },
        **{
            (engine, before.name): ("he", "said")
            for engine in ("whisper", "parakeet", "qwen")
        },
        **{
            (engine, after.name): ("then", "left")
            for engine in ("whisper", "parakeet", "qwen")
        },
    }
    calls: list[str] = []
    ensemble = _ensemble(transcripts, calls)

    evidence = await ensemble.verify_carrier_extraction(
        candidate_id=SHA_A,
        carrier_audio=carrier,
        extracted_audio=extracted,
        carrier_tokens=carrier_tokens,
        target_token_start=2,
        target_token_end=3,
        vad=BoundaryObservation("vad", 295, 405),
        waveform=BoundaryObservation("waveform", 290, 410),
        tolerance_frames=20,
    )

    assert evidence.carrier.verification.accepted
    assert evidence.extracted.verification.accepted
    assert evidence.extracted.forced_alignment is not None
    assert {item.engine for item in evidence.extracted.recognitions} == {
        "whisper",
        "parakeet",
        "qwen",
    }
    assert evidence.terminal_reason == "accepted_all_required_gates"
    bracket = await bracket_sentence_repair(
        ensemble,
        before=RepairAnchorInput(before, ("he", "said"), 0),
        after=RepairAnchorInput(after, ("then", "left"), 1_000),
        generated=evidence,
    )
    assert bracket.original_start_frame == 300
    assert bracket.original_end_frame == 1_100
    first_calls = tuple(calls)

    repeated = await ensemble.verify_carrier_extraction(
        candidate_id=SHA_A,
        carrier_audio=carrier,
        extracted_audio=extracted,
        carrier_tokens=carrier_tokens,
        target_token_start=2,
        target_token_end=3,
        vad=BoundaryObservation("vad", 295, 405),
        waveform=BoundaryObservation("waveform", 290, 410),
        tolerance_frames=20,
    )

    assert tuple(calls) == first_calls
    assert repeated.fingerprint == evidence.fingerprint
    assert repeated.carrier.cache.miss_stages == ()
    assert repeated.extracted.cache.miss_stages == ()
    assert "forced_alignment" in repeated.extracted.cache.hit_stages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expected", "recognized"),
    [
        (("no",), ("naaah", "no")),
        (("alone",), ("alon",)),
        (("no",), ("no", "liora", "added")),
    ],
)
async def test_known_short_audio_defects_reject_without_forced_only_rescue(
    tmp_path: Path,
    expected: tuple[str, ...],
    recognized: tuple[str, ...],
) -> None:
    audio = tmp_path / "defect.wav"
    _write_wav(audio)
    calls: list[str] = []
    ensemble = _ensemble(
        {
            (engine, audio.name): recognized
            for engine in ("whisper", "parakeet", "qwen")
        },
        calls,
    )

    result = await ensemble.analyze(
        audio,
        expected_tokens=expected,
        clip_class=ClipClass.ONE_WORD,
        scope=VerificationScope.CANDIDATE,
        high_risk=True,
        repair=True,
    )

    assert not result.verification.accepted
    assert result.forced_alignment is None
    assert not any(call.startswith("forced:") for call in calls)


def test_boundary_agreement_rejects_disagreement_without_averaging() -> None:
    with pytest.raises(SpeechAnalysisError, match="disagreement"):
        boundary_agreement(
            (
                BoundaryObservation("forced", 100, 200),
                BoundaryObservation("asr", 160, 200),
                BoundaryObservation("vad", 100, 200),
            ),
            tolerance_frames=20,
        )


@pytest.mark.asyncio
async def test_join_uses_pcm_consensus_and_forced_word_edges(tmp_path: Path) -> None:
    audio = tmp_path / "join.wav"
    _write_wav(audio)
    expected = ("wren", "asked")
    calls: list[str] = []
    ensemble = _ensemble(
        {(engine, audio.name): expected for engine in ("whisper", "parakeet", "qwen")},
        calls,
    )
    pcm = JoinPcmEvidence(0.01, 0.2, 40.0, True, SHA_A)
    store = JoinEvidenceStore(ensemble)

    result, hit = await store.verify(
        join_id="chunk-a:after",
        contextual_audio=audio,
        expected_tokens=expected,
        pcm=pcm,
    )

    assert result.accepted
    assert result.analysis.forced_alignment is not None
    assert {item.engine for item in result.analysis.recognitions} == {
        "whisper",
        "parakeet",
        "qwen",
    }
    Draft202012Validator(load_schema("speech-join-verification")).validate(
        join_report(result)
    )
    prior_calls = tuple(calls)
    repeated, repeated_hit = await store.verify(
        join_id="chunk-a:after",
        contextual_audio=audio,
        expected_tokens=expected,
        pcm=pcm,
    )
    assert not hit
    assert repeated_hit
    assert repeated.fingerprint == result.fingerprint
    assert tuple(calls) == prior_calls


@pytest.mark.asyncio
async def test_release_evidence_reuse(  # noqa: PLR0915 - full lifecycle proof
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.wav"
    master = tmp_path / "master.wav"
    decoded = tmp_path / "decoded.wav"
    container = tmp_path / "book.mp3"
    _write_wav(raw)
    _write_wav(master)
    decoded.write_bytes(master.read_bytes())
    container.write_bytes(b"fake-container")
    expected = ("wren", "asked")
    calls: list[str] = []
    ensemble = _ensemble(
        {
            (engine, name): expected
            for engine in ("whisper", "parakeet", "qwen")
            for name in (master.name, decoded.name)
        },
        calls,
    )
    timing = qualify_frame_timing_map(
        source_rate=1_000,
        destination_rate=1_000,
        source_frame_count=1_000,
        destination_frame_count=1_000,
        markers=(
            TimingMarkerObservation(100, 96),
            TimingMarkerObservation(500, 500),
            TimingMarkerObservation(900, 904),
        ),
        maximum_residual_frames=4,
    )
    mastering = mastering_identity(
        raw_audio=raw,
        mastered_audio=master,
        raw_format="pcm_s16le",
        mastered_format="pcm_s16le",
        ffmpeg_fingerprint=SHA_A,
        filter_graph_fingerprint=SHA_B,
        timing_map=timing,
        mapping_validation_fingerprint=SHA_C,
    )
    delivery = delivery_identity(
        mastered_audio=master,
        container=container,
        decoded_audio=decoded,
        stream_index=0,
        stream_codec="mp3",
        encoder_fingerprint=SHA_A,
        decoder_fingerprint=SHA_B,
        decoded_format="pcm_s16le",
        timing_map=timing,
        metadata_fingerprint=SHA_C,
        container_inspection_fingerprint=SHA_A,
    )
    technical = DeliveryTechnicalEvidence(
        delivery.decoded_pcm_digest,
        0,
        1,
        0,
        0,
        True,
        True,
        True,
        True,
        SHA_A,
    )
    chunks = (
        ExpectedChunkSpan(
            "chunk-a",
            0,
            2,
            "source/chapter.md",
            4,
            4,
            "wren",
            "nick",
            None,
            "chunk-a:after",
        ),
    )
    store = ReleaseEvidenceStore(ensemble)

    first = await store.verify(
        release_id="chapter-one",
        mastering=mastering,
        mastered_audio=master,
        expected_tokens=expected,
        chunk_spans=chunks,
        deliveries=((delivery, technical, container, decoded),),
    )
    first_calls = tuple(calls)
    repeated = await store.verify(
        release_id="chapter-one",
        mastering=mastering,
        mastered_audio=master,
        expected_tokens=expected,
        chunk_spans=chunks,
        deliveries=((delivery, technical, container, decoded),),
    )

    assert first.state is AnalyzedArtifactState.RELEASE_VERIFIED
    assert (
        "/release/verified/"
        in release_verification_path(tmp_path, "chapter-one").as_posix()
    )
    assert first.deliveries[0].reused_lexical_evidence
    assert repeated.master_cache_hit
    assert repeated.delivery_cache_hits == (True,)
    assert tuple(calls) == first_calls

    metadata_container = tmp_path / "book-metadata.mp3"
    metadata_container.write_bytes(b"fake-container-with-new-metadata")
    metadata_delivery = delivery_identity(
        mastered_audio=master,
        container=metadata_container,
        decoded_audio=decoded,
        stream_index=0,
        stream_codec="mp3",
        encoder_fingerprint=SHA_A,
        decoder_fingerprint=SHA_B,
        decoded_format="pcm_s16le",
        timing_map=timing,
        metadata_fingerprint=SHA_B,
        container_inspection_fingerprint=SHA_B,
    )
    metadata_technical = DeliveryTechnicalEvidence(
        metadata_delivery.decoded_pcm_digest,
        0,
        1,
        0,
        0,
        True,
        True,
        True,
        True,
        SHA_B,
    )
    metadata_release = await store.verify(
        release_id="chapter-one",
        mastering=mastering,
        mastered_audio=master,
        expected_tokens=expected,
        chunk_spans=chunks,
        deliveries=(
            (metadata_delivery, metadata_technical, metadata_container, decoded),
        ),
    )
    assert metadata_release.delivery_cache_hits == (False,)
    assert metadata_release.deliveries[0].reused_lexical_evidence
    assert tuple(calls) == first_calls

    bad_technical = DeliveryTechnicalEvidence(
        delivery.decoded_pcm_digest,
        0,
        1,
        0,
        0,
        True,
        True,
        False,
        True,
        SHA_A,
    )
    rejected = await store.verify(
        release_id="chapter-one",
        mastering=mastering,
        mastered_audio=master,
        expected_tokens=expected,
        chunk_spans=chunks,
        deliveries=((delivery, bad_technical, container, decoded),),
    )
    assert rejected.state is AnalyzedArtifactState.REPAIR_CANDIDATE
    assert not rejected.accepted
    assert release_status_message(rejected).startswith("Repair candidate")
    with pytest.raises(SpeechAnalysisError, match="not release verified"):
        require_release_verified(rejected)
    report = release_report(repeated)
    assert report["state"] == "release_verified"
    assert report["accepted"] is True
    Draft202012Validator(load_schema("speech-release-verification")).validate(report)
    invalid = deepcopy(report)
    invalid["state"] = "repair_candidate"
    with pytest.raises(SchemaError):
        Draft202012Validator(load_schema("speech-release-verification")).validate(
            invalid
        )
    Draft202012Validator(load_schema("speech-chapter-verification")).validate(
        chapter_report(repeated.master)
    )

    invalidated = invalidation_for_repair(
        raw_chunk_ids=("chunk-a",),
        affected_join_ids=("chunk-a:after",),
        raw_windows=((100, 200),),
        prior_master=mastering,
    )
    assert invalidated.full_master_stale
    assert invalidated.prior_master_digest == mastering.mastered_digest
    assert invalidated.post_master_windows == ((96, 204),)


def test_mismatch_maps_to_stable_repair_selector() -> None:
    chunks = (
        ExpectedChunkSpan(
            "chunk-a",
            0,
            2,
            "source/chapter.md",
            4,
            5,
            "narrator",
            "andy",
            None,
            "join-1",
        ),
        ExpectedChunkSpan(
            "chunk-b",
            2,
            4,
            "source/chapter.md",
            6,
            6,
            "wren",
            "nick",
            "join-1",
            None,
        ),
    )

    selector = selector_for_span(LexicalSpan(2, 3), chunks)

    assert selector.chunk_id == "chunk-b"
    assert selector.source_path == "source/chapter.md"
    assert selector.source_start_line == 6
    assert selector.speaker == "wren"
    assert selector.profile == "nick"
    assert selector.adjacent_join_ids == ("join-1",)

    with pytest.raises(ValidationError, match="unique"):
        boundary_agreement(
            (
                BoundaryObservation("asr", 100, 200),
                BoundaryObservation("asr", 101, 201),
                BoundaryObservation("vad", 102, 202),
            ),
            tolerance_frames=20,
        )


@pytest.mark.asyncio
async def test_hierarchical_chapter_escalates_only_disagreement_window(
    tmp_path: Path,
) -> None:
    first_audio = tmp_path / "window-one.wav"
    second_audio = tmp_path / "window-two.wav"
    _write_wav(first_audio, 400)
    _write_wav(second_audio, 410)
    calls: list[str] = []
    transcripts = {
        ("whisper", first_audio.name): ("he", "said"),
        ("parakeet", first_audio.name): ("he", "said"),
        ("qwen", first_audio.name): ("he", "said"),
        ("whisper", second_audio.name): ("no",),
        ("parakeet", second_audio.name): ("naaah", "no"),
        ("qwen", second_audio.name): ("no",),
    }
    ensemble = _ensemble(transcripts, calls)
    chunks = (
        ExpectedChunkSpan(
            "chunk-a",
            0,
            2,
            "source/chapter.md",
            4,
            4,
            "narrator",
            "andy",
            None,
            "join-1",
        ),
        ExpectedChunkSpan(
            "chunk-b",
            2,
            3,
            "source/chapter.md",
            5,
            5,
            "liora",
            "karen",
            "join-1",
            None,
        ),
    )

    result = await verify_chapter_hierarchical(
        ensemble,
        chapter_id="chapter-one",
        windows=(
            ChapterWindowInput("window-1", first_audio, ("he", "said"), 0, 2),
            ChapterWindowInput("window-2", second_audio, ("no",), 2, 3),
        ),
        chunk_spans=chunks,
    )

    assert not result.accepted
    assert result.defects[0].selector.chunk_id == "chunk-b"
    assert "recognition:qwen:window-one.wav" not in calls
    assert "recognition:qwen:window-two.wav" in calls
