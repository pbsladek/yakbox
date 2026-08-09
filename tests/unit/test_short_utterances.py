from __future__ import annotations

import math
import struct
import wave
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import override

import pytest

from yakbox._files import sha256_file
from yakbox.audio.crop import (
    SpeechIslandEvidence,
    SpeechRegion,
    crop_aligned_wav,
    detect_speech_regions,
    inspect_speech_islands,
    wav_duration_seconds,
)
from yakbox.errors import ArtifactError, BuildError
from yakbox.local_alignment import MlxWhisperAligner
from yakbox.speech.alignment import (
    AlignmentResult,
    AlignmentToken,
    DecodePassEvidence,
    lexical_tokens,
    validate_carrier_alignment,
    validate_extracted_alignment,
)
from yakbox.speech.models import AudioFormat, SpeechArtifact, SpeechSynthesisRequest
from yakbox.speech.phonemes import PhonemeAlignmentResult, PhonemeToken
from yakbox.speech.services import FakeSpeechService
from yakbox.speech.short_synthesis import (
    CandidateEvaluation,
    _acoustic_reason_codes,
    _apply_phoneme_gate,
    _rank_candidates,
    _speech_island_gap_ms,
    synthesize_short_utterance,
)
from yakbox.speech.short_utterances import (
    CarrierPosition,
    CarrierRecipe,
    ShortUtterancePolicy,
    ShortUtteranceStrategy,
    adapt_target_punctuation,
    carrier_recipes,
    classify_short_utterance,
)


def _rankable_candidate(
    tmp_path: Path,
    *,
    index: int,
    position: CarrierPosition,
    confidence: float,
    leading_ms: float,
    natural: bool = False,
    regions: tuple[SpeechRegion, ...] | None = None,
) -> CandidateEvaluation:
    audio = tmp_path / f"candidate-{index}.wav"
    resolved_regions = regions or (SpeechRegion(leading_ms / 1_000, 0.95),)
    primary = SpeechRegion(
        resolved_regions[0].start_seconds,
        resolved_regions[-1].end_seconds,
    )
    evidence = SpeechIslandEvidence(
        duration_seconds=1.0,
        regions=resolved_regions,
        islands=(primary,),
        primary_island=primary,
        detached_prefix=(),
        detached_suffix=(),
        leading_silence_ms=primary.start_seconds * 1_000,
        trailing_silence_ms=(1.0 - primary.end_seconds) * 1_000,
    )
    return CandidateEvaluation(
        recipe=CarrierRecipe(index, "No.", "No.", "test", position, natural, index),
        accepted=True,
        reason_codes=(),
        full_audio=audio,
        extracted_audio=audio,
        confidence=confidence,
        crop=None,
        acoustic=evidence,
    )


def test_ranking_uses_confidence_without_a_context_strategy_bias(
    tmp_path: Path,
) -> None:
    policy = ShortUtterancePolicy(prefer_natural_context=False)
    direct = _rankable_candidate(
        tmp_path,
        index=1,
        position=CarrierPosition.DIRECT,
        confidence=0.9,
        leading_ms=35,
    )
    contextual = _rankable_candidate(
        tmp_path,
        index=2,
        position=CarrierPosition.MIDDLE,
        confidence=0.8,
        leading_ms=30,
    )
    higher_confidence_context = _rankable_candidate(
        tmp_path,
        index=3,
        position=CarrierPosition.MIDDLE,
        confidence=0.99,
        leading_ms=80,
    )

    assert _rank_candidates((direct, contextual), policy=policy) is direct
    assert (
        _rank_candidates((direct, higher_confidence_context), policy=policy)
        is higher_confidence_context
    )


def test_ranking_prefers_natural_context_within_confidence_tolerance(
    tmp_path: Path,
) -> None:
    direct = _rankable_candidate(
        tmp_path,
        index=1,
        position=CarrierPosition.DIRECT,
        confidence=0.90,
        leading_ms=30,
    )
    natural = _rankable_candidate(
        tmp_path,
        index=2,
        position=CarrierPosition.MIDDLE,
        confidence=0.88,
        leading_ms=30,
        natural=True,
    )

    assert _rank_candidates((direct, natural), policy=ShortUtterancePolicy()) is natural
    assert (
        _rank_candidates(
            (direct, natural),
            policy=ShortUtterancePolicy(prefer_natural_context=False),
        )
        is direct
    )


def test_ranking_penalizes_internal_pauses_within_confidence_tolerance(
    tmp_path: Path,
) -> None:
    continuous = _rankable_candidate(
        tmp_path,
        index=1,
        position=CarrierPosition.MIDDLE,
        confidence=0.80,
        leading_ms=30,
        regions=(SpeechRegion(0.03, 0.95),),
    )
    fragmented = _rankable_candidate(
        tmp_path,
        index=2,
        position=CarrierPosition.MIDDLE,
        confidence=0.82,
        leading_ms=30,
        regions=(
            SpeechRegion(0.03, 0.20),
            SpeechRegion(0.40, 0.60),
            SpeechRegion(0.80, 0.95),
        ),
    )

    assert (
        _rank_candidates(
            (continuous, fragmented),
            policy=ShortUtterancePolicy(prefer_natural_context=False),
        )
        is continuous
    )


def test_acoustic_gate_rejects_audio_that_ends_on_active_speech() -> None:
    evidence = SpeechIslandEvidence(
        duration_seconds=0.54,
        regions=(SpeechRegion(0.03, 0.54),),
        islands=(SpeechRegion(0.03, 0.54),),
        primary_island=SpeechRegion(0.03, 0.54),
        detached_prefix=(),
        detached_suffix=(),
        leading_silence_ms=30,
        trailing_silence_ms=0,
    )

    assert "insufficient_trailing_silence" in _acoustic_reason_codes(
        evidence, ShortUtterancePolicy()
    )


def _result(
    *words: str,
    confidence: float | None = 0.95,
    regions: tuple[SpeechRegion, ...] = (),
) -> AlignmentResult:
    tokens = tuple(
        AlignmentToken(word, index * 0.25 + 0.1, index * 0.25 + 0.3, confidence)
        for index, word in enumerate(words)
    )
    return AlignmentResult(tokens, regions, "fake", "fake", "f" * 64)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("No.", ("no",)),
        ("You first in?", ("you", "first", "in")),
        ("don't stop", ("don't", "stop")),
        ("state-of-the-art A.I. 609", ("state-of-the-art", "a", "i", "609")),
        ("Élan\u2019s answer", ("élan\u2019s", "answer")),
    ],
)
def test_lexical_tokens_are_unicode_aware(text: str, expected: tuple[str, ...]) -> None:
    assert lexical_tokens(text) == expected


def test_risk_classification_applies_to_any_speaker_text() -> None:
    policy = ShortUtterancePolicy(maximum_words=3)

    assert classify_short_utterance("Wren asked.", policy).risky
    assert classify_short_utterance("You first in?", policy).risky
    assert not classify_short_utterance("Did anyone touch him?", policy).risky
    assert not classify_short_utterance(
        "__YAKBOX_PAUSE_MS=50__", policy, explicit_pause=True
    ).risky


def test_explicit_sentence_boundary_allows_bounded_rhetorical_pause() -> None:
    result = AlignmentResult(
        tokens=(
            AlignmentToken("static", 0.0, 0.2, 0.9),
            AlignmentToken("so", 0.75, 0.9, 0.9),
            AlignmentToken("what", 0.9, 1.1, 0.9),
        ),
        speech_regions=(SpeechRegion(0.0, 1.1),),
        backend="fake",
        model="fake",
        fingerprint="f" * 64,
    )

    punctuated = validate_extracted_alignment(
        result,
        target_text="Static? So what?",
        minimum_confidence=0.5,
        maximum_extra_speech_ms=60,
        minimum_duration_seconds=0.1,
        maximum_duration_seconds=2.0,
        maximum_internal_token_gap_ms=350,
    )
    unpunctuated = validate_extracted_alignment(
        result,
        target_text="Static so what?",
        minimum_confidence=0.5,
        maximum_extra_speech_ms=60,
        minimum_duration_seconds=0.1,
        maximum_duration_seconds=2.0,
        maximum_internal_token_gap_ms=350,
    )

    assert punctuated.accepted
    assert unpunctuated.reason_codes == ("excessive_internal_pause",)
    policy = ShortUtterancePolicy(speech_island_gap_ms=300)
    assert _speech_island_gap_ms("Static? So what?", policy) == 900
    assert _speech_island_gap_ms("You first in?", policy) == 300


def test_carrier_matrix_is_bounded_deterministic_and_adapts_commas() -> None:
    policy = ShortUtterancePolicy(
        strategy=ShortUtteranceStrategy.CONTEXT_EXTRACT,
        candidate_count=5,
        carrier_positions=(CarrierPosition.MIDDLE,),
    )

    first = carrier_recipes(
        "No,",
        policy,
        seed_material="chapter:24",
        previous_context="Liora studied the room.",
        next_context="She moved toward the window.",
    )
    repeated = carrier_recipes(
        "No,",
        policy,
        seed_material="chapter:24",
        previous_context="Liora studied the room.",
        next_context="She moved toward the window.",
    )

    assert first == repeated
    assert len(first) == 5
    assert len({item.seed for item in first}) == 5
    assert first[0].text == "No,"
    assert first[-1].natural
    assert "No." in first[-1].text
    assert adapt_target_punctuation("Third,") == "Third."
    assert all(item.text.count("No") == 1 for item in first)

    stable_dialogue = carrier_recipes(
        "You first in?",
        policy,
        seed_material="0001-chapter-one-room-609:17",
    )
    assert stable_dialogue[2].seed == 5_299_994

    one_sided = carrier_recipes(
        "Anyone touch him?",
        policy,
        seed_material="chapter:25",
        previous_context="You first in?",
    )
    assert one_sided[-1].natural
    assert one_sided[-1].text == (
        "You first in? Anyone touch him? Then the conversation continued."
    )


def test_carrier_matrix_retries_safe_layouts_with_new_seeds() -> None:
    policy = ShortUtterancePolicy(
        strategy=ShortUtteranceStrategy.CONTEXT_EXTRACT,
        candidate_count=7,
        carrier_positions=(CarrierPosition.MIDDLE,),
    )

    recipes = carrier_recipes(
        "Yes, Mara replied.",
        policy,
        seed_material="chapter:146",
        previous_context="Wren asked the question.",
        next_context="Then he waited.",
    )

    assert len(recipes) == 7
    assert recipes[5].text == recipes[0].text
    assert recipes[6].text == recipes[1].text
    assert len({item.seed for item in recipes}) == 7


@pytest.mark.parametrize(
    ("recognized", "target", "reason"),
    [
        (("or", "wren", "asked"), "Wren asked.", "unexpected_prefix"),
        (("liora", "added", "it"), "Liora added.", "unexpected_suffix"),
        (("wren", "asks"), "Wren asked.", "target_substituted"),
        (("no", "no"), "No.", "target_repeated"),
        ((), "No.", "target_missing"),
    ],
)
def test_extracted_alignment_rejects_reported_lexical_failures(
    recognized: tuple[str, ...], target: str, reason: str
) -> None:
    decision = validate_extracted_alignment(
        _result(*recognized),
        target_text=target,
        minimum_confidence=0.5,
        maximum_extra_speech_ms=60,
        minimum_duration_seconds=0.05,
        maximum_duration_seconds=2.0,
    )

    assert not decision.accepted
    assert reason in decision.reason_codes


def test_carrier_alignment_requires_exact_carrier_and_one_target() -> None:
    exact = validate_carrier_alignment(
        _result("the", "exchange", "continued", "wren", "asked", "then", "silence"),
        expected_text="The exchange continued. Wren asked. Then silence.",
        target_text="Wren asked.",
        minimum_confidence=0.8,
    )
    ambiguous = validate_carrier_alignment(
        _result("no", "then", "no"),
        expected_text="No. Then no.",
        target_text="No.",
        minimum_confidence=0.8,
    )

    assert exact.accepted
    assert exact.start_seconds == pytest.approx(0.85)
    assert exact.end_seconds == pytest.approx(1.3)
    assert ambiguous.reason_codes == ("target_ambiguous",)


def test_explicit_alignment_alias_accepts_spelling_without_hiding_extra_words() -> None:
    aliases = {"liora": ("leora",)}
    exact = validate_extracted_alignment(
        _result("leora", "added"),
        target_text="Liora added.",
        minimum_confidence=0.5,
        maximum_extra_speech_ms=60,
        minimum_duration_seconds=0.05,
        maximum_duration_seconds=2.0,
        token_aliases=aliases,
    )
    extra = validate_extracted_alignment(
        _result("leora", "added", "it"),
        target_text="Liora added.",
        minimum_confidence=0.5,
        maximum_extra_speech_ms=60,
        minimum_duration_seconds=0.05,
        maximum_duration_seconds=2.0,
        token_aliases=aliases,
    )

    assert exact.accepted
    assert extra.reason_codes == ("unexpected_suffix",)


@pytest.mark.parametrize("apostrophe", ["'", "\N{RIGHT SINGLE QUOTATION MARK}"])
def test_explicit_alignment_alias_accepts_possessive_inflection(
    apostrophe: str,
) -> None:
    aliases = {"liora": ("leora",)}
    decision = validate_extracted_alignment(
        _result(f"leora{apostrophe}s", "jaw", "tightened"),
        target_text=f"Liora{apostrophe}s jaw tightened.",
        minimum_confidence=0.5,
        maximum_extra_speech_ms=60,
        minimum_duration_seconds=0.05,
        maximum_duration_seconds=2.0,
        token_aliases=aliases,
    )

    assert decision.accepted


def test_alias_equivalent_decode_passes_resolve_spelling_only_consensus() -> None:
    decode_passes = (
        DecodePassEvidence("authority", "", ("or", "rex", "ant"), (), 0.8, False),
        DecodePassEvidence("sampled", "", ("or", "rex", "ant"), (), 0.8, False),
        DecodePassEvidence("prompt", "", ("or", "rex", "an"), (), 0.8, False),
    )
    result = replace(
        _result("or", "rex", "ant", confidence=0.8),
        decode_passes=decode_passes,
        consensus_reason_codes=("decode_consensus_mismatch",),
        prompt_sensitivity="prompt_changed",
    )

    decision = validate_extracted_alignment(
        result,
        target_text="Aw rex an",
        minimum_confidence=0.5,
        maximum_extra_speech_ms=60,
        minimum_duration_seconds=0.05,
        maximum_duration_seconds=3.0,
        token_aliases={"aw": ("or",), "an": ("ant",)},
    )

    assert decision.accepted


def test_aliases_do_not_hide_decode_pass_with_extra_word() -> None:
    decode_passes = (
        DecodePassEvidence("authority", "", ("or", "rex", "ant"), (), 0.8, False),
        DecodePassEvidence("sampled", "", ("or", "rex", "ant", "now"), (), 0.8, False),
        DecodePassEvidence("prompt", "", ("or", "rex", "an"), (), 0.8, False),
    )
    result = replace(
        _result("or", "rex", "ant", confidence=0.8),
        decode_passes=decode_passes,
        consensus_reason_codes=("decode_consensus_mismatch",),
        prompt_sensitivity="prompt_changed",
    )

    decision = validate_extracted_alignment(
        result,
        target_text="Aw rex an",
        minimum_confidence=0.5,
        maximum_extra_speech_ms=60,
        minimum_duration_seconds=0.05,
        maximum_duration_seconds=3.0,
        token_aliases={"aw": ("or",), "an": ("ant",)},
    )

    assert not decision.accepted
    assert "decode_consensus_mismatch" in decision.reason_codes
    assert "prompt_sensitive_transcript" in decision.reason_codes


def test_extracted_alignment_rejects_nonlexical_prefix_energy() -> None:
    decision = validate_extracted_alignment(
        _result("no", regions=(SpeechRegion(0.0, 0.08),)),
        target_text="No.",
        minimum_confidence=0.5,
        maximum_extra_speech_ms=60,
        minimum_duration_seconds=0.05,
        maximum_duration_seconds=2.0,
    )

    assert not decision.accepted
    assert "unexpected_prefix_speech" in decision.reason_codes


def test_low_confidence_and_duration_are_hard_gates() -> None:
    low = validate_extracted_alignment(
        _result("no", confidence=0.2),
        target_text="No.",
        minimum_confidence=0.5,
        maximum_extra_speech_ms=60,
        minimum_duration_seconds=0.05,
        maximum_duration_seconds=2.0,
    )
    long = validate_extracted_alignment(
        AlignmentResult(
            (AlignmentToken("no", 0.1, 2.5, 0.9),),
            (),
            "fake",
            "fake",
            "f" * 64,
        ),
        target_text="No.",
        minimum_confidence=0.5,
        maximum_extra_speech_ms=60,
        minimum_duration_seconds=0.05,
        maximum_duration_seconds=1.5,
    )

    assert low.reason_codes == ("low_confidence",)
    assert long.reason_codes == ("excessive_token_duration",)


def _write_tone_with_silence(path: Path) -> None:
    rate = 16_000
    samples: list[int] = []
    for index in range(rate):
        seconds = index / rate
        audible = 0.2 <= seconds < 0.7
        value = round(8_000 * math.sin(2 * math.pi * 220 * seconds)) if audible else 0
        samples.append(value)
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        writer.writeframes(b"".join(struct.pack("<h", value) for value in samples))


def _write_detached_tones(path: Path) -> None:
    rate = 16_000
    samples: list[int] = []
    for index in range(round(rate * 1.6)):
        seconds = index / rate
        audible = 0.02 <= seconds < 0.1 or 0.55 <= seconds < 1.45
        value = round(8_000 * math.sin(2 * math.pi * 220 * seconds)) if audible else 0
        samples.append(value)
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        writer.writeframes(b"".join(struct.pack("<h", value) for value in samples))


def test_speech_islands_find_detached_prefix_and_dominant_utterance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "detached.wav"
    _write_detached_tones(source)

    evidence = inspect_speech_islands(source, island_gap_ms=300)

    assert len(evidence.islands) == 2
    assert evidence.primary_island is not None
    assert evidence.primary_island.start_seconds == pytest.approx(0.55, abs=0.01)
    assert evidence.detached_prefix_ms == pytest.approx(80, abs=10)
    assert evidence.detached_suffix == ()


def test_crop_retains_padding_refines_edges_and_applies_fades(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    _write_tone_with_silence(source)
    regions = detect_speech_regions(source)

    evidence = crop_aligned_wav(
        source,
        target,
        start_seconds=0.22,
        end_seconds=0.68,
        pre_roll_ms=30,
        post_roll_ms=40,
        fade_ms=8,
        speech_regions=regions,
    )

    assert 0.49 <= wav_duration_seconds(target) <= 0.58
    assert evidence.crop_start_seconds <= 0.22
    assert evidence.crop_end_seconds >= 0.68
    with wave.open(str(target), "rb") as audio:
        content = audio.readframes(audio.getnframes())
    first = struct.unpack("<h", content[:2])[0]
    last = struct.unpack("<h", content[-2:])[0]
    assert abs(first) < 500
    assert abs(last) < 500


def test_crop_refuses_padding_that_reaches_separate_speech(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    _write_tone_with_silence(source)

    with pytest.raises(ArtifactError, match="adjacent speech"):
        crop_aligned_wav(
            source,
            target,
            start_seconds=0.4,
            end_seconds=0.6,
            pre_roll_ms=50,
            post_roll_ms=50,
            fade_ms=8,
            speech_regions=(SpeechRegion(0.34, 0.39),),
        )


def test_crop_protects_short_region_just_after_asr_word_end(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    _write_tone_with_silence(source)

    evidence = crop_aligned_wav(
        source,
        target,
        start_seconds=0.22,
        end_seconds=0.68,
        pre_roll_ms=30,
        post_roll_ms=40,
        fade_ms=8,
        speech_regions=(SpeechRegion(0.69, 0.75),),
    )

    assert evidence.crop_end_seconds >= 0.75


def test_crop_does_not_expand_across_one_continuous_carrier_region(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    _write_tone_with_silence(source)

    crop_aligned_wav(
        source,
        target,
        start_seconds=0.3,
        end_seconds=0.5,
        pre_roll_ms=20,
        post_roll_ms=20,
        fade_ms=8,
        speech_regions=(SpeechRegion(0.2, 0.9),),
    )

    assert wav_duration_seconds(target) < 0.35


class _LongFakeSpeechService(FakeSpeechService):
    @override
    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        if destination.exists() and not overwrite:
            raise ArtifactError(f"Output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        rate = 16_000
        with wave.open(str(destination), "wb") as writer:
            writer.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            writer.writeframes(struct.pack("<h", 1_000) * rate * 5)
        return SpeechArtifact(
            path=destination,
            backend="fake",
            voice=request.voice,
            output_format=AudioFormat.WAV,
            bytes_written=destination.stat().st_size,
            sha256=sha256_file(destination),
            duration_seconds=5.0,
            sample_rate=rate,
        )


class _DetachedSpeechService(FakeSpeechService):
    @override
    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        del request
        if destination.exists() and not overwrite:
            raise ArtifactError(f"Output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_detached_tones(destination)
        return SpeechArtifact(
            path=destination,
            backend="fake",
            voice="fake",
            output_format=AudioFormat.WAV,
            bytes_written=destination.stat().st_size,
            sha256=sha256_file(destination),
            duration_seconds=1.6,
            sample_rate=16_000,
        )


class _RefinementAligner:
    @property
    def fingerprint(self) -> str:
        return "b" * 64

    async def align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
    ) -> AlignmentResult:
        del language
        words = lexical_tokens(expected_text)
        refined = audio.name.endswith("-refined.wav")
        if not refined:
            words = ("or", *words)
        start = 0.03 if refined else 0.55
        end = wav_duration_seconds(audio) - 0.04 if refined else 1.45
        step = (end - start) / len(words)
        tokens = tuple(
            AlignmentToken(word, start + index * step, start + (index + 1) * step, 0.99)
            for index, word in enumerate(words)
        )
        return AlignmentResult(tokens, (), "fake", "fake", self.fingerprint)


class _ExpectedTextAligner:
    def __init__(self, *, prefix: str | None = None) -> None:
        self.prefix = prefix
        self.calls = 0

    @property
    def fingerprint(self) -> str:
        return "a" * 64

    async def align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
    ) -> AlignmentResult:
        del language
        self.calls += 1
        words = lexical_tokens(expected_text)
        if self.prefix is not None:
            words = (self.prefix, *words)
        duration = wav_duration_seconds(audio)
        step = duration * 0.8 / max(1, len(words))
        tokens = tuple(
            AlignmentToken(
                word,
                duration * 0.1 + index * step,
                duration * 0.1 + index * step + step * 0.45,
                0.99,
            )
            for index, word in enumerate(words)
        )
        return AlignmentResult(tokens, (), "fake", "fake", self.fingerprint)


class _LowFinalConsonantAligner:
    fingerprint = "p" * 64

    async def align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
    ) -> PhonemeAlignmentResult:
        del audio, expected_text
        return PhonemeAlignmentResult(
            phonemes=(
                PhonemeToken("θ", 0.1, 0.2, 0.9),
                PhonemeToken("ɜ", 0.2, 0.3, 0.9),
                PhonemeToken("d", 0.3, 0.4, 0.1),
            ),
            backend="wav2vec2-ctc",
            model="fake",
            fingerprint=self.fingerprint,
            language=language,
        )


class _StrongBoundaryPhonemeAligner:
    fingerprint = "q" * 64

    async def align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
    ) -> PhonemeAlignmentResult:
        del audio, expected_text
        return PhonemeAlignmentResult(
            phonemes=(
                PhonemeToken("j", 0.06, 0.12, 0.9),
                PhonemeToken("n", 0.82, 0.87, 0.95),
            ),
            backend="wav2vec2-ctc",
            model="fake",
            fingerprint=self.fingerprint,
            language=language,
        )


class _OneWordBoundaryPhonemeAligner:
    fingerprint = "r" * 64

    async def align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
    ) -> PhonemeAlignmentResult:
        del audio, expected_text
        return PhonemeAlignmentResult(
            phonemes=(
                PhonemeToken("j", 0.10, 0.20, 0.3),
                PhonemeToken("ɛ", 0.20, 0.30, 0.05),
                PhonemeToken("s", 0.30, 0.40, 0.95),
            ),
            backend="wav2vec2-ctc",
            model="fake",
            fingerprint=self.fingerprint,
            language=language,
        )


class _TooShortPhonemeAligner:
    fingerprint = "s" * 64

    async def align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
    ) -> PhonemeAlignmentResult:
        del audio, expected_text, language
        raise BuildError("Audio is too short for the expected phoneme sequence")


@pytest.mark.asyncio
async def test_candidate_pipeline_prefers_verified_context_and_writes_safe_report(
    tmp_path: Path,
) -> None:
    policy = ShortUtterancePolicy(
        strategy=ShortUtteranceStrategy.CONTEXT_EXTRACT,
        candidate_count=3,
        minimum_pause_ms=100,
        minimum_edge_silence_ms=0,
        pre_roll_ms=20,
        post_roll_ms=20,
        require_review_for_one_word=False,
        keep_candidates=True,
    )
    request = SpeechSynthesisRequest(text="Wren asked.", voice="nick")
    recipes = carrier_recipes(request.text, policy, seed_material="chapter:1")
    destination = tmp_path / "selected.wav"
    qa = tmp_path / "qa"
    aligner = _ExpectedTextAligner()

    result = await synthesize_short_utterance(
        service=_LongFakeSpeechService(),
        aligner=aligner,
        request=request,
        destination=destination,
        recipes=recipes,
        policy=policy,
        language="en-US",
        qa_directory=qa,
    )

    assert destination.is_file()
    assert result.selected.recipe.position is CarrierPosition.MIDDLE
    assert result.report == qa / "report.json"
    report = result.report.read_text(encoding="utf-8")
    assert "Wren asked" not in report
    assert '"target_word_count": 2' in report
    assert aligner.calls == 5


@pytest.mark.asyncio
async def test_candidate_pipeline_refines_detached_prefix_and_revalidates(
    tmp_path: Path,
) -> None:
    policy = ShortUtterancePolicy(
        strategy=ShortUtteranceStrategy.DIRECT,
        require_review_for_one_word=False,
        keep_candidates=True,
    )
    request = SpeechSynthesisRequest(text="Anyone touch him?", voice="caro")

    result = await synthesize_short_utterance(
        service=_DetachedSpeechService(),
        aligner=_RefinementAligner(),
        request=request,
        destination=tmp_path / "selected.wav",
        recipes=carrier_recipes(request.text, policy, seed_material="chapter:5"),
        policy=policy,
        language="en",
        qa_directory=tmp_path / "qa",
    )

    assert result.selected.acoustic_refined
    assert result.selected.acoustic is not None
    assert result.selected.acoustic.detached_prefix == ()
    assert result.selected.acoustic.leading_silence_ms <= 120
    assert wav_duration_seconds(tmp_path / "selected.wav") < 1.1


@pytest.mark.asyncio
async def test_candidate_pipeline_rejects_weak_final_phoneme(
    tmp_path: Path,
) -> None:
    policy = ShortUtterancePolicy(
        strategy=ShortUtteranceStrategy.DIRECT,
        candidate_count=1,
        minimum_edge_silence_ms=0,
        require_review_for_one_word=False,
        keep_candidates=True,
    )
    request = SpeechSynthesisRequest(text="Third.", voice="nick")

    with pytest.raises(BuildError, match="low_phoneme_confidence"):
        await synthesize_short_utterance(
            service=_LongFakeSpeechService(),
            aligner=_ExpectedTextAligner(),
            phoneme_aligner=_LowFinalConsonantAligner(),
            minimum_phoneme_confidence=0.2,
            request=request,
            destination=tmp_path / "selected.wav",
            recipes=carrier_recipes(request.text, policy, seed_material="chapter:3"),
            policy=policy,
            language="en-us",
            qa_directory=tmp_path / "qa",
        )

    report = (tmp_path / "qa" / "report.json").read_text(encoding="utf-8")
    assert '"symbol": "d"' in report
    assert '"phoneme_confidence": 0.1' in report


@pytest.mark.asyncio
async def test_phoneme_edge_can_clear_underestimated_whisper_suffix(
    tmp_path: Path,
) -> None:
    evaluation = _rankable_candidate(
        tmp_path,
        index=3,
        position=CarrierPosition.MIDDLE,
        confidence=0.8,
        leading_ms=30,
        regions=(SpeechRegion(0.03, 0.61), SpeechRegion(0.65, 0.999)),
    )
    evaluation = replace(
        evaluation,
        accepted=False,
        reason_codes=("unexpected_suffix_speech",),
    )

    result = await _apply_phoneme_gate(
        evaluation,
        aligner=_StrongBoundaryPhonemeAligner(),
        target_text="You first in?",
        language="en-us",
        minimum_confidence=0.2,
        policy=ShortUtterancePolicy(maximum_extra_speech_ms=60),
    )

    assert result.accepted
    assert result.reason_codes == ()


@pytest.mark.asyncio
async def test_phoneme_edge_does_not_clear_speech_beyond_word_tail_cap(
    tmp_path: Path,
) -> None:
    evaluation = _rankable_candidate(
        tmp_path,
        index=3,
        position=CarrierPosition.MIDDLE,
        confidence=0.8,
        leading_ms=30,
        regions=(SpeechRegion(0.03, 0.61), SpeechRegion(0.65, 1.01)),
    )
    evaluation = replace(
        evaluation,
        accepted=False,
        reason_codes=("unexpected_suffix_speech",),
    )

    result = await _apply_phoneme_gate(
        evaluation,
        aligner=_StrongBoundaryPhonemeAligner(),
        target_text="You first in?",
        language="en-us",
        minimum_confidence=0.2,
        policy=ShortUtterancePolicy(maximum_extra_speech_ms=60),
    )

    assert not result.accepted
    assert result.reason_codes == ("unexpected_suffix_speech",)


@pytest.mark.asyncio
async def test_strong_full_phoneme_path_can_clear_only_low_asr_confidence(
    tmp_path: Path,
) -> None:
    evaluation = _rankable_candidate(
        tmp_path,
        index=3,
        position=CarrierPosition.DIRECT,
        confidence=0.1,
        leading_ms=30,
    )
    evaluation = replace(
        evaluation,
        accepted=False,
        reason_codes=("low_confidence",),
    )

    result = await _apply_phoneme_gate(
        evaluation,
        aligner=_StrongBoundaryPhonemeAligner(),
        target_text="Truth, Wren thought.",
        language="en-us",
        minimum_confidence=0.2,
        policy=ShortUtterancePolicy(),
    )

    assert result.accepted
    assert result.reason_codes == ()
    assert result.phoneme_path_confidence == pytest.approx(0.925)


@pytest.mark.asyncio
async def test_strong_one_word_boundary_can_clear_marginal_asr_confidence(
    tmp_path: Path,
) -> None:
    evaluation = _rankable_candidate(
        tmp_path,
        index=1,
        position=CarrierPosition.DIRECT,
        confidence=0.53,
        leading_ms=30,
    )
    evaluation = replace(
        evaluation,
        accepted=False,
        reason_codes=("low_confidence",),
    )

    result = await _apply_phoneme_gate(
        evaluation,
        aligner=_OneWordBoundaryPhonemeAligner(),
        target_text="Yes.",
        language="en-us",
        minimum_confidence=0.2,
        policy=ShortUtterancePolicy(),
    )

    assert result.accepted
    assert result.reason_codes == ()
    assert result.phoneme_confidence == pytest.approx(0.95)
    assert result.phoneme_path_confidence == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_phoneme_failure_rejects_only_its_candidate(tmp_path: Path) -> None:
    evaluation = _rankable_candidate(
        tmp_path,
        index=1,
        position=CarrierPosition.DIRECT,
        confidence=0.9,
        leading_ms=30,
    )

    result = await _apply_phoneme_gate(
        evaluation,
        aligner=_TooShortPhonemeAligner(),
        target_text="Everybody did.",
        language="en-us",
        minimum_confidence=0.2,
        policy=ShortUtterancePolicy(),
    )

    assert not result.accepted
    assert result.reason_codes == ("phoneme_alignment_failed",)


@pytest.mark.asyncio
async def test_candidate_pipeline_rejects_every_prefixed_transcript(
    tmp_path: Path,
) -> None:
    policy = ShortUtterancePolicy(
        strategy=ShortUtteranceStrategy.CONTEXT_EXTRACT,
        candidate_count=2,
        minimum_pause_ms=0,
        require_review_for_one_word=False,
        keep_candidates=True,
    )
    request = SpeechSynthesisRequest(text="Wren asked.", voice="nick")

    with pytest.raises(BuildError, match="No short-utterance candidate"):
        await synthesize_short_utterance(
            service=_LongFakeSpeechService(),
            aligner=_ExpectedTextAligner(prefix="or"),
            request=request,
            destination=tmp_path / "selected.wav",
            recipes=carrier_recipes(request.text, policy, seed_material="chapter:1"),
            policy=policy,
            language="en",
            qa_directory=tmp_path / "qa",
        )

    assert not (tmp_path / "selected.wav").exists()
    assert (tmp_path / "qa" / "report.json").is_file()


@pytest.mark.asyncio
async def test_one_word_candidate_stops_for_review_without_committing(
    tmp_path: Path,
) -> None:
    policy = ShortUtterancePolicy(
        strategy=ShortUtteranceStrategy.CONTEXT_EXTRACT,
        candidate_count=2,
        minimum_pause_ms=0,
        minimum_edge_silence_ms=0,
        require_review_for_one_word=True,
        keep_candidates=True,
    )
    request = SpeechSynthesisRequest(text="No.", voice="caro")

    with pytest.raises(BuildError, match="requires listening review"):
        await synthesize_short_utterance(
            service=_LongFakeSpeechService(),
            aligner=_ExpectedTextAligner(),
            request=request,
            destination=tmp_path / "selected.wav",
            recipes=carrier_recipes(request.text, policy, seed_material="chapter:2"),
            policy=policy,
            language="en",
            qa_directory=tmp_path / "qa",
        )

    assert not (tmp_path / "selected.wav").exists()
    assert (tmp_path / "qa" / "report.json").is_file()
    review = tmp_path / "qa" / "listening-review.toml"
    assert review.is_file()

    review.write_text(
        review.read_text(encoding="utf-8").replace(
            'status = "pending"', 'status = "pass"'
        ),
        encoding="utf-8",
    )
    result = await synthesize_short_utterance(
        service=_LongFakeSpeechService(),
        aligner=_ExpectedTextAligner(),
        request=request,
        destination=tmp_path / "selected.wav",
        recipes=carrier_recipes(request.text, policy, seed_material="chapter:2"),
        policy=policy,
        language="en",
        qa_directory=tmp_path / "qa",
    )

    assert (tmp_path / "selected.wav").is_file()
    assert result.review_required is False


@pytest.mark.asyncio
async def test_mlx_aligner_is_lazy_local_and_does_not_prompt_expected_text(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "speech.wav"
    expected_audio = str(audio)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"test")
    _write_tone_with_silence(audio)

    class _FakeMlx:
        def __init__(self) -> None:
            self.options: dict[str, object] = {}

        def transcribe(self, audio: str, **options: object) -> Mapping[str, object]:
            assert audio == expected_audio
            self.options = options
            return {
                "segments": [
                    {
                        "start": 0.2,
                        "end": 0.7,
                        "avg_logprob": -0.2,
                        "compression_ratio": 1.0,
                        "no_speech_prob": 0.01,
                        "temperature": 0.0,
                        "words": [
                            {
                                "word": " Wren",
                                "start": 0.2,
                                "end": 0.45,
                                "probability": 0.9,
                            },
                            {
                                "word": " asked.",
                                "start": 0.45,
                                "end": 0.7,
                                "probability": 0.8,
                            },
                        ],
                    }
                ]
            }

    fake = _FakeMlx()
    aligner = MlxWhisperAligner(model=str(model))
    aligner._module = fake

    result = await aligner.align(audio, "Or Wren asked.", language="en-US")

    assert tuple(token.text for token in result.tokens) == ("wren", "asked")
    assert fake.options["path_or_hf_repo"] == str(model)
    assert fake.options["word_timestamps"] is True
    assert fake.options["condition_on_previous_text"] is False
    assert fake.options["language"] == "en"
    assert "initial_prompt" not in fake.options
    assert result.speech_regions


@pytest.mark.asyncio
async def test_mlx_aligner_uses_prompt_only_to_refine_an_exact_authority_pass(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "speech.wav"
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"test")
    _write_tone_with_silence(audio)

    class _FakeMlx:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def transcribe(self, audio: str, **options: object) -> Mapping[str, object]:
            del audio
            self.calls.append(options)
            prompted = "initial_prompt" in options
            start = 0.24 if prompted else 0.20
            return {
                "text": " Wren asked.",
                "language": "en",
                "segments": [
                    {
                        "start": start,
                        "end": 0.7,
                        "avg_logprob": -0.2,
                        "compression_ratio": 1.0,
                        "no_speech_prob": 0.01,
                        "temperature": 0.0,
                        "words": [
                            {
                                "word": " Wren",
                                "start": start,
                                "end": 0.45,
                                "probability": 0.9,
                            },
                            {
                                "word": " asked.",
                                "start": 0.45,
                                "end": 0.7,
                                "probability": 0.8,
                            },
                        ],
                    }
                ],
            }

    fake = _FakeMlx()
    aligner = MlxWhisperAligner(model=str(model))
    aligner._module = fake

    result = await aligner.align(audio, "Wren asked.", language="en")

    assert len(fake.calls) == 2
    assert "initial_prompt" not in fake.calls[0]
    assert fake.calls[1]["initial_prompt"] == "Wren asked."
    assert result.timing_source == "prompted_exact_match"
    assert result.tokens[0].start_seconds == 0.24


@pytest.mark.asyncio
async def test_mlx_aligner_preserves_malformed_words_as_rejection_evidence(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "speech.wav"
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"test")
    _write_tone_with_silence(audio)

    class _FakeMlx:
        def transcribe(self, audio: str, **options: object) -> Mapping[str, object]:
            del audio, options
            return {
                "segments": [
                    {
                        "start": 0.2,
                        "end": 0.7,
                        "avg_logprob": -0.2,
                        "compression_ratio": 1.0,
                        "no_speech_prob": 0.01,
                        "temperature": 0.0,
                        "words": [
                            {
                                "word": " Wren",
                                "start": 0.2,
                                "end": 0.45,
                                "probability": 0.9,
                            },
                            {"word": " hidden", "start": "bad", "end": 0.5},
                            {
                                "word": " asked.",
                                "start": 0.45,
                                "end": 0.7,
                                "probability": 0.8,
                            },
                        ],
                    }
                ]
            }

    aligner = MlxWhisperAligner(model=str(model), prompted_timing=False)
    aligner._module = _FakeMlx()
    result = await aligner.align(audio, "Wren asked.", language="en")
    decision = validate_extracted_alignment(
        result,
        target_text="Wren asked.",
        minimum_confidence=0.2,
        maximum_extra_speech_ms=60,
        minimum_duration_seconds=0.1,
        maximum_duration_seconds=2.0,
    )

    assert "segment_0_word_1_timing_missing" in result.issues
    assert decision.accepted is False
    assert "segment_0_word_1_timing_missing" in decision.reason_codes
