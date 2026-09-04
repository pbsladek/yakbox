from __future__ import annotations

import json
import math
import struct
import subprocess
import wave
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from tests.schema_helpers import validate_contract

from yakbox._files import sha256_file
from yakbox.audio.crop import SpeechRegion
from yakbox.local_alignment import MlxWhisperAligner
from yakbox.local_phoneme_alignment import (
    _ctc_forced_alignment,
    _phonemize,
    _target_tokens,
)
from yakbox.phoneme_models import phoneme_model_status
from yakbox.phoneme_qa import PhonemeInspection
from yakbox.short_testing import (
    list_short_reviews,
    review_audio_path,
    write_short_review,
)
from yakbox.speech.alignment import (
    AlignmentResult,
    AlignmentSegment,
    AlignmentToken,
    DecodePassEvidence,
    alignment_quality_reason_codes,
)
from yakbox.speech.phonemes import (
    PhonemeAlignmentResult,
    PhonemeToken,
    validate_phoneme_alignment,
)
from yakbox.whisper_calibration import calibrate_frozen_corpus
from yakbox.whisper_models import model_status, require_model_path
from yakbox.whisper_qa import (
    JoinSpecification,
    WhisperClipType,
    evaluate_alignment,
    inspect_joins,
    verify_manuscript,
)


def _alignment(
    *words: str,
    confidence: float = 0.9,
    fingerprint: str = "f" * 64,
) -> AlignmentResult:
    tokens = tuple(
        AlignmentToken(word, 0.2 + index * 0.3, 0.45 + index * 0.3, confidence)
        for index, word in enumerate(words)
    )
    return AlignmentResult(
        tokens=tokens,
        speech_regions=(SpeechRegion(0.2, max(0.45, len(words) * 0.3 + 0.2)),),
        backend="fake-whisper",
        model="fake",
        fingerprint=fingerprint,
        segments=(
            AlignmentSegment(
                0.2,
                max(0.5, len(words) * 0.3 + 0.2),
                -0.2,
                1.0,
                0.01,
                0.0,
            ),
        ),
        transcript=" ".join(words),
    )


def _write_join_wav(
    path: Path, *, click: bool = False, duration_seconds: int = 2
) -> None:
    rate = 16_000
    samples = [0] * (rate * duration_seconds)
    for index in range(round(rate * 0.2), round(rate * 0.8)):
        samples[index] = 3_000
    for index in range(round(rate * 1.2), round(rate * 1.8)):
        samples[index] = 3_000
    if click:
        samples[rate - 1] = -32_000
        samples[rate] = 32_000
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        writer.writeframes(b"".join(struct.pack("<h", item) for item in samples))


class _FakeQaAligner:
    def __init__(
        self,
        result: AlignmentResult,
        *,
        window_result: AlignmentResult | None = None,
        window_results: tuple[AlignmentResult, ...] | None = None,
    ) -> None:
        self.result = result
        self.window_result = window_result or result
        self.window_results = window_results
        self.windows: list[tuple[float, float, str]] = []
        self.align_calls = 0

    @property
    def fingerprint(self) -> str:
        return self.result.fingerprint

    async def align(
        self, audio: Path, expected_text: str, *, language: str
    ) -> AlignmentResult:
        del audio, expected_text, language
        self.align_calls += 1
        return self.result

    async def align_window(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
        start_seconds: float,
        end_seconds: float,
    ) -> AlignmentResult:
        del audio, language
        self.windows.append((start_seconds, end_seconds, expected_text))
        if self.window_results is not None:
            index = min(len(self.windows) - 1, len(self.window_results) - 1)
            return self.window_results[index]
        return self.window_result


def test_local_whisper_model_status_verifies_required_files(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")

    status = model_status(str(model), None)

    assert status.installed
    assert status.verified
    assert status.fingerprint
    assert require_model_path(str(model), None) == model.resolve()


def test_local_phoneme_model_status_accepts_pinned_ctc_files(
    tmp_path: Path,
) -> None:
    model = tmp_path / "phoneme-model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "vocab.json").write_text('{"<pad>": 0, "d": 1}', encoding="utf-8")
    (model / "pytorch_model.bin").write_bytes(b"weights")

    status = phoneme_model_status(str(model), None)

    assert status.installed
    assert status.verified
    assert status.local_path == model.resolve()


def test_frozen_whisper_calibration_has_no_false_accepts_or_rejects() -> None:
    report = calibrate_frozen_corpus()

    assert report["case_count"] == 7
    assert report["false_accepts"] == 0
    assert report["false_rejects"] == 0
    assert report["consonant_case_accuracy"] == 1.0
    assert report["benchmark_fingerprint"]
    validate_contract("whisper-calibration", report)


def test_confidence_is_calibrated_by_clip_type() -> None:
    result = _alignment("no", confidence=0.55)

    one_word = evaluate_alignment(
        result,
        clip_type=WhisperClipType.ONE_WORD,
        expected_text="No.",
    )
    sentence = evaluate_alignment(
        result,
        clip_type=WhisperClipType.SENTENCE,
        expected_text="No.",
    )

    assert not one_word.accepted
    assert one_word.reason_codes == ("low_confidence",)
    assert sentence.accepted


@pytest.mark.asyncio
async def test_chapter_verification_reports_timecoded_manuscript_edits(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio)
    manuscript.write_text("Expected manuscript", encoding="utf-8")
    aligner = _FakeQaAligner(_alignment("the", "wrong", "word"))

    report = await verify_manuscript(
        audio,
        manuscript,
        "The right word.",
        language="en",
        model="fake",
        revision=None,
        aligner=cast(MlxWhisperAligner, aligner),
    )
    payload = report.to_dict()

    assert not report.accepted
    assert report.token_accuracy == pytest.approx(2 / 3)
    assert report.mismatches[0].operation == "replace"
    assert report.mismatches[0].audio_start_seconds == pytest.approx(0.5)
    assert "manuscript_transcript_mismatch" in report.reason_codes
    validate_contract("whisper-manuscript-verification", payload)


@pytest.mark.asyncio
async def test_manuscript_verification_canonicalizes_aliases_and_hyphens(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio)
    manuscript.write_text("Liora called the hotline.", encoding="utf-8")
    aligner = _FakeQaAligner(_alignment("leora", "called", "the", "hot", "line"))

    report = await verify_manuscript(
        audio,
        manuscript,
        "Liora called the hot-line.",
        language="en",
        model="fake",
        revision=None,
        aligner=cast(MlxWhisperAligner, aligner),
        token_aliases={"liora": ("leora",)},
    )

    assert report.accepted, report.to_dict()
    assert report.token_accuracy == 1.0
    assert report.mismatches == ()
    assert "expected_transcript_mismatch" in report.diagnostic_reason_codes


@pytest.mark.asyncio
async def test_manuscript_aliases_are_directional_when_alias_is_canonical(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio)
    expected = "ha u"
    manuscript.write_text(expected, encoding="utf-8")

    report = await verify_manuscript(
        audio,
        manuscript,
        expected,
        language="en",
        model="fake",
        revision=None,
        aligner=cast(
            MlxWhisperAligner,
            _FakeQaAligner(_alignment("ha", "ha")),
        ),
        token_aliases={"ha": ("ah",), "u": ("ha",)},
    )

    assert report.accepted
    assert report.token_accuracy == 1.0
    assert report.mismatches == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expected", "recognized"),
    [
        ("Call the hotline.", ("call", "the", "hot", "line")),
        ("Call the hot-line.", ("call", "the", "hotline")),
    ],
)
async def test_manuscript_verification_accepts_compound_tokenization_variants(
    tmp_path: Path,
    expected: str,
    recognized: tuple[str, ...],
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio)
    manuscript.write_text(expected, encoding="utf-8")

    report = await verify_manuscript(
        audio,
        manuscript,
        expected,
        language="en",
        model="fake",
        revision=None,
        aligner=cast(MlxWhisperAligner, _FakeQaAligner(_alignment(*recognized))),
    )

    assert report.accepted
    assert report.token_accuracy == 1.0
    assert report.mismatches == ()


@pytest.mark.asyncio
async def test_manuscript_verification_accepts_joined_alias_compounds(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio)
    expected = "Me ha okhel."
    manuscript.write_text(expected, encoding="utf-8")

    report = await verify_manuscript(
        audio,
        manuscript,
        expected,
        language="en",
        model="fake",
        revision=None,
        aligner=cast(
            MlxWhisperAligner,
            _FakeQaAligner(_alignment("mihah", "okeh")),
        ),
        token_aliases={
            "me": ("mih",),
            "ha": ("ah",),
            "okhel": ("okeh",),
        },
    )

    assert report.accepted
    assert report.token_accuracy == 1.0
    assert report.mismatches == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expected", "recognized"),
    [
        ("Room six oh nine.", ("room", "609")),
        ("Room 609.", ("room", "six", "oh", "nine")),
    ],
)
async def test_manuscript_verification_accepts_digit_sequence_variants(
    tmp_path: Path,
    expected: str,
    recognized: tuple[str, ...],
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio)
    manuscript.write_text(expected, encoding="utf-8")

    report = await verify_manuscript(
        audio,
        manuscript,
        expected,
        language="en",
        model="fake",
        revision=None,
        aligner=cast(MlxWhisperAligner, _FakeQaAligner(_alignment(*recognized))),
    )

    assert report.accepted
    assert report.token_accuracy == 1.0
    assert report.mismatches == ()


@pytest.mark.asyncio
async def test_digit_normalization_allows_adjacent_compound_coalescing(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio)
    expected = "Outside six oh nine, Liora waited."
    manuscript.write_text(expected, encoding="utf-8")

    report = await verify_manuscript(
        audio,
        manuscript,
        expected,
        language="en",
        model="fake",
        revision=None,
        aligner=cast(
            MlxWhisperAligner,
            _FakeQaAligner(_alignment("outside", "609", "lio", "ra", "waited")),
        ),
    )

    assert report.accepted
    assert report.token_accuracy == 1.0
    assert report.mismatches == ()


@pytest.mark.asyncio
async def test_manuscript_alias_accepts_split_recognized_name_without_global_aliases(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio)
    expected = "Liora waited."
    manuscript.write_text(expected, encoding="utf-8")

    report = await verify_manuscript(
        audio,
        manuscript,
        expected,
        language="en",
        model="fake",
        revision=None,
        token_aliases={"liora": ("thera",)},
        aligner=cast(
            MlxWhisperAligner,
            _FakeQaAligner(_alignment("the", "ra", "waited")),
        ),
    )

    assert report.accepted
    assert report.mismatches == ()


@pytest.mark.asyncio
async def test_manuscript_verification_keeps_decoder_issues_diagnostic(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio)
    manuscript.write_text("Exact transcript.", encoding="utf-8")
    result = replace(
        _alignment("exact", "transcript"),
        issues=("segment_1_word_1_lexical_invalid",),
    )

    report = await verify_manuscript(
        audio,
        manuscript,
        "Exact transcript.",
        language="en",
        model="fake",
        revision=None,
        aligner=cast(MlxWhisperAligner, _FakeQaAligner(result)),
    )

    assert report.accepted
    assert report.reason_codes == ()
    assert "segment_1_word_1_lexical_invalid" in report.diagnostic_reason_codes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expected", "recognized", "insert_index", "confidence"),
    [
        ("His knees complained.", ("his", "knees", "complained", "mind"), 3, 0.01),
        ("At the mirror.", ("at", "the", "the", "mirror"), 2, 0.30),
    ],
)
async def test_manuscript_verification_ignores_short_decoder_insertions(
    tmp_path: Path,
    expected: str,
    recognized: tuple[str, ...],
    insert_index: int,
    confidence: float,
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio)
    result = _alignment(*recognized)
    tokens = list(result.tokens)
    inserted = tokens[insert_index]
    tokens[insert_index] = replace(
        inserted,
        end_seconds=inserted.start_seconds + 0.1,
        confidence=confidence,
    )

    report = await verify_manuscript(
        audio,
        manuscript,
        expected,
        language="en",
        model="fake",
        revision=None,
        aligner=cast(
            MlxWhisperAligner,
            _FakeQaAligner(replace(result, tokens=tuple(tokens))),
        ),
    )

    assert report.accepted
    assert report.token_accuracy == 1.0
    assert report.recognized_token_count == report.expected_token_count
    assert "low_confidence_insert_ignored" in report.diagnostic_reason_codes


@pytest.mark.asyncio
async def test_manuscript_verification_rejects_credible_extra_speech(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio)

    report = await verify_manuscript(
        audio,
        manuscript,
        "At the mirror.",
        language="en",
        model="fake",
        revision=None,
        aligner=cast(
            MlxWhisperAligner,
            _FakeQaAligner(_alignment("at", "the", "broken", "mirror")),
        ),
    )

    assert not report.accepted
    assert report.mismatches[0].operation == "insert"


@pytest.mark.asyncio
async def test_manuscript_verification_rechecks_long_form_mismatch_locally(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio)
    expected = "Window. The window looked."
    manuscript.write_text(expected, encoding="utf-8")
    local = replace(
        _alignment("window", "the", "window", "looked"),
        decode_passes=(
            DecodePassEvidence(
                "authority",
                "",
                ("window", "the", "window", "looked"),
                (),
                0.9,
                True,
            ),
            DecodePassEvidence(
                "sampled_consensus",
                "",
                ("window", "the", "window", "looked"),
                (),
                0.8,
                True,
            ),
        ),
    )
    aligner = _FakeQaAligner(
        _alignment("window", "window", "the", "window", "looked"),
        window_result=local,
    )

    report = await verify_manuscript(
        audio,
        manuscript,
        expected,
        language="en",
        model="fake",
        revision=None,
        aligner=cast(MlxWhisperAligner, aligner),
    )

    assert report.accepted
    assert report.token_accuracy == 1.0
    assert report.mismatches == ()
    assert aligner.windows
    assert "localized_mismatch_recheck_passed" in report.diagnostic_reason_codes


@pytest.mark.asyncio
async def test_manuscript_verification_retries_mismatch_with_narrow_context(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio)
    expected = "alpha beta gamma target delta epsilon zeta"
    manuscript.write_text(expected, encoding="utf-8")
    broad_tokens = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
    broad = replace(
        _alignment(*broad_tokens),
        decode_passes=(
            DecodePassEvidence("authority", "", broad_tokens, (), 0.9, False),
            DecodePassEvidence("sampled_consensus", "", broad_tokens, (), 0.9, False),
        ),
    )
    narrow_tokens = ("beta", "gamma", "target", "delta", "epsilon")
    narrow = replace(
        _alignment(*narrow_tokens),
        decode_passes=(
            DecodePassEvidence("authority", "", narrow_tokens, (), 0.9, True),
            DecodePassEvidence("sampled_consensus", "", narrow_tokens, (), 0.9, True),
        ),
    )
    aligner = _FakeQaAligner(
        broad,
        window_results=(broad, narrow),
    )

    report = await verify_manuscript(
        audio,
        manuscript,
        expected,
        language="en",
        model="fake",
        revision=None,
        aligner=cast(MlxWhisperAligner, aligner),
    )

    assert report.accepted
    assert report.mismatches == ()
    assert len(aligner.windows) == 2
    assert aligner.windows[0][2] == expected
    assert aligner.windows[1][2] == "beta gamma target delta epsilon"


@pytest.mark.asyncio
async def test_manuscript_verification_retries_boundary_mismatch_with_wide_context(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio, duration_seconds=20)
    expected_tokens = tuple(
        f"word{chr(97 + index // 26)}{chr(97 + index % 26)}" for index in range(40)
    )
    expected = " ".join(expected_tokens)
    manuscript.write_text(expected, encoding="utf-8")
    recognized_tokens = (*expected_tokens[:20], "wrong", *expected_tokens[21:])
    mismatch = replace(
        _alignment(*recognized_tokens),
        decode_passes=(
            DecodePassEvidence("authority", "", recognized_tokens, (), 0.9, False),
            DecodePassEvidence(
                "sampled_consensus", "", recognized_tokens, (), 0.9, False
            ),
        ),
    )
    wide = replace(
        _alignment(*expected_tokens[4:37]),
        decode_passes=(
            DecodePassEvidence("authority", "", expected_tokens[4:37], (), 0.9, True),
            DecodePassEvidence(
                "sampled_consensus", "", expected_tokens[4:37], (), 0.9, True
            ),
        ),
    )
    aligner = _FakeQaAligner(
        mismatch,
        window_results=(mismatch, mismatch, wide),
    )

    report = await verify_manuscript(
        audio,
        manuscript,
        expected,
        language="en",
        model="fake",
        revision=None,
        aligner=cast(MlxWhisperAligner, aligner),
    )

    assert report.accepted
    assert report.mismatches == ()
    assert len(aligner.windows) == 3
    assert aligner.windows[2][2] == " ".join(expected_tokens[4:37])


@pytest.mark.asyncio
async def test_manuscript_verification_retries_with_very_wide_context(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio, duration_seconds=30)
    expected_tokens = tuple(
        f"token{chr(97 + index // 26)}{chr(97 + index % 26)}" for index in range(70)
    )
    expected = " ".join(expected_tokens)
    manuscript.write_text(expected, encoding="utf-8")
    recognized_tokens = (*expected_tokens[:35], "wrong", *expected_tokens[36:])
    mismatch = replace(
        _alignment(*recognized_tokens),
        decode_passes=(
            DecodePassEvidence("authority", "", recognized_tokens, (), 0.9, False),
            DecodePassEvidence(
                "sampled_consensus", "", recognized_tokens, (), 0.9, False
            ),
        ),
    )
    exact = replace(
        _alignment(*expected_tokens[3:68]),
        decode_passes=(
            DecodePassEvidence("authority", "", expected_tokens[3:68], (), 0.9, True),
            DecodePassEvidence(
                "sampled_consensus", "", expected_tokens[3:68], (), 0.9, True
            ),
        ),
    )
    aligner = _FakeQaAligner(
        mismatch,
        window_results=(mismatch, mismatch, mismatch, exact),
    )

    report = await verify_manuscript(
        audio,
        manuscript,
        expected,
        language="en",
        model="fake",
        revision=None,
        aligner=cast(MlxWhisperAligner, aligner),
    )

    assert report.accepted
    assert len(aligner.windows) == 4
    assert aligner.windows[3][2] == " ".join(expected_tokens[3:68])


@pytest.mark.asyncio
async def test_manuscript_verification_requires_two_clean_local_decodes(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio)
    expected = "At the mirror."
    manuscript.write_text(expected, encoding="utf-8")
    local = replace(
        _alignment("at", "the", "mirror"),
        decode_passes=(
            DecodePassEvidence(
                "authority",
                "",
                ("at", "the", "mirror"),
                (),
                0.9,
                True,
            ),
        ),
    )

    report = await verify_manuscript(
        audio,
        manuscript,
        expected,
        language="en",
        model="fake",
        revision=None,
        aligner=cast(
            MlxWhisperAligner,
            _FakeQaAligner(
                _alignment("at", "the", "broken", "mirror"),
                window_result=local,
            ),
        ),
    )

    assert not report.accepted
    assert report.mismatches


@pytest.mark.asyncio
async def test_join_inspection_uses_targeted_window_and_detects_click(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "joined.wav"
    _write_join_wav(audio, click=True)
    aligner = _FakeQaAligner(_alignment("wren", "asked", confidence=0.9))

    report = await inspect_joins(
        audio,
        (JoinSpecification(1.0, "Wren", "asked", "sentence"),),
        language="en",
        model="fake",
        revision=None,
        window_seconds=0.5,
        aligner=cast(MlxWhisperAligner, aligner),
    )
    payload = report.to_dict()

    assert aligner.windows == [(0.5, 1.5, "Wren asked")]
    assert not report.accepted
    assert "join_click" in report.joins[0].reason_codes
    validate_contract("whisper-join-inspection", payload)


@pytest.mark.asyncio
async def test_join_inspection_coalesces_context_free_windows(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "joined.wav"
    _write_join_wav(audio)
    aligner = _FakeQaAligner(_alignment("steady", confidence=0.9))

    report = await inspect_joins(
        audio,
        (
            JoinSpecification(0.8),
            JoinSpecification(1.0),
            JoinSpecification(1.2),
        ),
        language="en",
        model="fake",
        revision=None,
        window_seconds=0.25,
        coalesce_gap_seconds=0.1,
        aligner=aligner,
    )

    assert len(aligner.windows) == 1
    assert report.alignment_window_count == 1
    assert report.coalesced_join_count == 2
    assert len(report.joins) == 3


@pytest.mark.asyncio
async def test_join_asr_instability_is_diagnostic_without_a_pcm_click(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "joined.wav"
    _write_join_wav(audio)
    unstable = replace(
        _alignment("steady", confidence=0.1),
        tokens=(AlignmentToken("steady", 0.8, 1.2, 0.1),),
    )
    aligner = _FakeQaAligner(unstable)

    report = await inspect_joins(
        audio,
        (JoinSpecification(1.0),),
        language="en",
        model="fake",
        revision=None,
        window_seconds=0.5,
        aligner=aligner,
    )

    assert report.accepted
    assert report.joins[0].reason_codes == ()
    assert "low_confidence" in report.joins[0].diagnostic_reason_codes
    assert "word_crosses_join" in report.joins[0].diagnostic_reason_codes


@pytest.mark.asyncio
async def test_chapter_verification_reuses_content_addressed_cache(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "chapter.wav"
    manuscript = tmp_path / "chapter.md"
    _write_join_wav(audio)
    manuscript.write_text("Steady words", encoding="utf-8")
    aligner = _FakeQaAligner(_alignment("steady", "words"))
    cache = tmp_path / "cache"

    for _ in range(2):
        report = await verify_manuscript(
            audio,
            manuscript,
            "Steady words.",
            language="en",
            model="fake",
            revision=None,
            aligner=aligner,
            cache_root=cache,
        )
        assert report.accepted

    assert aligner.align_calls == 1
    cache_text = next(cache.rglob("*.json")).read_text(encoding="utf-8")
    assert "Steady words" not in cache_text


def test_ctc_viterbi_forces_each_expected_phoneme() -> None:
    probabilities = (
        (0.90, 0.05, 0.05),
        (0.05, 0.90, 0.05),
        (0.90, 0.05, 0.05),
        (0.05, 0.05, 0.90),
        (0.90, 0.05, 0.05),
    )
    log_probabilities = [
        [math.log(value) for value in frame] for frame in probabilities
    ]

    spans = _ctc_forced_alignment(log_probabilities, (1, 2), blank_id=0)

    assert spans[0][:2] == (1, 2)
    assert spans[1][:2] == (3, 4)
    assert spans[0][2] == pytest.approx(0.9)


def test_espeak_phonemizer_uses_supported_quiet_and_voice_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_run(
        command: list[str], **_options: object
    ) -> subprocess.CompletedProcess[str]:
        captured.extend(command)
        return subprocess.CompletedProcess(
            command,
            0,
            "θ\N{MODIFIER LETTER VERTICAL LINE}ɜ"
            "\N{MODIFIER LETTER TRIANGULAR COLON}d\n",
            "",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = _phonemize(
        "Third",
        language="en-US",
        executable="espeak-ng",
        timeout_seconds=10,
    )

    assert result == "θɜ\N{MODIFIER LETTER TRIANGULAR COLON}d"
    assert captured == ["espeak-ng", "-q", "--ipa", "-v", "en-us", "Third"]


def test_phoneme_tokenization_backtracks_from_incomplete_compound() -> None:
    long_mark = "\N{MODIFIER LETTER TRIANGULAR COLON}"

    symbols, token_ids = _target_tokens(
        f"ju{long_mark}",
        {"j": 1, "ju": 2, f"u{long_mark}": 3},
    )

    assert symbols == ("j", f"u{long_mark}")
    assert token_ids == (1, 3)


def test_phoneme_gate_and_report_expose_timed_consonant_evidence(
    tmp_path: Path,
) -> None:
    result = PhonemeAlignmentResult(
        phonemes=(
            PhonemeToken("θ", 0.10, 0.18, 0.8),
            PhonemeToken("ɜː", 0.18, 0.42, 0.7),
            PhonemeToken("d", 0.42, 0.50, 0.1),
        ),
        backend="wav2vec2-ctc",
        model="fake",
        fingerprint="f" * 64,
        language="en-us",
    )
    decision = validate_phoneme_alignment(result, minimum_confidence=0.2)
    report = PhonemeInspection(
        audio=tmp_path / "third.wav",
        expected_sha256="a" * 64,
        result=result,
        decision=decision,
    )

    assert not decision.accepted
    assert decision.reason_codes == ("low_phoneme_confidence",)
    assert decision.end_seconds == 0.5
    validate_contract("phoneme-inspection", report.to_dict())


def test_final_consonant_cluster_uses_cluster_confidence() -> None:
    result = PhonemeAlignmentResult(
        phonemes=(
            PhonemeToken("æ", 0.10, 0.20, 0.01),
            PhonemeToken("s", 0.20, 0.30, 0.9),
            PhonemeToken("k", 0.30, 0.40, 0.4),
            PhonemeToken("t", 0.40, 0.50, 0.001),
        ),
        backend="wav2vec2-ctc",
        model="fake",
        fingerprint="f" * 64,
        language="en-us",
    )

    decision = validate_phoneme_alignment(result, minimum_confidence=0.2)

    assert decision.accepted
    assert decision.confidence == pytest.approx((0.9 + 0.4 + 0.001) / 3)


def test_vowel_final_alignment_retains_evidence_without_consonant_gate() -> None:
    result = PhonemeAlignmentResult(
        phonemes=(
            PhonemeToken("n", 0.10, 0.20, 0.9),
            PhonemeToken("oʊ", 0.20, 0.40, 0.001),
        ),
        backend="wav2vec2-ctc",
        model="fake",
        fingerprint="f" * 64,
        language="en-us",
    )

    decision = validate_phoneme_alignment(result, minimum_confidence=0.2)

    assert decision.accepted
    assert decision.confidence == pytest.approx(0.001)


@pytest.mark.asyncio
async def test_consensus_prompt_sensitivity_and_targeted_options_are_retained(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "speech.wav"
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"test")
    _write_join_wav(audio)

    class _FakeMlx:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def transcribe(self, audio: str, **options: object) -> Mapping[str, object]:
            del audio
            self.calls.append(options)
            word = " no"
            if options.get("best_of") == 5 and "initial_prompt" not in options:
                word = " nah"
            return {
                "text": word,
                "language": "en",
                "segments": [
                    {
                        "start": 0.8,
                        "end": 1.2,
                        "avg_logprob": -0.2,
                        "compression_ratio": 1.0,
                        "no_speech_prob": 0.01,
                        "temperature": 0.0,
                        "words": [
                            {
                                "word": word,
                                "start": 0.8,
                                "end": 1.2,
                                "probability": 0.9,
                            }
                        ],
                    }
                ],
            }

    fake = _FakeMlx()
    aligner = MlxWhisperAligner(
        model=str(model),
        decode_consensus=True,
        prompt_sensitivity=True,
    )
    aligner._module = fake

    result = await aligner.align_window(
        audio,
        "No.",
        language="en",
        start_seconds=0.5,
        end_seconds=1.5,
    )

    assert len(fake.calls) == 3
    assert all(call["clip_timestamps"] == [0.5, 1.5] for call in fake.calls)
    assert all(call["hallucination_silence_threshold"] == 0.8 for call in fake.calls)
    assert fake.calls[1]["temperature"] == 0.2
    assert fake.calls[1]["best_of"] == 5
    assert "beam_size" not in fake.calls[1]
    assert "beam_size" not in fake.calls[2]
    assert result.consensus_score == 0.0
    assert result.consensus_reason_codes == ("decode_consensus_mismatch",)
    assert result.prompt_sensitivity == "stable_match"
    assert result.timing_source == "prompted_exact_match"


@pytest.mark.asyncio
async def test_prompt_sensitivity_cannot_rescue_authoritative_mismatch(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "speech.wav"
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"test")
    _write_join_wav(audio)

    class _PromptRescueMlx:
        def transcribe(self, audio: str, **options: object) -> Mapping[str, object]:
            del audio
            word = " no" if "initial_prompt" in options else " nah"
            return {
                "text": word,
                "language": "en",
                "segments": [
                    {
                        "start": 0.5,
                        "end": 0.9,
                        "avg_logprob": -0.2,
                        "compression_ratio": 1.0,
                        "no_speech_prob": 0.01,
                        "temperature": 0.0,
                        "words": [
                            {
                                "word": word,
                                "start": 0.5,
                                "end": 0.9,
                                "probability": 0.9,
                            }
                        ],
                    }
                ],
            }

    aligner = MlxWhisperAligner(
        model=str(model),
        decode_consensus=True,
        prompt_sensitivity=True,
    )
    aligner._module = _PromptRescueMlx()

    result = await aligner.align(audio, "No.", language="en")
    decision = evaluate_alignment(
        result,
        clip_type=WhisperClipType.ONE_WORD,
        expected_text="No.",
    )

    assert tuple(token.text for token in result.tokens) == ("nah",)
    assert result.prompt_sensitivity == "prompt_rescued"
    assert result.timing_source == "unprompted"
    assert not decision.accepted
    assert "expected_transcript_mismatch" in decision.reason_codes


def test_repetition_loop_is_a_hard_quality_gate() -> None:
    result = _alignment("again", "again", "again")

    assert "probable_repetition_loop" in alignment_quality_reason_codes(result)


def test_high_temperature_fallback_is_a_hard_quality_gate() -> None:
    base = _alignment("uncertain")
    result = replace(
        base,
        segments=(AlignmentSegment(0.2, 0.5, -0.2, 1.0, 0.01, 0.4),),
    )

    assert "high_segment_temperature" in alignment_quality_reason_codes(result)


def test_short_review_decision_is_bound_to_report_and_audio_hashes(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "candidate-001-extracted.wav"
    audio.write_bytes(b"audio evidence")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "short_utterance_qa",
                "selected_candidate": 1,
                "candidates": [
                    {
                        "candidate_index": 1,
                        "extracted_audio": audio.name,
                        "extracted_audio_sha256": sha256_file(audio),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    review = write_short_review(report, approved=True, notes="clean ending")
    discovered = list_short_reviews(tmp_path)

    assert review_audio_path(report) == audio.resolve()
    assert 'status = "pass"' in review.read_text(encoding="utf-8")
    assert f'selected_audio_sha256 = "{sha256_file(audio)}"' in review.read_text(
        encoding="utf-8"
    )
    assert discovered[0]["status"] == "pass"
