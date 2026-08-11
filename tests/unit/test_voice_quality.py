from __future__ import annotations

import hashlib
import json
import math
import struct
import wave
from pathlib import Path

import pytest
from tests.schema_helpers import validate_contract

from yakbox.audio.crop import SpeechRegion
from yakbox.errors import ValidationError
from yakbox.speech.alignment import (
    AlignmentResult,
    AlignmentSegment,
    AlignmentToken,
    DecodePassEvidence,
)
from yakbox.voice_quality import qualify_audition_voices

EXPECTED = "The signal rang twice."


class _VoiceAligner:
    fingerprint = "a" * 64

    async def align(
        self, audio: Path, expected_text: str, *, language: str
    ) -> AlignmentResult:
        del expected_text, language
        words = (
            ("the", "signal", "rang", "twice")
            if audio.stem != "suspect"
            else ("a", "signal", "rang", "once")
        )
        tokens = tuple(
            AlignmentToken(word, 0.1 + index * 0.35, 0.35 + index * 0.35, 0.96)
            for index, word in enumerate(words)
        )
        return AlignmentResult(
            tokens=tokens,
            speech_regions=(SpeechRegion(0.1, 1.5),),
            backend="fake-whisper",
            model="fake",
            fingerprint=self.fingerprint,
            segments=(AlignmentSegment(0.1, 1.5, -0.1, 1.2, 0.01, 0.0),),
            language="en",
            transcript=" ".join(words),
            decode_passes=(
                DecodePassEvidence(
                    "authority", "", words, (), 0.96, audio.stem != "suspect"
                ),
                DecodePassEvidence(
                    "sampled_consensus", "", words, (), 0.96, audio.stem != "suspect"
                ),
            ),
            consensus_score=1.0,
            maximum_timing_delta_ms=0.0,
            prompt_sensitivity=(
                "stable_match" if audio.stem != "suspect" else "stable_mismatch"
            ),
        )


def _write_wav(path: Path) -> None:
    rate = 16_000
    samples = [
        round(2_500 * math.sin(2 * math.pi * 180 * index / rate))
        if round(rate * 0.1) <= index < round(rate * 1.5)
        else 0
        for index in range(rate * 2)
    ]
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        writer.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def _write_audition_report(path: Path, voices: tuple[str, ...]) -> None:
    comparisons = []
    for voice in voices:
        audio = path.parent / f"{voice}.wav"
        _write_wav(audio)
        comparisons.append(
            {
                "variant": voice,
                "artifact": {"path": f"artifacts/auditions/run/{audio.name}"},
            }
        )
    path.write_text(
        json.dumps(
            {
                "input_text_sha256": hashlib.sha256(EXPECTED.encode()).hexdigest(),
                "comparisons": comparisons,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_voice_qualification_marks_transcript_outlier_suspect(
    tmp_path: Path,
) -> None:
    audition = tmp_path / "audition.json"
    _write_audition_report(audition, ("baseline-a", "baseline-b", "good", "suspect"))

    report = await qualify_audition_voices(
        audition,
        EXPECTED,
        baseline_voices=("baseline-a", "baseline-b"),
        language="en",
        model="fake",
        revision=None,
        aligner=_VoiceAligner(),
    )
    payload = report.to_dict()
    by_voice = {item.voice: item for item in report.voices}

    assert by_voice["baseline-a"].status == "baseline"
    assert by_voice["good"].status == "high_quality"
    assert by_voice["suspect"].status == "suspect"
    assert "transcript_accuracy_below_baseline" in by_voice["suspect"].reason_codes
    assert payload["suspect_voices"] == ["suspect"]
    assert not report.accepted
    validate_contract("voice-quality", payload)


@pytest.mark.asyncio
async def test_voice_qualification_rejects_wrong_audition_text(tmp_path: Path) -> None:
    audition = tmp_path / "audition.json"
    _write_audition_report(audition, ("baseline-a", "baseline-b"))

    with pytest.raises(ValidationError, match="does not match"):
        await qualify_audition_voices(
            audition,
            "Different words.",
            baseline_voices=("baseline-a", "baseline-b"),
            language="en",
            model="fake",
            revision=None,
            aligner=_VoiceAligner(),
        )
