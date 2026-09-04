from __future__ import annotations

import math
import wave
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema import ValidationError as SchemaError

from yakbox._files import sha256_file
from yakbox.audiobook.repair_cache import RepairStageCache
from yakbox.errors import ValidationError
from yakbox.schemas import load_schema
from yakbox.speech.analysis_models import SpeechVerification, VerificationScope
from yakbox.speech.analysis_repair import (
    AnalyzedArtifactState,
    ApprovedReplacementTake,
    ChangedChunkEvidence,
    PostMasterWindowEvidence,
    RepairJoinEvidence,
    assemble_repair_batch,
    plan_repair_batch,
    qualify_repair_candidate,
    repair_candidate_output_path,
    repair_candidate_report,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def _write_wav(path: Path, *, frames: int, frequency: float) -> None:
    samples = tuple(
        round(4_000 * math.sin(2 * math.pi * frequency * index / 1_000))
        for index in range(frames)
    )
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(1_000)
        writer.writeframes(
            b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)
        )


def _approval(
    repair_id: str,
    base: Path,
    replacement: Path,
    *,
    start: int,
    end: int,
    chunk_id: str,
) -> ApprovedReplacementTake:
    forced = SHA_D
    verification = SpeechVerification(
        SHA_A,
        SHA_B,
        forced,
        SHA_C,
        sha256_file(replacement),
        VerificationScope.CANDIDATE,
        True,
        (),
    )
    return ApprovedReplacementTake(
        repair_id,
        1,
        chunk_id,
        sha256_file(base),
        start,
        end,
        replacement,
        sha256_file(replacement),
        ("whisper", "parakeet", "qwen"),
        SHA_A,
        SHA_B,
        forced,
        SHA_E,
        SHA_F,
        SHA_C,
        verification,
    )


def test_multi_repair_plan_is_order_independent_and_reconstructs_once(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.wav"
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    _write_wav(base, frames=2_000, frequency=20)
    _write_wav(first_audio, frames=180, frequency=50)
    _write_wav(second_audio, frames=220, frequency=70)
    first = _approval(
        "repair-a", base, first_audio, start=300, end=500, chunk_id="chunk-a"
    )
    second = _approval(
        "repair-b", base, second_audio, start=1_200, end=1_450, chunk_id="chunk-b"
    )

    plan = plan_repair_batch(
        (second, first),
        sample_rate=1_000,
        qa_margin_frames=100,
        splice_policy_fingerprint=SHA_A,
    )
    reordered = plan_repair_batch(
        (first, second),
        sample_rate=1_000,
        qa_margin_frames=100,
        splice_policy_fingerprint=SHA_A,
    )

    assert plan.fingerprint == reordered.fingerprint
    assert tuple(item.repair_id for item in plan.replacements) == (
        "repair-a",
        "repair-b",
    )
    assert plan.affected_chunk_ids == ("chunk-a", "chunk-b")
    assert len(plan.qa_windows) == 2
    assert plan.state is AnalyzedArtifactState.REPAIR_CANDIDATE

    cache = RepairStageCache(tmp_path / "cache")
    candidate = assemble_repair_batch(
        plan,
        base_audio=base,
        destination=tmp_path / "candidate.wav",
        cache=cache,
    )
    repeated = assemble_repair_batch(
        reordered,
        base_audio=base,
        destination=tmp_path / "candidate-repeated.wav",
        cache=cache,
    )

    assert candidate.state is AnalyzedArtifactState.REPAIR_CANDIDATE
    assert candidate.cache_miss_stage == "reconstruction"
    assert len(candidate.splice_evidence) == 2
    assert repeated.cache_event is not None and repeated.cache_event.hit
    assert repeated.splice_evidence == ()
    assert repeated.output_digest == candidate.output_digest

    joins = tuple(
        RepairJoinEvidence(join_id, SHA_B, True) for join_id in plan.affected_join_ids
    )
    windows = tuple(
        PostMasterWindowEvidence(
            window.start_frame,
            window.end_frame,
            window.start_frame + 1,
            window.end_frame + 1,
            window.repair_ids,
            SpeechVerification(
                SHA_A,
                SHA_B,
                SHA_D,
                SHA_C,
                candidate.output_digest,
                VerificationScope.CANDIDATE,
                True,
                (),
            ),
        )
        for window in plan.qa_windows
    )
    changed_chunks = tuple(
        ChangedChunkEvidence(
            chunk_id,
            SpeechVerification(
                SHA_A,
                SHA_B,
                SHA_D,
                SHA_C,
                candidate.output_digest,
                VerificationScope.CANDIDATE,
                True,
                (),
            ),
        )
        for chunk_id in plan.affected_chunk_ids
    )
    qualification = qualify_repair_candidate(
        candidate,
        changed_chunks=changed_chunks,
        joins=joins,
        post_master_windows=windows,
        technical_fingerprint=SHA_A,
        technical_accepted=True,
    )
    assert qualification.accepted
    assert qualification.state is AnalyzedArtifactState.REPAIR_CANDIDATE
    candidate_path = repair_candidate_output_path(tmp_path, plan.fingerprint)
    assert ".yakbox/repair-candidates" in candidate_path.as_posix()
    assert "/release/" not in candidate_path.as_posix()
    report = repair_candidate_report(qualification)
    Draft202012Validator(load_schema("speech-repair-candidate-verification")).validate(
        report
    )
    with pytest.raises(SchemaError):
        Draft202012Validator(load_schema("speech-release-verification")).validate(
            report
        )

    with pytest.raises(ValidationError, match="join evidence"):
        qualify_repair_candidate(
            candidate,
            changed_chunks=changed_chunks,
            joins=joins[:-1],
            post_master_windows=windows,
            technical_fingerprint=SHA_A,
            technical_accepted=True,
        )


def test_multi_repair_rejects_overlap_before_writing(tmp_path: Path) -> None:
    base = tmp_path / "base.wav"
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    _write_wav(base, frames=1_000, frequency=20)
    _write_wav(first_audio, frames=100, frequency=50)
    _write_wav(second_audio, frames=100, frequency=70)
    first = _approval(
        "repair-a", base, first_audio, start=200, end=400, chunk_id="chunk-a"
    )
    overlapping = _approval(
        "repair-b", base, second_audio, start=350, end=500, chunk_id="chunk-a"
    )

    with pytest.raises(ValidationError, match="overlap"):
        plan_repair_batch(
            (overlapping, first),
            sample_rate=1_000,
            qa_margin_frames=100,
            splice_policy_fingerprint=SHA_A,
        )

    assert not (tmp_path / "candidate.wav").exists()


def test_approved_take_requires_independent_recognition_and_forced_timing(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.wav"
    replacement = tmp_path / "replacement.wav"
    _write_wav(base, frames=1_000, frequency=20)
    _write_wav(replacement, frames=100, frequency=50)
    verification = SpeechVerification(
        SHA_A,
        SHA_B,
        SHA_D,
        SHA_C,
        sha256_file(replacement),
        VerificationScope.CANDIDATE,
        True,
        (),
    )

    with pytest.raises(ValidationError, match="Whisper and Qwen"):
        ApprovedReplacementTake(
            "repair",
            1,
            "chunk",
            sha256_file(base),
            100,
            200,
            replacement,
            sha256_file(replacement),
            ("parakeet",),
            SHA_A,
            SHA_B,
            SHA_D,
            SHA_E,
            SHA_F,
            SHA_C,
            verification,
        )
