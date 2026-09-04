from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator

from yakbox.errors import ValidationError
from yakbox.schemas import load_schema
from yakbox.speech.analysis_corpus_boundaries import (
    build_boundary_review_authoring,
    load_approved_corpus_boundaries,
    write_approved_boundary_report,
    write_boundary_review_authoring,
)
from yakbox.speech.analysis_corpus_partition import (
    CorpusPartitionAssignment,
    FrozenCorpusPartition,
    QualificationPartition,
)
from yakbox.speech.analysis_corpus_sources import (
    CorpusSourceInventory,
    CorpusSourceWindow,
)
from yakbox.speech.analysis_corpus_truth import (
    ApprovedCorpusTranscripts,
    ApprovedTranscriptCase,
)
from yakbox.speech.analysis_fingerprints import semantic_fingerprint, text_fingerprint
from yakbox.speech.analysis_models import (
    AlignmentPurpose,
    AudioSpan,
    ConversionIdentity,
    ExecutionIdentity,
    ForcedAlignmentResult,
    ForcedAlignmentUnit,
    ModelArtifactIdentity,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _sha(label: str) -> str:
    return semantic_fingerprint("test-boundary-truth-v1", label)


def _inputs() -> tuple[
    ApprovedCorpusTranscripts,
    FrozenCorpusPartition,
    dict[str, ForcedAlignmentResult],
]:
    windows = tuple(
        CorpusSourceWindow(
            f"voice-{index}-passage-01-window-01",
            f"voice-{index}-passage-01",
            f"voice-{index}",
            f"Reader {index}",
            f"cache/windows/voice-{index}.wav",
            _sha(f"audio-{index}"),
            16_000,
            48_000,
            _sha(f"canonical-{index}"),
            0,
            48_000,
            120_000,
            123_000,
            f"https://example.invalid/voice-{index}.wav",
            _sha(f"source-{index}"),
            "LicenseRef-LibriVox-Public-Domain-US",
            "https://librivox.org/pages/public-domain/",
        )
        for index in range(2)
    )
    inventory = CorpusSourceInventory(SHA_A, 1, windows)
    tokens = ("approved", "sample")
    token_hash = text_fingerprint("\u001f".join(tokens))
    transcripts = ApprovedCorpusTranscripts(
        inventory.fingerprint,
        SHA_B,
        tuple(
            ApprovedTranscriptCase(
                window.source_window_id,
                window.source_passage_group,
                window.voice,
                window.reader,
                window.relative_audio_path,
                window.audio_digest,
                "Approved sample.",
                tokens,
                token_hash,
                SHA_A,
                _sha(f"draft-{index}"),
            )
            for index, window in enumerate(windows)
        ),
    )
    partition = FrozenCorpusPartition(
        inventory.fingerprint,
        transcripts.fingerprint,
        SHA_C,
        1,
        (
            CorpusPartitionAssignment(
                windows[0].source_window_id,
                windows[0].source_passage_group,
                windows[0].voice,
                QualificationPartition.CALIBRATION,
            ),
            CorpusPartitionAssignment(
                windows[1].source_window_id,
                windows[1].source_passage_group,
                windows[1].voice,
                QualificationPartition.HELD_OUT,
            ),
        ),
    )
    alignments = {
        item.source_window_id: _alignment(item.audio_digest, token_hash)
        for item in transcripts.cases
    }
    return transcripts, partition, alignments


def _alignment(audio_digest: str, lexical_hash: str) -> ForcedAlignmentResult:
    model = ModelArtifactIdentity(
        "qwen-forced",
        "mlx-audio",
        "1.0",
        3,
        2,
        "example/qwen-forced",
        "1" * 40,
        SHA_A,
        "upstream/qwen-forced",
        "2" * 40,
        ConversionIdentity("upstream", "tool", "1", SHA_B, "bf16", True),
        "bf16",
        SHA_C,
    )
    execution = ExecutionIdentity(
        SHA_A,
        SHA_B,
        "3.14.0",
        "Darwin",
        "26.0",
        "arm64",
        "1.0",
        None,
        "test",
        "argmax",
        (),
    )
    return ForcedAlignmentResult(
        "qwen-forced",
        model,
        execution,
        AudioSpan(audio_digest, 0, 48_000, 16_000),
        AlignmentPurpose.VERIFIED_TARGET,
        text_fingerprint("Approved sample."),
        lexical_hash,
        (
            ForcedAlignmentUnit(text_fingerprint("approved"), 1_000, 10_000),
            ForcedAlignmentUnit(text_fingerprint("sample"), 11_000, 20_000),
        ),
        1.0,
        (),
    )


def _approve(raw: dict[str, object]) -> None:
    raw["review_status"] = "approved"
    cases = cast(list[dict[str, object]], raw["cases"])
    for case in cases:
        case["review_status"] = "approved"
        tokens = cast(list[dict[str, object]], case["tokens"])
        first = [
            {
                "token_index": token["token_index"],
                "start_frame": token["proposed_start_frame"],
                "end_frame": token["proposed_end_frame"],
            }
            for token in tokens
        ]
        second = [
            {
                **boundary,
                "start_frame": cast(int, boundary["start_frame"]) + 10,
                "end_frame": cast(int, boundary["end_frame"]) + 10,
            }
            for boundary in first
        ]
        case["passes"] = [
            {"pass_index": 1, "reviewer_fingerprint": SHA_A, "boundaries": first},
            {"pass_index": 2, "reviewer_fingerprint": SHA_B, "boundaries": second},
        ]
        case["accepted_boundaries"] = first
        case["adjudicator_fingerprint"] = SHA_A


def test_boundary_truth_requires_two_passes_and_records_disagreement(
    tmp_path: Path,
) -> None:
    transcripts, partition, alignments = _inputs()
    selected = (transcripts.cases[0].source_window_id,)
    raw = build_boundary_review_authoring(
        transcripts,
        partition,
        alignments,
        case_ids=selected,
    )
    authoring = tmp_path / "boundary-authoring.json"
    write_boundary_review_authoring(authoring, raw)
    Draft202012Validator(load_schema("speech-corpus-boundary-authoring")).validate(raw)

    with pytest.raises(ValidationError, match="approval metadata"):
        load_approved_corpus_boundaries(
            authoring,
            transcripts=transcripts,
            partition=partition,
        )

    _approve(raw)
    authoring.write_text(json.dumps(raw), encoding="utf-8")
    truth = load_approved_corpus_boundaries(
        authoring,
        transcripts=transcripts,
        partition=partition,
    )
    report_path = tmp_path / "boundary-truth.json"
    write_approved_boundary_report(report_path, truth)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    Draft202012Validator(load_schema("speech-corpus-boundary-truth")).validate(report)

    assert report["cases"][0]["boundary_disagreement"] == [
        {"token_index": 0, "start_difference_frames": 10, "end_difference_frames": 10},
        {"token_index": 1, "start_difference_frames": 10, "end_difference_frames": 10},
    ]
    assert "approved" not in json.dumps(report).casefold()
    assert "Approved sample." not in json.dumps(report)


def test_boundary_truth_rejects_missing_second_pass(tmp_path: Path) -> None:
    transcripts, partition, alignments = _inputs()
    raw = build_boundary_review_authoring(
        transcripts,
        partition,
        alignments,
        case_ids=(transcripts.cases[0].source_window_id,),
    )
    _approve(raw)
    cases = cast(list[dict[str, object]], raw["cases"])
    cases[0]["passes"] = cast(list[object], cases[0]["passes"])[:1]
    authoring = tmp_path / "boundary-authoring.json"
    authoring.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValidationError, match="requires two review passes"):
        load_approved_corpus_boundaries(
            authoring,
            transcripts=transcripts,
            partition=partition,
        )
