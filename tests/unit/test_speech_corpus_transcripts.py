from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator

from yakbox._files import sha256_file
from yakbox.errors import ValidationError
from yakbox.schemas import load_schema
from yakbox.speech.analysis_corpus_review import main as review_main
from yakbox.speech.analysis_corpus_sources import (
    CorpusSourceInventory,
    prepare_corpus_source_inventory,
    write_corpus_source_inventory,
)
from yakbox.speech.analysis_corpus_text_anchors import (
    build_corpus_text_anchoring,
    corpus_text_anchor_review_markdown,
    write_corpus_text_anchor_report,
)
from yakbox.speech.analysis_corpus_text_sources import (
    CorpusTextSource,
    CorpusTextSourceInventory,
)
from yakbox.speech.analysis_corpus_transcripts import (
    TranscriptAgreement,
    build_corpus_transcript_draft,
    collect_engine_recognitions,
    corpus_transcript_review_markdown,
    write_corpus_transcript_draft,
)
from yakbox.speech.analysis_corpus_truth import (
    approve_corpus_transcript_case,
    corpus_transcript_review_progress,
    load_approved_corpus_transcripts,
    load_corpus_transcript_review_draft,
    write_approved_transcript_report,
)
from yakbox.speech.analysis_fingerprints import text_fingerprint
from yakbox.speech.analysis_models import (
    AudioSpan,
    ConversionIdentity,
    ExecutionIdentity,
    ModelArtifactIdentity,
    ParakeetEvidence,
    QwenEvidence,
    RecognitionResult,
    RecognitionToken,
    ScoreKind,
    WhisperEvidence,
)
from yakbox.speech.analysis_services import FakeSpeechRecognizer

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _voice(path: Path) -> None:
    rate = 16_000
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        for second in range(20):
            amplitude = 0 if second in {6, 13} else 2_000
            writer.writeframesraw(
                b"".join(
                    int(amplitude).to_bytes(2, "little", signed=True)
                    for _ in range(rate)
                )
            )


def _registry(path: Path, audio: Path) -> None:
    digest = sha256_file(audio)
    path.write_text(
        f'''schema_version = 1
rights_policy = "Public domain only"

[voices.reader-one]
file = "voice.wav"
sha256 = "{digest}"
reader = "Reader One"
reader_url = "https://example.invalid/reader"
source_work = "Example"
catalog_url = "https://example.invalid/catalog"
source_url = "https://example.invalid/source.wav"
source_sha256 = "{SHA_A}"
license_id = "LicenseRef-LibriVox-Public-Domain-US"
rights_url = "https://librivox.org/pages/public-domain/"
source_start_seconds = 0
duration_seconds = 20
sample_rate_hz = 16000
channels = 1
pcm_bits = 16
filters = []
''',
        encoding="utf-8",
    )


def _inventory(tmp_path: Path) -> tuple[CorpusSourceInventory, Path]:
    audio = tmp_path / "voice.wav"
    registry = tmp_path / "voices.toml"
    output = tmp_path / "corpus"
    _voice(audio)
    _registry(registry, audio)
    return (
        prepare_corpus_source_inventory(
            registry,
            repository_root=tmp_path,
            output_root=output,
        ),
        output,
    )


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
        "test",
        "greedy",
        (),
    )


def _model(engine: str) -> ModelArtifactIdentity:
    return ModelArtifactIdentity(
        engine,
        f"{engine}-package",
        "1.0",
        1,
        2,
        f"example/{engine}",
        "1" * 40,
        SHA_A,
        f"upstream/{engine}",
        "2" * 40,
        ConversionIdentity("upstream", "tool", "1", SHA_B, "bf16", True),
        "bf16",
        SHA_C,
    )


def _evidence(engine: str) -> WhisperEvidence | ParakeetEvidence | QwenEvidence:
    if engine == "whisper":
        return WhisperEvidence(-0.1, 1.0, 0.01, 0.0)
    if engine == "parakeet":
        return ParakeetEvidence(0.99, "greedy", 1_920_000, 240_000)
    return QwenEvidence("stop", 0, 2)


def _recognition(
    engine: str,
    span: AudioSpan,
    tokens: tuple[str, ...],
) -> RecognitionResult:
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
            RecognitionToken(token, None, None, None, ScoreKind.UNAVAILABLE, SHA_C)
            for token in tokens
        ),
        _evidence(engine),
        (),
    )


def _authoring_source(
    tmp_path: Path,
    *,
    tokens_by_engine: dict[str, tuple[str, ...]],
) -> tuple[CorpusSourceInventory, Path, Path]:
    inventory, output = _inventory(tmp_path)
    recognitions = {
        engine: {
            window.source_window_id: _recognition(
                engine,
                AudioSpan(
                    window.audio_digest,
                    0,
                    window.frame_count,
                    window.sample_rate,
                ),
                tokens,
            )
            for window in inventory.windows
        }
        for engine, tokens in tokens_by_engine.items()
    }
    draft = build_corpus_transcript_draft(inventory, recognitions)
    authoring = output / "transcript-authoring.json"
    write_corpus_transcript_draft(
        authoring,
        output / "transcript-agreement.json",
        draft,
    )
    return inventory, output, authoring


def test_draft_records_exact_agreement_without_granting_truth(tmp_path: Path) -> None:
    inventory, output = _inventory(tmp_path)
    window_ids = tuple(item.source_window_id for item in inventory.windows)
    token_sets = (
        dict.fromkeys(("parakeet", "qwen", "whisper"), ("alpha",)),
        {"parakeet": ("beta",), "qwen": ("beta",), "whisper": ("better",)},
        {"parakeet": ("gamma",), "qwen": ("delta",), "whisper": ("epsilon",)},
    )
    drafts = []
    for tokens in token_sets:
        recognitions = {
            engine: {
                window.source_window_id: _recognition(
                    engine,
                    AudioSpan(
                        window.audio_digest,
                        0,
                        window.frame_count,
                        window.sample_rate,
                    ),
                    values,
                )
                for window in inventory.windows
            }
            for engine, values in tokens.items()
        }
        drafts.append(build_corpus_transcript_draft(inventory, recognitions))
    draft = drafts[0]
    authoring = output / "transcript-draft.json"
    report = output / "transcript-agreement.json"
    write_corpus_transcript_draft(authoring, report, draft)

    assert tuple(item.source_window_id for item in draft.cases) == window_ids
    assert tuple(item.cases[0].agreement for item in drafts) == (
        TranscriptAgreement.UNANIMOUS,
        TranscriptAgreement.MAJORITY,
        TranscriptAgreement.DISSENT,
    )
    assert draft.cases[0].proposed_text == "alpha"
    assert drafts[1].cases[0].proposed_text == "beta"
    assert drafts[2].cases[0].proposed_text == ""
    authoring_raw = json.loads(authoring.read_text(encoding="utf-8"))
    report_raw = json.loads(report.read_text(encoding="utf-8"))
    Draft202012Validator(load_schema("speech-corpus-transcript-draft")).validate(
        authoring_raw
    )
    Draft202012Validator(load_schema("speech-corpus-transcript-agreement")).validate(
        report_raw
    )
    assert "alpha" in json.dumps(authoring_raw)
    assert "alpha" not in json.dumps(report_raw)
    assert authoring_raw["cases"][0]["accepted_text"] == "alpha"
    dissent_cases = drafts[2].to_authoring_dict()["cases"]
    assert isinstance(dissent_cases, list)
    dissent_case = cast(dict[str, object], dissent_cases[0])
    assert dissent_case["accepted_text"] == ""
    review = corpus_transcript_review_markdown(
        draft,
        audio_prefix="corpus-sources",
    )
    assert window_ids[0] in review
    assert "## Reader One" in review
    assert "Voice key: `reader-one`" in review
    assert "Your task:" in review
    assert "](corpus-sources/cache/windows/" in review
    assert str(tmp_path) not in review

    with pytest.raises(ValidationError, match="approval metadata"):
        load_approved_corpus_transcripts(authoring, inventory=inventory)

    authoring_raw["review_status"] = "approved"
    for case in authoring_raw["cases"]:
        case["review_status"] = "approved"
        case["reviewer_fingerprint"] = SHA_A
        if not case["accepted_text"]:
            case["accepted_text"] = case["candidates"][0]["text"]
    authoring.write_text(json.dumps(authoring_raw), encoding="utf-8")
    approved = load_approved_corpus_transcripts(authoring, inventory=inventory)
    truth_report = output / "transcript-truth.json"
    write_approved_transcript_report(truth_report, approved)
    truth_raw = json.loads(truth_report.read_text(encoding="utf-8"))
    Draft202012Validator(load_schema("speech-corpus-transcript-truth")).validate(
        truth_raw
    )
    assert len(approved.cases) == 1
    assert "alpha" not in json.dumps(truth_raw)


def test_two_models_must_match_one_unique_pinned_source_span(
    tmp_path: Path,
) -> None:
    inventory, output, authoring = _authoring_source(
        tmp_path,
        tokens_by_engine={
            "parakeet": ("the", "unique", "spoken", "passage"),
            "qwen": ("the", "unique", "spoken-passage"),
            "whisper": ("a", "different", "passage"),
        },
    )
    plain = output / "plain" / "reader-one.txt"
    raw = output / "raw" / "reader-one.txt"
    plain.parent.mkdir()
    raw.parent.mkdir()
    source_text = (
        "opening words "
        + "filler words " * 60
        + "The unique spoken-passage appears here. "
        + "closing words " * 10
    )
    plain.write_text(source_text, encoding="utf-8")
    raw.write_text(source_text, encoding="utf-8")
    source = CorpusTextSource(
        "reader-one",
        "Reader One",
        "Example",
        "https://example.invalid/catalog",
        "https://example.invalid/book.txt",
        "LicenseRef-Public-Domain-US",
        "https://librivox.org/pages/public-domain/",
        "raw/reader-one.txt",
        sha256_file(raw),
        raw.stat().st_size,
        "plain/reader-one.txt",
        sha256_file(plain),
        146,
    )
    text_sources = CorpusTextSourceInventory(SHA_A, None, 1, (source,))
    draft = load_corpus_transcript_review_draft(authoring, inventory=inventory)

    anchoring = build_corpus_text_anchoring(
        draft,
        text_sources,
        text_root=output,
        audit_size=1,
    )
    report = output / "source-anchors.json"
    write_corpus_text_anchor_report(report, anchoring)
    markdown = corpus_text_anchor_review_markdown(
        draft,
        anchoring,
        audio_prefix="corpus",
    )

    assert len(anchoring.anchors) == 1
    assert not anchoring.unresolved
    assert anchoring.anchors[0].accepted_text == "The unique spoken-passage"
    assert anchoring.anchors[0].matched_engines == ("parakeet", "qwen")
    assert anchoring.audit_case_ids == (inventory.windows[0].source_window_id,)
    raw_report = json.loads(report.read_text(encoding="utf-8"))
    Draft202012Validator(load_schema("speech-corpus-text-anchors")).validate(raw_report)
    assert "unique spoken passage" not in json.dumps(raw_report)
    assert "Review type: **source-anchor audit**" in markdown
    assert "The unique spoken-passage" in markdown


def test_source_anchoring_leaves_disagreement_for_human_review(
    tmp_path: Path,
) -> None:
    inventory, output, authoring = _authoring_source(
        tmp_path,
        tokens_by_engine={
            "parakeet": ("first", "version"),
            "qwen": ("second", "version"),
            "whisper": ("third", "version"),
        },
    )
    plain = output / "source.txt"
    source_text = "first version " + "filler words " * 60
    plain.write_text(source_text, encoding="utf-8")
    source = CorpusTextSource(
        "reader-one",
        "Reader One",
        "Example",
        "https://example.invalid/catalog",
        "https://example.invalid/book.txt",
        "LicenseRef-Public-Domain-US",
        "https://librivox.org/pages/public-domain/",
        "source.txt",
        sha256_file(plain),
        plain.stat().st_size,
        "source.txt",
        sha256_file(plain),
        122,
    )
    text_sources = CorpusTextSourceInventory(SHA_A, None, 1, (source,))
    draft = load_corpus_transcript_review_draft(authoring, inventory=inventory)

    anchoring = build_corpus_text_anchoring(
        draft,
        text_sources,
        text_root=output,
    )

    assert not anchoring.anchors
    assert anchoring.unresolved[0].reason == "insufficient-model-corroboration"


def test_engine_major_checkpoints_are_reused_and_corruption_is_recomputed(
    tmp_path: Path,
) -> None:
    inventory, output = _inventory(tmp_path)
    spans_by_path = {
        (output / item.relative_audio_path).resolve(): AudioSpan(
            item.audio_digest,
            0,
            item.frame_count,
            item.sample_rate,
        )
        for item in inventory.windows
    }
    calls: list[Path] = []

    def result_factory(
        path: Path,
        _language: str,
        span: AudioSpan | None,
    ) -> RecognitionResult:
        calls.append(path)
        assert span == spans_by_path[path.resolve()]
        return _recognition("parakeet", span, ("draft", "text"))

    recognizer = FakeSpeechRecognizer(SHA_A, result_factory)
    checkpoint_root = output / "checkpoints"
    first = asyncio.run(
        collect_engine_recognitions(
            inventory,
            engine="parakeet",
            recognizer=recognizer,
            audio_root=output,
            checkpoint_root=checkpoint_root,
        )
    )
    second = asyncio.run(
        collect_engine_recognitions(
            inventory,
            engine="parakeet",
            recognizer=recognizer,
            audio_root=output,
            checkpoint_root=checkpoint_root,
        )
    )
    assert second == first
    assert len(calls) == len(inventory.windows)

    corrupted = (
        checkpoint_root
        / "parakeet"
        / SHA_A
        / f"{inventory.windows[0].source_window_id}.json"
    )
    corrupted.write_text("{}\n", encoding="utf-8")
    asyncio.run(
        collect_engine_recognitions(
            inventory,
            engine="parakeet",
            recognizer=recognizer,
            audio_root=output,
            checkpoint_root=checkpoint_root,
        )
    )
    assert len(calls) == len(inventory.windows) + 1


def test_review_progress_and_approval_never_emit_transcript_text(
    tmp_path: Path,
) -> None:
    inventory, _output, authoring = _authoring_source(
        tmp_path,
        tokens_by_engine={
            "parakeet": ("private", "words"),
            "qwen": ("private", "words"),
            "whisper": ("other", "words"),
        },
    )
    case_id = inventory.windows[0].source_window_id

    pending = corpus_transcript_review_progress(authoring, inventory=inventory)
    approved = approve_corpus_transcript_case(
        authoring,
        inventory=inventory,
        source_window_id=case_id,
        reviewer_label="Local Reviewer",
    )

    assert pending.pending_case_ids == (case_id,)
    assert pending.review_status == "pending"
    assert approved.review_status == "approved"
    assert approved.pending_count == 0
    assert "private" not in json.dumps(pending.to_dict())
    raw = json.loads(authoring.read_text(encoding="utf-8"))
    assert raw["cases"][0]["accepted_text"] == "private words"
    assert raw["cases"][0]["reviewer_fingerprint"] not in {
        "",
        "Local Reviewer",
    }
    load_approved_corpus_transcripts(authoring, inventory=inventory)


def test_dissent_review_requires_private_corrected_text(tmp_path: Path) -> None:
    inventory, _output, authoring = _authoring_source(
        tmp_path,
        tokens_by_engine={
            "parakeet": ("first",),
            "qwen": ("second",),
            "whisper": ("third",),
        },
    )

    with pytest.raises(ValidationError, match="requires corrected text"):
        approve_corpus_transcript_case(
            authoring,
            inventory=inventory,
            source_window_id=inventory.windows[0].source_window_id,
            reviewer_label="Local Reviewer",
        )

    approved = approve_corpus_transcript_case(
        authoring,
        inventory=inventory,
        source_window_id=inventory.windows[0].source_window_id,
        reviewer_label="Local Reviewer",
        accepted_text="The corrected human transcript.",
    )
    assert approved.review_status == "approved"


def test_transcript_review_rejects_stale_expected_authoring_digest(
    tmp_path: Path,
) -> None:
    inventory, _output, authoring = _authoring_source(
        tmp_path,
        tokens_by_engine=dict.fromkeys(
            ("parakeet", "qwen", "whisper"),
            ("agreed",),
        ),
    )

    with pytest.raises(ValidationError, match="source changed"):
        approve_corpus_transcript_case(
            authoring,
            inventory=inventory,
            source_window_id=inventory.windows[0].source_window_id,
            reviewer_label="Local Reviewer",
            expected_authoring_digest=SHA_A,
        )
    assert (
        corpus_transcript_review_progress(
            authoring,
            inventory=inventory,
        ).pending_count
        == 1
    )


def test_internal_review_cli_uses_files_and_emits_text_free_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inventory, output, authoring = _authoring_source(
        tmp_path,
        tokens_by_engine={
            "parakeet": ("secret", "sentence"),
            "qwen": ("different",),
            "whisper": ("another",),
        },
    )
    inventory_path = output / "inventory.json"
    write_corpus_source_inventory(inventory_path, inventory)
    reviewer = tmp_path / "reviewer.txt"
    correction = tmp_path / "correction.txt"
    reviewer.write_text("Local Reviewer\n", encoding="utf-8")
    correction.write_text("Human approved wording.\n", encoding="utf-8")

    exit_code = review_main(
        [
            "--authoring",
            str(authoring),
            "--inventory",
            str(inventory_path),
            "--audio-root",
            str(output),
            "approve",
            inventory.windows[0].source_window_id,
            "--reviewer-label-file",
            str(reviewer),
            "--accepted-text-file",
            str(correction),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert report["review_status"] == "approved"
    assert report["pending_count"] == 0
    assert "secret" not in captured.out
    assert "Human approved" not in captured.out
    assert "Local Reviewer" not in captured.out


def test_internal_review_cli_writes_a_voice_labeled_packet(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inventory, output, authoring = _authoring_source(
        tmp_path,
        tokens_by_engine=dict.fromkeys(
            ("parakeet", "qwen", "whisper"),
            ("spoken", "words"),
        ),
    )
    inventory_path = output / "inventory.json"
    packet = output / "review.md"
    write_corpus_source_inventory(inventory_path, inventory)

    exit_code = review_main(
        [
            "--authoring",
            str(authoring),
            "--inventory",
            str(inventory_path),
            "--audio-root",
            str(output),
            "packet",
            "--output",
            str(packet),
            "--audio-prefix",
            "corpus",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    content = packet.read_text(encoding="utf-8")
    assert exit_code == 0
    assert report["review_packet"] == str(packet.resolve())
    assert "## Reader One" in content
    assert "Play Reader One" in content
    assert "CASE_ID: corrected text here" in content
