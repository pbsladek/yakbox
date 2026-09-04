"""Draft qualification-corpus transcripts from independent recognizers."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from yakbox._files import atomic_write_bytes, atomic_write_json, safe_child
from yakbox.contracts import runtime_metadata
from yakbox.errors import ValidationError, WorkerProtocolError
from yakbox.speech.analysis_adapters import (
    MlxAudioQwenRecognizer,
    MlxWhisperRecognizer,
    ParakeetMlxRecognizer,
)
from yakbox.speech.analysis_corpus_sources import (
    CorpusSourceInventory,
    CorpusSourceWindow,
    load_corpus_source_inventory,
)
from yakbox.speech.analysis_fingerprints import semantic_fingerprint, text_fingerprint
from yakbox.speech.analysis_models import AudioSpan, RecognitionResult
from yakbox.speech.analysis_runtime import WorkerBackedSpeechRecognizer
from yakbox.speech.analysis_runtime_identity import execution_identity_from_digests
from yakbox.speech.analysis_runtime_install import (
    AnalysisRuntimeInstaller,
    default_analysis_runtime_root,
)
from yakbox.speech.analysis_serialization import (
    recognition_from_report,
    recognition_report,
)
from yakbox.speech.analysis_services import SpeechRecognizer
from yakbox.speech.model_registry import ModelRegistry, default_model_root

_ENGINES = ("parakeet", "qwen", "whisper")
_MINIMUM_CONSENSUS_ENGINES = 2
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CALIBRATION_DRAFT_FINGERPRINT = "0" * 64


class TranscriptAgreement(StrEnum):
    """Exact lexical agreement available to help a human author corpus truth."""

    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    DISSENT = "dissent"
    UNUSABLE = "unusable"


@dataclass(frozen=True, slots=True)
class TranscriptCandidate:
    engine: str
    recognition_fingerprint: str
    model_fingerprint: str
    execution_fingerprint: str
    normalized_transcript_hash: str
    tokens: tuple[str, ...]
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.recognition_fingerprint, "recognition fingerprint"),
            (self.model_fingerprint, "model fingerprint"),
            (self.execution_fingerprint, "execution fingerprint"),
            (self.normalized_transcript_hash, "normalized transcript hash"),
        ):
            _require_sha256(value, label)
        if self.engine not in _ENGINES:
            raise ValidationError("Transcript candidate engine is unsupported")
        if any(not token or token != token.casefold() for token in self.tokens):
            raise ValidationError("Transcript candidate tokens are not normalized")
        if (
            text_fingerprint("\u001f".join(self.tokens))
            != self.normalized_transcript_hash
        ):
            raise ValidationError("Transcript candidate hash differs from its tokens")

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def eligible(self) -> bool:
        return bool(self.tokens) and not self.issues

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-transcript-candidate-v1", self)


@dataclass(frozen=True, slots=True)
class CorpusTranscriptDraftCase:
    source_window_id: str
    source_passage_group: str
    voice: str
    reader: str
    relative_audio_path: str
    audio_digest: str
    candidates: tuple[TranscriptCandidate, ...]
    agreement: TranscriptAgreement
    agreement_count: int
    proposed_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        engines = tuple(item.engine for item in self.candidates)
        if (
            not self.source_window_id
            or not self.source_passage_group
            or not self.voice
            or not self.reader
            or not self.relative_audio_path
            or engines != _ENGINES
            or not 0 <= self.agreement_count <= len(_ENGINES)
        ):
            raise ValidationError("Corpus transcript draft case is inconsistent")
        _require_sha256(self.audio_digest, "transcript draft audio digest")
        expected_agreement, expected_count, expected_tokens = _agreement(
            self.candidates
        )
        if (
            self.agreement is not expected_agreement
            or self.agreement_count != expected_count
            or self.proposed_tokens != expected_tokens
        ):
            raise ValidationError("Corpus transcript draft agreement differs")

    @property
    def proposed_text(self) -> str:
        return " ".join(self.proposed_tokens)

    @property
    def proposed_tokens_hash(self) -> str | None:
        if not self.proposed_tokens:
            return None
        return text_fingerprint("\u001f".join(self.proposed_tokens))

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-transcript-draft-case-v1", self)


@dataclass(frozen=True, slots=True)
class CorpusTranscriptDraft:
    source_inventory_fingerprint: str
    cases: tuple[CorpusTranscriptDraftCase, ...]

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_inventory_fingerprint,
            "source inventory fingerprint",
        )
        identifiers = tuple(item.source_window_id for item in self.cases)
        if not self.cases or identifiers != tuple(sorted(set(identifiers))):
            raise ValidationError("Corpus transcript draft cases are inconsistent")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-transcript-draft-v1", self)

    def to_authoring_dict(self) -> dict[str, object]:
        """Return the private corpus-authoring source with normalized text."""
        return {
            **runtime_metadata("speech-corpus-transcript-draft"),
            "fingerprint": self.fingerprint,
            "source_inventory_fingerprint": self.source_inventory_fingerprint,
            "review_status": "pending",
            "case_count": len(self.cases),
            "cases": [
                {
                    "source_window_id": item.source_window_id,
                    "source_passage_group": item.source_passage_group,
                    "voice": item.voice,
                    "reader": item.reader,
                    "relative_audio_path": item.relative_audio_path,
                    "audio_digest": item.audio_digest,
                    "agreement": item.agreement.value,
                    "agreement_count": item.agreement_count,
                    "proposed_text": item.proposed_text,
                    "accepted_text": item.proposed_text,
                    "review_status": "pending",
                    "reviewer_fingerprint": "",
                    "candidates": [
                        {
                            "engine": candidate.engine,
                            "text": candidate.text,
                            "recognition_fingerprint": (
                                candidate.recognition_fingerprint
                            ),
                            "model_fingerprint": candidate.model_fingerprint,
                            "execution_fingerprint": (candidate.execution_fingerprint),
                            "normalized_transcript_hash": (
                                candidate.normalized_transcript_hash
                            ),
                            "token_count": len(candidate.tokens),
                            "issues": list(candidate.issues),
                        }
                        for candidate in item.candidates
                    ],
                    "fingerprint": item.fingerprint,
                }
                for item in self.cases
            ],
        }

    def to_agreement_report(self) -> dict[str, object]:
        """Return text-free evidence about the draft's recognition agreement."""
        counts = Counter(item.agreement.value for item in self.cases)
        return {
            **runtime_metadata("speech-corpus-transcript-agreement"),
            "fingerprint": self.fingerprint,
            "source_inventory_fingerprint": self.source_inventory_fingerprint,
            "case_count": len(self.cases),
            "agreement_counts": {
                status.value: counts[status.value] for status in TranscriptAgreement
            },
            "cases": [
                {
                    "source_window_id": item.source_window_id,
                    "source_passage_group": item.source_passage_group,
                    "audio_digest": item.audio_digest,
                    "agreement": item.agreement.value,
                    "agreement_count": item.agreement_count,
                    "proposed_tokens_hash": item.proposed_tokens_hash,
                    "proposed_token_count": len(item.proposed_tokens),
                    "recognitions": [
                        {
                            "engine": candidate.engine,
                            "recognition_fingerprint": (
                                candidate.recognition_fingerprint
                            ),
                            "model_fingerprint": candidate.model_fingerprint,
                            "execution_fingerprint": (candidate.execution_fingerprint),
                            "normalized_transcript_hash": (
                                candidate.normalized_transcript_hash
                            ),
                            "token_count": len(candidate.tokens),
                            "issues": list(candidate.issues),
                        }
                        for candidate in item.candidates
                    ],
                    "fingerprint": item.fingerprint,
                }
                for item in self.cases
            ],
        }


async def collect_engine_recognitions(
    inventory: CorpusSourceInventory,
    *,
    engine: str,
    recognizer: SpeechRecognizer,
    audio_root: Path,
    checkpoint_root: Path,
) -> dict[str, RecognitionResult]:
    """Recognize one engine-major pass, reusing exact validated checkpoints."""
    if engine not in _ENGINES:
        raise ValidationError("Corpus transcript recognizer engine is unsupported")
    _require_sha256(recognizer.fingerprint, "corpus transcript recognizer fingerprint")
    audio_root = audio_root.resolve()
    cache = safe_child(
        checkpoint_root,
        checkpoint_root / engine / recognizer.fingerprint,
    )
    indexed = _recognition_checkpoint_index(cache, engine=engine)
    results: dict[str, RecognitionResult] = {}
    for window in inventory.windows:
        expected_span = AudioSpan(
            window.audio_digest,
            0,
            window.frame_count,
            window.sample_rate,
        )
        checkpoint = safe_child(cache, cache / f"{window.source_window_id}.json")
        result = _load_recognition_checkpoint(checkpoint)
        if result is not None and not _recognition_matches(
            result,
            engine=engine,
            span=expected_span,
        ):
            result = None
        if result is None:
            result = indexed.get(expected_span)
        if result is None:
            audio = safe_child(audio_root, audio_root / window.relative_audio_path)
            result = await recognizer.recognize(
                audio,
                language="en",
                span=expected_span,
            )
            _validate_recognition(result, engine=engine, span=expected_span)
            atomic_write_json(checkpoint, recognition_report(result))
        else:
            _validate_recognition(result, engine=engine, span=expected_span)
            if not checkpoint.exists():
                atomic_write_json(checkpoint, recognition_report(result))
        results[window.source_window_id] = result
    return results


def build_corpus_transcript_draft(
    inventory: CorpusSourceInventory,
    recognitions: Mapping[str, Mapping[str, RecognitionResult]],
) -> CorpusTranscriptDraft:
    """Build a deterministic authoring draft without granting transcript truth."""
    if set(recognitions) != set(_ENGINES):
        raise ValidationError("Corpus transcript draft requires all recognizers")
    expected_windows = {item.source_window_id for item in inventory.windows}
    if any(set(items) != expected_windows for items in recognitions.values()):
        raise ValidationError("Corpus transcript recognitions are incomplete")
    cases: list[CorpusTranscriptDraftCase] = []
    for window in inventory.windows:
        candidates = tuple(
            _candidate(
                recognitions[engine][window.source_window_id],
                engine=engine,
                window=window,
            )
            for engine in _ENGINES
        )
        agreement, count, proposed = _agreement(candidates)
        cases.append(
            CorpusTranscriptDraftCase(
                window.source_window_id,
                window.source_passage_group,
                window.voice,
                window.reader,
                window.relative_audio_path,
                window.audio_digest,
                candidates,
                agreement,
                count,
                proposed,
            )
        )
    return CorpusTranscriptDraft(inventory.fingerprint, tuple(cases))


def write_corpus_transcript_draft(
    authoring_path: Path,
    report_path: Path,
    draft: CorpusTranscriptDraft,
) -> None:
    """Atomically write the private authoring source and text-free report."""
    atomic_write_json(authoring_path, draft.to_authoring_dict())
    atomic_write_json(report_path, draft.to_agreement_report())


def corpus_transcript_review_markdown(
    draft: CorpusTranscriptDraft,
    *,
    audio_prefix: str,
) -> str:
    """Render a bounded listening worksheet without approving any transcript."""
    prefix = audio_prefix.strip("/")
    if not prefix or ".." in Path(prefix).parts:
        raise ValidationError("Corpus transcript review audio prefix is invalid")
    counts = Counter(item.agreement.value for item in draft.cases)
    lines = [
        "# Speech corpus transcript review",
        "",
        f"Draft fingerprint: `{draft.fingerprint}`",
        "",
        (
            f"There are {len(draft.cases)} original LibriVox clips from "
            f"{len({item.voice for item in draft.cases})} readers: "
            f"{counts['dissent']} need a manual transcript, {counts['majority']} "
            f"have a two-model proposal, and {counts['unanimous']} have a "
            "three-model proposal."
        ),
        "",
        "## What you are reviewing",
        "",
        "Each section names the human reader whose voice is in the WAV.",
        "",
        (
            "For each clip, decide which words the reader actually says. Ignore "
            "capitalization and sentence punctuation. Do check wording, word "
            "order, contractions, names, numbers, missing words, and extra words."
        ),
        "",
        (
            "If the audio starts or ends in the middle of a word, or you cannot "
            "transcribe it confidently, mark the clip `reselect` instead of "
            "guessing."
        ),
        "",
        "Return one line per clip in one of these forms:",
        "",
        "- `CASE_ID: approve` — the proposed transcript is exact.",
        "- `CASE_ID: corrected text here` — use the exact words you hear.",
        "- `CASE_ID: reselect — reason` — the clip cannot provide reliable truth.",
        "",
        (
            "A model proposal is only a shortcut. It does not become ground truth "
            "until you listen and approve it."
        ),
    ]
    priority = {
        TranscriptAgreement.DISSENT: 0,
        TranscriptAgreement.MAJORITY: 1,
        TranscriptAgreement.UNANIMOUS: 2,
        TranscriptAgreement.UNUSABLE: 3,
    }
    cases_by_voice: dict[str, list[CorpusTranscriptDraftCase]] = defaultdict(list)
    for item in draft.cases:
        cases_by_voice[item.voice].append(item)
    for voice in sorted(cases_by_voice):
        voice_cases = cases_by_voice[voice]
        reader = voice_cases[0].reader
        lines.extend(("", f"## {reader}", "", f"Voice key: `{voice}`"))
        ordered = sorted(
            voice_cases,
            key=lambda item: (priority[item.agreement], item.source_window_id),
        )
        for item in ordered:
            audio = f"{prefix}/{item.relative_audio_path}"
            task = _transcript_review_task(item.agreement)
            passage = item.source_passage_group.removeprefix(f"{voice}-").replace(
                "-", " "
            )
            lines.extend(
                (
                    "",
                    f"### {passage.title()}",
                    "",
                    f"Clip ID: `{item.source_window_id}`",
                    "",
                    f"[Play {reader} — {passage.title()}]({audio})",
                    "",
                    f"Your task: {task}",
                    "",
                    (
                        f"Model agreement: `{item.agreement.value}` "
                        f"({item.agreement_count} matching models)"
                    ),
                    "",
                    f"Proposed transcript: `{item.proposed_text or '[none]'}`",
                    "",
                    "Model transcripts:",
                    "",
                )
            )
            lines.extend(
                f"- {candidate.engine}: `{candidate.text or '[no tokens]'}`"
                for candidate in item.candidates
            )
            decision = (
                f"`{item.source_window_id}: approve`"
                if item.proposed_text
                else f"`{item.source_window_id}: corrected text here`"
            )
            lines.extend(("", f"Decision to return: {decision} or `reselect`."))
    return "\n".join(lines) + "\n"


def _transcript_review_task(agreement: TranscriptAgreement) -> str:
    if agreement is TranscriptAgreement.UNANIMOUS:
        return "All three models agree. Confirm that the proposal is exact."
    if agreement is TranscriptAgreement.MAJORITY:
        return (
            "Two models agree. Compare the proposal with the audio; the outlier "
            "may still be right."
        )
    if agreement is TranscriptAgreement.DISSENT:
        return "The models disagree. Write the exact words you hear."
    return "The models produced no usable transcript. Transcribe or mark `reselect`."


def write_corpus_transcript_review(
    path: Path,
    draft: CorpusTranscriptDraft,
    *,
    audio_prefix: str,
) -> None:
    content = corpus_transcript_review_markdown(draft, audio_prefix=audio_prefix)
    atomic_write_bytes(path, content.encode(), overwrite=True)


async def draft_with_managed_runtimes(
    *,
    inventory_path: Path,
    audio_root: Path,
    checkpoint_root: Path,
    authoring_path: Path,
    report_path: Path,
    review_path: Path | None = None,
    review_audio_prefix: str = "corpus-sources",
    runtime_root: Path | None = None,
    model_root: Path | None = None,
) -> CorpusTranscriptDraft:
    """Run the three verified managed recognizers sequentially and checkpoint."""
    audio_root = audio_root.resolve()
    inventory = load_corpus_source_inventory(inventory_path, audio_root=audio_root)
    installer = AnalysisRuntimeInstaller(
        runtime_root or default_analysis_runtime_root()
    )
    runtime_report = installer.verify(_ENGINES)
    if not runtime_report.verified:
        raise ValidationError("Every managed analysis runtime must be verified")
    registry = ModelRegistry(model_root or default_model_root())
    if not all(registry.status(engine).verified for engine in _ENGINES):
        raise ValidationError("Every default recognition model must be verified")
    runtime_by_family = {item.family: item for item in runtime_report.runtimes}
    recognitions: dict[str, Mapping[str, RecognitionResult]] = {}
    adapter_types = {
        "whisper": MlxWhisperRecognizer,
        "parakeet": ParakeetMlxRecognizer,
        "qwen": MlxAudioQwenRecognizer,
    }
    for engine in _ENGINES:
        runtime = runtime_by_family[engine]
        execution = execution_identity_from_digests(
            worker_artifact_digest=runtime.worker_artifact_digest,
            lock_digest=runtime.lock_digest,
        )
        adapter = adapter_types[engine](
            registry=registry,
            audio_root=audio_root,
            execution=execution,
            calibration_fingerprint=_CALIBRATION_DRAFT_FINGERPRINT,
        )
        worker = installer.create_worker(
            engine,
            audio_root=audio_root,
            model_root=registry.root,
            calibration_fingerprint=_CALIBRATION_DRAFT_FINGERPRINT,
        )
        recognizer = WorkerBackedSpeechRecognizer(
            engine=engine,
            worker=worker,
            audio_root=audio_root,
            adapter_fingerprint=adapter.fingerprint,
            timeout_seconds=300,
        )
        try:
            recognitions[engine] = await collect_engine_recognitions(
                inventory,
                engine=engine,
                recognizer=recognizer,
                audio_root=audio_root,
                checkpoint_root=checkpoint_root,
            )
        finally:
            await worker.close()
    draft = build_corpus_transcript_draft(inventory, recognitions)
    write_corpus_transcript_draft(authoring_path, report_path, draft)
    if review_path is not None:
        write_corpus_transcript_review(
            review_path,
            draft,
            audio_prefix=review_audio_prefix,
        )
    return draft


def _candidate(
    result: RecognitionResult,
    *,
    engine: str,
    window: CorpusSourceWindow,
) -> TranscriptCandidate:
    _validate_recognition(
        result,
        engine=engine,
        span=AudioSpan(
            window.audio_digest,
            0,
            window.frame_count,
            window.sample_rate,
        ),
    )
    return TranscriptCandidate(
        engine,
        result.fingerprint,
        result.model.fingerprint,
        result.execution.fingerprint,
        result.normalized_transcript_hash,
        tuple(token.text for token in result.tokens),
        result.issues,
    )


def _agreement(
    candidates: Sequence[TranscriptCandidate],
) -> tuple[TranscriptAgreement, int, tuple[str, ...]]:
    eligible = [item.tokens for item in candidates if item.eligible]
    if len(eligible) < _MINIMUM_CONSENSUS_ENGINES:
        return TranscriptAgreement.UNUSABLE, len(eligible), ()
    counts = Counter(eligible)
    proposed, count = min(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    if len(eligible) == len(_ENGINES) and count == len(_ENGINES):
        return TranscriptAgreement.UNANIMOUS, count, proposed
    if count >= _MINIMUM_CONSENSUS_ENGINES:
        return TranscriptAgreement.MAJORITY, count, proposed
    return TranscriptAgreement.DISSENT, count, ()


def _load_recognition_checkpoint(path: Path) -> RecognitionResult | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return recognition_from_report(value)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValidationError,
        WorkerProtocolError,
    ):
        return None


def _recognition_checkpoint_index(
    root: Path,
    *,
    engine: str,
) -> dict[AudioSpan, RecognitionResult]:
    if not root.is_dir():
        return {}
    indexed: dict[AudioSpan, RecognitionResult] = {}
    for path in sorted(root.glob("*.json")):
        result = _load_recognition_checkpoint(path)
        if result is None or result.engine != engine:
            continue
        previous = indexed.get(result.span)
        if previous is not None and previous != result:
            raise ValidationError("Corpus transcript checkpoints conflict for one span")
        indexed[result.span] = result
    return indexed


def _validate_recognition(
    result: RecognitionResult,
    *,
    engine: str,
    span: AudioSpan,
) -> None:
    if not _recognition_matches(result, engine=engine, span=span):
        raise ValidationError("Corpus transcript recognition identity differs")


def _recognition_matches(
    result: RecognitionResult,
    *,
    engine: str,
    span: AudioSpan,
) -> bool:
    return (
        result.engine == engine
        and result.requested_language == "en"
        and result.span == span
    )


def _require_sha256(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label.capitalize()} must be a SHA-256")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draft corpus transcripts with verified managed runtimes"
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--authoring-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--review-audio-prefix", default="corpus-sources")
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the internal qualification authoring command."""
    arguments = _parser().parse_args(argv)
    draft = asyncio.run(
        draft_with_managed_runtimes(
            inventory_path=arguments.inventory,
            audio_root=arguments.audio_root,
            checkpoint_root=arguments.checkpoint_root,
            authoring_path=arguments.authoring_output,
            report_path=arguments.report_output,
            review_path=arguments.review_output,
            review_audio_prefix=arguments.review_audio_prefix,
            runtime_root=arguments.runtime_root,
            model_root=arguments.model_root,
        )
    )
    counts = Counter(item.agreement.value for item in draft.cases)
    sys.stdout.write(
        json.dumps(
            {
                "case_count": len(draft.cases),
                "draft_fingerprint": draft.fingerprint,
                "agreement_counts": dict(sorted(counts.items())),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CorpusTranscriptDraft",
    "CorpusTranscriptDraftCase",
    "TranscriptAgreement",
    "TranscriptCandidate",
    "build_corpus_transcript_draft",
    "collect_engine_recognitions",
    "corpus_transcript_review_markdown",
    "draft_with_managed_runtimes",
    "main",
    "write_corpus_transcript_draft",
    "write_corpus_transcript_review",
]
