from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.unit.test_short_utterances import (
    _ExpectedTextAligner,
    _LongFakeSpeechService,
)

from yakbox._files import sha256_file
from yakbox.audiobook import load_manifest, repair
from yakbox.audiobook.repair_cache import RepairCacheEvent, RepairStageCache
from yakbox.errors import BuildError
from yakbox.speech.alignment import AlignmentResult
from yakbox.speech.models import SpeechSynthesisRequest


class _WindowAligner(_ExpectedTextAligner):
    async def align_window(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
        start_seconds: float,
        end_seconds: float,
    ) -> AlignmentResult:
        result = await self.align(audio, expected_text, language=language)
        return replace(
            result, clip_start_seconds=start_seconds, clip_end_seconds=end_seconds
        )


@pytest.fixture
def splice_inputs(
    book_workspace: Path,
) -> tuple[repair.AudiobookManifest, repair.RepairPlan]:
    path = book_workspace / "yakbox.toml"
    path.write_text(path.read_text().replace("chunk_chars = 100", "chunk_chars = 500"))
    source = book_workspace / "source/book.md"
    source.write_text("# One\n\nHe entered. Micah spoke softly. The room listened.\n")
    manifest = load_manifest(path)
    plan = repair.plan_repair(
        manifest, text_match="Micah spoke", mode=repair.RepairMode.SENTENCE
    )
    assert plan.chunks[0].replacement_text == "Micah spoke softly."
    return manifest, plan


@pytest.mark.asyncio
async def test_subchunk_repair_splices_real_pcm_and_reuses_verified_dsp_stages(
    splice_inputs: tuple[repair.AudiobookManifest, repair.RepairPlan],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, plan = splice_inputs
    chunk = plan.chunks[0]
    service = _LongFakeSpeechService()
    original = manifest.root / "original.wav"
    request = SpeechSynthesisRequest(text=chunk.text, voice="narrator")
    await service.synthesize_to_file(request, original)
    original_digest = sha256_file(original)
    monkeypatch.setattr(repair, "_current_chunk_audio", lambda *_args: original)
    qa = AsyncMock(return_value=())
    monkeypatch.setattr(repair, "_qa_direct_candidate", qa)
    joins = AsyncMock(return_value=SimpleNamespace(accepted=True))
    monkeypatch.setattr(repair, "inspect_joins", joins)
    cache = RepairStageCache(manifest.root / ".yakbox/cache/repair-stages")
    events: list[RepairCacheEvent] = []
    outputs: list[Path] = []
    for number in (1, 2):
        destination = manifest.root / f"take-{number}.wav"
        await repair._synthesize_subchunk_candidate(
            manifest,
            plan,
            chunk,
            request=request,
            service=service,
            aligner=_WindowAligner(),
            destination=destination,
            stage_cache=cache,
            cache_events=events,
        )
        outputs.append(destination)
        evidence = json.loads(
            destination.with_name(f"take-{number}-splice.json").read_text()
        )
        assert evidence["reconstructed_qa"]["accepted"]
        assert evidence["join_qa"]["accepted"]
        assert not destination.with_name(f"take-{number}-carrier.wav").exists()
        assert not destination.with_name(f"take-{number}-replacement.wav").exists()
    assert sha256_file(outputs[0]) == sha256_file(outputs[1])
    assert sha256_file(original) == original_digest
    assert [event.hit for event in events] == [False, False, True, True]
    assert qa.await_count == 4
    assert joins.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["replacement", "reconstructed", "join"])
async def test_subchunk_repair_refuses_failed_quality_gates_and_cleans_temporary_audio(
    splice_inputs: tuple[repair.AudiobookManifest, repair.RepairPlan],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    manifest, plan = splice_inputs
    chunk = plan.chunks[0]
    service = _LongFakeSpeechService()
    original = manifest.root / "original.wav"
    request = SpeechSynthesisRequest(text=chunk.text, voice="narrator")
    await service.synthesize_to_file(request, original)
    monkeypatch.setattr(repair, "_current_chunk_audio", lambda *_args: original)
    reasons = (
        [("unverified",)]
        if failure == "replacement"
        else [(), ("unverified",)]
        if failure == "reconstructed"
        else [(), ()]
    )
    monkeypatch.setattr(repair, "_qa_direct_candidate", AsyncMock(side_effect=reasons))
    monkeypatch.setattr(
        repair, "inspect_joins", AsyncMock(return_value=SimpleNamespace(accepted=False))
    )
    destination = manifest.root / "take.wav"
    with pytest.raises(BuildError, match=r"failed QA|unsafe audio join"):
        await repair._synthesize_subchunk_candidate(
            manifest,
            plan,
            chunk,
            request=request,
            service=service,
            aligner=_WindowAligner(),
            destination=destination,
            stage_cache=RepairStageCache(manifest.root / "cache"),
            cache_events=[],
        )
    assert not (manifest.root / "take-splice.json").exists()
    assert not (manifest.root / "take-carrier.wav").exists()
    assert not (manifest.root / "take-replacement.wav").exists()
    assert original.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
async def test_full_chunk_sentence_repair_still_requires_qa(
    splice_inputs: tuple[repair.AudiobookManifest, repair.RepairPlan],
    monkeypatch: pytest.MonkeyPatch,
    accepted: bool,
) -> None:
    manifest, plan = splice_inputs
    chunk = replace(plan.chunks[0], replacement_text=plan.chunks[0].text)
    monkeypatch.setattr(
        repair,
        "_qa_direct_candidate",
        AsyncMock(return_value=() if accepted else ("unverified",)),
    )
    destination = manifest.root / "whole.wav"
    expectation = (
        nullcontext() if accepted else pytest.raises(BuildError, match="failed QA")
    )
    with expectation:
        await repair._synthesize_subchunk_candidate(
            manifest,
            plan,
            chunk,
            request=SpeechSynthesisRequest(text=chunk.text, voice="narrator"),
            service=_LongFakeSpeechService(),
            aligner=_WindowAligner(),
            destination=destination,
            stage_cache=RepairStageCache(manifest.root / "cache"),
            cache_events=[],
        )
    assert destination.is_file()
