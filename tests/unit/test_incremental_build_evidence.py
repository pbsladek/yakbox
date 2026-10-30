from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.integration.test_repair_splice_pipeline import _WindowAligner

import yakbox.audiobook.build as builds
from yakbox._files import sha256_file
from yakbox.audiobook import load_manifest, normalize_sources, plan_audiobook
from yakbox.audiobook.manifest import AudiobookManifest
from yakbox.audiobook.planner import PlanNode
from yakbox.errors import BuildError
from yakbox.speech.models import SpeechSynthesisRequest
from yakbox.speech.services import FakeSpeechService
from yakbox.speech.waves import WavJoinPart


@pytest.fixture
def planned_chunks(book_workspace: Path) -> tuple[AudiobookManifest, PlanNode]:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    plan = plan_audiobook(manifest, normalize_sources(manifest.sources))
    return manifest, next(
        node for node in plan.nodes if node.stage.value == "synthesize"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
async def test_changed_region_verification_checks_speech_and_keeps_private_evidence(
    planned_chunks: tuple[AudiobookManifest, PlanNode],
    monkeypatch: pytest.MonkeyPatch,
    accepted: bool,
) -> None:
    manifest, node = planned_chunks
    parts: list[WavJoinPart] = []
    for index in range(len(node.chunks)):
        path = manifest.root / f"chunk-{index}.wav"
        await FakeSpeechService().synthesize_to_file(
            SpeechSynthesisRequest(text="Fixture", voice="narrator"), path
        )
        parts.append(WavJoinPart(path))
    verification = SimpleNamespace(
        accepted=accepted,
        reason_codes=() if accepted else ("mismatch",),
        diagnostic_reason_codes=(),
        token_accuracy=1 if accepted else 0,
        alignment_cache_hits=1,
        alignment_cache_misses=0,
    )
    verify = AsyncMock(return_value=verification)
    monkeypatch.setattr(builds, "verify_manuscript", verify)
    target = manifest.target("default")
    if accepted:
        await builds._verify_changed_synthesis_chunks(
            node,
            join_parts=tuple(parts),
            changed_indices=frozenset(range(len(parts))),
            aligner=_WindowAligner(),
            manifest=manifest,
            target=target,
        )
    else:
        with pytest.raises(BuildError, match="Changed-region verification rejected"):
            await builds._verify_changed_synthesis_chunks(
                node,
                join_parts=tuple(parts),
                changed_indices=frozenset(range(len(parts))),
                aligner=_WindowAligner(),
                manifest=manifest,
                target=target,
            )
    destination = (
        target.output_root / "qa/changed-regions" / f"{node.chapter_id}.chunks.json"
    )
    report = json.loads(destination.read_text())
    assert report["accepted"] is accepted
    assert report["changed_chunk_count"] == verify.await_count == 2
    assert all(row["audio_sha256"] for row in report["results"])
    assert all(
        text not in destination.read_text()
        for text in node.chunks
        if not text.startswith("[")
    )
    assert report["results"][0]["audio_sha256"] == sha256_file(parts[0].path)


@pytest.mark.parametrize(
    "prior,reason",
    [
        ({}, "previous_chunk_had_no_cache_entry"),
        ({"cache_fingerprint": "a" * 64}, "cache_entry_missing"),
        ({"cache_fingerprint": "b" * 64}, "prior_metadata_unavailable"),
    ],
)
def test_synthesis_cache_explains_missing_prior_entries(
    tmp_path: Path, prior: dict[str, object], reason: str
) -> None:
    reasons = builds._prior_cache_difference(
        tmp_path,
        expected="a" * 64,
        request=SpeechSynthesisRequest(text="Hello", voice="narrator"),
        previous=prior,
    )
    assert reason in reasons


@pytest.mark.parametrize("changed", [False, True])
def test_synthesis_cache_differences_identify_text_changes_or_policy_only_changes(
    tmp_path: Path, changed: bool
) -> None:
    request = SpeechSynthesisRequest(text="Hello", voice="narrator")
    values = builds._cache_request_metadata(request)
    if changed:
        values["text_sha256"] = "changed"
    _, metadata = builds._chunk_cache_paths(tmp_path, "b" * 64)
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"request": values}))
    reasons = builds._prior_cache_difference(
        tmp_path,
        expected="a" * 64,
        request=request,
        previous={"cache_fingerprint": "b" * 64},
    )
    assert reasons == (
        ("text_sha256_changed",) if changed else ("context_or_qa_policy_changed",)
    )


def test_synthesis_cache_distinguishes_invalid_entries_and_missing_request_metadata(
    tmp_path: Path,
) -> None:
    request = SpeechSynthesisRequest(text="Hello", voice="narrator")
    _, metadata = builds._chunk_cache_paths(tmp_path, "b" * 64)
    metadata.parent.mkdir(parents=True)
    metadata.write_text("[]")
    reasons = builds._prior_cache_difference(
        tmp_path,
        expected="a" * 64,
        request=request,
        previous={"cache_fingerprint": "b" * 64},
    )
    assert "prior_request_metadata_unavailable" in reasons
    reasons = builds._cache_miss_reasons(
        tmp_path, expected="b" * 64, request=request, previous=None, repair=None
    )
    assert reasons == ("cache_entry_invalid",)


def test_failed_node_recovery_ignores_torn_tail_and_non_failure_records(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "journal.ndjson"
    journal.write_text(
        "\n".join(
            [
                "[]",
                '{"event":"node_started","node_id":"untouched"}',
                '{"event":"node_failed","node_id":"retry-me"}',
                '{"event":"node_failed","node_id":42}',
                '{"torn":',
            ]
        )
    )
    assert builds._failed_node_ids(journal) == {"retry-me"}
    journal.unlink()
    with pytest.raises(BuildError, match="Cannot read run journal"):
        builds._failed_node_ids(journal)
