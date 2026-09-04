from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path
from typing import cast

import pytest
from tests.schema_helpers import validate_contract

import yakbox.audiobook.build as build_module
from yakbox.audio.crop import inspect_speech_islands
from yakbox.audiobook import (
    AudiobookManifest,
    build_audiobook,
    load_manifest,
    normalize_sources,
    plan_audiobook,
)
from yakbox.audiobook.assembly_manifest import (
    assembly_manifest_path,
    locate_assembly_time,
)
from yakbox.audiobook.planner import PlanNode
from yakbox.audiobook.repair import (
    RepairMode,
    _signal_reasons,
    _trim_direct_candidate_edges,
    approve_repair_session,
    begin_repair_batch,
    finalize_repair_batch,
    generate_repair_batch,
    generate_repair_session,
    plan_repair,
    stage_repair_batch_entry,
)
from yakbox.audiobook.repairs import load_approved_repairs
from yakbox.errors import ValidationError
from yakbox.speech import ChatterboxSynthesisOptions


def test_chunk_identity_and_seed_survive_earlier_paragraph_insertion(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    source = book_workspace / "source" / "book.md"
    source.write_text(
        "# One\n\nThe first paragraph.\n\nThe passage that must stay stable.\n",
        encoding="utf-8",
    )
    first = _synthesis_node(manifest)
    stable_index = first.chunks.index("The passage that must stay stable.")
    stable_id = first.chunk_ids[stable_index]
    first_seed = build_module._chunk_chatterbox(
        ChatterboxSynthesisOptions(seed=42),
        chunk_id=stable_id,
        chapter_id=first.chapter_id,
        chunk_index=stable_index + 1,
        text=first.chunks[stable_index],
    )

    source.write_text(
        "# One\n\nA newly inserted opening.\n\nThe first paragraph.\n\n"
        "The passage that must stay stable.\n",
        encoding="utf-8",
    )
    second = _synthesis_node(manifest)
    moved_index = second.chunks.index("The passage that must stay stable.")
    second_seed = build_module._chunk_chatterbox(
        ChatterboxSynthesisOptions(seed=42),
        chunk_id=second.chunk_ids[moved_index],
        chapter_id=second.chapter_id,
        chunk_index=moved_index + 1,
        text=second.chunks[moved_index],
    )

    assert second.chunk_ids[moved_index] == stable_id
    assert first_seed == second_seed


def test_identical_repeated_paragraphs_receive_unique_stable_chunk_ids(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    source = book_workspace / "source" / "book.md"
    source.write_text("# One\n\nAgain.\n\nAgain.\n", encoding="utf-8")

    node = _synthesis_node(manifest)

    assert node.chunks == ("Again.", "Again.")
    assert len(set(node.chunk_ids)) == 2


def test_chunk_id_does_not_change_when_planning_one_selected_chapter(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    source = book_workspace / "source" / "book.md"
    source.write_text(
        "# First\n\nShared words.\n\n# Second\n\nShared words.\n",
        encoding="utf-8",
    )
    document = normalize_sources(manifest.sources)
    full = plan_audiobook(manifest, document)
    second = document.chapters[1]
    full_node = next(node for node in full.nodes if node.chapter_id == second.id)
    selected = plan_audiobook(manifest, document, chapter_selector=second.id)
    selected_node = next(
        node for node in selected.nodes if node.stage.value == "synthesize"
    )

    assert selected_node.chunk_ids == full_node.chunk_ids


@pytest.mark.asyncio
async def test_repair_take_is_approved_and_rebuild_reuses_other_chunks(
    book_workspace: Path,
) -> None:
    manifest_path = book_workspace / "yakbox.toml"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8")
        + "\n[short_utterances]\nautomatic_join_inspection = false\n",
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    source = book_workspace / "source" / "book.md"
    source.write_text(
        "# One\n\nHe knelt beside the body.\n\nMicah Levi waited at the doorway.\n",
        encoding="utf-8",
    )
    initial = await build_audiobook(manifest, through_stage="synthesize")
    assert initial.status == "complete"
    node = _synthesis_node(manifest)
    target_index = node.chunks.index("Micah Levi waited at the doorway.")
    repair = plan_repair(
        manifest,
        chunk_id=node.chunk_ids[target_index],
        mode=RepairMode.TARGET_ONLY,
    )

    session = await generate_repair_session(
        manifest,
        repair,
        takes=2,
        whisper_qa=False,
    )
    assert len(session.takes) == 2
    assert all(take.accepted for take in session.takes)
    assert "Micah Levi" not in session.report_path.read_text(encoding="utf-8")
    validate_contract(
        "audiobook-repair-session",
        json.loads(session.report_path.read_text(encoding="utf-8")),
    )

    approval = approve_repair_session(
        manifest,
        repair_id=session.id,
        take=1,
    )
    assert approval.chapter_id == node.chapter_id
    assert [item.chunk_id for item in approval.repairs] == [
        node.chunk_ids[target_index]
    ]
    preflight = build_module.preflight_audiobook_build(
        manifest,
        through_stage="synthesize",
    )
    assert preflight.pending_synthesis_chunks == 0
    assert preflight.reusable_synthesis_chunks == 2
    assert preflight.affected_join_count == 1

    rebuilt = await build_audiobook(manifest, through_stage="synthesize")
    assert rebuilt.status == "complete"
    assembly_path = assembly_manifest_path(
        manifest.root,
        target="default",
        chapter_id=node.chapter_id,
    )
    assembly = json.loads(assembly_path.read_text(encoding="utf-8"))
    validate_contract("audiobook-assembly", assembly)
    repaired = assembly["chunks"][target_index]
    assert repaired["cache_fingerprint"] == approval.repairs[0].fingerprint

    location = locate_assembly_time(
        manifest.root,
        target="default",
        chapter_id=node.chapter_id,
        at_seconds=repaired["start_frame"] / assembly["sample_rate"],
    )
    validate_contract("audiobook-repair-location", location)
    located_chunk = cast(dict[str, object], location["chunk"])
    assert located_chunk["id"] == node.chunk_ids[target_index]


@pytest.mark.asyncio
async def test_identical_repair_reuses_verified_candidate(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    node = _synthesis_node(manifest)
    plan = plan_repair(
        manifest,
        chunk_id=node.chunk_ids[0],
        mode=RepairMode.TARGET_ONLY,
    )

    first = await generate_repair_session(
        manifest,
        plan,
        takes=1,
        whisper_qa=False,
    )
    second = await generate_repair_session(
        manifest,
        plan,
        takes=1,
        whisper_qa=False,
    )

    assert not first.takes[0].cache_events[0].hit
    assert second.takes[0].cache_events[0].hit
    assert second.takes[0].cache_events[0].reason == "content_match"
    assert first.takes[0].chunk_sha256s == second.takes[0].chunk_sha256s


@pytest.mark.asyncio
async def test_repair_approval_rejects_a_tampered_scope(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    source = book_workspace / "source" / "book.md"
    source.write_text(
        "# One\n\nFirst passage.\n\nSecond passage.\n\nThird passage.\n",
        encoding="utf-8",
    )
    node = _synthesis_node(manifest)
    repair = plan_repair(
        manifest,
        chunk_id=node.chunk_ids[1],
        mode=RepairMode.NEIGHBORS,
    )
    session = await generate_repair_session(
        manifest,
        repair,
        takes=1,
        whisper_qa=False,
    )
    report = json.loads(session.report_path.read_text(encoding="utf-8"))
    report["plan"]["chunks"][1] = report["plan"]["chunks"][0]
    session.report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValidationError, match="scope changed"):
        approve_repair_session(manifest, repair_id=session.id, take=1)


@pytest.mark.asyncio
async def test_repair_batch_stages_without_mutation_and_commits_once(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    source = book_workspace / "source" / "book.md"
    source.write_text(
        "# One\n\nFirst passage.\n\nSecond passage.\n",
        encoding="utf-8",
    )
    node = _synthesis_node(manifest)
    sessions = []
    for chunk_id in node.chunk_ids:
        session = await generate_repair_session(
            manifest,
            plan_repair(manifest, chunk_id=chunk_id, mode=RepairMode.TARGET_ONLY),
            takes=1,
            whisper_qa=False,
        )
        sessions.append(session)

    batch = begin_repair_batch(manifest)
    validate_contract(
        "audiobook-repair-batch",
        json.loads(batch.report_path.read_text(encoding="utf-8")),
    )
    for session in sessions:
        batch = stage_repair_batch_entry(
            manifest,
            batch_id=batch.id,
            repair_id=session.id,
            take=1,
        )
    assert load_approved_repairs(manifest.root, "default") == ()
    validate_contract(
        "audiobook-repair-batch",
        json.loads(batch.report_path.read_text(encoding="utf-8")),
    )

    approval = finalize_repair_batch(manifest, batch_id=batch.id)

    assert approval.chapter_id == node.chapter_id
    assert {item.chunk_id for item in approval.repairs} == set(node.chunk_ids)
    assert len(load_approved_repairs(manifest.root, "default")) == 2


@pytest.mark.asyncio
async def test_batch_generation_creates_one_review_package(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    source = book_workspace / "source" / "book.md"
    source.write_text("# One\n\nFirst passage.\n\nSecond passage.\n", encoding="utf-8")
    node = _synthesis_node(manifest)
    plans = tuple(
        plan_repair(manifest, chunk_id=chunk_id, mode=RepairMode.TARGET_ONLY)
        for chunk_id in node.chunk_ids
    )

    result = await generate_repair_batch(
        manifest,
        plans,
        takes=1,
        whisper_qa=False,
    )

    assert len(result.sessions) == 2
    assert result.review_playlist.is_file()
    assert result.report_path.is_file()
    assert result.batch.entries == ()
    validate_contract(
        "audiobook-repair-batch-generation",
        json.loads(result.report_path.read_text(encoding="utf-8")),
    )


def test_direct_repair_trims_only_excessive_edge_silence(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    path = book_workspace / "direct-repair.wav"
    sample_rate = 24_000
    silence = (0,) * (sample_rate // 2)
    speech = tuple(
        round(8_000 * math.sin(2 * math.pi * 220 * index / sample_rate))
        for index in range(sample_rate // 2)
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(
            struct.pack(
                f"<{len((*silence, *speech, *silence))}h", *silence, *speech, *silence
            )
        )

    _trim_direct_candidate_edges(manifest, path)

    evidence = inspect_speech_islands(path)
    assert (
        evidence.leading_silence_ms <= manifest.short_utterances.maximum_edge_silence_ms
    )
    assert (
        evidence.trailing_silence_ms
        <= manifest.short_utterances.maximum_edge_silence_ms
    )
    assert evidence.duration_seconds > 0.5


def test_direct_repair_does_not_treat_long_prose_pauses_as_edge_speech(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    path = book_workspace / "multi-sentence-repair.wav"
    sample_rate = 24_000
    silence = (0,) * (sample_rate // 2)
    long_pause = (0,) * (sample_rate * 3 // 5)
    speech = tuple(
        round(8_000 * math.sin(2 * math.pi * 220 * index / sample_rate))
        for index in range(sample_rate * 2 // 5)
    )
    samples = (*silence, *speech, *long_pause, *speech, *silence)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    short_reasons = _signal_reasons(manifest, path)
    prose_reasons = _signal_reasons(
        manifest,
        path,
        enforce_single_utterance=False,
    )

    assert "unexpected_suffix_speech" in short_reasons
    assert "unexpected_suffix_speech" not in prose_reasons
    assert "excessive_leading_silence" not in prose_reasons
    assert "excessive_trailing_silence" not in prose_reasons


def test_sentence_repair_selects_only_matching_sentence(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    source = book_workspace / "source" / "book.md"
    source.write_text(
        "# One\n\nHe entered. Micah spoke softly. The room listened.\n",
        encoding="utf-8",
    )

    repair = plan_repair(
        manifest,
        text_match="Micah spoke",
        mode=RepairMode.SENTENCE,
    )

    assert len(repair.chunks) == 1
    assert repair.chunks[0].replacement_text == "Micah spoke softly."
    assert repair.chunks[0].replacement_text != repair.chunks[0].text


def test_clause_repair_selects_only_matching_clause(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    source = book_workspace / "source" / "book.md"
    source.write_text(
        "# One\n\nHe waited, Micah answered, and the room listened.\n",
        encoding="utf-8",
    )

    repair = plan_repair(
        manifest,
        text_match="Micah answered",
        mode=RepairMode.CLAUSE,
    )

    assert repair.chunks[0].replacement_text == "Micah answered,"


def _synthesis_node(manifest: AudiobookManifest) -> PlanNode:
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
    )
    return next(
        node
        for node in plan_audiobook(manifest, document).nodes
        if node.stage.value == "synthesize"
    )
