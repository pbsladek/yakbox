from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from yakbox.errors import ValidationError
from yakbox.schemas import load_schema
from yakbox.speech.analysis_corpus_boundaries import select_boundary_review_case_ids
from yakbox.speech.analysis_corpus_partition import (
    QualificationPartition,
    freeze_corpus_partition,
    load_frozen_corpus_partition,
    write_frozen_corpus_partition,
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

SHA_A = "a" * 64
SHA_B = "b" * 64


def _sha(label: str) -> str:
    return semantic_fingerprint("test-corpus-partition-v1", label)


def _corpus() -> tuple[CorpusSourceInventory, ApprovedCorpusTranscripts]:
    windows: list[CorpusSourceWindow] = []
    for voice_index in range(25):
        voice = f"voice-{voice_index:02d}"
        for passage_index in range(1, 4):
            group = f"{voice}-passage-{passage_index:02d}"
            window_id = f"{group}-window-01"
            windows.append(
                CorpusSourceWindow(
                    window_id,
                    group,
                    voice,
                    f"Reader {voice_index:02d}",
                    f"cache/windows/{window_id}.wav",
                    _sha(f"audio-{window_id}"),
                    16_000,
                    320_000,
                    _sha(f"canonical-{window_id}"),
                    0,
                    320_000,
                    passage_index * 120_000,
                    passage_index * 120_000 + 20_000,
                    f"https://example.invalid/{voice}.wav",
                    _sha(f"source-{voice}"),
                    "LicenseRef-LibriVox-Public-Domain-US",
                    "https://librivox.org/pages/public-domain/",
                )
            )
    inventory = CorpusSourceInventory(
        SHA_A,
        1,
        tuple(sorted(windows, key=lambda item: item.source_window_id)),
    )
    tokens = ("approved", "sample")
    token_hash = text_fingerprint("\u001f".join(tokens))
    approved = ApprovedCorpusTranscripts(
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
                _sha(f"draft-{window.source_window_id}"),
            )
            for window in inventory.windows
        ),
    )
    return inventory, approved


def test_frozen_partition_keeps_voices_disjoint_and_held_out_large_enough(
    tmp_path: Path,
) -> None:
    inventory, approved = _corpus()

    partition = freeze_corpus_partition(
        inventory,
        approved,
        partition_seed="private qualification seed",
    )
    repeated = freeze_corpus_partition(
        inventory,
        approved,
        partition_seed="private qualification seed",
    )

    assert partition == repeated
    assert partition.calibration_voice_count == 6
    assert partition.held_out_voice_count == 19
    assert partition.calibration_cluster_count == 18
    assert partition.held_out_cluster_count == 57
    by_voice: dict[str, set[QualificationPartition]] = {}
    for item in partition.assignments:
        by_voice.setdefault(item.voice, set()).add(item.partition)
    assert {len(values) for values in by_voice.values()} == {1}
    selected = select_boundary_review_case_ids(partition)
    selected_voices = {
        item.voice
        for item in partition.assignments
        if item.source_window_id in selected
    }
    assert len(selected) == 12
    assert len(selected_voices) == 12

    output = tmp_path / "partition.json"
    write_frozen_corpus_partition(output, partition)
    raw = json.loads(output.read_text(encoding="utf-8"))
    Draft202012Validator(load_schema("speech-corpus-partition")).validate(raw)
    assert (
        load_frozen_corpus_partition(
            output,
            inventory=inventory,
            transcripts=approved,
        )
        == partition
    )


def test_partition_rejects_a_split_that_cannot_keep_52_held_out_clusters() -> None:
    inventory, approved = _corpus()

    with pytest.raises(ValidationError, match="insufficient clusters"):
        freeze_corpus_partition(
            inventory,
            approved,
            partition_seed="private qualification seed",
            calibration_voice_count=8,
        )


def test_partition_loader_rejects_changed_assignment(tmp_path: Path) -> None:
    inventory, approved = _corpus()
    partition = freeze_corpus_partition(
        inventory,
        approved,
        partition_seed="private qualification seed",
    )
    output = tmp_path / "partition.json"
    write_frozen_corpus_partition(output, partition)
    raw = json.loads(output.read_text(encoding="utf-8"))
    first = partition.assignments[0]
    changed = replace(first, voice="substituted-voice")
    raw["assignments"][0]["voice"] = changed.voice
    raw["assignments"][0]["fingerprint"] = changed.fingerprint
    output.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValidationError, match="identity differs"):
        load_frozen_corpus_partition(
            output,
            inventory=inventory,
            transcripts=approved,
        )
