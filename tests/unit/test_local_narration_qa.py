from __future__ import annotations

import shutil
import struct
import tomllib
import wave
from pathlib import Path
from typing import cast

from tests.live.test_local_chatterbox_e2e import _splice_step_dbfs

from yakbox.audiobook.manifest import load_manifest
from yakbox.audiobook.planner import plan_audiobook
from yakbox.audiobook.sources import apply_pronunciations, normalize_sources
from yakbox.speech.chunking import CHATTERBOX_CHUNK_CHARACTERS

ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "local-chatterbox-e2e"
VOICE_ASSETS = ROOT / "examples" / "local-chatterbox" / "voices"


def test_local_narration_qa_fixture_covers_semantic_chunking(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "local-chatterbox-e2e"
    shutil.copytree(FIXTURE, workspace)
    shutil.copytree(VOICE_ASSETS, workspace / "voices")
    manifest = load_manifest(workspace / "yakbox.toml")
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
    )
    plan = plan_audiobook(manifest, document)
    synthesis = next(node for node in plan.nodes if node.stage == "synthesize")
    qa = cast(
        dict[str, object],
        tomllib.loads((workspace / "qa.toml").read_text(encoding="utf-8")),
    )

    assert qa["schema_version"] == 1
    assert {profile.name for profile in manifest.profiles} == set(
        cast(list[str], qa["profiles"])
    )
    manual = cast(dict[str, object], qa["manual"])
    assert manual["chapter_profile"] == "character-routed"
    assert manifest.character("narrator").profile == manifest.target("default").profile
    assert {route.speaker for route in synthesis.chunk_routes} >= {
        "narrator",
        "mara",
        "wren",
    }
    assert set(cast(list[str], qa["required_boundaries"])) <= set(
        synthesis.chunk_boundaries
    )
    maximum = qa["max_chunk_characters"]
    assert isinstance(maximum, int) and not isinstance(maximum, bool)
    assert max(map(len, synthesis.chunks)) <= maximum
    assert any("Asterion" in chunk for chunk in synthesis.chunks)
    assert any("Wren" in chunk for chunk in synthesis.chunks)
    assert any("trust me long enough" in chunk for chunk in synthesis.chunks)
    assert all("As tear ee on" not in chunk for chunk in synthesis.chunks)
    assert f"__YAKBOX_PAUSE_MS={qa['explicit_pause_ms']}__" in synthesis.chunks


def test_narration_audition_is_one_sustained_production_equivalent_paragraph() -> None:
    audition = (FIXTURE / "audition.txt").read_text(encoding="utf-8").strip()
    normalized = apply_pronunciations(
        audition,
        FIXTURE / "pronunciations.toml",
    )

    assert "\n" not in audition
    assert len(audition.split()) >= 65
    assert "morning" in audition
    assert "Asterion array" in audition
    assert normalized == audition
    assert len(normalized) <= CHATTERBOX_CHUNK_CHARACTERS


def test_dialogue_audition_is_bounded_clear_and_configured() -> None:
    qa = tomllib.loads((FIXTURE / "qa.toml").read_text(encoding="utf-8"))
    passages = cast(list[dict[str, object]], qa["audition_passages"])
    dialogue = (FIXTURE / "dialogue.txt").read_text(encoding="utf-8").strip()

    assert {str(item["id"]): str(item["text_file"]) for item in passages} == {
        "narration": "audition.txt",
        "dialogue": "dialogue.txt",
    }
    assert len(dialogue) <= CHATTERBOX_CHUNK_CHARACTERS
    assert dialogue.count("Wren") >= 3
    assert dialogue.count("Mara") >= 2
    assert dialogue.count("“") >= 7
    assert "Wren warned" in dialogue
    assert "trust, not your protection" in dialogue


def test_splice_step_metric_distinguishes_smooth_and_abrupt_edges(
    tmp_path: Path,
) -> None:
    path = tmp_path / "splice.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(struct.pack("<hhhh", 1_000, 0, 0, 1_000))

    with wave.open(str(path), "rb") as audio:
        abrupt = _splice_step_dbfs(audio, 1)
        smooth = _splice_step_dbfs(audio, 2)

    assert abrupt == -30.31
    assert smooth == -120.0
