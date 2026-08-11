from __future__ import annotations

import hashlib
import re
import shutil
import tomllib
import wave
from pathlib import Path

import pytest

from yakbox.audiobook.manifest import ChatterboxOptions, load_manifest

ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "examples" / "local-chatterbox"
VOICE_DIR = EXAMPLE / "voices"
REGISTRY = tomllib.loads((VOICE_DIR / "voices.toml").read_text(encoding="utf-8"))
QUALITY = tomllib.loads((VOICE_DIR / "quality.toml").read_text(encoding="utf-8"))
EXPECTED_VOICES = {
    "amanda-friday",
    "andy-minter",
    "bill-boerst",
    "bob-neufeld",
    "caro-davy",
    "cori-samuel",
    "david-barnes",
    "gregg-margarite",
    "john-greenman",
    "john-burlinson",
    "karen-savage",
    "kirsten-ferreri",
    "laurie-anne-walden",
    "lucy-burgoyne",
    "mark-f-smith",
    "mark-nelson",
    "martin-geeson",
    "mil-nicholson",
    "nick-whitley",
    "peter-yearsley",
    "phil-chenevert",
    "ruth-golding",
    "sibella-denton",
    "simon-evers",
    "tony-foster",
}
REMOVED_VOICES = {
    "adrian-praetzellis",
    "david-clarke",
    "elizabeth-klett",
    "kara-shallenberg",
    "lee-ann-howlett",
    "stuart-bell",
}
ALLOWED_MEDIA_LICENSES = {
    "CC-BY-4.0",
    "CC0-1.0",
    "LicenseRef-LibriVox-Public-Domain-US",
}


@pytest.mark.parametrize("selected", sorted(EXPECTED_VOICES))
def test_local_chatterbox_voice_profiles_are_configured_and_switchable(
    tmp_path: Path, selected: str
) -> None:
    workspace = tmp_path / "local-chatterbox"
    shutil.copytree(EXAMPLE, workspace)
    manifest_path = workspace / "yakbox.toml"
    content = (
        manifest_path.read_text(encoding="utf-8")
        .replace(
            '[characters.narrator]\ndisplay_name = "Narrator"\nprofile = "andy-minter"',
            f'[characters.narrator]\ndisplay_name = "Narrator"\nprofile = "{selected}"',
        )
        .replace(
            '[targets.default]\nprofile = "andy-minter"',
            f'[targets.default]\nprofile = "{selected}"',
        )
    )
    manifest_path.write_text(content, encoding="utf-8")
    manifest = load_manifest(manifest_path)

    assert manifest.target("default").profile == selected
    assert manifest.character("narrator").profile == selected
    assert {character.name for character in manifest.characters} == {
        "narrator",
        "character-1",
        "character-2",
        "character-3",
        "character-4",
        "character-5",
        "character-6",
        "character-7",
        "character-8",
        "character-9",
        "character-10",
        "character-11",
        "character-12",
        "character-13",
        "character-14",
        "character-15",
        "character-16",
        "character-17",
        "character-18",
        "character-19",
        "character-20",
        "character-21",
        "character-22",
        "character-23",
        "character-24",
    }
    assert manifest.character("character-1").gender == "female"
    assert manifest.character("character-2").gender == "male"
    assert manifest.character("character-3").gender == "female"
    assert manifest.character("character-4").gender == "male"
    assert manifest.character("character-5").gender == "male"
    assert all(
        manifest.character(f"character-{index}").gender == "female"
        for index in range(6, 11)
    )
    assert all(
        manifest.character(f"character-{index}").gender == "male"
        for index in range(11, 21)
    )
    assert all(
        manifest.character(f"character-{index}").gender == "female"
        for index in range(21, 24)
    )
    assert manifest.character("character-24").gender == "male"
    assert {voice.name for voice in manifest.voices} == EXPECTED_VOICES
    assert {profile.name for profile in manifest.profiles} == EXPECTED_VOICES
    for name in EXPECTED_VOICES:
        voice = manifest.voice(name)
        profile = manifest.profile(name)
        assert voice.rights_basis == "public_domain"
        assert voice.reference_audio == workspace / "voices" / f"{name}.wav"
        assert profile.voice == name
        assert profile.backend == "chatterbox-local"
        assert isinstance(profile.options, ChatterboxOptions)


def test_every_bundled_voice_has_complete_verified_provenance() -> None:
    records = REGISTRY["voices"]
    media = {
        path.name
        for path in VOICE_DIR.iterdir()
        if path.is_file()
        and path.suffix.casefold() in {".flac", ".mp3", ".ogg", ".wav"}
    }

    assert REGISTRY["schema_version"] == 1
    assert set(records) == EXPECTED_VOICES
    assert media == {f"{name}.wav" for name in EXPECTED_VOICES}
    assert records["caro-davy"]["filters"] == ["loudnorm=I=-23:TP=-3:LRA=11"]
    assert all(
        records[name]["filters"] == [] for name in EXPECTED_VOICES - {"caro-davy"}
    )
    for name, record in records.items():
        path = VOICE_DIR / record["file"]
        assert record["file"] == f"{name}.wav"
        assert record["license_id"] in ALLOWED_MEDIA_LICENSES
        assert re.fullmatch(r"[0-9a-f]{64}", record["source_sha256"])
        assert record["source_url"].startswith("https://")
        assert record["catalog_url"].startswith("https://")
        assert record["reader_url"].startswith("https://librivox.org/")
        assert record["rights_url"] == "https://librivox.org/pages/public-domain/"
        assert isinstance(record["filters"], list)
        assert _sha256(path) == record["sha256"]
        with wave.open(str(path), "rb") as audio:
            assert audio.getframerate() == record["sample_rate_hz"]
            assert audio.getnchannels() == record["channels"]
            assert audio.getsampwidth() * 8 == record["pcm_bits"]
            assert (
                audio.getnframes() / audio.getframerate() == record["duration_seconds"]
            )


def test_every_bundled_voice_has_an_explicit_quality_status() -> None:
    records = QUALITY["voices"]
    baselines = set(QUALITY["baseline_voices"])
    high_quality = set(QUALITY["high_quality_voices"])
    suspect = set(QUALITY["suspect_voices"])

    assert QUALITY["schema_version"] == 1
    assert QUALITY["scope"] == "automated_intelligibility_and_signal_quality"
    assert baselines == {
        "andy-minter",
        "caro-davy",
        "nick-whitley",
        "ruth-golding",
    }
    assert baselines | high_quality == EXPECTED_VOICES
    assert not suspect
    assert not (
        baselines & high_quality or baselines & suspect or high_quality & suspect
    )
    assert set(records) == EXPECTED_VOICES
    assert all(records[name]["status"] == "baseline" for name in baselines)
    assert all(records[name]["status"] == "high_quality" for name in high_quality)
    assert all(not records[name]["reason_codes"] for name in baselines | high_quality)
    assert set(QUALITY["removed_voice_names"]) == REMOVED_VOICES
    assert set(QUALITY["removed_voices"]) == REMOVED_VOICES
    assert {
        name: record["replacement"]
        for name, record in QUALITY["removed_voices"].items()
        if "replacement" in record
    } == {
        "adrian-praetzellis": "simon-evers",
        "david-clarke": "tony-foster",
        "elizabeth-klett": "amanda-friday",
        "kara-shallenberg": "lee-ann-howlett",
        "stuart-bell": "john-greenman",
    }
    assert QUALITY["removed_voices"]["lee-ann-howlett"]["reason_codes"] == [
        "failed_manual_listening_review"
    ]
    assert all(
        QUALITY["removed_voices"][name]["reason_codes"] for name in REMOVED_VOICES
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
