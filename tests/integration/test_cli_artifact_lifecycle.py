from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result
from tests.schema_helpers import validate_contract

from yakbox.cli import main


def _invoke(*arguments: str, exit_code: int = 0) -> Result:
    result = CliRunner().invoke(main, ["--json", *arguments])
    assert result.exit_code == exit_code, (arguments, result.output, result.exception)
    validate_contract("cli-output", json.loads(result.output))
    return result


def test_direct_speech_batch_and_conversion_share_real_audio_contracts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("A local sentence.")
    audio = tmp_path / "speech.wav"
    direct = _invoke(
        "tts", "--text-file", str(source), "--backend", "fake", "--out", str(audio)
    )
    speech = json.loads(direct.output)["data"]
    assert speech["bytes_written"] == audio.stat().st_size
    assert speech["hosted_usage"] is None
    _invoke("verify", str(audio))
    converted = tmp_path / "converted.wav"
    result = _invoke("vc", str(audio), "--backend", "fake", "--out", str(converted))
    assert json.loads(result.output)["data"]["sha256"] == speech["sha256"]
    script = tmp_path / "batch.txt"
    script.write_text("First line.\nSecond line.\n")
    output = tmp_path / "batch"
    result = _invoke(
        "batch", str(script), "--backend", "fake", "--out-dir", str(output)
    )
    assert len(json.loads(result.output)["data"]["artifacts"]) == 2
    assert len(tuple(output.glob("*.wav"))) == 2
    capabilities = json.loads(_invoke("backends", "capabilities", "fake").output)[
        "data"
    ]
    assert capabilities["synthesis"] and capabilities["transformation"]
    assert not capabilities["hosted"]
    result = _invoke("models")
    assert isinstance(json.loads(result.output)["data"]["chatterbox_installed"], bool)


def test_cli_build_cache_inventory_and_selective_rebuild(book_workspace: Path) -> None:
    manifest = str(book_workspace / "yakbox.toml")
    _invoke("build", manifest)
    listed = json.loads(
        _invoke(
            "artifacts", "list", manifest, "--voice", "narrator", "--limit", "1"
        ).output
    )["data"]
    assert len(listed["artifacts"]) == 1
    assert (
        json.loads(_invoke("artifacts", "usage", manifest).output)["data"][
            "managed_bytes"
        ]
        > 0
    )
    _invoke("artifacts", "verify", manifest)
    cache = json.loads(_invoke("artifacts", "cache", "list", manifest).output)["data"]
    entry = cache["entries"][0]
    inspected = json.loads(
        _invoke("artifacts", "cache", "inspect", entry["fingerprint"], manifest).output
    )["data"]
    assert inspected["valid"] is True
    explained = json.loads(_invoke("explain", manifest).output)["data"]
    node = explained["nodes"][0]
    assert node["state"] == "reusable"
    filtered = json.loads(
        _invoke("explain", manifest, "--artifact", node["id"]).output
    )["data"]
    assert len(filtered["nodes"]) == 1
    unchanged = json.loads(_invoke("build", manifest, "--changed").output)["data"]
    assert unchanged["status"] == "up_to_date"
    for selection in ("--missing", "--failed"):
        assert (
            json.loads(_invoke("build", manifest, selection).output)["data"]["status"]
            == "up_to_date"
        )
    _invoke("artifacts", "cache", "inspect", "unknown", manifest, exit_code=1)
    _invoke("artifacts", "cache", "why-miss", "unknown", manifest, exit_code=1)


def test_cli_cleanup_requires_apply_and_trash_can_be_restored_and_purged(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = str(book_workspace / "yakbox.toml")
    monkeypatch.chdir(book_workspace)
    _invoke("build", manifest)
    _invoke("preview", manifest, "--text", "A disposable preview.")
    dry = json.loads(
        _invoke(
            "artifacts", "clean", manifest, "--kind", "preview", "--older-than", "0"
        ).output
    )["data"]
    assert not dry["applied"] and dry["candidates"]
    applied = json.loads(
        _invoke(
            "artifacts",
            "clean",
            manifest,
            "--kind",
            "preview",
            "--older-than",
            "0",
            "--apply",
        ).output
    )["data"]
    trash = Path(applied["trash"])
    entries = json.loads(_invoke("artifacts", "trash", "list", manifest).output)[
        "data"
    ]["entries"]
    assert trash.name in entries
    restored = json.loads(
        _invoke("artifacts", "trash", "restore", trash.name, manifest).output
    )["data"]
    assert restored["restored"] == 1
    _invoke(
        "artifacts",
        "clean",
        manifest,
        "--kind",
        "preview",
        "--older-than",
        "0",
        "--apply",
    )
    purged = json.loads(
        _invoke("artifacts", "trash", "purge", "--all", "--yes").output
    )["data"]
    assert purged["purged"] > 0
    assert not json.loads(_invoke("artifacts", "trash", "list", manifest).output)[
        "data"
    ]["entries"]
    cache = book_workspace / ".yakbox/cache"
    orphan = cache / "synthesis/00" / ("0" * 64 + ".wav")
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"unreferenced-invalid-cache")
    before = sorted(cache.rglob("*.wav"))
    assert before
    _invoke("artifacts", "cache", "clean", manifest, "--max-bytes", "0")
    assert all(path.exists() for path in before)
    removed = json.loads(
        _invoke(
            "artifacts", "cache", "clean", manifest, "--max-bytes", "0", "--apply"
        ).output
    )["data"]
    assert removed["removed"] > 0
    assert not orphan.exists()
    assert all(path.exists() for path in before if path != orphan)


def test_cli_shards_roundtrip_and_tampering_is_rejected(book_workspace: Path) -> None:
    manifest = str(book_workspace / "yakbox.toml")
    source = book_workspace / "source/book.md"
    source.write_text(source.read_text() + "\n# Another Chapter\n\nAnother sentence.\n")
    _invoke("build", manifest)
    exported = json.loads(_invoke("shards", "export", manifest, "--count", "2").output)[
        "data"
    ]["shards"]
    assert len(exported) == 2
    verified = json.loads(_invoke("shards", "verify", manifest, *exported).output)[
        "data"
    ]
    assert verified["verified"] == 2
    path = Path(exported[0])
    raw = json.loads(path.read_text())
    raw["plan_fingerprint"] = "0" * 64
    path.write_text(json.dumps(raw))
    _invoke("shards", "verify", manifest, *exported, exit_code=1)


@pytest.mark.parametrize(
    "options",
    [
        ["--stage", "master", "--from", "synthesize"],
        ["--changed", "--failed"],
        ["--since", "missing"],
        ["--mode", "draft", "--target", "other"],
    ],
)
def test_cli_build_selection_rejects_conflicting_options_before_work(
    book_workspace: Path, options: list[str]
) -> None:
    _invoke("build", str(book_workspace / "yakbox.toml"), *options, exit_code=2)
    assert not (book_workspace / ".yakbox").exists()


def test_cli_release_manifest_can_be_compared_to_itself(book_workspace: Path) -> None:
    manifest = str(book_workspace / "yakbox.toml")
    _invoke("build", manifest)
    _invoke("assemble", manifest)
    _invoke("release", "check", manifest, "--write-manifest")
    manifests = sorted((book_workspace / "build/yakbox/release").glob("*/release.json"))
    assert manifests
    result = _invoke(
        "release", "diff", str(manifests[0]), str(manifests[0]), "--manifest", manifest
    )
    assert json.loads(result.output)["data"]["identical"] is True
