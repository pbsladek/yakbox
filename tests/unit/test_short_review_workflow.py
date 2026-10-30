from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner

import yakbox.short_testing as short
from yakbox._files import sha256_file
from yakbox.cli import main
from yakbox.errors import BuildError, ValidationError
from yakbox.speech.models import SpeechSynthesisRequest
from yakbox.speech.services import FakeSpeechService


def _report(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    audio = root / "selected.wav"
    audio.write_bytes(b"reviewed-audio")
    report = root / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "short_utterance_qa",
                "selected_candidate": 1,
                "candidates": [
                    {
                        "candidate_index": 1,
                        "extracted_audio": audio.name,
                        "extracted_audio_sha256": sha256_file(audio),
                    }
                ],
            }
        )
    )
    return report


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True])
async def test_short_test_writes_a_complete_bound_diagnostic_package(
    book_workspace: Path, monkeypatch: pytest.MonkeyPatch, persistent: bool
) -> None:
    manifest = book_workspace / "yakbox.toml"
    if persistent:
        manifest.write_text(manifest.read_text() + "\n[runtime]\nenabled = true\n")
    aligner = object()
    monkeypatch.setattr(short, "open_local_aligner", Mock(return_value=aligner))
    monkeypatch.setattr(
        "yakbox.local_runtime.PersistentMlxWhisperAligner", Mock(return_value=aligner)
    )

    async def select(**options: object) -> SimpleNamespace:
        destination = options["destination"]
        assert isinstance(destination, Path)
        assert options["aligner"] is aligner
        await FakeSpeechService().synthesize_to_file(
            SpeechSynthesisRequest(text="Hello", voice="narrator"), destination
        )
        report = _report(destination.parent / "candidate-report")
        return SimpleNamespace(
            selected=SimpleNamespace(recipe=SimpleNamespace(candidate_index=1)),
            report=report,
        )

    synthesis = AsyncMock(side_effect=select)
    monkeypatch.setattr(short, "synthesize_short_utterance", synthesis)
    directory = await short.run_short_test(
        manifest,
        profile_name="default",
        text="Hello",
        output_root=book_workspace / "qa",
        previous_context="Earlier.",
        next_context="Later.",
    )
    report = json.loads((directory / "short-test.json").read_text())
    assert report["selected_candidate"] == 1
    assert report["selected_audio_sha256"] == sha256_file(
        directory / report["selected_audio"]
    )
    assert report["previous_context"] == "Earlier."
    assert report["recipes"]
    assert synthesis.await_count == 1


def test_short_review_decisions_bind_report_and_audio_and_surface_in_cli(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path / "qa")
    assert short.list_short_reviews(tmp_path)[0]["status"] == "unreviewed"
    assert short.review_audio_path(report) == report.parent / "selected.wav"
    review = short.write_short_review(
        report, approved=True, notes='Listened: "clear".\nGood.'
    )
    assert review.is_file()
    assert short.list_short_reviews(tmp_path)[0]["status"] == "pass"
    runner = CliRunner()
    for command in [
        ["short-review", "list", str(tmp_path)],
        ["short-review", "reject", str(report), "--notes", "Retry"],
        ["short-review", "approve", str(report), "--notes", "Reviewed"],
    ]:
        result = runner.invoke(main, ["--json", *command])
        assert result.exit_code == 0, result.output
        json.loads(result.output)
    assert short.list_short_reviews(tmp_path)[0]["status"] == "pass"
    (report.parent / "selected.wav").write_bytes(b"tampered")
    with pytest.raises(BuildError, match="checksum"):
        short.review_audio_path(report)
    result = runner.invoke(main, ["--json", "short-review", "approve", str(report)])
    assert result.exit_code == 1


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        "[]",
        '{"schema_version":2}',
        '{"schema_version":1,"kind":"other","candidates":[]}',
    ],
)
def test_short_review_rejects_invalid_reports_and_discovery_skips_them(
    tmp_path: Path, content: str
) -> None:
    report = tmp_path / "report.json"
    report.write_text(content)
    with pytest.raises((BuildError, ValidationError)):
        short.review_audio_path(report)
    assert short.list_short_reviews(tmp_path) == ()
    assert short.list_short_reviews(tmp_path / "missing") == ()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-selection",
        "missing-candidate",
        "missing-audio",
        "escaped-audio",
        "missing-digest",
    ],
)
def test_short_review_requires_safe_complete_audio_evidence(
    tmp_path: Path, mutation: str
) -> None:
    report = _report(tmp_path / "qa")
    raw = json.loads(report.read_text())
    if mutation == "missing-selection":
        raw["selected_candidate"] = None
    elif mutation == "missing-candidate":
        raw["selected_candidate"] = 99
    elif mutation == "missing-audio":
        (report.parent / "selected.wav").unlink()
    elif mutation == "escaped-audio":
        raw["candidates"][0]["extracted_audio"] = "../outside.wav"
        (tmp_path / "outside.wav").write_bytes(b"reviewed-audio")
    else:
        raw["candidates"][0].pop("extracted_audio_sha256")
    report.write_text(json.dumps(raw))
    with pytest.raises(BuildError):
        short.write_short_review(report, approved=True, notes="Must not approve")
    assert not (report.parent / "listening-review.toml").exists()


@pytest.mark.parametrize(
    "platform,opener", [("darwin", "open"), ("linux", "xdg-open"), ("win32", "cmd")]
)
def test_short_review_playback_uses_the_integrity_checked_take(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str, opener: str
) -> None:
    report = _report(tmp_path)
    launch = Mock()
    monkeypatch.setattr(short.sys, "platform", platform)
    monkeypatch.setattr(short.subprocess, "run", launch)
    audio = short.play_short_review(report)
    assert launch.call_args.args[0][0] == opener
    assert launch.call_args.args[0][-1] == str(audio)
    launch.side_effect = subprocess.CalledProcessError(1, opener)
    with pytest.raises(BuildError, match="Could not open"):
        short.play_short_review(report)
