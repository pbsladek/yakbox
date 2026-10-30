from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner
from tests.schema_helpers import validate_contract
from tests.unit.test_short_review_workflow import _report

from yakbox import cli_whisper
from yakbox.cli import main
from yakbox.errors import BackendUnavailableError


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("command", ["inspect", "reinspect", "inspect-phonemes"])
def test_inspection_commands_forward_options_write_reports_and_propagate_qa_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accepted: bool, command: str
) -> None:
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"audio-boundary-fixture")
    destination = tmp_path / "inspection.json"
    payload = {
        "accepted": accepted,
        "recognized_text": "Hello",
        "fixture": "diagnostic",
    }
    report = SimpleNamespace(
        accepted=accepted,
        reason_codes=() if accepted else ("low_confidence",),
        result=SimpleNamespace(
            timing_source="unprompted", tokens=("hello",), phonemes=("h",)
        ),
        decision=SimpleNamespace(confidence=0.9),
        to_dict=lambda: payload,
    )
    operation = AsyncMock(return_value=report)
    name = (
        "inspect_phonemes" if command == "inspect-phonemes" else "inspect_with_whisper"
    )
    monkeypatch.setattr(cli_whisper, name, operation)
    arguments = [
        "--json",
        "whisper",
        command,
        str(audio),
        "--expected",
        "Hello",
        "--out",
        str(destination),
    ]
    if command == "reinspect":
        arguments.extend(["--start", "0.1", "--end", "0.9"])
    result = CliRunner().invoke(main, arguments)
    assert result.exit_code == (0 if accepted else 1), result.output
    envelope = json.loads(result.output)
    validate_contract("cli-output", envelope)
    assert envelope["status"] == ("ok" if accepted else "partial_failure")
    assert json.loads(destination.read_text()) == payload
    assert operation.call_args.kwargs["language"] == (
        "en-us" if command == "inspect-phonemes" else "en"
    )
    if command == "reinspect":
        assert operation.call_args.kwargs["start_seconds"] == 0.1
        assert operation.call_args.kwargs["end_seconds"] == 0.9


@pytest.mark.parametrize("command", ["inspect", "reinspect", "inspect-phonemes"])
def test_inspection_cli_translates_missing_model_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    audio = tmp_path / "speech.wav"
    audio.touch()
    name = (
        "inspect_phonemes" if command == "inspect-phonemes" else "inspect_with_whisper"
    )
    monkeypatch.setattr(
        cli_whisper,
        name,
        AsyncMock(side_effect=BackendUnavailableError("Install the pinned model")),
    )
    arguments = ["--json", "whisper", command, str(audio), "--expected", "Hello"]
    if command == "reinspect":
        arguments.extend(["--start", "0", "--end", "1"])
    result = CliRunner().invoke(main, arguments)
    assert result.exit_code == 1
    assert json.loads(result.output)["error"]["code"] == "backend_unavailable"


@pytest.mark.parametrize(
    "family,commands",
    [
        ("models", ("status", "verify", "path", "install")),
        ("phoneme-models", ("status", "install")),
    ],
)
def test_cli_model_management_accepts_local_snapshots_without_downloads(
    tmp_path: Path, family: str, commands: tuple[str, ...]
) -> None:
    model = tmp_path / "local-model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "vocab.json").write_text('{"<pad>":0,"a":1}')
    (model / "weights.safetensors").write_bytes(b"local-model-fixture")
    for command in commands:
        result = CliRunner().invoke(
            main, ["--json", "whisper", family, command, "--model", str(model)]
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)["data"]
        assert data["verified"] is True
        assert data["local_path"] == str(model)


@pytest.mark.parametrize("command", ["status", "verify", "path"])
def test_cli_missing_whisper_models_fail_without_attempting_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    missing = SimpleNamespace(
        verified=False,
        local_path=None,
        size_bytes=0,
        issues=("model_not_cached",),
        to_dict=lambda: {"verified": False},
    )
    monkeypatch.setattr(cli_whisper, "model_status", lambda *_args: missing)
    result = CliRunner().invoke(
        main,
        ["--json", "whisper", "models", command, "--model", str(tmp_path / "missing")],
    )
    assert result.exit_code == 1, result.output
    validate_contract("cli-output", json.loads(result.output))


@pytest.mark.parametrize(
    "family,function",
    [("models", "install_model"), ("phoneme-models", "install_phoneme_model")],
)
def test_cli_explicit_model_install_failures_remain_structured(
    monkeypatch: pytest.MonkeyPatch, family: str, function: str
) -> None:
    def fail(*_args: object) -> None:
        raise BackendUnavailableError("Offline model unavailable")

    monkeypatch.setattr(cli_whisper, function, fail)
    result = CliRunner().invoke(main, ["--json", "whisper", family, "install"])
    assert result.exit_code == 1
    assert json.loads(result.output)["error"]["code"] == "backend_unavailable"


@pytest.mark.parametrize("accepted", [False, True])
def test_voice_qualification_cli_retains_report_and_returns_correct_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accepted: bool
) -> None:
    report = tmp_path / "audition.json"
    report.write_text("{}")
    expected = tmp_path / "text.txt"
    expected.write_text("Fixed words")
    output = tmp_path / "qualification.json"
    payload = {
        "accepted": accepted,
        "voices": [{"voice": "narrator", "accepted": accepted}],
    }
    operation = AsyncMock(
        return_value=SimpleNamespace(
            accepted=accepted,
            voices=(SimpleNamespace(voice="narrator", accepted=accepted),),
            to_dict=lambda: payload,
        )
    )
    monkeypatch.setattr(cli_whisper, "qualify_audition_voices", operation)
    result = CliRunner().invoke(
        main,
        [
            "--json",
            "whisper",
            "qualify-voices",
            str(report),
            "--expected-file",
            str(expected),
            "--baseline",
            "narrator",
            "--no-cache",
            "--out",
            str(output),
        ],
    )
    assert result.exit_code == (0 if accepted else 1), result.output
    assert json.loads(output.read_text()) == payload
    assert operation.call_args.kwargs["cache_root"] is None
    assert operation.call_args.kwargs["baseline_voices"] == ("narrator",)


def test_short_review_play_cli_uses_checked_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(tmp_path)
    paths: list[Path] = []
    monkeypatch.setattr(
        "yakbox.short_testing.subprocess.run",
        lambda command, **_kwargs: paths.append(Path(command[-1])),
    )
    result = CliRunner().invoke(main, ["--json", "short-review", "play", str(report)])
    assert result.exit_code == 0, result.output
    assert paths == [tmp_path / "selected.wav"]


def test_cli_batch_repair_generation_uses_yaml_selectors_and_writes_review_bundle(
    book_workspace: Path,
) -> None:
    manifest = book_workspace / "yakbox.toml"
    requests = book_workspace / "repairs.yaml"
    requests.write_text(
        "repairs:\n  - text: opening\n    mode: target-only\n"
        "  - text: ending\n    mode: target-only\n"
    )
    runner = CliRunner()
    planned = runner.invoke(
        main,
        [
            "--json",
            "repair",
            "plan",
            str(manifest),
            "--text",
            "opening",
            "--mode",
            "target-only",
        ],
    )
    assert planned.exit_code == 0, planned.output
    result = runner.invoke(
        main,
        [
            "--json",
            "repair",
            "batch",
            "generate",
            str(requests),
            str(manifest),
            "--takes",
            "1",
            "--no-whisper",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)["data"]
    assert len(data["sessions"]) == 2
    assert (book_workspace / data["report_path"]).is_file()
    assert (book_workspace / data["review_playlist"]).is_file()
    session = data["sessions"][0]
    cache = runner.invoke(
        main,
        [
            "--json",
            "repair",
            "cache",
            "why-miss",
            session["repair_id"],
            str(manifest),
            "--take",
            "1",
        ],
    )
    assert cache.exit_code == 0, cache.output
    assert json.loads(cache.output)["data"]["events"]


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        "repairs: []",
        "repairs: [false]",
        "repairs: [{text: hello, unknown: value}]",
        "repairs: [{text: hello, line: 1}]",
        "repairs: [{line: false}]",
        "repairs: [{text: 7}]",
    ],
)
def test_cli_batch_repair_rejects_invalid_selectors_before_generation(
    book_workspace: Path, content: str
) -> None:
    request = book_workspace / "repairs.yaml"
    request.write_text(content)
    result = CliRunner().invoke(
        main,
        [
            "--json",
            "repair",
            "batch",
            "generate",
            str(request),
            str(book_workspace / "yakbox.toml"),
            "--no-whisper",
        ],
    )
    assert result.exit_code == 1, result.output
    validate_contract("cli-output", json.loads(result.output))
    assert not (book_workspace / ".yakbox/repair-sessions").exists()
