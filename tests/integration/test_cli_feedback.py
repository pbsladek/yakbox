from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner
from rich.console import Console
from tests.schema_helpers import validate_contract

from yakbox._files import atomic_write_json
from yakbox.audiobook import load_manifest, normalize_sources, plan_audiobook
from yakbox.audiobook.journal import RunJournal, target_lock
from yakbox.audiobook.manifest import ChatterboxOptions, FakeOptions
from yakbox.cli import main
from yakbox.cli_status import watch_status
from yakbox.errors import BuildError


@pytest.mark.parametrize("backend", ["fake", "chatterbox-local"])
def test_init_backend_produces_a_valid_profile_without_loading_or_contacting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    def reject_process(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("Initialization must not start a backend")

    workspace = tmp_path / "book"
    arguments = ["--json", "init", str(workspace), "--backend", backend]
    title, author = "A book 📖", "Author 𐐀"
    arguments.extend(["--title", title, "--author", author])
    with monkeypatch.context() as initialization:
        initialization.setattr("subprocess.run", reject_process)
        result = CliRunner().invoke(main, arguments)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    manifest = load_manifest(workspace / "yakbox.toml")
    assert manifest.book.title == title
    assert manifest.book.author == author
    profile = manifest.profile("default")
    assert profile.backend == backend
    assert all(target.profile == "default" for target in manifest.targets)
    if backend == "fake":
        assert isinstance(profile.options, FakeOptions)
        assert profile.options.sample_rate == 16000
    else:
        assert isinstance(profile.options, ChatterboxOptions)
        assert profile.options.device == "cpu"
        assert profile.options.max_processes == profile.options.threads_per_process == 1
    assert CliRunner().invoke(main, ["validate", str(manifest.path)]).exit_code == 0


@pytest.mark.parametrize(
    "options",
    [
        ["--backend", "unknown"],
        ["--backend", "resemble"],
        ["--backend", "resemble", "--voice-uuid", " "],
        ["--backend", "fake", "--voice-uuid", "id"],
        ["--backend", "chatterbox-local", "--voice-uuid", "id"],
    ],
)
def test_init_rejects_invalid_backend_options_before_creating_files(
    tmp_path: Path, options: list[str]
) -> None:
    workspace = tmp_path / "book"
    result = CliRunner().invoke(main, ["--json", "init", str(workspace), *options])
    assert result.exit_code == 2
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    assert payload["error"]["code"] == (
        "unknown_option" if "--voice-uuid" in options else "invalid_argument"
    )
    assert not workspace.exists()


@pytest.mark.parametrize("command", ["preview", "audition"])
@pytest.mark.parametrize("play", [False, True])
@pytest.mark.parametrize("machine", [False, True])
def test_play_is_explicit_and_preserves_normal_artifact_output(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    play: bool,
    machine: bool,
) -> None:
    manifest = book_workspace / "yakbox.toml"
    if command == "audition":
        with manifest.open("a") as stream:
            stream.write('\n[profiles.other]\nbackend = "fake"\nvoice = "narrator"\n')
    opened: list[Path] = []

    def open_audio(path: Path) -> None:
        assert path.is_file()
        opened.append(path)

    monkeypatch.setattr("yakbox.cli.play_audio", open_audio)
    arguments = ["--json"] if machine else []
    arguments.extend([command, str(manifest), "--text", "A short listening check."])
    if command == "audition":
        arguments.extend(["--profile", "default", "--profile", "other"])
    if play:
        arguments.append("--play")
    result = CliRunner().invoke(main, arguments)
    assert result.exit_code == 0, result.output
    count = 2 if command == "audition" else 1
    assert len(opened) == (count if play else 0)
    if machine:
        payload = json.loads(result.output)
        validate_contract("cli-output", payload)
        records = (
            payload["data"]["artifacts"]
            if command == "audition"
            else [payload["data"]["artifact"]]
        )
        assert len(records) == count
        for record in records:
            validate_contract("audiobook-artifact", record)
        if play:
            assert [book_workspace / record["path"] for record in records] == opened
    elif play:
        assert all(path.name in result.output for path in opened)


@pytest.mark.parametrize("command", ["preview", "audition"])
def test_player_failure_returns_a_structured_error_and_keeps_generated_audio(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    opened: list[Path] = []

    def fail(path: Path) -> None:
        opened.append(path)
        raise BuildError("Default audio player unavailable")

    monkeypatch.setattr("yakbox.cli.play_audio", fail)
    arguments = [
        "--json",
        command,
        str(book_workspace / "yakbox.toml"),
        "--text",
        "A sample.",
        "--play",
    ]
    if command == "audition":
        arguments.extend(["--profile", "default"])
    result = CliRunner().invoke(main, arguments)
    assert result.exit_code == 1
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    assert payload["error"]["code"] == "build_error"
    assert len(opened) == 1 and opened[0].is_file()


@pytest.mark.parametrize("command", ["preview", "audition"])
def test_cli_keeps_player_diagnostics_outside_json(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    calls: list[tuple[str, ...]] = []

    def launch(
        arguments: tuple[str, ...],
        *,
        check: bool,
        stdout: int,
        stderr: int,
        timeout: int,
    ) -> subprocess.CompletedProcess[bytes]:
        assert check and timeout == 15
        assert stdout == stderr == subprocess.DEVNULL
        calls.append(arguments)
        return subprocess.CompletedProcess(
            arguments, 0, b"player chatter", b"desktop warning"
        )

    monkeypatch.setattr("yakbox.playback.subprocess.run", launch)
    arguments = [
        "--json",
        command,
        str(book_workspace / "yakbox.toml"),
        "--text",
        "A sample.",
        "--play",
    ]
    if command == "audition":
        arguments.extend(["--profile", "default"])
    result = CliRunner().invoke(main, arguments)
    assert result.exit_code == 0, result.output
    validate_contract("cli-output", json.loads(result.stdout))
    assert "player chatter" not in result.output
    assert "desktop warning" not in result.output
    assert len(calls) == 1


def _active_run(workspace: Path) -> RunJournal:
    manifest = load_manifest(workspace / "yakbox.toml")
    plan = plan_audiobook(manifest, normalize_sources(manifest.sources))
    directory = workspace / ".yakbox/runs/live-run"
    directory.mkdir(parents=True)
    node = plan.nodes[0]
    atomic_write_json(
        directory / "plan.json",
        {**plan.to_dict(root=workspace), "execution": {"node_ids": [node.id]}},
    )
    journal = RunJournal(directory / "journal.ndjson", "live-run")
    journal.append("run_started")
    journal.append("node_started", node_id=node.id)
    return journal


def test_watch_refreshes_active_and_completed_journal_progress_and_ctrl_c_exits_130(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = _active_run(book_workspace)
    sleeps = 0

    def refresh(seconds: float) -> None:
        nonlocal sleeps
        assert seconds == 2
        sleeps += 1
        if sleeps == 1:
            node_id = str(journal.events()[-1]["node_id"])
            journal.append("node_completed", node_id=node_id)
            journal.append("run_complete")
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr("yakbox.cli_status.time.sleep", refresh)
    with target_lock(book_workspace / ".yakbox", "default"):
        result = CliRunner().invoke(
            main, ["status", str(book_workspace / "yakbox.toml"), "--watch"]
        )
    assert result.exit_code == 130, result.output
    assert "State: running" in result.output
    assert "State: complete" in result.output
    assert "Stages: 0/1 complete" in result.output
    assert "Stages: 1/1 complete" in result.output
    assert "synthesize" in result.output
    assert "Aborted!" in result.output
    assert "\x1b" not in result.output
    assert not journal.path.with_name("run.json").exists()


def test_watch_waits_for_the_first_build_without_creating_files(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("yakbox.cli_status.time.sleep", interrupt)
    result = CliRunner().invoke(
        main, ["status", str(book_workspace / "yakbox.toml"), "--watch"]
    )
    assert result.exit_code == 130
    assert "Waiting for a build" in result.output
    assert not (book_workspace / ".yakbox").exists()


def test_watch_refreshes_interactive_terminal_display(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _active_run(book_workspace)
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system=None, width=120)

    def interrupt(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("yakbox.cli_status.time.sleep", interrupt)
    with (
        target_lock(book_workspace / ".yakbox", "default"),
        pytest.raises(KeyboardInterrupt),
    ):
        watch_status(book_workspace, "default", console)
    assert "State: running" in output.getvalue()
    assert "Ctrl+C to stop" in output.getvalue()


@pytest.mark.parametrize("flag", ["--json", "--quiet"])
def test_watch_rejects_incompatible_output_modes_before_reading_a_manifest(
    flag: str,
) -> None:
    result = CliRunner().invoke(
        main, [flag, "status", "/missing/yakbox.toml", "--watch"]
    )
    assert result.exit_code == 2
    assert "--watch cannot be combined" in result.output
    if flag == "--json":
        payload = json.loads(result.output)
        validate_contract("cli-output", payload)
        assert payload["error"]["code"] == "usage_error"


def test_plain_status_preserves_its_single_json_document(book_workspace: Path) -> None:
    result = CliRunner().invoke(
        main, ["--json", "status", str(book_workspace / "yakbox.toml")]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    assert payload["data"] == {
        "runs": [],
        "managed_artifacts": 0,
        "managed_bytes": 0,
        "unknown_files": [],
    }


def test_watch_rejects_a_missing_target_before_polling(book_workspace: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "status",
            str(book_workspace / "yakbox.toml"),
            "--target",
            "missing",
            "--watch",
        ],
    )
    assert result.exit_code == 1
    assert "missing" in result.output
    assert not (book_workspace / ".yakbox").exists()


def test_watch_stops_on_journal_corruption_without_repairing_it(
    book_workspace: Path,
) -> None:
    journal = _active_run(book_workspace)
    with journal.path.open("ab") as stream:
        stream.write(b"invalid record\n")
    original = journal.path.read_bytes()
    result = CliRunner().invoke(
        main, ["status", str(book_workspace / "yakbox.toml"), "--watch"]
    )
    assert result.exit_code == 1
    assert "Corrupt journal record" in result.output
    assert journal.path.read_bytes() == original
    assert not journal.path.with_suffix(".torn").exists()


def test_unlocked_stopped_run_is_not_displayed_as_currently_running(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _active_run(book_workspace)

    def interrupt(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("yakbox.cli_status.time.sleep", interrupt)
    result = CliRunner().invoke(
        main, ["status", str(book_workspace / "yakbox.toml"), "--watch"]
    )
    assert result.exit_code == 130
    assert "State: stopped" in result.output
    assert "Running:" not in result.output
