from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.shell_completion import BashComplete
from click.testing import CliRunner
from tests.schema_helpers import validate_contract

from yakbox.audiobook import load_manifest
from yakbox.cli import main


def test_init_preserves_existing_pronunciations_and_manuscript(tmp_path: Path) -> None:
    source = tmp_path / "source" / "book.md"
    source.parent.mkdir()
    source.write_bytes(b"# My chapter\n\nMy existing manuscript.\n")
    pronunciation = tmp_path / "pronunciations.toml"
    pronunciation.write_bytes(b"schema_version = 1\n# Keep my dictionary.\n")
    original = {path: path.read_bytes() for path in (source, pronunciation)}
    result = CliRunner().invoke(main, ["--json", "init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    validate_contract("cli-output", json.loads(result.output))
    assert {path: path.read_bytes() for path in original} == original
    assert (
        CliRunner().invoke(main, ["validate", str(tmp_path / "yakbox.toml")]).exit_code
        == 0
    )
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize("relative", ["source", "pronunciations.toml"])
def test_init_rejects_paths_outside_workspace_before_writing(
    tmp_path: Path, relative: str
) -> None:
    workspace = tmp_path / "book"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / relative).symlink_to(outside, target_is_directory=True)
    result = CliRunner().invoke(main, ["--json", "init", str(workspace)])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    assert payload["error"]["code"] == "artifact_error"
    assert not (workspace / "yakbox.toml").exists()
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("narrator", "language"),
    [('Zoë "Reader"\\name\n\t\x7f', "fr-CA"), ("Narrator Name", "en")],
)
def test_init_narrator_and_language_round_trip(
    tmp_path: Path, narrator: str, language: str
) -> None:
    result = CliRunner().invoke(
        main,
        ["init", str(tmp_path), "--narrator", narrator, "--language", language],
    )
    assert result.exit_code == 0, result.output
    loaded = load_manifest(tmp_path / "yakbox.toml")
    assert loaded.book.narrator == narrator
    assert loaded.book.language == language


@pytest.mark.parametrize("option", ["--narrator", "--language"])
def test_init_rejects_blank_extra_metadata(tmp_path: Path, option: str) -> None:
    workspace = tmp_path / "book"
    result = CliRunner().invoke(main, ["--json", "init", str(workspace), option, " \t"])
    assert result.exit_code == 2
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    assert payload["error"]["code"] == "invalid_argument"
    assert not workspace.exists()


@pytest.mark.parametrize("machine", [False, True])
def test_explain_rejects_unknown_artifact(book_workspace: Path, machine: bool) -> None:
    args = ["--json"] if machine else []
    args.extend(["explain", str(book_workspace / "yakbox.toml"), "--artifact", "typo"])
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 1
    assert "No planned artifact matches: typo" in result.output
    if machine:
        payload = json.loads(result.output)
        validate_contract("cli-output", payload)
        assert payload["error"]["code"] == "validation_error"


@pytest.mark.parametrize("selector", ["id", "output"])
def test_explain_accepts_node_id_and_output_path(
    book_workspace: Path, selector: str
) -> None:
    runner = CliRunner()
    manifest = str(book_workspace / "yakbox.toml")
    result = runner.invoke(main, ["--json", "explain", manifest])
    assert result.exit_code == 0, result.output
    node = json.loads(result.output)["data"]["nodes"][0]
    selected = runner.invoke(
        main, ["--json", "explain", manifest, "--artifact", node[selector]]
    )
    assert selected.exit_code == 0, selected.output
    assert json.loads(selected.output)["data"]["nodes"] == [node]


@pytest.mark.parametrize(
    "options", [["plan", "--target", "defalt"], ["preview", "--profile", "defalt"]]
)
def test_cli_suggests_close_manifest_names(
    book_workspace: Path, options: list[str]
) -> None:
    command, *flags = options
    result = CliRunner().invoke(
        main, ["--json", command, str(book_workspace / "yakbox.toml"), *flags]
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    assert payload["error"]["code"] == "validation_error"
    assert "Did you mean 'default'?" in payload["error"]["message"]


@pytest.mark.parametrize("command", ["tts", "preview", "audition"])
@pytest.mark.parametrize("problem", ["missing", "directory", "invalid_utf8"])
@pytest.mark.parametrize("machine", [False, True])
def test_text_file_errors_are_actionable_and_do_not_quote_contents(
    book_workspace: Path, command: str, problem: str, machine: bool
) -> None:
    text_file = book_workspace / "input.txt"
    if problem == "directory":
        text_file.mkdir()
    elif problem == "invalid_utf8":
        text_file.write_bytes(b"private manuscript text \xff")
    args = ["--json"] if machine else []
    args.append(command)
    if command != "tts":
        args.append(str(book_workspace / "yakbox.toml"))
    if command == "audition":
        args.extend(["--profile", "default"])
    args.extend(["--text-file", str(text_file)])
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 2, result.output
    assert "--text-file" in result.output
    assert "private manuscript text" not in result.output
    assert (
        "valid UTF-8" if problem == "invalid_utf8" else "exists and is readable"
    ) in result.output
    if machine:
        payload = json.loads(result.output)
        validate_contract("cli-output", payload)
        assert payload["error"]["code"] == "invalid_argument"
        assert payload["exit_code"] == result.exit_code


@pytest.mark.parametrize("stdin", [False, True])
def test_utf8_text_input_still_generates_audio(tmp_path: Path, stdin: bool) -> None:
    text_file = tmp_path / "input.txt"
    text_file.write_text("A short café visit.", encoding="utf-8")
    output = tmp_path / "speech.wav"
    result = CliRunner().invoke(
        main,
        [
            "--json",
            "tts",
            "--backend",
            "fake",
            "--text-file",
            "-" if stdin else str(text_file),
            "--out",
            str(output),
        ],
        input=text_file.read_text(encoding="utf-8") if stdin else None,
    )
    assert result.exit_code == 0, result.output
    validate_contract("cli-output", json.loads(result.output))
    assert output.is_file()


@pytest.mark.parametrize(
    "options",
    [
        ["--limit", "0"],
        ["--limit", "-1"],
        ["--limit", "bad"],
        ["--latest", "--limit", "1"],
        ["--watch", "--latest"],
        ["--watch", "--limit", "2"],
        ["--interval", "5"],
        ["--exit-when-done"],
        ["--watch", "--interval", "0"],
        ["--watch", "--interval", "-1"],
        ["--watch", "--interval", "nan"],
        ["--watch", "--interval", "inf"],
    ],
)
@pytest.mark.parametrize("machine", [False, True])
def test_status_rejects_invalid_options_before_manifest_access(
    options: list[str],
    machine: bool,
) -> None:
    args = ["--json"] if machine else []
    result = CliRunner().invoke(main, [*args, "status", "/missing.toml", *options])
    assert result.exit_code == 2, result.output
    assert "Unknown target" not in result.output
    if machine:
        payload = json.loads(result.output)
        validate_contract("cli-output", payload)
        assert payload["error"]["code"] in {"usage_error", "invalid_argument"}


@pytest.mark.parametrize(
    ("command", "option", "prefix", "expected"),
    [
        ("build", "--target", "d", ["default", "draft"]),
        ("plan", "--target", "p", ["proof"]),
        ("preview", "--profile", "d", ["default"]),
        ("audition", "--profiles", "", ["default", "guest"]),
        ("repair plan", "--target", "r", ["release"]),
    ],
)
def test_completion_uses_selected_manifest(
    tmp_path: Path, command: str, option: str, prefix: str, expected: list[str]
) -> None:
    assert CliRunner().invoke(main, ["init", str(tmp_path)]).exit_code == 0
    manifest = tmp_path / "yakbox.toml"
    with manifest.open("a") as stream:
        stream.write('\n[profiles.guest]\nbackend = "fake"\nvoice = "narrator"\n')
    completion = BashComplete(main, {}, "yakbox", "_YAKBOX_COMPLETE")
    items = completion.get_completions(
        [*command.split(), str(manifest), option], prefix
    )
    assert [item.value for item in items] == expected


def test_completion_uses_working_directory_and_handles_missing_or_invalid_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    completion = BashComplete(main, {}, "yakbox", "_YAKBOX_COMPLETE")
    args = ["build", "--target"]
    assert completion.get_completions(args, "") == []
    assert CliRunner().invoke(main, ["init"]).exit_code == 0
    assert [item.value for item in completion.get_completions(args, "d")] == [
        "default",
        "draft",
    ]
    (tmp_path / "yakbox.toml").write_text("not valid toml", encoding="utf-8")
    assert completion.get_completions(args, "") == []


@pytest.mark.parametrize(
    "arguments",
    [
        ["release", "diff", "old", "new", "--manifest", "MANIFEST", "--target"],
        ["repair", "locate", "chapter-id", "0.5", "MANIFEST", "--target"],
        ["build", "MANIFEST", "--target=dr"],
        ["build", "--target", "default", "MANIFEST", "--profile"],
    ],
)
def test_completion_resolves_manifest_among_other_arguments(
    tmp_path: Path, arguments: list[str]
) -> None:
    assert CliRunner().invoke(main, ["init", str(tmp_path)]).exit_code == 0
    args = [
        str(tmp_path / "yakbox.toml") if value == "MANIFEST" else value
        for value in arguments
    ]
    completion = BashComplete(main, {}, "yakbox", "_YAKBOX_COMPLETE")
    if args[-1] == "--target=dr":
        items = completion.get_completions(args[:-1], args[-1])
        expected = ["draft"]
    else:
        items = completion.get_completions(args, "d")
        expected = ["default"] if args[-1] == "--profile" else ["default", "draft"]
    assert [item.value for item in items] == expected


def test_completion_protocol_emits_names_without_running_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert CliRunner().invoke(main, ["init"]).exit_code == 0
    result = CliRunner().invoke(
        main,
        [],
        prog_name="yakbox",
        env={
            "_YAKBOX_COMPLETE": "bash_complete",
            "COMP_WORDS": "yakbox build --target d",
            "COMP_CWORD": "3",
        },
    )
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["plain,default", "plain,draft"]
    assert not (tmp_path / ".yakbox").exists()
