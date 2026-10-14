from __future__ import annotations

import json
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import NoReturn

import pytest
from click.testing import CliRunner
from tests.schema_helpers import validate_contract

from yakbox._files import atomic_write_bytes, sha256_file
from yakbox.audiobook.artifacts import (
    ArtifactKind,
    ArtifactRecord,
    write_artifact_record,
)
from yakbox.audiobook.manifest import load_manifest
from yakbox.cli import main
from yakbox.diagnostics import (
    Diagnostic,
    DiagnosticSeverity,
    DiagnosticStatus,
    DoctorReport,
)


@pytest.mark.parametrize(
    ("title", "author"),
    [
        ("My Audiobook", "Author Name"),
        ('A "quoted" book\\volume', "Zoë\nSecond line\t\x7f"),
    ],
)
def test_init_custom_metadata_round_trips_and_validates(
    tmp_path: Path, title: str, author: str
) -> None:
    workspace = tmp_path / "book"
    result = CliRunner().invoke(
        main, ["--json", "init", str(workspace), "--title", title, "--author", author]
    )
    assert result.exit_code == 0, result.output
    validate_contract("cli-output", json.loads(result.output))
    manifest = load_manifest(workspace / "yakbox.toml")
    assert manifest.book.title == title
    assert manifest.book.author == author
    assert (
        tomllib.loads((workspace / "yakbox.toml").read_text())["book"]["title"] == title
    )
    validated = CliRunner().invoke(main, ["validate", str(manifest.path)])
    assert validated.exit_code == 0, validated.output


@pytest.mark.parametrize("option", ["--title", "--author"])
def test_init_rejects_blank_metadata_before_writing(
    tmp_path: Path, option: str
) -> None:
    workspace = tmp_path / "book"
    result = CliRunner().invoke(main, ["--json", "init", str(workspace), option, "  "])
    assert result.exit_code == 2
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    assert payload["error"]["code"] == "invalid_argument"
    assert not workspace.exists()


def test_init_refuses_to_replace_an_existing_project(tmp_path: Path) -> None:
    runner = CliRunner()
    assert runner.invoke(main, ["init", str(tmp_path)]).exit_code == 0
    before = (tmp_path / "yakbox.toml").read_bytes()
    result = runner.invoke(main, ["init", str(tmp_path), "--title", "Changed"])
    assert result.exit_code == 1
    assert (tmp_path / "yakbox.toml").read_bytes() == before


@pytest.fixture
def listed_artifacts(book_workspace: Path) -> Path:
    output = book_workspace / "build" / "yakbox"
    for name, kind, voice, voices in (
        ("a", ArtifactKind.RAW, None, ()),
        ("b", ArtifactKind.RAW, "narrator", ()),
        ("c", ArtifactKind.DELIVERY, None, ("narrator", "guest")),
        ("d", ArtifactKind.RAW, "guest", ()),
    ):
        path = output / f"{name}.wav"
        atomic_write_bytes(path, b"artifact fixture")
        record = ArtifactRecord(
            schema_version=1,
            id=name,
            kind=kind,
            path=path,
            sha256=sha256_file(path),
            size=path.stat().st_size,
            fingerprint="b" * 64,
            target="default",
            run_id="test-run",
            protected=False,
            logical_voice=voice,
            logical_voices=voices,
            media_type="audio/mpeg" if kind is ArtifactKind.DELIVERY else "audio/wav",
        )
        write_artifact_record(record, root=output)
    return book_workspace / "yakbox.toml"


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ([], ["a", "b", "c", "d"]),
        (["--limit", "2"], ["a", "b"]),
        (["--limit", "10"], ["a", "b", "c", "d"]),
        (["--voice", "narrator"], ["b", "c"]),
        (["--voice", "guest"], ["c", "d"]),
        (["--voice", "missing"], []),
        (["--voice", "narrator", "--limit", "1"], ["b"]),
        (["--kind", "delivery", "--voice", "narrator", "--limit", "1"], ["c"]),
    ],
)
def test_artifact_list_filters_before_limit_without_changing_files(
    listed_artifacts: Path, options: list[str], expected: list[str]
) -> None:
    output = listed_artifacts.parent / "build" / "yakbox"
    before = {path: path.read_bytes() for path in output.iterdir()}
    result = CliRunner().invoke(
        main, ["--json", "artifacts", "list", str(listed_artifacts), *options]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    assert [record["id"] for record in payload["data"]["artifacts"]] == expected
    for record in payload["data"]["artifacts"]:
        validate_contract("audiobook-artifact", record)
    assert {path: path.read_bytes() for path in output.iterdir()} == before


@pytest.mark.parametrize("limit", ["0", "-1", "not-a-number"])
def test_artifact_list_rejects_invalid_limits(limit: str) -> None:
    result = CliRunner().invoke(main, ["--json", "artifacts", "list", "--limit", limit])
    assert result.exit_code == 2
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    assert payload["error"]["code"] == "invalid_argument"


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize(
    ("status", "strict", "exit_code"),
    [
        (DiagnosticStatus.WARN, False, 0),
        (DiagnosticStatus.WARN, True, 1),
        (DiagnosticStatus.PASS, True, 0),
        (DiagnosticStatus.SKIP, True, 0),
        (DiagnosticStatus.FAIL, False, 1),
        (DiagnosticStatus.FAIL, True, 1),
    ],
)
def test_doctor_strict_policy_preserves_diagnostics_and_json_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    status: DiagnosticStatus,
    strict: bool,
    exit_code: int,
    json_output: bool,
) -> None:
    report = DoctorReport(
        schema_version=1,
        diagnostics=(
            Diagnostic("check", status, DiagnosticSeverity.WARNING, "Check result"),
        ),
    )

    async def fake_doctor(*_args: object, **_kwargs: object) -> DoctorReport:
        return report

    monkeypatch.setattr("yakbox.cli.run_doctor", fake_doctor)
    arguments = ["--json"] if json_output else []
    arguments.append("doctor")
    if strict:
        arguments.append("--strict")
    result = CliRunner().invoke(main, arguments)
    assert result.exit_code == exit_code, result.output
    if json_output:
        payload = json.loads(result.output)
        validate_contract("cli-output", payload)
        validate_contract("doctor-report", payload["data"])
        assert payload["exit_code"] == exit_code
        assert payload["status"] == ("ok" if exit_code == 0 else "partial_failure")
        assert payload["data"]["healthy"] == report.healthy
        assert payload["data"]["diagnostics"][0]["status"] == status.value
    else:
        assert "Check result" in result.output


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_completion_outputs_usable_raw_script_and_json(shell: str) -> None:
    runner = CliRunner()
    raw = runner.invoke(main, ["completion", "--shell", shell], terminal_width=20)
    assert raw.exit_code == 0, raw.output
    assert f"{shell}_complete" in raw.output
    assert "_YAKBOX_COMPLETE" in raw.output
    encoded = runner.invoke(main, ["--json", "completion", "--shell", shell])
    assert encoded.exit_code == 0, encoded.output
    payload = json.loads(encoded.output)
    validate_contract("cli-output", payload)
    assert payload["command"] == "completion"
    assert payload["data"]["shell"] == shell
    assert payload["data"]["script"] + "\n" == raw.output
    executable = shutil.which(shell)
    if executable is not None:
        syntax = subprocess.run(  # noqa: S603 - installed shell, generated completion on stdin
            [executable, "-n"],
            input=raw.output,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        assert syntax.returncode == 0, syntax.stderr
    quiet = runner.invoke(main, ["--quiet", "completion", "--shell", shell])
    assert quiet.exit_code == 0
    assert quiet.output == ""


@pytest.mark.parametrize("arguments", [[], ["--shell", "unknown"]])
def test_completion_requires_a_supported_shell(arguments: list[str]) -> None:
    result = CliRunner().invoke(main, ["--json", "completion", *arguments])
    assert result.exit_code == 2
    validate_contract("cli-output", json.loads(result.output))


def test_completion_generation_does_not_probe_the_installed_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_probe(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("Script generation must not invoke a local shell")

    monkeypatch.setattr("subprocess.run", reject_probe)
    result = CliRunner().invoke(main, ["--json", "completion", "--shell", "bash"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    validate_contract("cli-output", payload)
    assert "bash_complete" in payload["data"]["script"]
    assert result.stderr == ""
