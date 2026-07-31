from __future__ import annotations

import importlib.util
import json
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from tests.schema_helpers import validate_contract

from yakbox.cli import main
from yakbox.cloud import Page


def test_help_does_not_import_torch() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "build" in result.output
    assert "chatterbox" not in result.output.casefold()


def test_json_usage_errors_use_stable_envelope_and_exit_two() -> None:
    result = CliRunner().invoke(
        main,
        ["--json", "build", "--from", "not-a-stage"],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 2
    assert payload["error"]["code"] == "BadParameter"
    validate_contract("cli-output", payload)


def test_ctrl_c_uses_exit_130_in_human_and_json_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "script.txt"
    script.write_text("Tiny line.\n", encoding="utf-8")
    monkeypatch.setenv("RESEMBLE_API_KEY", "test")

    def interrupted(coroutine: Coroutine[object, object, object]) -> object:
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr("yakbox.cli.asyncio.run", interrupted)
    human = CliRunner().invoke(main, ["cloud", "batch", str(script)])
    assert human.exit_code == 130
    assert "Aborted!" in human.output

    machine = CliRunner().invoke(
        main,
        ["--json", "cloud", "batch", str(script)],
    )
    assert machine.exit_code == 130
    payload = json.loads(machine.output)
    assert payload["status"] == "aborted"
    assert payload["exit_code"] == 130
    validate_contract("cli-output", payload)


def test_init_validate_plan_and_doctor_json(tmp_path: Path) -> None:
    runner = CliRunner()
    workspace = tmp_path / "book"
    initialized = runner.invoke(main, ["init", str(workspace)])
    assert initialized.exit_code == 0, initialized.output

    validated = runner.invoke(main, ["validate", str(workspace / "yakbox.toml")])
    assert validated.exit_code == 0, validated.output

    planned = runner.invoke(main, ["--json", "plan", str(workspace / "yakbox.toml")])
    assert planned.exit_code == 0, planned.output
    payload = json.loads(planned.output)
    assert payload["schema_version"] == 1
    assert payload["data"]["nodes"]
    validate_contract("cli-output", payload)
    validate_contract("audiobook-plan", payload["data"])

    doctor = runner.invoke(main, ["--json", "doctor", str(workspace / "yakbox.toml")])
    assert doctor.exit_code == 0, doctor.output
    report = json.loads(doctor.output)
    assert report["data"]["healthy"] is True
    validate_contract("cli-output", report)
    validate_contract("doctor-report", report["data"])


def test_build_dry_run_and_preview_are_audiobook_first(tmp_path: Path) -> None:
    runner = CliRunner()
    workspace = tmp_path / "book"
    assert runner.invoke(main, ["init", str(workspace)]).exit_code == 0
    manifest = workspace / "yakbox.toml"

    planned = runner.invoke(
        main,
        ["--json", "build", str(manifest), "--dry-run"],
    )
    assert planned.exit_code == 0, planned.output
    build_payload = json.loads(planned.output)
    assert build_payload["exit_code"] == 0
    assert build_payload["data"]["status"] == "planned"
    assert build_payload["data"]["preflight"]["pending_nodes"] == 4
    assert not (workspace / ".yakbox").exists()
    validate_contract("cli-output", build_payload)

    draft = runner.invoke(
        main,
        ["--json", "build", str(manifest), "--mode", "draft", "--dry-run"],
    )
    assert draft.exit_code == 0, draft.output
    draft_payload = json.loads(draft.output)
    assert draft_payload["data"]["target"] == "draft"
    assert draft_payload["data"]["preflight"]["planned_nodes"] == 1

    preview = runner.invoke(
        main,
        ["--json", "preview", str(manifest), "--text", "A tiny preview."],
    )
    assert preview.exit_code == 0, preview.output
    preview_payload = json.loads(preview.output)
    artifact = preview_payload["data"]["artifact"]
    assert artifact["kind"] == "preview"
    assert "previews" in Path(artifact["path"]).parts
    validate_contract("cli-output", preview_payload)
    validate_contract("audiobook-artifact", artifact)


def test_cloud_batch_confirmation_happens_before_credentials_or_network(
    tmp_path: Path,
) -> None:
    script = tmp_path / "script.txt"
    script.write_text("Hello hosted world.\n", encoding="utf-8")
    output = tmp_path / "output"
    runner = CliRunner()

    refused = runner.invoke(
        main,
        [
            "cloud",
            "batch",
            str(script),
            "--out-dir",
            str(output),
            "--voice-uuid",
            "voice",
            "--confirm-above-characters",
            "0",
        ],
    )
    assert refused.exit_code == 1
    assert "--yes" in refused.output
    assert not output.exists()

    dry_run = runner.invoke(
        main,
        [
            "--json",
            "cloud",
            "batch",
            str(script),
            "--out-dir",
            str(output),
            "--voice-uuid",
            "voice",
            "--confirm-above-characters",
            "0",
            "--dry-run",
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    payload = json.loads(dry_run.output)
    assert payload["data"]["preflight"]["logical_items"] == 1
    assert payload["data"]["results"][0]["status"] == "not_run"
    assert not output.exists()
    validate_contract("cli-output", payload)
    validate_contract("batch-report", payload["data"])


def test_cloud_batch_json_reports_partial_failure_and_effective_exit_code(
    tmp_path: Path,
) -> None:
    script = tmp_path / "script.txt"
    script.write_text("x" * 3_001, encoding="utf-8")
    runner = CliRunner()
    common = [
        "--json",
        "cloud",
        "batch",
        str(script),
        "--voice-uuid",
        "voice",
        "--dry-run",
        "--no-report",
    ]

    failed = runner.invoke(main, common)
    assert failed.exit_code == 1, failed.output
    failed_payload = json.loads(failed.output)
    assert failed_payload["status"] == "partial_failure"
    assert failed_payload["exit_code"] == 1
    assert failed_payload["data"]["summary"]["failed"] == 1
    validate_contract("cli-output", failed_payload)

    ignored = runner.invoke(main, [*common, "--ignore-errors"])
    assert ignored.exit_code == 0, ignored.output
    ignored_payload = json.loads(ignored.output)
    assert ignored_payload["status"] == "partial_failure"
    assert ignored_payload["exit_code"] == 0
    assert ignored_payload["data"]["summary"]["failed"] == 1
    validate_contract("cli-output", ignored_payload)


def test_cloud_profile_reads_optional_keyring_without_exposing_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESEMBLE_API_KEY", raising=False)
    monkeypatch.setenv("YAKBOX_CONFIG", str(tmp_path / "missing.toml"))
    captured: list[str] = []
    keyring = SimpleNamespace(
        get_password=lambda service, profile: (
            "keyring-secret"
            if (service, profile) == ("yakbox/resemble", "studio")
            else None
        )
    )
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        "yakbox.cli.importlib.util.find_spec",
        lambda name: object() if name == "keyring" else real_find_spec(name),
    )
    monkeypatch.setattr(
        "yakbox.cli.importlib.import_module",
        lambda name: keyring if name == "keyring" else __import__(name),
    )

    async def list_projects(api_key: str, page: int, page_size: int) -> Page[object]:
        captured.append(api_key)
        assert (page, page_size) == (1, 10)
        return Page(items=(), page=1, page_count=1, total_results=0)

    monkeypatch.setattr("yakbox.cli._list_projects", list_projects)
    result = CliRunner().invoke(
        main,
        ["--json", "cloud", "--profile", "studio", "projects", "list"],
    )

    assert result.exit_code == 0, result.output
    assert captured == ["keyring-secret"]
    assert "keyring-secret" not in result.output
