from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner
from tests.schema_helpers import validate_contract

from yakbox.audiobook import build_audiobook, load_manifest
from yakbox.cli import main
from yakbox.config import load_config
from yakbox.diagnostics import run_doctor
from yakbox.errors import BackendUnavailableError, ValidationError
from yakbox.speech import open_speech_backend

RETIRED_BACKENDS = ("resemble", "cloud", "remote", "chatterbox-remote")


@pytest.fixture
def no_provider_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_client(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Local workflows must not open provider clients")

    monkeypatch.setattr(httpx, "AsyncClient", reject_client)
    monkeypatch.setattr(httpx, "Client", reject_client)
    monkeypatch.setenv("RESEMBLE_API_KEY", "unused-retired-token")


@pytest.mark.usefixtures("no_provider_clients")
@pytest.mark.parametrize("backend", RETIRED_BACKENDS)
@pytest.mark.asyncio
async def test_retired_backend_factory_rejects_without_network(backend: str) -> None:
    with pytest.raises(BackendUnavailableError, match=r"removed.*chatterbox-local"):
        async with open_speech_backend(backend.upper()):
            pytest.fail("A retired backend must never be opened")


@pytest.mark.usefixtures("no_provider_clients")
@pytest.mark.parametrize("backend", RETIRED_BACKENDS)
def test_retired_manifest_profiles_fail_before_creating_state(
    book_workspace: Path, backend: str
) -> None:
    path = book_workspace / "yakbox.toml"
    path.write_text(
        path.read_text().replace('backend = "fake"', f'backend = "{backend}"')
    )
    with pytest.raises(ValidationError, match=r"no longer supported.*chatterbox-local"):
        load_manifest(path)
    result = CliRunner().invoke(main, ["--json", "build", str(path)])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    assert payload["error"]["code"] == "validation_error"
    assert "chatterbox-local" in payload["error"]["message"]
    assert not (book_workspace / ".yakbox").exists()
    assert not (book_workspace / "build").exists()


@pytest.mark.usefixtures("no_provider_clients")
@pytest.mark.parametrize(
    "command", [["cloud", "tts", "hello"], ["config", "auth", "status"]]
)
def test_retired_command_families_are_usage_errors(command: list[str]) -> None:
    result = CliRunner().invoke(main, ["--json", *command])
    assert result.exit_code == 2, result.output
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    assert payload["error"]["code"] == "usage_error"
    assert "No such command" in payload["error"]["message"]


@pytest.mark.usefixtures("no_provider_clients")
@pytest.mark.parametrize("backend", RETIRED_BACKENDS)
def test_retired_global_default_cannot_enable_hosted_tts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        f'[defaults]\nbackend = "{backend}"\n[cloud]\napi_key = "retired-secret"\n'
    )
    monkeypatch.setenv("YAKBOX_CONFIG", str(config))
    destination = tmp_path / "speech.wav"
    result = CliRunner().invoke(
        main, ["--json", "tts", "hello", "--out", str(destination)]
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "backend_unavailable"
    assert "retired-secret" not in result.output
    assert not destination.exists()


@pytest.mark.usefixtures("no_provider_clients")
@pytest.mark.asyncio
async def test_local_build_preserves_report_contracts_and_ignores_old_cloud_settings(
    book_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = book_workspace / "config.toml"
    config.write_text(
        '[defaults]\nbackend = "fake"\n[cloud]\n'
        'api_key = "retired-secret"\nconcurrency = -1\n'
    )
    monkeypatch.setenv("YAKBOX_CONFIG", str(config))
    monkeypatch.setenv("YAKBOX_CLOUD_CONCURRENCY", "invalid-retired-setting")
    assert load_config().default_backend == "fake"
    result = await build_audiobook(load_manifest(book_workspace / "yakbox.toml"))
    assert result.status == "complete"
    assert result.hosted_usage is None
    assert result.preflight.hosted_work is None
    report = json.loads((result.run_directory / "run.json").read_text())
    validate_contract("audiobook-run", report)
    assert report["hosted_usage"] is None
    assert report["preflight"]["hosted_work"] is None
    assert "retired-secret" not in json.dumps(report)


@pytest.mark.usefixtures("no_provider_clients")
def test_cli_lists_only_local_synthesis_backends_and_keeps_whisper_commands() -> None:
    result = CliRunner().invoke(main, ["--json", "backends", "list"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {item["name"] for item in payload["data"]["backends"]} == {
        "fake",
        "chatterbox-local",
    }
    help_result = CliRunner().invoke(main, ["whisper", "--help"])
    assert help_result.exit_code == 0
    assert "models" in help_result.output
    assert "inspect" in help_result.output


@pytest.mark.usefixtures("no_provider_clients")
@pytest.mark.parametrize("backend", RETIRED_BACKENDS)
@pytest.mark.asyncio
async def test_doctor_reports_retired_backends_as_unavailable(backend: str) -> None:
    report = await run_doctor(backend=backend.upper())
    diagnostic = next(
        item for item in report.diagnostics if item.id == "backend.unsupported"
    )
    assert diagnostic.status == "fail"
    assert not report.healthy
    validate_contract("doctor-report", report.to_dict())


@pytest.mark.parametrize(
    "command",
    [
        ["build", "--max-provider-requests", "1"],
        ["tts", "hello", "--max-estimated-spend", "1"],
        ["preview", "--api-key", "unused"],
        ["audition", "--api-key", "unused"],
        ["doctor", "--network"],
    ],
)
def test_hosted_only_options_are_removed(command: list[str]) -> None:
    result = CliRunner().invoke(main, ["--json", *command])
    assert result.exit_code == 2, result.output
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    assert payload["error"]["code"] == "unknown_option"
