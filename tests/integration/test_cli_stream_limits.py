from __future__ import annotations

import json
from pathlib import Path

import pytest
import respx
from click.testing import CliRunner
from tests.schema_helpers import validate_contract

from yakbox.cli import main
from yakbox.cloud import ClientOptions, ResembleClient


@pytest.mark.parametrize("length", ["17", "invalid"])
def test_oversized_or_malformed_provider_response_is_a_structured_cli_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    length: str,
) -> None:
    monkeypatch.setenv("RESEMBLE_API_KEY", "test-secret")
    monkeypatch.setenv("YAKBOX_CONFIG", str(tmp_path / "missing.toml"))
    route = respx_mock.get("https://app.resemble.ai/api/v2/voices").respond(
        200, headers={"content-length": length}, content=b"private response"
    )

    def client(api_key: str) -> ResembleClient:
        return ResembleClient(
            api_key, options=ClientOptions(max_json_response_bytes=16)
        )

    monkeypatch.setattr("yakbox.cli.ResembleClient", client)

    result = CliRunner().invoke(main, ["--json", "cloud", "voices", "list"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "provider_protocol_error"
    assert payload["command"] == "cloud voices list"
    assert payload["exit_code"] == result.exit_code
    validate_contract("cli-output", payload)
    assert "private response" not in result.output
    assert "test-secret" not in result.output
    assert route.call_count == 1
