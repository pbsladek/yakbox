from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from tests.schema_helpers import validate_contract

from yakbox.cli import main
from yakbox.errors import ValidationError


@pytest.mark.parametrize(
    "command",
    [
        ["validate"],
        ["plan"],
        ["preview", "--text", "Hello"],
        ["audition", "--text", "Hello", "--profile", "default"],
        ["inspect"],
        ["assemble"],
        ["release", "check"],
        ["release", "diff", "left", "right"],
    ],
)
def test_cli_domain_failures_return_structured_errors_before_producing_artifacts(
    monkeypatch: pytest.MonkeyPatch, command: list[str]
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise ValidationError("Invalid workspace fixture")

    monkeypatch.setattr("yakbox.cli.load_manifest", fail)
    result = CliRunner().invoke(main, ["--json", *command])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    validate_contract("cli-output", payload)
    assert payload["error"]["code"] == "validation_error"
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"
