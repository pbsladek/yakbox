from __future__ import annotations

from collections.abc import Callable

import pytest
from jsonschema import Draft202012Validator, ValidationError

from yakbox.schemas import load_schema, schema_names


def test_all_packaged_schemas_are_valid_draft_2020_12() -> None:
    names = schema_names()
    assert names
    for name in names:
        schema = load_schema(name)
        Draft202012Validator.check_schema(schema)
        assert schema["$id"] == (f"https://yakbox.dev/schemas/{name}-v1.schema.json")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("exit_code"),
        lambda value: value.pop("data"),
        lambda value: value.update(error={"code": "failure", "message": "bad"}),
        lambda value: value.update(unexpected=True),
        lambda value: value.update(**{"$schema": "https://example.invalid/schema"}),
        lambda value: value.update(exit_code=1),
    ],
)
def test_cli_output_schema_rejects_contract_violations(
    mutation: Callable[[dict[str, object]], object],
) -> None:
    value: dict[str, object] = {
        "$schema": "https://yakbox.dev/schemas/cli-output-v1.schema.json",
        "schema_version": 1,
        "yakbox_version": "0.1.0",
        "timestamp": "2026-07-31T00:00:00+00:00",
        "command": "plan",
        "status": "ok",
        "exit_code": 0,
        "data": {},
    }
    mutation(value)

    with pytest.raises(ValidationError):
        Draft202012Validator(load_schema("cli-output")).validate(value)


def test_cli_error_shape_requires_stable_code_and_message() -> None:
    value = {
        "$schema": "https://yakbox.dev/schemas/cli-output-v1.schema.json",
        "schema_version": 1,
        "yakbox_version": "0.1.0",
        "timestamp": "2026-07-31T00:00:00+00:00",
        "command": "build",
        "status": "error",
        "exit_code": 1,
        "error": {"anything": 42},
    }

    with pytest.raises(ValidationError):
        Draft202012Validator(load_schema("cli-output")).validate(value)
