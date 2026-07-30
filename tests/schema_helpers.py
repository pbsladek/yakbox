from __future__ import annotations

from jsonschema import Draft202012Validator, FormatChecker

from yakbox.schemas import load_schema


def validate_contract(name: str, value: object) -> None:
    Draft202012Validator(
        load_schema(name),
        format_checker=FormatChecker(),
    ).validate(value)
