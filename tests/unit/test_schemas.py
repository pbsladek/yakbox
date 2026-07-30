from __future__ import annotations

from jsonschema import Draft202012Validator

from yakbox.schemas import load_schema, schema_names


def test_all_packaged_schemas_are_valid_draft_2020_12() -> None:
    names = schema_names()
    assert names
    for name in names:
        schema = load_schema(name)
        Draft202012Validator.check_schema(schema)
        assert schema["$id"] == (f"https://yakbox.dev/schemas/{name}-v1.schema.json")
