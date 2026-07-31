from __future__ import annotations

import importlib
import inspect
import json
import re
from contextlib import suppress
from dataclasses import is_dataclass
from enum import Enum
from pathlib import Path
from typing import cast

import yakbox.cloud
import yakbox.speech

ROOT = Path(__file__).parents[2]
MODULES = (
    "yakbox",
    "yakbox.audio",
    "yakbox.audiobook",
    "yakbox.cloud",
    "yakbox.diagnostics",
    "yakbox.speech",
)


def _public_kind(value: object) -> str:
    if inspect.isclass(value) and issubclass(value, Exception):
        return "exception"
    if inspect.isclass(value) and issubclass(value, Enum):
        return "enum"
    if inspect.isclass(value):
        return "class"
    if inspect.isfunction(value):
        return "function"
    return "callable_type" if callable(value) else "type_alias"


def _public_item(value: object) -> dict[str, object]:
    kind = _public_kind(value)
    item: dict[str, object] = {"kind": kind}
    if callable(value):
        with suppress(TypeError, ValueError):
            item["signature"] = str(inspect.signature(value))
    if kind == "enum":
        enum_type = cast(type[Enum], value)
        item["members"] = {member.name: member.value for member in enum_type}
    if kind == "exception":
        item["code"] = getattr(value, "code", None)
    return item


def _module_exports(module_name: str) -> dict[str, object]:
    module = importlib.import_module(module_name)
    return {
        name: {"kind": "value"}
        if name == "__version__"
        else _public_item(getattr(module, name))
        for name in module.__all__
    }


def _public_api() -> dict[str, object]:
    modules = {
        module_name: {"exports": _module_exports(module_name)}
        for module_name in MODULES
    }
    return {"schema_version": 1, "modules": modules}


def test_public_api_matches_reviewed_manifest() -> None:
    expected = json.loads(
        (ROOT / "tests" / "public-api-v1.json").read_text(encoding="utf-8")
    )

    assert _public_api() == expected


def test_every_public_callable_has_a_docstring() -> None:
    missing: list[str] = []
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            value = getattr(module, name)
            if callable(value) and not inspect.getdoc(value):
                missing.append(f"{module_name}.{name}")

    assert missing == []


def test_public_dataclasses_have_semantic_docstrings() -> None:
    generated: list[str] = []
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            value = getattr(module, name)
            if inspect.isclass(value) and is_dataclass(value):
                docstring = value.__doc__ or ""
                if docstring.startswith(f"{value.__name__}("):
                    generated.append(f"{module_name}.{name}")

    assert generated == []


def test_every_public_sdk_method_has_a_docstring() -> None:
    missing: list[str] = []
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            value = getattr(module, name)
            if not inspect.isclass(value):
                continue
            for method_name, member in value.__dict__.items():
                if method_name.startswith("_"):
                    continue
                method = member.fget if isinstance(member, property) else member
                if inspect.isfunction(method) and not inspect.getdoc(method):
                    missing.append(f"{module_name}.{name}.{method_name}")

    assert missing == []


def test_sdk_reference_names_every_public_export() -> None:
    documentation = (ROOT / "docs" / "python-api.md").read_text(encoding="utf-8")
    missing = [
        f"{module_name}.{name}"
        for module_name in MODULES
        for name in importlib.import_module(module_name).__all__
        if f"`{name}`" not in documentation
    ]

    assert missing == []


def test_error_codes_and_shared_audio_types_are_stable() -> None:
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            value = getattr(module, name)
            if inspect.isclass(value) and issubclass(value, Exception):
                assert re.fullmatch(r"[a-z][a-z0-9_]*", value.code)

    assert yakbox.cloud.AudioFormat is yakbox.speech.AudioFormat
    assert yakbox.cloud.Precision is yakbox.speech.Precision


def test_typed_distribution_marker_exists() -> None:
    assert (ROOT / "src" / "yakbox" / "py.typed").is_file()
