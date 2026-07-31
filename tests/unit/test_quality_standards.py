from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
SCHEDULED = (ROOT / ".github" / "workflows" / "scheduled.yml").read_text(
    encoding="utf-8"
)


def test_quality_gates_are_configured() -> None:
    tool = PYPROJECT["tool"]
    lint = tool["ruff"]["lint"]

    assert tool["ruff"]["target-version"] == "py313"
    assert tool["ty"]["environment"]["python-version"] == "3.13"
    assert tool["ruff"]["lint"]["mccabe"]["max-complexity"] <= 10
    assert tool["ruff"]["lint"]["pylint"]["max-args"] <= 20
    assert tool["ruff"]["lint"]["pylint"]["max-positional-args"] <= 5
    assert tool["coverage"]["run"]["branch"] is True
    assert tool["coverage"]["report"]["fail_under"] >= 75

    import_linter = tool["importlinter"]
    assert import_linter["root_package"] == "yakbox"
    contracts = {contract["id"]: contract for contract in import_linter["contracts"]}
    assert {
        "audio-is-backend-neutral",
        "backends-do-not-directly-couple",
        "cli-is-an-entrypoint",
        "speech-is-backend-neutral",
        "subpackage-imports-are-acyclic",
    } <= contracts.keys()
    assert contracts["cli-is-an-entrypoint"]["ignore_imports"] == [
        "yakbox.__main__ -> yakbox.cli"
    ]

    mutmut = tool["mutmut"]
    assert mutmut["source_paths"] == ["src/yakbox"]
    assert {
        "src/yakbox/_files.py",
        "src/yakbox/contracts.py",
        "src/yakbox/audiobook/artifacts.py",
        "src/yakbox/audiobook/journal.py",
        "src/yakbox/speech/guardrails.py",
    } <= set(mutmut["only_mutate"])
    assert mutmut["mutate_only_covered_lines"] is True
    assert "uv run lint-imports --no-cache" in CI
    assert "tests/package_sdk_consumer.py" in CI
    assert "Type-check a public SDK consumer against the wheel" in CI
    assert "--project /tmp" in CI
    assert "--python .sdk-venv/bin/python" in CI
    assert "uv run mutmut run" in SCHEDULED
    assert "s['killed'] / s['total'] >= 0.80" in SCHEDULED
    assert {"ANN", "ARG", "BLE", "C90", "RET", "S"} <= set(lint["select"])


def test_production_code_has_no_file_wide_lint_exemptions() -> None:
    ignored = PYPROJECT["tool"]["ruff"]["lint"].get("per-file-ignores", {})

    assert not [pattern for pattern in ignored if pattern.startswith("src/")]


def test_inline_lint_exemptions_are_narrow_and_explained() -> None:
    invalid: list[str] = []
    forbidden = {"C901", "PLR0911", "PLR0912", "PLR0915"}
    pattern = re.compile(r"# noqa: (?P<codes>[A-Z0-9,]+) - (?P<reason>.+)$")

    for path in (ROOT / "src").rglob("*.py"):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if "# noqa" not in line:
                continue
            match = pattern.search(line)
            codes = set(match.group("codes").split(",")) if match else set()
            if match is None or codes & forbidden:
                invalid.append(f"{path.relative_to(ROOT)}:{line_number}")

    assert invalid == []
