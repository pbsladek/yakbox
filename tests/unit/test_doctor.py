from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
import respx
from tests.schema_helpers import validate_contract

from yakbox.diagnostics import run_doctor


@pytest.mark.asyncio
async def test_doctor_is_offline_by_default_and_marks_policy_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAKBOX_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("RESEMBLE_API_KEY", "secret")
    with patch(
        "yakbox.diagnostics.checks.httpx.AsyncClient",
        side_effect=AssertionError("offline Doctor must not construct an HTTP client"),
    ):
        report = await run_doctor(backend="resemble")

    network = next(
        item for item in report.diagnostics if item.id == "backend.resemble.network"
    )
    assert network.skipped_by_policy
    assert network.status == "skip"
    validate_contract("doctor-report", report.to_dict())


@pytest.mark.asyncio
async def test_doctor_network_uses_only_read_only_voice_list_and_redacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "doctor-secret"  # noqa: S105 - synthetic redaction fixture
    monkeypatch.setenv("YAKBOX_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("RESEMBLE_API_KEY", secret)
    with respx.mock(assert_all_called=True) as router:
        route = router.get("https://app.resemble.ai/api/v2/voices").mock(
            return_value=httpx.Response(200, json={"items": []})
        )
        report = await run_doctor(backend="resemble", network=True)

    assert route.called
    request = route.calls[0].request
    assert request.method == "GET"
    assert dict(request.url.params) == {"page": "1", "page_size": "1"}
    value = report.to_dict()
    assert secret not in str(value)
    network = next(
        item for item in report.diagnostics if item.id == "backend.resemble.network"
    )
    assert network.evidence is not None
    assert network.evidence["mutating"] is False
    validate_contract("doctor-report", value)


@pytest.mark.asyncio
async def test_deep_local_doctor_inspects_runtime_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    torch = SimpleNamespace(
        __version__="test",
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setattr(
        "yakbox.diagnostics.checks.importlib.util.find_spec",
        lambda name: object() if name == "chatterbox" else None,
    )

    def imported_module(name: str) -> object:
        imported.append(name)
        assert name == "torch"
        return torch

    monkeypatch.setattr(
        "yakbox.diagnostics.checks.importlib.import_module",
        imported_module,
    )
    report = await run_doctor(backend="chatterbox-local", deep=True)

    assert imported == ["torch"]
    runtime = next(
        item for item in report.diagnostics if item.id == "backend.chatterbox.runtime"
    )
    assert runtime.status == "pass"
    assert runtime.evidence is not None
    assert runtime.evidence["model_loaded"] is False
