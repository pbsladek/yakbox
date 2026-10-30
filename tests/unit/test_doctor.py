from __future__ import annotations

from types import SimpleNamespace

import pytest

from yakbox.diagnostics import run_doctor


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
