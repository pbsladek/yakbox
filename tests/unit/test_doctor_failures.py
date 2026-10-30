from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from tests.unit.test_whisper_tooling import _alignment

from yakbox.diagnostics import checks, run_doctor
from yakbox.errors import BackendUnavailableError


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_doctor_deep_whisper_smoke_is_bounded_and_reports_model_failures(
    monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    align = AsyncMock(return_value=_alignment())
    if failure:
        align.side_effect = BackendUnavailableError("Runtime unavailable")
    factory = Mock(return_value=SimpleNamespace(align=align))
    monkeypatch.setattr(checks, "MlxWhisperAligner", factory)
    result = await checks._whisper_deep("fixture", "pinned")
    assert result.status == ("fail" if failure else "pass")
    assert factory.call_args.kwargs["timeout_seconds"] == 180
    assert factory.call_args.kwargs["prompted_timing"] is False
    assert align.call_args.args[1] == ""
    assert not align.call_args.args[0].exists()
    if failure:
        assert result.action
    else:
        assert result.evidence and result.evidence["backend"] == "fake-whisper"


@pytest.mark.asyncio
async def test_doctor_reports_invalid_configuration_without_loading_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[invalid")
    monkeypatch.setenv("YAKBOX_CONFIG", str(config))
    report = await run_doctor(tmp_path / "missing.toml")
    failed = {item.id for item in report.diagnostics if item.status == "fail"}
    assert {"config.load", "workspace.manifest"} <= failed
    assert not report.healthy


@pytest.mark.asyncio
async def test_doctor_reports_workspace_read_and_write_failures(
    book_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checks.os, "access", lambda *_args: False)
    monkeypatch.setattr(
        checks.shutil, "disk_usage", lambda _root: SimpleNamespace(free=1)
    )
    monkeypatch.setattr(
        checks.tempfile,
        "NamedTemporaryFile",
        Mock(side_effect=OSError("disk write refused")),
    )
    report = await run_doctor(book_workspace / "yakbox.toml")
    statuses = {item.id: item.status for item in report.diagnostics}
    assert statuses["workspace.sources"] == "fail"
    assert statuses["workspace.storage"] == "warn"
    assert statuses["workspace.atomic_write"] == "fail"
    assert statuses["workspace.outputs"] == "fail"
    assert (
        checks._nearest_existing_parent(book_workspace / "missing/nested")
        == book_workspace
    )


@pytest.mark.parametrize("value", [OSError("unsupported"), "unknown"])
def test_doctor_memory_probe_tolerates_missing_or_non_numeric_sysconf(
    monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    probe = (
        Mock(side_effect=value)
        if isinstance(value, Exception)
        else Mock(return_value=value)
    )
    monkeypatch.setattr(checks.os, "sysconf", probe)
    assert checks._physical_memory_bytes() is None


def test_doctor_local_runtime_import_failure_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        checks.importlib,
        "import_module",
        Mock(side_effect=ImportError("missing torch")),
    )
    result = checks._local_deep()
    assert result.status == "fail"
    assert result.detail == "missing torch"
    assert checks._redact("token token", "token") == "[REDACTED] [REDACTED]"
    assert len(checks._redact("x" * 3000)) == 2048
