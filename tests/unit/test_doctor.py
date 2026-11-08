from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.unit.test_chatterbox_models import snapshot_files

from yakbox.chatterbox_models import record_snapshot
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


@pytest.mark.asyncio
@pytest.mark.parametrize("tampered", [False, True])
async def test_deep_mlx_doctor_selects_mlx_and_verifies_local_snapshot(
    book_workspace: Path, monkeypatch: pytest.MonkeyPatch, tampered: bool
) -> None:
    root = snapshot_files(book_workspace / "model", "mlx-turbo")
    record = record_snapshot(
        root,
        variant="mlx-turbo",
        repository="owner/model",
        revision="a" * 40,
        license_id="Apache-2.0",
    )
    manifest = book_workspace / "yakbox.toml"
    manifest.write_text(
        manifest.read_text()
        .replace('backend = "fake"', 'backend = "chatterbox-local"')
        .replace(
            "sample_rate = 16000",
            'variant = "mlx-turbo"\ndevice = "auto"\nmodel_directory = "model"\n'
            f'model_fingerprint = "{record["fingerprint"]}"',
        )
    )
    if tampered:
        (root / "weights.safetensors").write_bytes(b"changed")
    monkeypatch.setattr(
        "yakbox.diagnostics.checks.importlib.util.find_spec",
        lambda name: object() if name == "mlx_audio" else None,
    )
    monkeypatch.setattr("yakbox.diagnostics.checks.platform.system", lambda: "Darwin")
    monkeypatch.setattr("yakbox.diagnostics.checks.platform.machine", lambda: "arm64")
    imported: list[str] = []

    def imported_module(name: str) -> object:
        imported.append(name)
        assert name == "mlx.core"
        return SimpleNamespace()

    monkeypatch.setattr(
        "yakbox.diagnostics.checks.importlib.import_module", imported_module
    )
    report = await run_doctor(manifest, deep=True)
    checks = {item.id: item for item in report.diagnostics}
    assert imported == ["mlx.core"]
    assert checks["backend.chatterbox.mlx"].status == "pass"
    assert checks["backend.chatterbox.mlx"].evidence == {
        "package_available": True,
        "platform_compatible": True,
        "model_loaded": False,
    }
    assert checks["backend.chatterbox.snapshot"].status == (
        "fail" if tampered else "pass"
    )
