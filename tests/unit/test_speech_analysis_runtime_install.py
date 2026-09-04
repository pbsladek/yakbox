from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator

from yakbox.errors import BackendUnavailableError
from yakbox.schemas import load_schema
from yakbox.speech.analysis_protocol import encode_worker_handshake
from yakbox.speech.analysis_runtime import BUILT_IN_WORKERS
from yakbox.speech.analysis_runtime_install import (
    AnalysisRuntimeInstaller,
    load_runtime_project,
)
from yakbox.speech.analysis_scheduler import build_worker_handshake


def test_packaged_runtime_projects_are_frozen_and_family_specific() -> None:
    whisper = load_runtime_project("whisper")
    parakeet = load_runtime_project("parakeet")
    qwen = load_runtime_project("qwen")

    assert whisper.engines == ("whisper",)
    assert parakeet.engines == ("parakeet",)
    assert qwen.engines == ("qwen", "qwen-forced")
    assert b'requires-python = ">=3.14,<3.15"' in whisper.pyproject_toml
    assert b"mlx-whisper==0.4.3" in whisper.pyproject_toml
    assert b"parakeet-mlx==0.5.2" in parakeet.pyproject_toml
    assert b"mlx-audio[stt]==0.4.8" in qwen.pyproject_toml
    assert all(b"version = 1" in item.lockfile for item in (whisper, parakeet, qwen))
    assert len({item.lock_digest for item in (whisper, parakeet, qwen)}) == 3


class _FakeRunner:
    def __init__(self, *, python_version: bytes = b"3.14\n") -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, str]]] = []
        self.python_version = python_version

    def __call__(
        self, arguments: list[str], **options: object
    ) -> subprocess.CompletedProcess[bytes]:
        environment = options.get("env")
        assert isinstance(environment, dict)
        self.calls.append((tuple(arguments), cast(dict[str, str], environment)))
        if arguments[1:] == ["--version"]:
            return subprocess.CompletedProcess(arguments, 0, b"uv 0.12.0\n", b"")
        if arguments[1:3] == ["-I", "-c"]:
            return subprocess.CompletedProcess(arguments, 0, self.python_version, b"")
        if "sync" in arguments:
            root = Path(arguments[arguments.index("--project") + 1])
            python = (
                root
                / ".venv"
                / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            )
            python.parent.mkdir(parents=True)
            python.write_bytes(b"fake-python")
            return subprocess.CompletedProcess(arguments, 0, b"", b"")
        family = arguments[arguments.index("--family") + 1]
        artifact = arguments[arguments.index("--worker-artifact-digest") + 1]
        lock = arguments[arguments.index("--lock-digest") + 1]
        definition = arguments[arguments.index("--definition-fingerprint") + 1]
        handshake = build_worker_handshake(
            family=family,
            engines=BUILT_IN_WORKERS[family].engines,
            worker_artifact_fingerprint=artifact,
            environment_lock_fingerprint=lock,
            adapter_fingerprint=definition,
        )
        return subprocess.CompletedProcess(
            arguments, 0, encode_worker_handshake(handshake) + b"\n", b""
        )


def test_runtime_install_is_explicit_immutable_and_verifiable(tmp_path: Path) -> None:
    runner = _FakeRunner()
    installer = AnalysisRuntimeInstaller(
        tmp_path / "runtimes",
        uv_executable=Path("/usr/local/bin/uv"),
        python_executable=Path("/usr/local/bin/python3.14"),
        runner=runner,
        host_validator=lambda: None,
    )

    installed = installer.install(("whisper", "qwen"))
    repeated = installer.install(("whisper", "qwen"))

    assert installed.verified
    assert repeated.verified
    assert tuple(item.family for item in installed.runtimes) == ("whisper", "qwen")
    sync_calls = [call for call, _environment in runner.calls if "sync" in call]
    assert len(sync_calls) == 2
    assert all(
        "--frozen" in call and "--no-managed-python" in call for call in sync_calls
    )
    for _call, environment in runner.calls:
        assert "HTTP_PROXY" not in environment
        assert "HTTPS_PROXY" not in environment
        assert "PYTHONPATH" not in environment
        assert "UV_INDEX" not in environment
    public = installed.to_dict()
    Draft202012Validator(load_schema("speech-analysis-runtimes")).validate(public)
    assert "root" not in public
    runtimes = cast(list[dict[str, object]], public["runtimes"])
    assert all("path" not in item and "python_path" not in item for item in runtimes)


def test_runtime_verification_detects_lock_tampering(tmp_path: Path) -> None:
    runner = _FakeRunner()
    installer = AnalysisRuntimeInstaller(
        tmp_path / "runtimes",
        uv_executable=Path("/usr/local/bin/uv"),
        python_executable=Path("/usr/local/bin/python3.14"),
        runner=runner,
        host_validator=lambda: None,
    )
    installed = installer.install(("parakeet",)).runtimes[0]
    (installed.path / "uv.lock").write_bytes(b"tampered")

    report = installer.verify(("parakeet",))

    assert not report.verified
    assert "uv_lock_mismatch" in report.runtimes[0].issues


def test_verified_runtime_creates_digest_bound_worker(tmp_path: Path) -> None:
    installer = AnalysisRuntimeInstaller(
        tmp_path / "runtimes",
        uv_executable=Path("/usr/local/bin/uv"),
        python_executable=Path("/usr/local/bin/python3.14"),
        runner=_FakeRunner(),
        host_validator=lambda: None,
    )
    installed = installer.install(("whisper",)).runtimes[0]

    worker = installer.create_worker(
        "whisper",
        audio_root=tmp_path / "audio",
        model_root=tmp_path / "models",
        calibration_fingerprint="a" * 64,
    )

    assert worker.python_executable == installed.python_path
    assert worker.worker_artifact_path == installed.path / "analysis-worker.pyz"
    assert worker.worker_artifact_digest == installed.worker_artifact_digest
    assert worker.lock_digest == installed.lock_digest


def test_runtime_install_rejects_non_314_interpreter_before_uv_sync(
    tmp_path: Path,
) -> None:
    runner = _FakeRunner(python_version=b"3.13\n")
    installer = AnalysisRuntimeInstaller(
        tmp_path / "runtimes",
        uv_executable=Path("/usr/local/bin/uv"),
        python_executable=Path("/usr/local/bin/python3.13"),
        runner=runner,
        host_validator=lambda: None,
    )

    with pytest.raises(BackendUnavailableError, match=r"Python 3\.14"):
        installer.install(("whisper",))

    assert not any("sync" in call for call, _environment in runner.calls)
