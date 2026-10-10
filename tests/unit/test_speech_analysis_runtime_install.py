from __future__ import annotations

import os
import subprocess
import tomllib
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


@pytest.mark.parametrize(
    ("family", "engines", "requirement_name"),
    [
        ("whisper", ("whisper",), "mlx-whisper"),
        ("parakeet", ("parakeet",), "parakeet-mlx"),
        ("qwen", ("qwen", "qwen-forced"), "mlx-audio[stt]"),
    ],
)
def test_packaged_runtime_projects_are_frozen_and_family_specific(
    family: str, engines: tuple[str, ...], requirement_name: str
) -> None:
    runtime = load_runtime_project(family)
    project = tomllib.loads(runtime.pyproject_toml.decode())["project"]
    lock = tomllib.loads(runtime.lockfile.decode())
    adapter_requirements = [
        dependency
        for dependency in project["dependencies"]
        if dependency.startswith(requirement_name)
    ]

    assert runtime.engines == engines
    assert project["requires-python"] == ">=3.14,<3.15"
    assert len(adapter_requirements) == 1
    name, separator, version = adapter_requirements[0].partition("==")
    assert name == requirement_name
    assert separator == "=="
    assert version and all(character not in version for character in "*,;<>!= ")
    assert lock["version"] == 1
    assert [
        package["version"]
        for package in lock["package"]
        if package["name"] == runtime.adapter
    ] == [version]
    assert {
        package["name"]
        for package in lock["package"]
        if package["name"] in {"mlx-whisper", "parakeet-mlx", "mlx-audio"}
    } == {runtime.adapter}


def test_packaged_runtime_locks_are_family_specific() -> None:
    assert len(
        {load_runtime_project(family).lock_digest for family in BUILT_IN_WORKERS}
    ) == len(BUILT_IN_WORKERS)


class _FakeRunner:
    def __init__(
        self,
        *,
        python_version: bytes = b"3.14\n",
        uv_version: bytes = b"uv 0.13.0\n",
        uv_returncode: int = 0,
    ) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, str]]] = []
        self.python_version = python_version
        self.uv_version = uv_version
        self.uv_returncode = uv_returncode

    def __call__(
        self, arguments: list[str], **options: object
    ) -> subprocess.CompletedProcess[bytes]:
        environment = options.get("env")
        assert isinstance(environment, dict)
        self.calls.append((tuple(arguments), cast(dict[str, str], environment)))
        if arguments[1:] == ["--version"]:
            return subprocess.CompletedProcess(
                arguments, self.uv_returncode, self.uv_version, b""
            )
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


@pytest.mark.parametrize(
    "version",
    [
        b"uv 0.13.0\n",
        b"uv 0.13.7 (Homebrew)\n",
        b"uv 0.14.0\n",
        b"uv 1.0.0\n",
    ],
)
def test_runtime_install_accepts_supported_uv_versions(
    tmp_path: Path, version: bytes
) -> None:
    runner = _FakeRunner(uv_version=version)
    installer = AnalysisRuntimeInstaller(
        tmp_path / "runtimes",
        uv_executable=Path("/usr/local/bin/uv"),
        runner=runner,
        host_validator=lambda: None,
    )

    assert installer.install(("parakeet",)).verified
    assert any("sync" in call for call, _environment in runner.calls)


@pytest.mark.parametrize(
    ("version", "returncode"),
    [
        (b"uv 0.12.99\n", 0),
        (b"uv 0.13\n", 0),
        (b"uv unknown\n", 0),
        (b"", 0),
        (b"uv 0.13.0\n", 1),
    ],
)
def test_runtime_install_rejects_unsupported_uv_before_sync(
    tmp_path: Path, version: bytes, returncode: int
) -> None:
    runner = _FakeRunner(uv_version=version, uv_returncode=returncode)
    installer = AnalysisRuntimeInstaller(
        tmp_path / "runtimes",
        uv_executable=Path("/usr/local/bin/uv"),
        runner=runner,
        host_validator=lambda: None,
    )

    with pytest.raises(BackendUnavailableError, match=r"uv 0\.13\.0 or newer"):
        installer.install(("parakeet",))

    assert not any("sync" in call for call, _environment in runner.calls)


def test_runtime_install_reports_minimum_version_when_uv_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "yakbox.speech.analysis_runtime_install.shutil.which", lambda _name: None
    )

    with pytest.raises(BackendUnavailableError, match=r"uv 0\.13\.0 or newer"):
        AnalysisRuntimeInstaller(tmp_path / "runtimes")
