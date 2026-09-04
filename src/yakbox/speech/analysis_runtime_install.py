"""Explicit installer for frozen, dependency-family analysis runtimes."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import cast

from yakbox._files import (
    atomic_write_bytes,
    atomic_write_json,
    safe_child,
    sha256_bytes,
)
from yakbox.contracts import runtime_metadata
from yakbox.errors import (
    BackendUnavailableError,
    ModelIntegrityError,
    ValidationError,
    WorkerProtocolError,
)
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_protocol import parse_worker_handshake
from yakbox.speech.analysis_runtime import BUILT_IN_WORKERS, IsolatedAnalysisWorker
from yakbox.speech.analysis_scheduler import build_worker_handshake
from yakbox.speech.analysis_worker_artifact import (
    verify_worker_artifact,
    worker_artifact_bytes,
)

_FAMILIES = ("whisper", "parakeet", "qwen")
_FAMILY_ADAPTERS = {
    "whisper": "mlx-whisper",
    "parakeet": "parakeet-mlx",
    "qwen": "mlx-audio",
}
_UV_VERSION = re.compile(r"uv 0\.12(?:\.[0-9]+)?(?:\s.*)?")


@dataclass(frozen=True, slots=True)
class RuntimeProject:
    family: str
    engines: tuple[str, ...]
    adapter: str
    runtime_toml: bytes
    pyproject_toml: bytes
    lockfile: bytes

    @property
    def runtime_digest(self) -> str:
        return sha256_bytes(self.runtime_toml)

    @property
    def project_digest(self) -> str:
        return sha256_bytes(self.pyproject_toml)

    @property
    def lock_digest(self) -> str:
        return sha256_bytes(self.lockfile)

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "analysis-runtime-project-v1",
            {
                "family": self.family,
                "engines": self.engines,
                "adapter": self.adapter,
                "runtime_digest": self.runtime_digest,
                "project_digest": self.project_digest,
                "lock_digest": self.lock_digest,
                "worker_artifact_digest": sha256_bytes(worker_artifact_bytes()),
            },
        )


@dataclass(frozen=True, slots=True)
class InstalledAnalysisRuntime:
    family: str
    install_fingerprint: str
    path: Path
    python_path: Path
    lock_digest: str
    worker_artifact_digest: str
    verified: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "install_fingerprint": self.install_fingerprint,
            "lock_digest": self.lock_digest,
            "worker_artifact_digest": self.worker_artifact_digest,
            "verified": self.verified,
            "issues": list(self.issues),
        }


@dataclass(frozen=True, slots=True)
class AnalysisRuntimeReport:
    root: Path
    runtimes: tuple[InstalledAnalysisRuntime, ...]

    @property
    def verified(self) -> bool:
        return bool(self.runtimes) and all(item.verified for item in self.runtimes)

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("speech-analysis-runtimes"),
            "verified": self.verified,
            "runtimes": [item.to_dict() for item in self.runtimes],
        }


type CommandRunner = Callable[..., subprocess.CompletedProcess[bytes]]


class AnalysisRuntimeInstaller:
    """Install immutable runtime directories using one reviewed uv invocation."""

    def __init__(
        self,
        root: Path,
        *,
        uv_executable: Path | None = None,
        python_executable: Path | None = None,
        runner: CommandRunner = subprocess.run,
        host_validator: Callable[[], None] | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.uv_executable = (uv_executable or _find_uv()).expanduser().resolve()
        self.python_executable = (python_executable or Path(sys.executable)).absolute()
        self._runner = runner
        self._host_validator = host_validator or _require_supported_host

    def install(self, families: Sequence[str]) -> AnalysisRuntimeReport:
        """Install requested built-ins explicitly; builds never call this path."""
        self._host_validator()
        requested = _validated_families(families)
        self._verify_python()
        self._verify_uv()
        installed = tuple(
            self._install_one(load_runtime_project(name)) for name in requested
        )
        return AnalysisRuntimeReport(self.root, installed)

    def verify(self, families: Sequence[str] = _FAMILIES) -> AnalysisRuntimeReport:
        requested = _validated_families(families)
        return AnalysisRuntimeReport(
            self.root,
            tuple(self._status(load_runtime_project(name)) for name in requested),
        )

    def create_worker(
        self,
        family: str,
        *,
        audio_root: Path,
        model_root: Path,
        calibration_fingerprint: str,
        environment: Mapping[str, str] | None = None,
    ) -> IsolatedAnalysisWorker:
        """Create a supervisor bound to one verified managed runtime."""
        project = load_runtime_project(family)
        runtime = self._status(project)
        if not runtime.verified:
            raise BackendUnavailableError(
                f"Analysis runtime {family!r} is not installed and verified"
            )
        return IsolatedAnalysisWorker(
            BUILT_IN_WORKERS[family],
            audio_root=audio_root,
            model_root=model_root,
            calibration_fingerprint=calibration_fingerprint,
            worker_artifact_digest=runtime.worker_artifact_digest,
            lock_digest=runtime.lock_digest,
            python_executable=runtime.python_path,
            worker_artifact_path=runtime.path / "analysis-worker.pyz",
            environment=environment,
        )

    def _install_one(self, project: RuntimeProject) -> InstalledAnalysisRuntime:
        destination = self._install_path(project)
        existing = self._status(project)
        if existing.verified:
            return existing
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{project.family}.", dir=destination.parent)
        ).resolve()
        try:
            self._write_project(temporary, project)
            completed = self._runner(
                [
                    str(self.uv_executable),
                    "sync",
                    "--frozen",
                    "--no-dev",
                    "--no-managed-python",
                    "--python",
                    str(self.python_executable),
                    "--project",
                    str(temporary),
                ],
                check=False,
                capture_output=True,
                env=_installer_environment(self.root),
                timeout=1_800,
            )
            if completed.returncode != 0:
                raise BackendUnavailableError(
                    f"Could not install the {project.family} analysis runtime"
                )
            python_path = _runtime_python(temporary)
            if not python_path.is_file():
                raise BackendUnavailableError(
                    "uv did not create the runtime interpreter"
                )
            atomic_write_json(
                temporary / "installation.json",
                _installation_record(project),
            )
            try:
                temporary.replace(destination)
            except FileExistsError:
                raced = self._status(project)
                if not raced.verified:
                    raise ModelIntegrityError(
                        "Concurrent runtime installation produced different evidence"
                    ) from None
            status = self._status(project)
            if not status.verified:
                raise ModelIntegrityError(
                    f"Installed {project.family} runtime failed verification"
                )
            return status
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _write_project(self, root: Path, project: RuntimeProject) -> None:
        atomic_write_bytes(root / "runtime.toml", project.runtime_toml)
        atomic_write_bytes(root / "pyproject.toml", project.pyproject_toml)
        atomic_write_bytes(root / "uv.lock", project.lockfile)
        atomic_write_bytes(root / "analysis-worker.pyz", worker_artifact_bytes())

    def _status(self, project: RuntimeProject) -> InstalledAnalysisRuntime:
        root = self._install_path(project)
        python_path = _runtime_python(root)
        artifact = root / "analysis-worker.pyz"
        issues: list[str] = []
        expected = _installation_record(project)
        try:
            record_raw = json.loads(
                (root / "installation.json").read_text(encoding="utf-8")
            )
            if record_raw != expected:
                issues.append("installation_identity_mismatch")
        except OSError, UnicodeError, json.JSONDecodeError:
            issues.append("installation_record_missing")
        for name, digest in (
            ("runtime.toml", project.runtime_digest),
            ("pyproject.toml", project.project_digest),
            ("uv.lock", project.lock_digest),
        ):
            path = root / name
            if not path.is_file() or sha256_bytes(path.read_bytes()) != digest:
                issues.append(f"{name.replace('.', '_')}_mismatch")
        try:
            verified_artifact = verify_worker_artifact(artifact)
            artifact_digest = verified_artifact.sha256
        except ModelIntegrityError:
            artifact_digest = sha256_bytes(worker_artifact_bytes())
            issues.append("worker_artifact_mismatch")
        if not python_path.is_file():
            issues.append("python_missing")
        elif not issues:
            issues.extend(
                self._probe_worker(project, root, python_path, artifact_digest)
            )
        return InstalledAnalysisRuntime(
            project.family,
            project.fingerprint,
            root,
            python_path,
            project.lock_digest,
            artifact_digest,
            not issues,
            tuple(issues),
        )

    def _probe_worker(
        self,
        project: RuntimeProject,
        root: Path,
        python_path: Path,
        artifact_digest: str,
    ) -> tuple[str, ...]:
        definition = BUILT_IN_WORKERS[project.family]
        completed = self._runner(
            [
                str(python_path),
                "-I",
                str(root / "analysis-worker.pyz"),
                "--family",
                project.family,
                "--audio-root",
                str(root / "audio"),
                "--model-root",
                str(self.root / "models"),
                "--calibration-fingerprint",
                "0" * 64,
                "--worker-artifact-digest",
                artifact_digest,
                "--lock-digest",
                project.lock_digest,
                "--definition-fingerprint",
                definition.fingerprint,
            ],
            input=b"",
            check=False,
            capture_output=True,
            env=_worker_environment(),
            timeout=30,
        )
        if completed.returncode != 0:
            return ("worker_probe_failed",)
        try:
            first_line = completed.stdout.splitlines()[0]
            actual = parse_worker_handshake(first_line)
        except IndexError, WorkerProtocolError:
            return ("worker_handshake_invalid",)
        expected = build_worker_handshake(
            family=project.family,
            engines=project.engines,
            worker_artifact_fingerprint=artifact_digest,
            environment_lock_fingerprint=project.lock_digest,
            adapter_fingerprint=definition.fingerprint,
        )
        return () if actual == expected else ("worker_handshake_mismatch",)

    def _install_path(self, project: RuntimeProject) -> Path:
        return safe_child(
            self.root,
            self.root / project.family / project.fingerprint,
        )

    def _verify_uv(self) -> None:
        completed = self._runner(
            [str(self.uv_executable), "--version"],
            check=False,
            capture_output=True,
            env=_installer_environment(self.root),
            timeout=30,
        )
        output = completed.stdout.decode("utf-8", errors="replace").strip()
        if completed.returncode != 0 or _UV_VERSION.fullmatch(output) is None:
            raise BackendUnavailableError("Yakbox requires reviewed uv 0.12.x")

    def _verify_python(self) -> None:
        completed = self._runner(
            [
                str(self.python_executable),
                "-I",
                "-c",
                "import sys; "
                "print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            check=False,
            capture_output=True,
            env=_worker_environment(),
            timeout=30,
        )
        if completed.returncode != 0 or completed.stdout.strip() != b"3.14":
            raise BackendUnavailableError(
                "Analysis-runtime installation requires a Python 3.14 interpreter"
            )


def load_runtime_project(family: str) -> RuntimeProject:
    """Load one package-owned definition and frozen lock as immutable bytes."""
    if family not in _FAMILIES:
        raise ValidationError(f"Unknown analysis runtime family: {family}")
    root = files("yakbox").joinpath("runtimes", family)
    runtime_toml = root.joinpath("runtime.toml").read_bytes()
    pyproject_toml = root.joinpath("pyproject.toml").read_bytes()
    lockfile = root.joinpath("uv.lock").read_bytes()
    raw = tomllib.loads(runtime_toml.decode())
    if (
        raw.get("schema_version") != 1
        or raw.get("family") != family
        or raw.get("python") != "3.14"
        or raw.get("worker_artifact") != "analysis-worker.pyz"
        or raw.get("adapter") != _FAMILY_ADAPTERS[family]
        or not isinstance(raw.get("engines"), list)
    ):
        raise ModelIntegrityError("Packaged analysis runtime definition is invalid")
    engines = tuple(cast(list[str], raw["engines"]))
    if engines != BUILT_IN_WORKERS[family].engines:
        raise ModelIntegrityError("Runtime engines differ from worker definition")
    return RuntimeProject(
        family,
        engines,
        cast(str, raw["adapter"]),
        runtime_toml,
        pyproject_toml,
        lockfile,
    )


def default_analysis_runtime_root() -> Path:
    """Return the platform-owned runtime root; manifests cannot override it."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "yakbox" / "analysis-runtimes"
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "yakbox" / "analysis-runtimes"
    return Path.home() / ".cache" / "yakbox" / "analysis-runtimes"


def _installation_record(project: RuntimeProject) -> dict[str, object]:
    return {
        "schema_version": 1,
        "family": project.family,
        "engines": list(project.engines),
        "adapter": project.adapter,
        "runtime_digest": project.runtime_digest,
        "project_digest": project.project_digest,
        "lock_digest": project.lock_digest,
        "worker_artifact_digest": sha256_bytes(worker_artifact_bytes()),
        "install_fingerprint": project.fingerprint,
    }


def _validated_families(values: Sequence[str]) -> tuple[str, ...]:
    if not values:
        raise ValidationError("Choose at least one analysis runtime")
    result = tuple(dict.fromkeys(values))
    unknown = tuple(item for item in result if item not in _FAMILIES)
    if unknown:
        raise ValidationError(f"Unknown analysis runtime family: {unknown[0]}")
    return result


def _runtime_python(root: Path) -> Path:
    return root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _find_uv() -> Path:
    value = shutil.which("uv")
    if value is None:
        raise BackendUnavailableError(
            "uv 0.12.x is required for explicit analysis-runtime installation"
        )
    return Path(value)


def _require_supported_host() -> None:
    if (
        sys.version_info[:2] != (3, 14)
        or sys.platform != "darwin"
        or platform.machine() != "arm64"
    ):
        raise BackendUnavailableError(
            "Speech-analysis runtimes currently require Apple Silicon macOS "
            "and Python 3.14"
        )


def _installer_environment(root: Path) -> Mapping[str, str]:
    environment = {
        "PATH": os.defpath,
        "PYTHONNOUSERSITE": "1",
        "UV_CACHE_DIR": str(root / "uv-cache"),
        "UV_DEFAULT_INDEX": "https://pypi.org/simple",
        "UV_NO_CONFIG": "1",
    }
    if os.name == "nt" and "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return environment


def _worker_environment() -> Mapping[str, str]:
    environment = {
        "PATH": os.defpath,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if os.name == "nt" and "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return environment


__all__ = [
    "AnalysisRuntimeInstaller",
    "AnalysisRuntimeReport",
    "InstalledAnalysisRuntime",
    "RuntimeProject",
    "default_analysis_runtime_root",
    "load_runtime_project",
]
