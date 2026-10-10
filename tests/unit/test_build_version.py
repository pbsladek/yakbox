from __future__ import annotations

import shutil
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from yakbox import __version__, _build_commit
from yakbox._build import embedded_commit
from yakbox._build import main as build_main
from yakbox.cli import main
from yakbox.errors import BuildError

COMMIT = "a" * 40
ORIGINAL = b"COMMIT: str | None = None\n"


def _metadata(root: Path) -> Path:
    metadata = root / "src" / "yakbox" / "_build_commit.py"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(ORIGINAL)
    return metadata


@pytest.mark.parametrize("commit", [COMMIT, "b" * 64, None])
def test_cli_version_prints_embedded_commit_without_git(
    monkeypatch: pytest.MonkeyPatch, commit: str | None
) -> None:
    monkeypatch.setattr(_build_commit, "COMMIT", commit)

    def unexpected_git(*_args: object, **_kwargs: object) -> str:
        pytest.fail("Version output must not consult Git")

    monkeypatch.setattr(subprocess, "check_output", unexpected_git)
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output == (
        f"yakbox, version {__version__} (commit {commit or 'unknown'})\n"
    )


@pytest.mark.parametrize("fail", [False, True])
def test_commit_stamp_restores_source_and_permissions_on_failure(
    tmp_path: Path, fail: bool
) -> None:
    metadata = _metadata(tmp_path)
    metadata.chmod(0o640)

    def build() -> None:
        with embedded_commit(tmp_path, COMMIT):
            assert COMMIT in metadata.read_text()
            if fail:
                raise BuildError("Fixture build failed")

    if fail:
        with pytest.raises(BuildError, match="Fixture build failed"):
            build()
    else:
        build()
    assert metadata.read_bytes() == ORIGINAL
    assert stat.S_IMODE(metadata.stat().st_mode) == 0o640


@pytest.mark.parametrize("commit", ["short", "a" * 39, "g" * 40, "a" * 40 + "\n"])
def test_invalid_commit_is_rejected_before_stamping(
    tmp_path: Path, commit: str
) -> None:
    metadata = _metadata(tmp_path)
    with (
        pytest.raises(BuildError, match="full Git commit hash"),
        embedded_commit(tmp_path, commit),
    ):
        pytest.fail("Invalid commit should not reach the build")
    assert metadata.read_bytes() == ORIGINAL


def test_stamp_rejects_symlink_to_unmanaged_file(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path / "project")
    external = tmp_path / "external.py"
    external.write_bytes(ORIGINAL)
    metadata.unlink()
    metadata.symlink_to(external)
    with (
        pytest.raises(BuildError, match="inside the source root"),
        embedded_commit(tmp_path / "project", COMMIT),
    ):
        pytest.fail("Unmanaged file should not be stamped")
    assert external.read_bytes() == ORIGINAL


@pytest.mark.parametrize("status", [0, 7])
def test_build_command_propagates_exit_status_and_restores_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    metadata = _metadata(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: f"/tools/{name}")

    def git(command: tuple[str, ...], **_kwargs: object) -> str:
        return "" if command[1] == "status" else COMMIT

    monkeypatch.setattr(subprocess, "check_output", git)

    def run(
        command: tuple[str, ...], **_kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        assert command[:3] == ("/tools/uv", "build", "--no-sources")
        assert COMMIT in metadata.read_text()
        return subprocess.CompletedProcess(command, status)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(SystemExit) as error:
        build_main(["--root", str(tmp_path)])
    assert error.value.code == status
    assert metadata.read_bytes() == ORIGINAL


def test_build_rejects_dirty_sources_before_stamping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_args, **_kwargs: " M src/yakbox/cli.py\n"
    )
    with pytest.raises(SystemExit) as error:
        build_main(["--root", str(tmp_path)])
    assert error.value.code == 1
    assert metadata.read_bytes() == ORIGINAL


def test_real_wheel_and_source_archive_keep_the_embedded_commit(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    (metadata.parent / "__init__.py").write_text('"""Release fixture."""\n')
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["uv_build>=0.13.0,<0.14"]\n'
        'build-backend = "uv_build"\n'
        '[project]\nname = "yakbox"\nversion = "0.2.0"\n'
        'requires-python = ">=3.14,<3.15"\n'
    )
    uv = shutil.which("uv")
    assert uv is not None
    with embedded_commit(tmp_path, COMMIT):
        subprocess.run(  # noqa: S603 - installed uv and fixed offline build flags
            (uv, "build", "--offline", "--no-sources"),
            cwd=tmp_path,
            capture_output=True,
            check=True,
            timeout=30,
        )
    assert metadata.read_bytes() == ORIGINAL
    with zipfile.ZipFile(next((tmp_path / "dist").glob("*.whl"))) as wheel:
        assert COMMIT.encode() in wheel.read("yakbox/_build_commit.py")
    with tarfile.open(next((tmp_path / "dist").glob("*.tar.gz"))) as source:
        file = source.extractfile("yakbox-0.2.0/src/yakbox/_build_commit.py")
        assert file is not None
        assert COMMIT.encode() in file.read()
