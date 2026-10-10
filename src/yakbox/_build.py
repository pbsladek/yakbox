"""Embed the source commit in distributions built with the native uv backend."""

from __future__ import annotations

import argparse
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

from yakbox._files import atomic_write_bytes
from yakbox.errors import BuildError


@contextmanager
def embedded_commit(root: Path, commit: str) -> Generator[None]:
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None:
        raise BuildError("Build commit must be a full Git commit hash")
    root = root.resolve()
    metadata = root / "src" / "yakbox" / "_build_commit.py"
    if metadata.is_symlink() or not metadata.resolve().is_relative_to(root):
        raise BuildError("Build commit metadata must stay inside the source root")
    try:
        original = metadata.read_bytes()
        mode = stat.S_IMODE(metadata.stat().st_mode)
    except OSError as error:
        raise BuildError("Cannot read source build commit metadata") from error
    stamped = (
        '"""Immutable source commit embedded during packaging."""\n\n'
        f"COMMIT: str | None = {commit!r}\n"
    ).encode()
    try:
        atomic_write_bytes(metadata, stamped, overwrite=True)
        metadata.chmod(mode)
        yield
    finally:
        atomic_write_bytes(metadata, original, overwrite=True)
        metadata.chmod(mode)


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build distributions with their commit."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=Path("dist"))
    options = parser.parse_args(arguments)
    root = options.root.resolve()
    output = (root / options.out_dir).resolve()
    if output == root or not output.is_relative_to(root):
        parser.error("Build output must be a child of the project root")
    git = shutil.which("git")
    uv = shutil.which("uv")
    if git is None or uv is None:
        parser.error("Building commit metadata requires Git and uv")
    try:
        status = subprocess.check_output(  # noqa: S603 - fixed Git argv; no shell
            (git, "status", "--porcelain", "--untracked-files=all"),
            cwd=root,
            text=True,
            timeout=10,
        )
        if status.strip():
            raise BuildError("Commit-stamped builds require a clean Git worktree")
        commit = subprocess.check_output(  # noqa: S603 - fixed Git argv; no shell
            (git, "rev-parse", "--verify", "HEAD"),
            cwd=root,
            text=True,
            timeout=10,
        ).strip()
        with embedded_commit(root, commit):
            result = subprocess.run(  # noqa: S603 - fixed uv argv and managed paths
                (
                    uv,
                    "build",
                    "--no-sources",
                    "--clear",
                    "--no-create-gitignore",
                    "--out-dir",
                    str(output),
                ),
                cwd=root,
                check=False,
            )
    except (BuildError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Build failed: {error}\n")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
