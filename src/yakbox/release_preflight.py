"""Typed, fail-closed package release preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

type CommandRunner = Callable[[tuple[str, ...], Path], str]


class ReleasePreflightError(RuntimeError):
    """A package release invariant was not satisfied."""


@dataclass(frozen=True, slots=True)
class ReleaseFile:
    path: Path
    sha256: str
    size: int

    def to_dict(self, *, root: Path) -> dict[str, object]:
        return {
            "path": self.path.relative_to(root).as_posix(),
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class ReleasePreflightReport:
    version: str
    tag: str
    head_sha: str
    distributions: tuple[ReleaseFile, ...]
    sbom: ReleaseFile
    report_path: Path
    checksums_path: Path

    def to_dict(self, *, root: Path) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": "ok",
            "version": self.version,
            "tag": self.tag,
            "head_sha": self.head_sha,
            "generated_at": datetime.now(UTC).isoformat(),
            "distributions": [
                artifact.to_dict(root=root) for artifact in self.distributions
            ],
            "sbom": self.sbom.to_dict(root=root),
            "checksums_path": self.checksums_path.relative_to(root).as_posix(),
        }


def project_version(root: Path) -> str:
    try:
        raw = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        project = raw["project"]
        if not isinstance(project, dict):
            raise TypeError
        value = cast(dict[str, object], project).get("version")
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ReleasePreflightError("Cannot read project version") from error
    if not isinstance(value, str) or not value.strip():
        raise ReleasePreflightError("Project version must be a non-empty string")
    return value


def validate_release_tag(tag: str, version: str) -> None:
    expected = f"v{version}"
    if tag != expected:
        raise ReleasePreflightError(
            f"Release tag {tag!r} does not match project version {version!r}; "
            f"expected {expected!r}"
        )


def ensure_clean_worktree(root: Path, *, runner: CommandRunner) -> str:
    status = runner(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        root,
    )
    if status.strip():
        raise ReleasePreflightError("Release preflight requires a clean Git worktree")
    head = runner(("git", "rev-parse", "HEAD"), root).strip()
    if not head:
        raise ReleasePreflightError("Cannot resolve the release commit")
    return head


def ensure_tag_points_to_head(
    root: Path,
    tag: str,
    head_sha: str,
    *,
    runner: CommandRunner,
) -> None:
    tagged = runner(
        ("git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"),
        root,
    ).strip()
    if tagged != head_sha:
        raise ReleasePreflightError(
            f"Tag {tag!r} does not point to release commit {head_sha}"
        )


def run_release_preflight(
    root: Path,
    *,
    tag: str,
    output_dir: Path = Path("dist"),
    metadata_dir: Path = Path("release-metadata"),
    require_tag_ref: bool = False,
    runner: CommandRunner | None = None,
) -> ReleasePreflightReport:
    root = root.resolve()
    command_runner = runner or _run_command
    version = project_version(root)
    validate_release_tag(tag, version)
    head_sha = ensure_clean_worktree(root, runner=command_runner)
    if require_tag_ref:
        ensure_tag_points_to_head(root, tag, head_sha, runner=command_runner)

    distribution_root = _managed_directory(root, output_dir)
    metadata_root = _managed_directory(root, metadata_dir)
    _clear_directory(distribution_root)
    _clear_directory(metadata_root)

    quality_commands = (
        ("uv", "lock", "--check"),
        ("uv", "run", "ruff", "format", "--check", "."),
        ("uv", "run", "ruff", "check", "."),
        ("uv", "run", "ty", "check"),
        ("uv", "run", "lint-imports", "--no-cache"),
        ("uv", "run", "pytest"),
        ("uv", "audit", "--frozen"),
    )
    for command in quality_commands:
        command_runner(command, root)

    command_runner(
        (
            "uv",
            "build",
            "--no-sources",
            "--clear",
            "--no-create-gitignore",
            "--out-dir",
            str(distribution_root),
        ),
        root,
    )
    sbom_path = metadata_root / "yakbox.cdx.json"
    command_runner(
        (
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--format",
            "cyclonedx1.5",
            "--output-file",
            str(sbom_path),
        ),
        root,
    )

    distributions = _distribution_files(distribution_root, version)
    _verify_sbom(sbom_path, version)
    wheel = next(item.path for item in distributions if item.path.suffix == ".whl")
    source = next(
        item.path for item in distributions if item.path.name.endswith(".tar.gz")
    )
    for artifact in (wheel, source):
        command_runner(
            (
                "uv",
                "run",
                "--isolated",
                "--no-project",
                "--with",
                str(artifact),
                "yakbox",
                "--version",
            ),
            root,
        )

    sbom = _release_file(sbom_path)
    report_path = metadata_root / "release-preflight.json"
    checksums_path = metadata_root / "SHA256SUMS"
    report = ReleasePreflightReport(
        version=version,
        tag=tag,
        head_sha=head_sha,
        distributions=distributions,
        sbom=sbom,
        report_path=report_path,
        checksums_path=checksums_path,
    )
    report_path.write_text(
        json.dumps(report.to_dict(root=root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_checksums((*distributions, sbom, _release_file(report_path)), checksums_path)
    return report


def _managed_directory(root: Path, value: Path) -> Path:
    resolved = value if value.is_absolute() else root / value
    resolved = resolved.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ReleasePreflightError(
            f"Release output must be a child of the project root: {value}"
        )
    return resolved


def _clear_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ReleasePreflightError(f"Release output is not a directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _distribution_files(path: Path, version: str) -> tuple[ReleaseFile, ...]:
    wheels = tuple(path.glob(f"yakbox-{version}-*.whl"))
    sources = tuple(path.glob(f"yakbox-{version}.tar.gz"))
    unknown = tuple(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate not in {*wheels, *sources}
    )
    if len(wheels) != 1 or len(sources) != 1 or unknown:
        raise ReleasePreflightError(
            "Release build must produce exactly one yakbox wheel and one source archive"
        )
    return tuple(_release_file(item) for item in (*wheels, *sources))


def _verify_sbom(path: Path, version: str) -> None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError
        document = cast(dict[str, object], raw)
        metadata = document["metadata"]
        if not isinstance(metadata, dict):
            raise TypeError
        component = cast(dict[str, object], metadata).get("component")
        if not isinstance(component, dict):
            raise TypeError
        typed_component = cast(dict[str, object], component)
        valid = (
            document.get("bomFormat") == "CycloneDX"
            and document.get("specVersion") == "1.5"
            and typed_component.get("name") == "yakbox"
            and typed_component.get("version") == version
        )
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise ReleasePreflightError(
            "Cannot validate generated CycloneDX SBOM"
        ) from error
    if not valid:
        raise ReleasePreflightError(
            "Generated SBOM does not describe this yakbox release"
        )


def _release_file(path: Path) -> ReleaseFile:
    return ReleaseFile(
        path=path.resolve(),
        sha256=_sha256(path),
        size=path.stat().st_size,
    )


def _write_checksums(files: Sequence[ReleaseFile], destination: Path) -> None:
    names = [item.path.name for item in files]
    if len(names) != len(set(names)):
        raise ReleasePreflightError("Release asset filenames must be unique")
    lines = [f"{item.sha256}  {item.path.name}" for item in files]
    destination.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(command: tuple[str, ...], cwd: Path) -> str:
    try:
        result = subprocess.run(  # noqa: S603 - internal release argv; no shell
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ReleasePreflightError(
            f"Cannot run release command {command[0]!r}: {error}"
        ) from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-4_096:]
        raise ReleasePreflightError(
            f"Release command failed ({' '.join(command)}): {detail}"
        )
    return result.stdout


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run every package release gate and build artifacts once."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("release-metadata"),
    )
    parser.add_argument("--require-tag-ref", action="store_true")
    options = parser.parse_args(arguments)
    try:
        report = run_release_preflight(
            options.root,
            tag=options.tag,
            output_dir=options.output_dir,
            metadata_dir=options.metadata_dir,
            require_tag_ref=options.require_tag_ref,
        )
    except ReleasePreflightError as error:
        parser.exit(1, f"release preflight failed: {error}\n")
    sys.stdout.write(
        f"release preflight passed for {report.tag}: "
        f"{len(report.distributions)} distributions, SBOM, and checksums\n"
    )


if __name__ == "__main__":
    main()
