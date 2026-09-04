from __future__ import annotations

import json
from pathlib import Path

import pytest

from yakbox.release_preflight import (
    ReleaseFile,
    ReleasePreflightError,
    _write_checksums,
    ensure_clean_worktree,
    ensure_tag_points_to_head,
    project_version,
    run_release_preflight,
    validate_release_tag,
)


def test_project_version_and_tag_contract(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "yakbox"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )

    assert project_version(tmp_path) == "1.2.3"
    validate_release_tag("v1.2.3", "1.2.3")
    with pytest.raises(ReleasePreflightError, match=r"expected 'v1\.2\.3'"):
        validate_release_tag("v1.2.2", "1.2.3")


def test_clean_worktree_and_tag_must_resolve_to_head(tmp_path: Path) -> None:
    responses = {
        ("git", "status", "--porcelain", "--untracked-files=all"): "",
        ("git", "rev-parse", "HEAD"): "abc123\n",
        ("git", "rev-parse", "--verify", "refs/tags/v1.0.0^{commit}"): "abc123\n",
    }

    def runner(command: tuple[str, ...], _cwd: Path) -> str:
        return responses[command]

    head = ensure_clean_worktree(tmp_path, runner=runner)
    ensure_tag_points_to_head(tmp_path, "v1.0.0", head, runner=runner)

    responses[("git", "status", "--porcelain", "--untracked-files=all")] = (
        " M README.md\n"
    )
    with pytest.raises(ReleasePreflightError, match="clean Git worktree"):
        ensure_clean_worktree(tmp_path, runner=runner)

    responses[("git", "status", "--porcelain", "--untracked-files=all")] = ""
    responses[("git", "rev-parse", "--verify", "refs/tags/v1.0.0^{commit}")] = (
        "different\n"
    )
    with pytest.raises(ReleasePreflightError, match="does not point"):
        ensure_tag_points_to_head(tmp_path, "v1.0.0", head, runner=runner)


def test_checksum_manifest_is_deterministic_and_rejects_name_collisions(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.whl"
    second = tmp_path / "second.tar.gz"
    first.write_bytes(b"wheel")
    second.write_bytes(b"source")
    checksums = tmp_path / "SHA256SUMS"
    files = (
        ReleaseFile(first, "a" * 64, first.stat().st_size),
        ReleaseFile(second, "b" * 64, second.stat().st_size),
    )

    _write_checksums(files, checksums)

    assert checksums.read_text(encoding="utf-8").splitlines() == [
        f"{'a' * 64}  first.whl",
        f"{'b' * 64}  second.tar.gz",
    ]
    duplicate = tmp_path / "other" / "first.whl"
    duplicate.parent.mkdir()
    duplicate.write_bytes(b"different")
    with pytest.raises(ReleasePreflightError, match="filenames must be unique"):
        _write_checksums(
            (*files, ReleaseFile(duplicate, "c" * 64, duplicate.stat().st_size)),
            checksums,
        )


def test_cyclonedx_shape_fixture_documents_release_contract(tmp_path: Path) -> None:
    fixture = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "metadata": {"component": {"name": "yakbox", "version": "0.1.0"}},
    }
    path = tmp_path / "yakbox.cdx.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    assert json.loads(path.read_text(encoding="utf-8"))["specVersion"] == "1.5"


def test_preflight_runs_all_gates_builds_once_and_writes_release_evidence(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "yakbox"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    commands: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], _cwd: Path) -> str:
        commands.append(command)
        if command[:2] == ("git", "status"):
            return ""
        if command == ("git", "rev-parse", "HEAD"):
            return "commit-sha\n"
        if command[:2] == ("uv", "build"):
            output = Path(command[command.index("--out-dir") + 1])
            (output / "yakbox-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
            (output / "yakbox-0.1.0.tar.gz").write_bytes(b"source")
        if command[:2] == ("uv", "export"):
            output = Path(command[command.index("--output-file") + 1])
            if "--project" in command:
                family = Path(command[command.index("--project") + 1]).name
                component_name = f"yakbox-analysis-{family}-runtime"
                component_version = "0"
            else:
                component_name = "yakbox"
                component_version = "0.1.0"
            output.write_text(
                json.dumps(
                    {
                        "bomFormat": "CycloneDX",
                        "specVersion": "1.5",
                        "metadata": {
                            "component": {
                                "name": component_name,
                                "version": component_version,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
        return ""

    report = run_release_preflight(
        tmp_path,
        tag="v0.1.0",
        runner=runner,
    )

    assert report.head_sha == "commit-sha"
    assert len(report.distributions) == 2
    assert report.report_path.is_file()
    checksums = report.checksums_path.read_text(encoding="utf-8")
    assert "yakbox-0.1.0-py3-none-any.whl" in checksums
    assert "yakbox-0.1.0.tar.gz" in checksums
    assert "yakbox.cdx.json" in checksums
    assert "yakbox-worker-whisper.cdx.json" in checksums
    assert "yakbox-worker-parakeet.cdx.json" in checksums
    assert "yakbox-worker-qwen.cdx.json" in checksums
    assert "release-preflight.json" in checksums
    assert (
        commands.count(
            (
                "uv",
                "build",
                "--no-sources",
                "--clear",
                "--no-create-gitignore",
                "--out-dir",
                str(tmp_path / "dist"),
            )
        )
        == 1
    )
    assert ("uv", "run", "pytest") in commands
    assert ("uv", "run", "lint-imports", "--no-cache") in commands
    assert ("uv", "audit", "--frozen") in commands
    assert len(report.worker_sboms) == 3
    assert sum(command[:2] == ("uv", "audit") for command in commands) == 4
    assert sum(command[:2] == ("uv", "export") for command in commands) == 4
    assert sum(command[:2] == ("uv", "run") for command in commands) >= 6
    assert {
        command[command.index("--project") + 1]
        for command in commands
        if command[:2] in {("uv", "audit"), ("uv", "export")} and "--project" in command
    } == {
        "src/yakbox/runtimes/whisper",
        "src/yakbox/runtimes/parakeet",
        "src/yakbox/runtimes/qwen",
    }
