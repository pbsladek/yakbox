from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from yakbox.release_preflight import (
    CommandRunner,
    ReleaseFile,
    ReleasePreflightError,
    _write_checksums,
    ensure_clean_worktree,
    ensure_tag_points_to_head,
    project_version,
    run_release_preflight,
    validate_release_tag,
)
from yakbox.release_preflight import (
    main as preflight_main,
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


def test_preflight_explains_how_to_release_new_changes_after_a_tag_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.2.0"\n')
    tagged_commit = "a" * 40
    release_commit = "b" * 40
    responses = {
        ("git", "status", "--porcelain", "--untracked-files=all"): "",
        ("git", "rev-parse", "HEAD"): release_commit,
        ("git", "rev-parse", "--verify", "refs/tags/v0.2.0^{commit}"): tagged_commit,
    }

    def runner(command: tuple[str, ...], _cwd: Path) -> str:
        return responses[command]

    monkeypatch.setattr("yakbox.release_preflight._run_command", runner)

    with pytest.raises(SystemExit) as result:
        preflight_main(
            ["--root", str(tmp_path), "--tag", "v0.2.0", "--require-tag-ref"]
        )

    assert result.value.code == 1
    diagnostic = capsys.readouterr().err
    assert tagged_commit in diagnostic
    assert release_commit in diagnostic
    assert "uv version --bump patch" in diagnostic
    assert "commit and merge" in diagnostic
    assert not (tmp_path / "dist").exists()
    assert not (tmp_path / "release-metadata").exists()


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


def _preflight_runner(
    commands: list[tuple[str, ...]],
    commit: str,
    metadata: Path,
    *,
    bad_suffix: str | None = None,
) -> CommandRunner:
    def runner(command: tuple[str, ...], _cwd: Path) -> str:
        commands.append(command)
        if command[:2] == ("git", "status"):
            return ""
        if command == ("git", "rev-parse", "HEAD"):
            return f"{commit}\n"
        if command[:2] == ("uv", "build"):
            assert commit in metadata.read_text()
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
        if command[-2:] == ("yakbox", "--version"):
            if bad_suffix and any(item.endswith(bad_suffix) for item in command):
                return "yakbox, version 0.1.0 (commit unknown)\n"
            return f"yakbox, version 0.1.0 (commit {commit})\n"
        return ""

    return runner


@pytest.mark.parametrize("suffix", [".whl", ".tar.gz"])
def test_preflight_rejects_installs_without_the_release_commit(
    tmp_path: Path, suffix: str
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n')
    metadata = tmp_path / "src" / "yakbox" / "_build_commit.py"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("COMMIT: str | None = None\n")
    runner = _preflight_runner([], "a" * 40, metadata, bad_suffix=suffix)
    with pytest.raises(ReleasePreflightError, match="release version and commit"):
        run_release_preflight(tmp_path, tag="v0.1.0", runner=runner)
    assert not (tmp_path / "release-metadata" / "release-preflight.json").exists()
    assert metadata.read_text() == "COMMIT: str | None = None\n"


def test_preflight_runs_all_gates_builds_once_and_writes_release_evidence(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "yakbox"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    commands: list[tuple[str, ...]] = []
    commit = "a" * 40
    metadata = tmp_path / "src" / "yakbox" / "_build_commit.py"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("COMMIT: str | None = None\n")

    runner = _preflight_runner(commands, commit, metadata)

    report = run_release_preflight(
        tmp_path,
        tag="v0.1.0",
        runner=runner,
    )

    assert report.head_sha == commit
    assert metadata.read_text() == "COMMIT: str | None = None\n"
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
    assets = {
        item.path.name: item.path
        for item in (*report.distributions, report.sbom, *report.worker_sboms)
    }
    assets[report.report_path.name] = report.report_path
    entries = [line.split("  ", 1) for line in checksums.splitlines()]
    assert {name for _, name in entries} == set(assets)
    assert all(
        digest == hashlib.sha256(assets[name].read_bytes()).hexdigest()
        for digest, name in entries
    )
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
    assert ("uv", "run", "pytest", "--cov=yakbox", "--cov-branch") in commands
    assert (
        "uv",
        "run",
        "twine",
        "check",
        "--strict",
        str(tmp_path / "dist" / "yakbox-0.1.0-py3-none-any.whl"),
        str(tmp_path / "dist" / "yakbox-0.1.0.tar.gz"),
    ) in commands
    assert sum(command[:2] == ("uv", "lock") for command in commands) == 4
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
