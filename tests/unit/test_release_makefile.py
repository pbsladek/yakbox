from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

ROOT = Path(__file__).parents[2]


def _make(
    tmp_path: Path, *arguments: str, exit_code: int = 0
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    make = shutil.which("make")
    assert make is not None
    log = tmp_path / "commands.jsonl"
    executable = tmp_path / "cli"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with Path(os.environ['CLI_LOG']).open('a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:] == ['version', '--short']:\n"
        "    print('0.2.0')\n"
        "sys.exit(int(os.environ['CLI_EXIT']))\n"
    )
    executable.chmod(0o755)
    settings = {"RELEASE_REPO", "RELEASE_REF", "RELEASE_TAG", "PUBLISH_TO", "RUN_ID"}
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in settings | {"MAKEFLAGS", "MFLAGS", "MAKELEVEL"}
    }
    result = subprocess.run(  # noqa: S603 - reviewed Makefile and fixture CLIs only
        (make, f"UV={executable}", f"GH={executable}", *arguments),
        cwd=ROOT,
        env={**environment, "CLI_LOG": str(log), "CLI_EXIT": str(exit_code)},
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    commands = (
        [cast(list[str], json.loads(line)) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return result, commands


def test_make_default_only_shows_help(tmp_path: Path) -> None:
    result, commands = _make(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "make release" in result.stdout
    assert commands == []


def test_make_release_defaults_to_github_and_remote_version(tmp_path: Path) -> None:
    result, commands = _make(tmp_path, "release")
    assert result.returncode == 0, result.stderr
    assert commands == [
        [
            "workflow",
            "run",
            "release.yml",
            "--repo",
            "pbsladek/yakbox",
            "--ref",
            "main",
            "-f",
            "tag=",
            "-f",
            "publish_to=github",
        ]
    ]


@pytest.mark.parametrize("destination", ["pypi", "testpypi"])
def test_make_release_preserves_explicit_settings(
    tmp_path: Path, destination: str
) -> None:
    tag = "v0.2.0; literal text"
    result, commands = _make(
        tmp_path,
        "release",
        "RELEASE_REPO=example/yakbox",
        "RELEASE_REF=v0.2.0",
        f"RELEASE_TAG={tag}",
        f"PUBLISH_TO={destination}",
    )
    assert result.returncode == 0, result.stderr
    assert commands[0] == [
        "workflow",
        "run",
        "release.yml",
        "--repo",
        "example/yakbox",
        "--ref",
        "v0.2.0",
        "-f",
        f"tag={tag}",
        "-f",
        f"publish_to={destination}",
    ]


def test_make_release_rejects_unknown_destination_before_dispatch(
    tmp_path: Path,
) -> None:
    result, commands = _make(tmp_path, "release", "PUBLISH_TO=unknown")
    assert result.returncode != 0
    assert "PUBLISH_TO must be" in result.stderr
    assert commands == []


def test_make_preflight_derives_local_version(tmp_path: Path) -> None:
    result, commands = _make(tmp_path, "release-preflight")
    assert result.returncode == 0, result.stderr
    assert commands == [
        ["version", "--short"],
        ["run", "yakbox-release-preflight", "--tag", "v0.2.0"],
    ]


def test_make_preflight_uses_explicit_tag(tmp_path: Path) -> None:
    result, commands = _make(tmp_path, "release-preflight", "RELEASE_TAG=v0.2.0")
    assert result.returncode == 0, result.stderr
    assert commands == [["run", "yakbox-release-preflight", "--tag", "v0.2.0"]]


def test_make_preflight_stops_when_version_lookup_fails(tmp_path: Path) -> None:
    result, commands = _make(tmp_path, "release-preflight", exit_code=1)
    assert result.returncode != 0
    assert commands == [["version", "--short"]]


def test_make_lists_only_release_runs(tmp_path: Path) -> None:
    result, commands = _make(tmp_path, "release-runs")
    assert result.returncode == 0, result.stderr
    assert commands == [
        ["run", "list", "--repo", "pbsladek/yakbox", "--workflow", "release.yml"]
    ]


@pytest.mark.parametrize("run_id", ["", "latest", "12; echo unexpected"])
def test_make_watch_requires_numeric_run_id(tmp_path: Path, run_id: str) -> None:
    result, commands = _make(tmp_path, "release-watch", f"RUN_ID={run_id}")
    assert result.returncode != 0
    assert "Set RUN_ID" in result.stderr
    assert commands == []


def test_make_watch_returns_workflow_failure(tmp_path: Path) -> None:
    result, commands = _make(tmp_path, "release-watch", "RUN_ID=12345", exit_code=1)
    assert result.returncode != 0
    assert commands == [
        ["run", "watch", "12345", "--repo", "pbsladek/yakbox", "--exit-status"]
    ]


def test_make_release_returns_dispatch_failure(tmp_path: Path) -> None:
    result, commands = _make(tmp_path, "release", exit_code=1)
    assert result.returncode != 0
    assert len(commands) == 1
    assert "Release dispatched." not in result.stdout
