from __future__ import annotations

import hashlib
import itertools
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import cast

import pytest
import yaml
from packaging.specifiers import SpecifierSet

from yakbox import __version__

ROOT = Path(__file__).parents[2]


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _workflow(name: str) -> dict[str, object]:
    content = (ROOT / ".github" / "workflows" / name).read_text()
    # YAML 1.1 treats bare `on` as a boolean; Actions uses it as a string key.
    return _mapping(yaml.safe_load(content.replace("\non:", '\n"on":', 1)))


def _job(workflow: dict[str, object], name: str) -> dict[str, object]:
    return _mapping(_mapping(workflow["jobs"])[name])


def test_release_version_and_platform_support_are_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked = next(package for package in lock["package"] if package["name"] == "yakbox")

    assert project["version"] == locked["version"] == __version__
    assert "Operating System :: MacOS :: MacOS X" in project["classifiers"]
    assert "Operating System :: POSIX :: Linux" in project["classifiers"]
    assert not any("Windows" in classifier for classifier in project["classifiers"])
    for name in ("ci.yml", "release.yml"):
        assert (
            "windows-latest" not in (ROOT / ".github" / "workflows" / name).read_text()
        )


def test_latest_uv_is_supported_and_all_workflows_use_it() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    versions = SpecifierSet(project["tool"]["uv"]["required-version"])
    assert "0.13.0" in versions
    assert "0.14.0" in versions
    assert "0.12.0" not in versions
    for path in (ROOT / ".github" / "workflows").glob("*.yml"):
        workflow = _workflow(path.name)
        for job in _mapping(workflow["jobs"]).values():
            for step in cast(list[dict[str, object]], _mapping(job)["steps"]):
                if str(step.get("uses", "")).startswith("astral-sh/setup-uv@"):
                    assert _mapping(step["with"])["version"] == "latest"


def test_ci_checks_prs_main_and_merge_queue_with_one_required_gate() -> None:
    workflow = _workflow("ci.yml")
    events = _mapping(workflow["on"])
    assert _mapping(events["push"])["branches"] == ["main"]
    assert _mapping(events["pull_request"])["branches"] == ["main"]
    assert "merge_group" in events
    gate = _job(workflow, "gate")
    assert gate["if"] == "always()"
    assert gate["name"] == "CI gate"
    assert set(cast(list[str], gate["needs"])) == {"quality", "test", "package"}


@pytest.mark.parametrize(
    "results",
    [("success", "success", "success")]
    + [
        tuple(state if index == job else "success" for index in range(3))
        for job, state in itertools.product(
            range(3), ("failure", "cancelled", "skipped")
        )
    ],
)
def test_ci_gate_rejects_every_unsuccessful_required_job(
    results: tuple[str, str, str],
) -> None:
    gate = _job(_workflow("ci.yml"), "gate")
    step = cast(list[dict[str, object]], gate["steps"])[0]
    shell = shutil.which("bash")
    assert shell is not None
    result = subprocess.run(  # noqa: S603 - reviewed gate shell and fixed result values
        (shell, "-e", "-o", "pipefail", "-c", str(step["run"])),
        env={
            **os.environ,
            "QUALITY_RESULT": results[0],
            "TEST_RESULT": results[1],
            "PACKAGE_RESULT": results[2],
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert (result.returncode == 0) == all(state == "success" for state in results)


def test_ci_audits_worker_locks_and_dependabot_tracks_them() -> None:
    steps = cast(list[dict[str, object]], _job(_workflow("ci.yml"), "quality")["steps"])
    commands = "\n".join(str(step.get("run", "")) for step in steps)
    assert "whisper parakeet qwen" in commands
    assert 'uv lock --project "src/yakbox/runtimes/$family" --check' in commands
    assert (
        'uv audit --project "src/yakbox/runtimes/$family" --frozen --no-dev' in commands
    )
    assert "actionlint@v1.7.12" in commands
    dependabot = _mapping(
        yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text())
    )
    update = next(
        item
        for item in cast(list[dict[str, object]], dependabot["updates"])
        if item["package-ecosystem"] == "uv"
    )
    assert set(cast(list[str], update["directories"])) == {
        "/",
        "/src/yakbox/runtimes/whisper",
        "/src/yakbox/runtimes/parakeet",
        "/src/yakbox/runtimes/qwen",
    }


def test_each_publication_checks_assets_before_uploading() -> None:
    workflow = _workflow("release.yml")
    for name in ("github-release", "publish-pypi", "publish-testpypi"):
        steps = cast(list[dict[str, object]], _job(workflow, name)["steps"])
        verify = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Verify release checksums"
        )
        assert "sha256sum --check SHA256SUMS" in str(steps[verify]["run"])
        assert "release-metadata/*" in str(steps[verify]["run"])
        assert verify < len(steps) - 1


@pytest.mark.parametrize("job", ["github-release", "publish-pypi", "publish-testpypi"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_publication_checksum_gate_rejects_modified_assets(
    tmp_path: Path, job: str, corrupt: bool
) -> None:
    dist = tmp_path / "dist"
    metadata = tmp_path / "release-metadata"
    dist.mkdir()
    metadata.mkdir()
    wheel = dist / "yakbox-0.2.0-py3-none-any.whl"
    report = metadata / "release-preflight.json"
    wheel.write_bytes(b"reviewed wheel")
    report.write_bytes(b"reviewed report")
    (metadata / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in (wheel, report)
        )
    )
    if corrupt:
        wheel.write_bytes(b"modified wheel")
    steps = cast(list[dict[str, object]], _job(_workflow("release.yml"), job)["steps"])
    step = next(
        item for item in steps if item.get("name") == "Verify release checksums"
    )
    script = str(step["run"])
    if shutil.which("sha256sum") is None:
        tool = shutil.which("shasum")
        assert tool is not None
        script = script.replace("sha256sum", f"'{tool}' -a 256")
    shell = shutil.which("bash")
    assert shell is not None
    result = subprocess.run(  # noqa: S603 - reviewed checksum shell and fixture assets
        (shell, "-e", "-o", "pipefail", "-c", script),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert (result.returncode == 0) is not corrupt


@pytest.mark.parametrize(
    ("event", "destination", "exists", "require_tag"),
    [
        ("workflow_dispatch", "github", False, False),
        ("workflow_dispatch", "github", True, True),
        ("push", "github", False, True),
        ("workflow_dispatch", "pypi", False, True),
        ("workflow_dispatch", "testpypi", False, True),
    ],
)
def test_release_preflight_requires_existing_tags_for_package_indexes(
    tmp_path: Path, event: str, destination: str, exists: bool, require_tag: bool
) -> None:
    steps = cast(
        list[dict[str, object]], _job(_workflow("release.yml"), "build")["steps"]
    )
    step = next(item for item in steps if item.get("name") == "Run release preflight")
    commands = tmp_path / "commands"
    (tmp_path / "uv").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{commands}"\n')
    (tmp_path / "git").write_text(f"#!/bin/sh\nexit {0 if exists else 1}\n")
    for name in ("git", "uv"):
        (tmp_path / name).chmod(0o755)
    shell = shutil.which("bash")
    assert shell is not None
    result = subprocess.run(  # noqa: S603 - reviewed preflight shell with fake uv/Git
        (shell, "-e", "-o", "pipefail", "-c", str(step["run"])),
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_EVENT_NAME": event,
            "PUBLISH_TO": destination,
            "RELEASE_TAG": "v0.2.0",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    expected = ["run", "yakbox-release-preflight", "--tag", "v0.2.0"]
    if require_tag:
        expected.append("--require-tag-ref")
    assert commands.read_text().splitlines() == expected


def test_github_release_is_dispatchable_without_pypi_and_waits_for_both_platforms() -> (
    None
):
    workflow = _workflow("release.yml")
    inputs = _mapping(_mapping(_mapping(workflow["on"])["workflow_dispatch"])["inputs"])
    destination = _mapping(inputs["publish_to"])
    assert destination["default"] == "github"
    assert destination["options"] == ["github", "pypi", "testpypi"]
    assert _mapping(inputs["tag"])["required"] is False

    release = _job(workflow, "github-release")
    assert set(cast(list[str], release["needs"])) == {"build", "test"}
    assert release["if"] == (
        "github.event_name == 'push' || inputs.publish_to == 'github'"
    )
    assert _mapping(_mapping(_job(workflow, "test")["strategy"])["matrix"])["os"] == [
        "ubuntu-latest",
        "macos-latest",
    ]
    for name in ("publish-pypi", "publish-testpypi"):
        job = _job(workflow, name)
        assert set(cast(list[str], job["needs"])) == {"build", "test"}
        assert "workflow_dispatch" in str(job["if"])
        assert "inputs.publish_to ==" in str(job["if"])


def test_release_uploads_tested_assets_and_targets_the_resolved_commit() -> None:
    workflow = _workflow("release.yml")
    steps = cast(list[dict[str, object]], _job(workflow, "github-release")["steps"])
    create = next(step for step in steps if step.get("name") == "Create GitHub Release")
    environment = _mapping(create["env"])
    assert environment["GH_REPO"] == "${{ github.repository }}"
    assert environment["RELEASE_TAG"] == "${{ needs.build.outputs.tag }}"
    assert environment["RELEASE_COMMIT"] == "${{ needs.build.outputs.commit }}"
    command = str(create["run"])
    assert 'gh release create "$RELEASE_TAG" dist/* release-metadata/*' in command
    assert '--target "$RELEASE_COMMIT"' in command

    test_steps = cast(list[dict[str, object]], _job(workflow, "test")["steps"])
    checkout = test_steps[0]
    assert _mapping(checkout["with"])["ref"] == "${{ needs.build.outputs.commit }}"
    assert any("--cov-branch" in str(step.get("run", "")) for step in test_steps)
    assert any(
        "tests/package_worker_consumer.py" in str(step.get("run", ""))
        for step in test_steps
    )
    install = next(
        step
        for step in test_steps
        if step.get("name") == "Install and verify the built wheel"
    )
    assert _mapping(install["env"])["RELEASE_COMMIT"] == (
        "${{ needs.build.outputs.commit }}"
    )
    assert "${RELEASE_TAG#v} (commit ${RELEASE_COMMIT})" in str(install["run"])
    assert 'test "$actual" = "$expected"' in str(install["run"])


def test_ci_packages_the_commit_and_checks_installed_wheel_and_source() -> None:
    steps = cast(list[dict[str, object]], _job(_workflow("ci.yml"), "package")["steps"])
    assert any("python -m yakbox._build" in str(step.get("run", "")) for step in steps)
    verify = next(
        step
        for step in steps
        if step.get("name") == "Verify installed wheel and source version metadata"
    )
    command = str(verify["run"])
    assert "git rev-parse HEAD" in command
    assert 'test "$wheel_version" = "$expected"' in command
    assert 'test "$source_version" = "$expected"' in command


def _release_resolver_script() -> str:
    steps = cast(
        list[dict[str, object]], _job(_workflow("release.yml"), "build")["steps"]
    )
    step = next(item for item in steps if item.get("id") == "release")
    script = str(step["run"])
    return script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


@pytest.mark.parametrize(
    ("requested", "accepted"),
    [("", True), ("v0.2.0", True), ("v0.1.0", False), ("v0.2.0\ncommit=wrong", False)],
)
def test_manual_dispatch_resolves_the_version_and_commit_before_publication(
    tmp_path: Path, requested: str, accepted: bool
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.2.0"\n')
    git = shutil.which("git")
    assert git is not None
    for command in (
        (git, "init", "--quiet"),
        (git, "add", "pyproject.toml"),
        (
            git,
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=release@example.test",
            "commit",
            "--quiet",
            "--no-gpg-sign",
            "-m",
            "Release fixture",
        ),
    ):
        subprocess.run(  # noqa: S603 - fixed Git fixture argv; no shell
            command, cwd=tmp_path, check=True, capture_output=True, timeout=10
        )
    output = tmp_path / "outputs"
    result = subprocess.run(  # noqa: S603 - execute reviewed workflow Python; no shell
        (sys.executable, "-c", _release_resolver_script()),
        cwd=tmp_path,
        env={**os.environ, "REQUESTED_TAG": requested, "GITHUB_OUTPUT": str(output)},
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    if accepted:
        assert result.returncode == 0, result.stderr
        commit = subprocess.check_output(  # noqa: S603 - fixed Git fixture argv
            (git, "rev-parse", "HEAD"), cwd=tmp_path, text=True, timeout=10
        ).strip()
        assert output.read_text() == f"tag=v0.2.0\ncommit={commit}\n"
    else:
        assert result.returncode != 0
        assert not output.exists()
