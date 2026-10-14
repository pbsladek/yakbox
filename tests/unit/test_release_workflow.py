from __future__ import annotations

import itertools
import json
import os
import shlex
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


def test_every_release_publishes_to_both_destinations_before_completion() -> None:
    workflow = _workflow("release.yml")
    inputs = _mapping(_mapping(_mapping(workflow["on"])["workflow_dispatch"])["inputs"])
    assert "publish_to" not in inputs
    assert "publish-testpypi" not in _mapping(workflow["jobs"])
    assert _mapping(inputs["tag"])["required"] is False
    github = _job(workflow, "github-release")
    pypi = _job(workflow, "publish-pypi")
    finish = _job(workflow, "finish")
    assert set(cast(list[str], github["needs"])) == {"build", "test"}
    assert set(cast(list[str], pypi["needs"])) == {"build", "test", "github-release"}
    assert set(cast(list[str], finish["needs"])) == {
        "build",
        "github-release",
        "publish-pypi",
    }
    for job in (github, pypi, finish):
        assert "if" not in job
        steps = cast(list[dict[str, object]], job["steps"])
        assert _mapping(steps[0]["with"])["ref"] == "${{ needs.build.outputs.commit }}"
        assert steps[1]["uses"] == "./.github/actions/release-assets"
    assert _mapping(_mapping(_job(workflow, "test")["strategy"])["matrix"])["os"] == [
        "ubuntu-latest",
        "macos-latest",
    ]


def test_shared_download_action_verifies_assets_before_the_publish_step() -> None:
    action = _mapping(
        yaml.safe_load((ROOT / ".github/actions/release-assets/action.yml").read_text())
    )
    steps = cast(list[dict[str, object]], _mapping(action["runs"])["steps"])
    downloads = [
        step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/download-artifact@")
    ]
    assert {_mapping(step["with"])["path"] for step in downloads} == {
        "dist/",
        "release-metadata/",
    }
    assert str(steps[-1]["run"]).endswith("yakbox.release_publication verify-assets")


def test_index_publication_uses_oidc_and_checks_existing_files_before_retry() -> None:
    job = _job(_workflow("release.yml"), "publish-pypi")
    assert _mapping(job["environment"])["name"] == "pypi"
    assert _mapping(job["permissions"])["id-token"] == "write"
    steps = cast(list[dict[str, object]], job["steps"])
    publish = steps[-1]
    assert str(publish["uses"]).startswith("pypa/gh-action-pypi-publish@")
    configuration = _mapping(publish["with"])
    assert configuration == {"skip-existing": True}
    assert str(steps[-2]["run"]).endswith("yakbox.release_publication check-pypi")


def test_release_uploads_tested_assets_and_targets_the_resolved_commit() -> None:
    workflow = _workflow("release.yml")
    steps = cast(list[dict[str, object]], _job(workflow, "github-release")["steps"])
    environment = _mapping(_job(workflow, "github-release")["env"])
    assert environment["GH_REPO"] == "${{ github.repository }}"
    assert environment["RELEASE_TAG"] == "${{ needs.build.outputs.tag }}"
    assert environment["RELEASE_COMMIT"] == "${{ needs.build.outputs.commit }}"
    assert str(steps[-1]["run"]).endswith("yakbox.release_publication publish-github")

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


@pytest.mark.parametrize(
    ("requested", "accepted"),
    [("", True), ("v0.2.0", True), ("v0.1.0", False), ("v0.2.0\ncommit=wrong", False)],
)
@pytest.mark.parametrize("requested_commit", ["", "c" * 40])
def test_manual_dispatch_resolves_the_version_and_commit_before_publication(
    tmp_path: Path, requested: str, accepted: bool, requested_commit: str
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
    result = subprocess.run(
        (sys.executable, "-m", "yakbox.release_workflow", "resolve"),
        cwd=tmp_path,
        env={
            **os.environ,
            "REQUESTED_TAG": requested,
            "REQUESTED_COMMIT": requested_commit,
            "GITHUB_OUTPUT": str(output),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    if accepted and not requested_commit:
        assert result.returncode == 0, result.stderr
        commit = subprocess.check_output(  # noqa: S603 - fixed Git fixture argv
            (git, "rev-parse", "HEAD"), cwd=tmp_path, text=True, timeout=10
        ).strip()
        assert output.read_text() == f"tag=v0.2.0\ncommit={commit}\n"
    else:
        assert result.returncode != 0
        assert not output.exists()


def test_release_please_prepares_generated_files_and_explicitly_runs_ci() -> None:
    config = json.loads((ROOT / "release-please-config.json").read_text())
    assert config["packages"]["."]["release-type"] == "python"
    assert config["include-component-in-tag"] is False
    prepare = _job(_workflow("release-please.yml"), "prepare")
    steps = cast(list[dict[str, object]], prepare["steps"])
    action = next(
        step
        for step in steps
        if str(step.get("uses", "")).startswith("googleapis/release-please-action@")
    )
    assert _mapping(action["with"])["skip-github-release"] is True
    commands = "\n".join(str(step.get("run", "")) for step in steps)
    assert "yakbox.release_workflow refresh-pr" in commands
    assert "yakbox.release_workflow start-ci" in commands
    assert "yakbox.release_workflow find-pr" in commands
    assert _mapping(prepare["permissions"])["actions"] == "write"
    publish = _job(_workflow("release-publish.yml"), "dispatch")
    publication = "\n".join(
        str(step.get("run", ""))
        for step in cast(list[dict[str, object]], publish["steps"])
    )
    assert '-f "commit=$RELEASE_COMMIT"' in publication
    assert "publish_to=" not in publication
    assert "request_id=" not in publication


def test_workflow_titles_show_release_stages_and_versions_without_hashes() -> None:
    assert (
        _workflow("release.yml")["run-name"]
        == "Release ${{ inputs.tag || github.ref_name }}"
    )
    assert (
        _workflow("release-please.yml")["run-name"]
        == "Prepare release · ${{ github.ref_name }}"
    )
    assert (
        _workflow("release-publish.yml")["run-name"]
        == "Start release · ${{ github.ref_name }}"
    )
    assert "release preparation #{0}" in str(_workflow("ci.yml")["run-name"])


def test_ci_package_rejects_stale_lock_without_rewriting_it(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nname = "release-fixture"\nversion = "0.2.1"\n'
        'requires-python = ">=3.14,<3.15"\n'
    )
    environment = {**os.environ, "UV_OFFLINE": "true", "UV_PYTHON": sys.executable}
    subprocess.run(  # noqa: S603 - installed uv, dependency-free temporary project
        [uv, "lock"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        timeout=30,
    )
    lock = tmp_path / "uv.lock"
    original = lock.read_bytes()
    project.write_text(
        project.read_text().replace('version = "0.2.1"', 'version = "0.2.2"')
    )
    steps = cast(list[dict[str, object]], _job(_workflow("ci.yml"), "package")["steps"])
    script = next(
        str(step["run"])
        for step in steps
        if step.get("name") == "Build distributions with their source commit"
    )
    arguments = shlex.split(script)
    result = subprocess.run(  # noqa: S603 - checked-in CI argv, installed uv, temporary project
        [uv, *arguments[1:]],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "lockfile" in result.stderr.lower()
    assert lock.read_bytes() == original


@pytest.mark.parametrize(
    ("workflow", "job"),
    [("ci.yml", "quality"), ("ci.yml", "test"), ("release.yml", "build")],
)
def test_jobs_check_lock_freshness_before_uv_can_refresh_it(
    workflow: str, job: str
) -> None:
    steps = cast(list[dict[str, object]], _job(_workflow(workflow), job)["steps"])
    commands = [str(step["run"]) for step in steps if "run" in step]
    check = commands.index("uv lock --check")
    assert check < next(
        index
        for index, command in enumerate(commands)
        if command.startswith(("uv run", "uv sync"))
    )
