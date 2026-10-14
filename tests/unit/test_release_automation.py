from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from yakbox.release_automation import _GitHub, automate_release, main
from yakbox.release_preflight import ReleasePreflightError

ROOT = Path(__file__).parents[2]
HEAD = "a" * 40
MERGED = "b" * 40


class FakeGitHub:
    def __init__(self, *, failed: str = "", resume: bool = False) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.request = ""
        self.failed = failed
        self.resume = resume
        self.head = HEAD

    def pr(self, *, merged: bool = False) -> dict[str, object]:
        return {
            "number": 25,
            "headRefName": "release-please--branches--main",
            "headRefOid": HEAD,
            "mergeCommit": {"oid": MERGED} if merged else None,
            "isCrossRepository": False,
            "state": "MERGED" if merged else "OPEN",
        }

    def __call__(self, command: tuple[str, ...]) -> str:
        self.commands.append(command)
        if command[1:3] == ("workflow", "run"):
            self.request = next(
                part.removeprefix("request_id=")
                for part in command
                if part.startswith("request_id=")
            )
            return ""
        if command[1:3] == ("run", "list"):
            workflow = command[command.index("--workflow") + 1]
            identifier = {
                "release-please.yml": 10,
                "ci.yml": 20,
                "release.yml": 30,
                "release-publish.yml": 0,
            }[workflow]
            title = {
                "release-please.yml": f"Release preparation {self.request}",
                "ci.yml": "CI release-10",
                "release-publish.yml": "",
                "release.yml": f"Release {self.request}"
                if self.resume
                else f"Release publish-{MERGED}",
            }[workflow]
            return json.dumps(
                []
                if identifier == 0
                else [
                    {
                        "databaseId": identifier,
                        "displayTitle": title,
                        "status": "completed",
                        "headSha": self.head,
                        "conclusion": "failure"
                        if self.failed == workflow
                        else "success",
                    }
                ]
            )
        if command[1] == "api" and "--paginate" in command:
            state = "merged" if "state=closed" in command else "open"
            return json.dumps(
                [self.pr(merged=True)]
                if state == "merged" and self.resume
                else [self.pr()]
                if state == "open"
                else []
            )
        if command[1:3] == ("pr", "merge"):
            assert command[command.index("--match-head-commit") + 1] == HEAD
            if self.failed == "merge":
                raise ReleasePreflightError("Required review missing")
            return ""
        if command[1:3] == ("pr", "view"):
            return json.dumps(self.pr(merged=True))
        if command[1] == "api":
            return (
                "c" * 40
                if command[2].endswith("/commits/main")
                else json.dumps({".": "0.2.2"})
            )
        raise AssertionError(command)


def test_one_command_prepares_tests_merges_and_waits_for_publication() -> None:
    github = FakeGitHub()

    assert automate_release("example/yakbox", runner=github) == 30

    merge = next(
        i
        for i, command in enumerate(github.commands)
        if command[1:3] == ("pr", "merge")
    )
    assert any("ci.yml" in command for command in github.commands[:merge])
    assert any("release.yml" in command for command in github.commands[merge:])
    assert all("--admin" not in command for command in github.commands)


@pytest.mark.parametrize("failed", ["release-please.yml", "ci.yml", "merge"])
def test_unsuccessful_gates_do_not_reach_publication(failed: str) -> None:
    github = FakeGitHub(failed=failed)

    with pytest.raises(ReleasePreflightError):
        automate_release("example/yakbox", runner=github)

    assert not any("release.yml" in command for command in github.commands)
    if failed != "merge":
        assert not any(command[1:3] == ("pr", "merge") for command in github.commands)


def test_ci_for_another_commit_cannot_authorize_the_merge() -> None:
    github = FakeGitHub()
    github.head = "c" * 40

    with pytest.raises(ReleasePreflightError, match="different release commit"):
        automate_release("example/yakbox", runner=github)

    assert not any(command[1:3] == ("pr", "merge") for command in github.commands)


def test_publication_retry_uses_corrected_main_with_the_same_pending_version() -> None:
    github = FakeGitHub(resume=True)

    assert automate_release("example/yakbox", runner=github) == 30

    dispatch = next(
        command for command in github.commands if command[1:3] == ("workflow", "run")
    )
    assert "release.yml" in dispatch
    assert f"commit={'c' * 40}" in dispatch
    assert "tag=v0.2.2" in dispatch
    assert "release_pr=25" in dispatch
    assert not any(command[1:3] == ("pr", "merge") for command in github.commands)


@pytest.mark.parametrize("status", ["queued", "in_progress", "completed"])
def test_make_release_joins_existing_publication_instead_of_retrying(
    status: str,
) -> None:
    github = FakeGitHub(resume=True)
    lookups = 0

    def runner(command: tuple[str, ...]) -> str:
        nonlocal lookups
        result = github(command)
        if command[1:3] == ("run", "list") and "release.yml" in command:
            lookups += 1
            return json.dumps(
                [
                    {
                        "databaseId": 30,
                        "displayTitle": f"Release publish-{MERGED}",
                        "status": status if lookups == 1 else "completed",
                        "conclusion": "success",
                    }
                ]
            )
        return result

    assert automate_release("example/yakbox", runner=runner) == 30
    assert not any(command[1:3] == ("workflow", "run") for command in github.commands)


@pytest.mark.parametrize("status", ["queued", "in_progress", "completed"])
def test_make_release_waits_for_publication_dispatch_to_become_visible(
    status: str,
) -> None:
    github = FakeGitHub(resume=True)
    lookups = 0

    def runner(command: tuple[str, ...]) -> str:
        nonlocal lookups
        result = github(command)
        if command[1:3] == ("run", "list") and "release.yml" in command:
            lookups += 1
            return json.dumps(
                []
                if lookups == 1
                else [
                    {
                        "databaseId": 30,
                        "displayTitle": f"Release publish-{MERGED}",
                        "status": "completed",
                        "conclusion": "success",
                    }
                ]
            )
        if command[1:3] == ("run", "list") and "release-publish.yml" in command:
            return json.dumps([{"status": status, "conclusion": "success"}])
        return result

    assert automate_release("example/yakbox", runner=runner) == 30
    assert not any(command[1:3] == ("workflow", "run") for command in github.commands)


def test_failed_existing_publication_allows_an_explicit_retry() -> None:
    github = FakeGitHub(resume=True)
    lookups = 0

    def runner(command: tuple[str, ...]) -> str:
        nonlocal lookups
        result = github(command)
        if command[1:3] == ("run", "list") and "release.yml" in command:
            lookups += 1
            if lookups == 1:
                return json.dumps(
                    [
                        {
                            "databaseId": 29,
                            "displayTitle": f"Release publish-{MERGED}",
                            "status": "completed",
                            "conclusion": "failure",
                        }
                    ]
                )
        return result

    assert automate_release("example/yakbox", runner=runner) == 30
    assert sum(command[1:3] == ("workflow", "run") for command in github.commands) == 1


def test_publication_retry_rejects_a_different_version_on_main() -> None:
    github = FakeGitHub(resume=True)

    def runner(command: tuple[str, ...]) -> str:
        result = github(command)
        if command[1] == "api" and command[2].endswith(f"ref={'c' * 40}"):
            return json.dumps({".": "0.3.0"})
        return result

    with pytest.raises(ReleasePreflightError, match="pending release version"):
        automate_release("example/yakbox", runner=runner)

    assert not any(command[1:3] == ("workflow", "run") for command in github.commands)


def test_pending_release_lookup_combines_direct_api_pages() -> None:
    pr = FakeGitHub().pr(merged=True)
    github = _GitHub("example/yakbox", "gh", lambda _: f"[]\n{json.dumps([pr])}\n[]\n")

    pending = github.pending("main", "merged")

    assert pending is not None
    assert pending.number == 25
    assert pending.merged == MERGED


@pytest.mark.parametrize("payload", ["not JSON", "{}"])
def test_pending_release_lookup_rejects_invalid_pages(payload: str) -> None:
    github = _GitHub("example/yakbox", "gh", lambda _: payload)
    with pytest.raises(ReleasePreflightError, match="invalid release PR"):
        github.pending("main", "open")


def test_run_discovery_waits_for_the_matching_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            [],
            [{"databaseId": 1, "displayTitle": "other", "status": "completed"}],
            [
                {
                    "databaseId": 2,
                    "displayTitle": "wanted",
                    "status": "completed",
                    "conclusion": "success",
                }
            ],
        ]
    )
    monkeypatch.setattr("yakbox.release_automation.time.sleep", lambda _: None)
    github = _GitHub("example/yakbox", "gh", lambda _: json.dumps(next(responses)))

    assert github.wait_run("ci.yml", "wanted") == 2


def test_waits_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([0, 5_000])
    monkeypatch.setattr("yakbox.release_automation.time.monotonic", lambda: next(times))
    github = _GitHub("example/yakbox", "gh", lambda _: "[]")

    with pytest.raises(ReleasePreflightError, match="Timed out"):
        github.wait_run("ci.yml", "never starts")


def test_cli_reports_invalid_repository_without_dispatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as result:
        main(["--repo", "not-a-repository"])
    assert result.value.code == 1
    assert "owner/name" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("previous", "current", "project", "expected"),
    [
        (None, "0.2.1", "0.2.1", ""),
        ("0.2.1", "0.2.1", "0.2.1", ""),
        ("0.2.1", "0.2.2", "0.2.2", "tag=v0.2.2\n"),
        ("0.2.1", "0.2.2", "0.2.1", None),
        ("0.2.1", "0.2.0", "0.2.0", None),
    ],
)
def test_merged_manifest_dispatch_is_bootstrap_safe_and_version_bound(
    tmp_path: Path,
    previous: str | None,
    current: str,
    project: str,
    expected: str | None,
) -> None:
    git = shutil.which("git")
    assert git is not None
    manifest = tmp_path / ".release-please-manifest.json"
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.2.1"\n')
    if previous is not None:
        manifest.write_text(json.dumps({".": previous}))
    for command in (
        (git, "init", "-q"),
        (git, "add", "."),
        (
            git,
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.name=Release test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--no-gpg-sign",
            "-qm",
            "Fixture",
        ),
    ):
        subprocess.run(  # noqa: S603 - fixed Git fixture commands
            command, cwd=tmp_path, check=True, capture_output=True, timeout=10
        )
    commit = subprocess.check_output(  # noqa: S603 - fixed Git fixture command
        (git, "rev-parse", "HEAD"), cwd=tmp_path, text=True, timeout=10
    ).strip()
    manifest.write_text(json.dumps({".": current}))
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{project}"\n')
    output = tmp_path / "outputs"

    result = subprocess.run(  # noqa: S603 - checked-in script and temporary repository
        (sys.executable, str(ROOT / ".github/scripts/merged_release.py")),
        cwd=tmp_path,
        env={**os.environ, "PREVIOUS_COMMIT": commit, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert (result.returncode == 0) is (expected is not None), result.stderr
    assert (output.read_text() if output.exists() else "") == (expected or "")
