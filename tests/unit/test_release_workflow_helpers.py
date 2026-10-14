from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from yakbox import release_workflow as workflow
from yakbox.release_automation import _GitHub
from yakbox.release_preflight import ReleasePreflightError

HEAD = "a" * 40


def _pr(**changes: object) -> dict[str, object]:
    return {
        "number": 24,
        "headRefName": "release-please--branches--main",
        "headRefOid": HEAD,
        "mergeCommit": None,
        "isCrossRepository": False,
        "state": "OPEN",
        "baseRefName": "main",
        **changes,
    }


@pytest.mark.parametrize("created", ['{"number":24}', "", "null"])
def test_find_release_pr_uses_direct_api_and_writes_validated_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, created: str
) -> None:
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    commands: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        commands.append(command)
        return (
            json.dumps(_pr())
            if command[1] == "pr"
            else f"[]\n{json.dumps([_pr()])}\n[]"
        )

    workflow.find_pr(_GitHub("example/yakbox", "gh", runner), "main", created)

    assert output.read_text() == "number=24\nbranch=release-please--branches--main\n"
    assert len(commands) == 1
    if created.startswith("{"):
        assert commands[0][1:4] == ("pr", "view", "24")
    else:
        assert "--paginate" in commands[0]
        assert "--slurp" not in commands[0]


@pytest.mark.parametrize("pending", [False, True])
def test_find_pr_defers_pending_publication_without_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pending: bool,
) -> None:
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    def runner(command: tuple[str, ...]) -> str:
        return json.dumps(
            [_pr(mergeCommit={"oid": HEAD})]
            if pending and "state=closed" in command
            else []
        )

    workflow.find_pr(_GitHub("example/yakbox", "gh", runner), "main", "null")
    message = capsys.readouterr().out
    assert ("::notice::" in message) is pending
    assert ("No release PR:" in message) is not pending
    assert not output.exists()


@pytest.mark.parametrize(
    "created", ["not JSON", "{}", '{"number":true}', '{"number":-1}']
)
def test_find_pr_rejects_invalid_action_output(created: str) -> None:
    with pytest.raises(ReleasePreflightError):
        workflow.find_pr(_GitHub("example/yakbox", "gh", lambda _: ""), "main", created)


@pytest.mark.parametrize(
    "changes",
    [
        {"isCrossRepository": True},
        {"baseRefName": "other"},
        {"state": "CLOSED"},
        {"headRefName": "unrelated"},
        {"headRefName": "release-please--main\nnumber=99"},
    ],
)
def test_find_pr_rejects_untrusted_or_injectable_prs(
    changes: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    with pytest.raises(ReleasePreflightError):
        workflow.find_pr(
            _GitHub("example/yakbox", "gh", lambda _: json.dumps(_pr(**changes))),
            "main",
            '{"number":24}',
        )
    assert not output.exists()


def test_find_pr_does_not_hide_api_failures() -> None:
    def fail(_: tuple[str, ...]) -> str:
        raise ReleasePreflightError("API unavailable")

    with pytest.raises(ReleasePreflightError, match="API unavailable"):
        workflow.find_pr(_GitHub("example/yakbox", "gh", fail), "main", "null")


@pytest.mark.parametrize(
    "event,exists,required",
    [
        ("workflow_dispatch", False, False),
        ("workflow_dispatch", True, True),
        ("push", False, True),
        ("push", True, True),
    ],
)
def test_preflight_checks_an_existing_tag_and_all_tag_pushes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    event: str,
    exists: bool,
    required: bool,
) -> None:
    monkeypatch.setattr(
        workflow,
        "_command",
        lambda *args, **_kwargs: subprocess.CompletedProcess(
            args, 0 if exists else 1, "", ""
        ),
    )
    calls: list[bool] = []

    def preflight(root: Path, *, tag: str, require_tag_ref: bool) -> None:
        assert root == tmp_path
        assert tag == "v0.3.2"
        calls.append(require_tag_ref)

    monkeypatch.setattr(workflow, "run_release_preflight", preflight)
    workflow.preflight(tmp_path, "v0.3.2", event)
    assert calls == [required]


@pytest.mark.parametrize("status", [0, 1, 2])
def test_refresh_pr_pushes_only_valid_generated_changes(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    commands: list[tuple[str, ...]] = []

    def command(
        *arguments: str, allow_failure: bool = False
    ) -> subprocess.CompletedProcess[str]:
        commands.append(arguments)
        return subprocess.CompletedProcess(
            arguments, status if allow_failure else 0, "", ""
        )

    monkeypatch.setattr(workflow, "_command", command)
    if status == 2:
        with pytest.raises(ReleasePreflightError):
            workflow.refresh_pr("release-please--branches--main")
    else:
        workflow.refresh_pr("release-please--branches--main")
    assert (any(args[:2] == ("git", "push") for args in commands)) is (status == 1)
    assert commands.index(("uv", "lock")) < commands.index(
        ("uv", "sync", "--frozen", "--all-groups")
    )
    assert any("--write-packaged" in args for args in commands)


def test_refresh_pr_rejects_an_unrelated_branch() -> None:
    with pytest.raises(ReleasePreflightError):
        workflow.refresh_pr("main")


def test_start_ci_uses_a_readable_preparation_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GH_REPO", "example/yakbox")
    monkeypatch.setenv("RELEASE_BRANCH", "release-please--branches--main")
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "17")
    commands: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        commands.append(command)
        return '{"workflow_run_id":123}'

    monkeypatch.setattr(workflow, "_run", runner)
    workflow.main(["start-ci"])
    assert "inputs[release_preparation]=17" in commands[0]
    assert "ref=release-please--branches--main" in commands[0]


def test_workflow_cli_reports_expected_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RELEASE_BRANCH", "main")
    with pytest.raises(SystemExit) as error:
        workflow.main(["refresh-pr"])
    assert error.value.code == 1
    assert "Expected a Release Please branch" in capsys.readouterr().err
