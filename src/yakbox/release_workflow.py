"""Small, testable commands used by the release preparation and build jobs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from yakbox.release_automation import _PR_FIELDS, _GitHub, _mapping, _pull_request, _run
from yakbox.release_preflight import (
    ReleasePreflightError,
    project_version,
    run_release_preflight,
    validate_release_tag,
)


def _command(
    *arguments: str, allow_failure: bool = False
) -> subprocess.CompletedProcess[str]:
    executable = shutil.which(arguments[0])
    if executable is None:
        raise ReleasePreflightError(
            f"Required release command is missing: {arguments[0]}"
        )
    try:
        result = subprocess.run(  # noqa: S603 - fixed release tools and validated arguments
            (executable, *arguments[1:]),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ReleasePreflightError(
            f"Release command failed: {arguments[0]}"
        ) from error
    if result.returncode and not allow_failure:
        raise ReleasePreflightError((result.stderr or result.stdout).strip()[:2048])
    return result


def _outputs(**values: str) -> None:
    if any("\n" in value or "\r" in value for value in values.values()):
        raise ReleasePreflightError("Invalid multiline workflow output")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.writelines(f"{key}={value}\n" for key, value in values.items())


def find_pr(github: _GitHub, ref: str, created: str) -> None:
    """Resolve the action's PR directly, with a paginated API fallback."""
    try:
        value = json.loads(created or "null")
    except json.JSONDecodeError as error:
        raise ReleasePreflightError(
            "Release Please returned invalid PR JSON"
        ) from error
    if value is None:
        pr = github.pending(ref, "open")
    else:
        number = _mapping(value).get("number")
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise ReleasePreflightError("Release Please returned an invalid PR number")
        raw = _mapping(
            github.json(
                "pr", "view", str(number), "--json", f"{_PR_FIELDS},baseRefName"
            )
        )
        if raw.get("baseRefName") != ref or raw.get("state") != "OPEN":
            raise ReleasePreflightError(
                "Release PR must be open and target the selected branch"
            )
        pr = _pull_request(raw)
    if pr is not None:
        _outputs(number=str(pr.number), branch=pr.branch)
        return
    pending = github.pending(ref, "merged")
    if pending is not None:
        sys.stdout.write(
            f"::notice::Release PR #{pending.number} is awaiting publication. "
            "Preparation resumes after publication; make release waits or retries.\n"
        )
    else:
        sys.stdout.write(
            "No release PR: no releasable Conventional Commits were found.\n"
        )


def refresh_pr(branch: str) -> None:
    """Refresh version-dependent files and push only an actual change."""
    if not branch.startswith("release-please--"):
        raise ReleasePreflightError("Expected a Release Please branch")
    _command("git", "check-ref-format", "--branch", branch)
    _command("uv", "lock")
    _command("uv", "sync", "--frozen", "--all-groups")
    _command(
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "yakbox.speech.analysis_worker_artifact",
        "--write-packaged",
    )
    _command(
        "git",
        "add",
        "--",
        "uv.lock",
        "src/yakbox/__init__.py",
        "src/yakbox/runtimes/analysis-worker.pyz",
    )
    changed = _command("git", "diff", "--cached", "--quiet", allow_failure=True)
    if changed.returncode == 0:
        return
    if changed.returncode != 1:
        raise ReleasePreflightError("Cannot inspect generated release files")
    _command("git", "config", "user.name", "github-actions[bot]")
    _command(
        "git",
        "config",
        "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com",
    )
    _command(
        "git",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "chore: refresh release artifacts",
    )
    _command("gh", "auth", "setup-git")
    _command("git", "push", "origin", f"HEAD:refs/heads/{branch}")


def resolve_release(root: Path, tag: str, requested_commit: str) -> None:
    """Bind the tag and every downstream job to the checked-out commit."""
    tag = tag or f"v{project_version(root)}"
    validate_release_tag(tag, project_version(root))
    commit = _command("git", "rev-parse", "HEAD").stdout.strip()
    if requested_commit and requested_commit != commit:
        raise ReleasePreflightError(
            "Release checkout does not match the requested commit"
        )
    _outputs(tag=tag, commit=commit)


def preflight(root: Path, tag: str, event: str) -> None:
    """Require an exact existing tag when retrying or responding to a tag push."""
    exists = _command(
        "git", "show-ref", "--verify", "--quiet", f"refs/tags/{tag}", allow_failure=True
    )
    if exists.returncode not in (0, 1):
        raise ReleasePreflightError("Cannot inspect the release tag")
    run_release_preflight(
        root, tag=tag, require_tag_ref=event == "push" or exists.returncode == 0
    )


def main(arguments: Sequence[str] | None = None) -> None:
    """Run one release workflow step using GitHub's environment values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step", choices=("find-pr", "refresh-pr", "start-ci", "resolve", "preflight")
    )
    options = parser.parse_args(arguments)
    try:
        _step(options.step)
    except ReleasePreflightError as error:
        parser.exit(1, f"release workflow failed: {error}\n")


def _step(step: str) -> None:
    if step == "refresh-pr":
        refresh_pr(os.environ["RELEASE_BRANCH"])
    elif step == "resolve":
        resolve_release(
            Path.cwd(),
            os.environ.get("REQUESTED_TAG", ""),
            os.environ.get("REQUESTED_COMMIT", ""),
        )
    elif step == "preflight":
        preflight(
            Path.cwd(), os.environ["RELEASE_TAG"], os.environ["GITHUB_EVENT_NAME"]
        )
    else:
        github = _GitHub(os.environ["GH_REPO"], "gh", _run)
        if step == "find-pr":
            find_pr(
                github, os.environ["TARGET_BRANCH"], os.environ.get("CREATED_PR", "")
            )
        else:
            github.dispatch(
                "ci.yml",
                os.environ["RELEASE_BRANCH"],
                release_preparation=os.environ["GITHUB_RUN_NUMBER"],
            )


if __name__ == "__main__":
    main()
