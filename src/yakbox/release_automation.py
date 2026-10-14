"""Drive Google's release PR through CI and the existing publication gates."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from yakbox.release_preflight import ReleasePreflightError

type Runner = Callable[[tuple[str, ...]], str]
_WAIT_SECONDS = 3_900
_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_PR_FIELDS = "number,headRefName,headRefOid,mergeCommit,isCrossRepository,state"


@dataclass(frozen=True, slots=True)
class _ReleasePR:
    number: int
    branch: str
    head: str
    merged: str | None


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ReleasePreflightError("GitHub returned an invalid release response")
    return value


def _pull_request(raw: object) -> _ReleasePR:
    value = _mapping(raw)
    number, branch, head = (
        value.get("number"),
        value.get("headRefName"),
        value.get("headRefOid"),
    )
    merge = value.get("mergeCommit")
    merged = _mapping(merge).get("oid") if merge is not None else None
    if (
        not isinstance(number, int)
        or number < 1
        or not isinstance(branch, str)
        or not branch.startswith("release-please--")
        or value.get("isCrossRepository") is not False
        or not isinstance(head, str)
        or _SHA.fullmatch(head) is None
        or (
            merged is not None
            and (not isinstance(merged, str) or _SHA.fullmatch(merged) is None)
        )
    ):
        raise ReleasePreflightError("Expected a same-repository Release Please PR")
    return _ReleasePR(number, branch, head, merged)


class _GitHub:
    def __init__(self, repository: str, executable: str, runner: Runner) -> None:
        self.repository = repository
        self.executable = executable
        self.runner = runner

    def command(self, *arguments: str) -> str:
        return self.runner((self.executable, *arguments, "--repo", self.repository))

    def json(self, *arguments: str) -> object:
        try:
            return json.loads(self.command(*arguments))
        except json.JSONDecodeError as error:
            raise ReleasePreflightError(
                "GitHub returned invalid release JSON"
            ) from error

    def pending(self, ref: str, state: str) -> _ReleasePR | None:
        query = (
            '[.[] | select(any(.labels[]; .name == "autorelease: pending"))'
            + (" | select(.merged_at != null)" if state == "merged" else "")
            + " | {number, headRefName: .head.ref, headRefOid: .head.sha,"
            " isCrossRepository: (.head.repo.full_name != .base.repo.full_name),"
            " mergeCommit: (if .merged_at then {oid: .merge_commit_sha} else null end),"
            ' state: (if .merged_at then "MERGED" else "OPEN" end)}] | @json'
        )
        raw = self.runner(
            (
                self.executable,
                "api",
                "--method",
                "GET",
                f"repos/{self.repository}/pulls",
                "-f",
                f"base={ref}",
                "-f",
                f"state={'closed' if state == 'merged' else 'open'}",
                "-f",
                "per_page=100",
                "--paginate",
                "--jq",
                query,
            )
        )
        try:
            values: list[object] = []
            for line in raw.splitlines():
                page = json.loads(line)
                if not isinstance(page, list):
                    raise ReleasePreflightError(
                        "GitHub returned invalid release PR pages"
                    )
                values.extend(page)
        except json.JSONDecodeError as error:
            raise ReleasePreflightError(
                "GitHub returned invalid release PR JSON"
            ) from error
        if len(values) > 1:
            raise ReleasePreflightError("Expected at most one pending release PR")
        return _pull_request(values[0]) if values else None

    def wait_run(self, workflow: str, title: str, *, head: str | None = None) -> int:
        deadline = time.monotonic() + _WAIT_SECONDS
        while time.monotonic() < deadline:
            runs = self.json(
                "run",
                "list",
                "--workflow",
                workflow,
                "--event",
                "workflow_dispatch",
                "--limit",
                "50",
                "--json",
                "databaseId,displayTitle,status,conclusion,headSha",
            )
            if not isinstance(runs, list):
                raise ReleasePreflightError("GitHub returned invalid workflow runs")
            for raw in runs:
                run = _mapping(raw)
                if run.get("displayTitle") != title:
                    continue
                if run.get("status") != "completed":
                    break  # Prefer the latest matching run over older retries.
                if head is not None and run.get("headSha") != head:
                    raise ReleasePreflightError("CI ran for a different release commit")
                identifier = run.get("databaseId")
                if not isinstance(identifier, int):
                    raise ReleasePreflightError("GitHub returned an invalid run ID")
                if run.get("conclusion") != "success":
                    raise ReleasePreflightError(
                        f"{workflow} failed; inspect https://github.com/"
                        f"{self.repository}/actions/runs/{identifier}"
                    )
                return identifier
            time.sleep(5)
        raise ReleasePreflightError(
            f"Timed out waiting for {workflow}; no checks bypassed"
        )

    def merge(self, pr: _ReleasePR) -> str:
        self.command(
            "pr", "merge", str(pr.number), "--squash", "--match-head-commit", pr.head
        )
        deadline = time.monotonic() + _WAIT_SECONDS
        while time.monotonic() < deadline:
            raw = self.json("pr", "view", str(pr.number), "--json", _PR_FIELDS)
            current = _pull_request(raw)
            if current.merged is not None:
                return current.merged
            if current.head != pr.head or _mapping(raw).get("state") == "CLOSED":
                raise ReleasePreflightError(
                    "Release PR changed or closed before merging"
                )
            time.sleep(5)
        raise ReleasePreflightError("Timed out waiting for the release merge queue")

    def manifest_version(self, commit: str) -> str:
        # gh api has no --repo option; keep the repository in its fixed endpoint.
        output = self.runner(
            (
                self.executable,
                "api",
                f"repos/{self.repository}/contents/.release-please-manifest.json?ref={commit}",
                "--jq",
                ".content | @base64d",
            )
        )
        try:
            version = _mapping(json.loads(output)).get(".")
        except json.JSONDecodeError as error:
            raise ReleasePreflightError("Invalid merged release manifest") from error
        if (
            not isinstance(version, str)
            or re.fullmatch(r"\d+\.\d+\.\d+", version) is None
        ):
            raise ReleasePreflightError("Invalid merged release version")
        return version

    def retry_publication(self, ref: str, pr: _ReleasePR) -> str:
        if pr.merged is None:
            raise ReleasePreflightError(
                "Publication retries require a merged release PR"
            )
        version = self.manifest_version(pr.merged)
        commit = self.runner(
            (
                self.executable,
                "api",
                f"repos/{self.repository}/commits/{ref}",
                "--jq",
                ".sha",
            )
        ).strip()
        if _SHA.fullmatch(commit) is None:
            raise ReleasePreflightError("Cannot resolve the publication retry commit")
        if self.manifest_version(commit) != version:
            raise ReleasePreflightError(
                "The selected ref no longer matches the pending release version"
            )
        request = uuid.uuid4().hex
        self.command(
            "workflow",
            "run",
            "release.yml",
            "--ref",
            ref,
            "-f",
            f"tag=v{version}",
            "-f",
            f"commit={commit}",
            "-f",
            "publish_to=github",
            "-f",
            f"request_id={request}",
            "-f",
            f"release_pr={pr.number}",
        )
        return f"Release {request}"


def automate_release(
    repository: str,
    ref: str = "main",
    *,
    executable: str = "gh",
    runner: Runner | None = None,
) -> int:
    """Prepare, test, merge, and publish one release under repository rules."""
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None:
        raise ReleasePreflightError("Release repository must be owner/name")
    if ref != "main":
        raise ReleasePreflightError(
            "Automatic releases target main; use an explicit RELEASE_TAG for other refs"
        )
    github = _GitHub(repository, executable, runner or _run)
    pending = github.pending(ref, "merged")
    if pending is not None and pending.merged is not None:
        title = github.retry_publication(ref, pending)
    else:
        request = uuid.uuid4().hex
        github.command(
            "workflow",
            "run",
            "release-please.yml",
            "--ref",
            ref,
            "-f",
            f"request_id={request}",
        )
        sys.stdout.write("Preparing the Release Please PR and generated files...\n")
        preparation = github.wait_run(
            "release-please.yml", f"Release preparation {request}"
        )
        pr = github.pending(ref, "open")
        if pr is None:
            raise ReleasePreflightError(
                "No release PR. Merge Conventional Commits such as "
                "fix:, feat:, or perf: first."
            )
        sys.stdout.write(f"Waiting for CI on release PR #{pr.number}...\n")
        github.wait_run("ci.yml", f"CI release-{preparation}", head=pr.head)
        merged = github.merge(pr)
        title = f"Release publish-{merged}"
    sys.stdout.write("Waiting for the checked GitHub publication...\n")
    return github.wait_run("release.yml", title)


def _run(command: tuple[str, ...]) -> str:
    try:
        result = subprocess.run(  # noqa: S603 - fixed gh verbs; arguments never use a shell
            command, capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ReleasePreflightError("Cannot run GitHub release automation") from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[:2048]
        raise ReleasePreflightError(f"GitHub release command failed: {detail}")
    return result.stdout


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the make release automation without changing the local checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ref", default="main")
    parser.add_argument("--gh", default="gh")
    options = parser.parse_args(arguments)
    try:
        identifier = automate_release(options.repo, options.ref, executable=options.gh)
    except ReleasePreflightError as error:
        parser.exit(1, f"release automation failed: {error}\n")
    sys.stdout.write(
        f"Release published: https://github.com/{options.repo}/actions/runs/{identifier}\n"
    )


if __name__ == "__main__":
    main()
