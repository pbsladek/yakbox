"""Verify release assets and finish publication to GitHub and PyPI safely."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from yakbox._files import sha256_file
from yakbox.release_automation import _PR_FIELDS, _GitHub, _mapping, _pull_request, _run
from yakbox.release_preflight import ReleasePreflightError


@dataclass(frozen=True)
class _Assets:
    tag: str
    commit: str
    distributions: tuple[Path, ...]
    files: tuple[Path, ...]


def _assets(root: Path, tag: str, commit: str) -> _Assets:
    if (
        re.fullmatch(r"v\d+\.\d+\.\d+", tag) is None
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None
    ):
        raise ReleasePreflightError("Invalid release tag or source commit")
    version = tag[1:]
    distributions = tuple(sorted((root / "dist").iterdir()))
    if {path.name for path in distributions} != {
        f"yakbox-{version}-py3-none-any.whl",
        f"yakbox-{version}.tar.gz",
    }:
        raise ReleasePreflightError("Release distributions do not match the version")
    metadata = root / "release-metadata"
    expected_metadata = {
        "SHA256SUMS",
        "release-preflight.json",
        "yakbox.cdx.json",
        *(
            f"yakbox-worker-{family}.cdx.json"
            for family in ("whisper", "parakeet", "qwen")
        ),
    }
    if {path.name for path in metadata.iterdir()} != expected_metadata:
        raise ReleasePreflightError("Release metadata is incomplete or unexpected")
    files = (*distributions, *sorted(metadata.iterdir()))
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ReleasePreflightError("Release assets must be regular files")
    checksums: dict[str, str] = {}
    for line in (metadata / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if (
            name != Path(name).name
            or name in checksums
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ReleasePreflightError("Invalid release checksum manifest")
        checksums[name] = digest
    actual = {
        path.name: sha256_file(path) for path in files if path.name != "SHA256SUMS"
    }
    if actual != checksums:
        raise ReleasePreflightError("Release asset checksums do not match")
    report = _mapping(json.loads((metadata / "release-preflight.json").read_text()))
    if (
        report.get("status"),
        report.get("version"),
        report.get("tag"),
        report.get("head_sha"),
    ) != (
        "ok",
        version,
        tag,
        commit,
    ):
        raise ReleasePreflightError("Preflight metadata does not match the release")
    return _Assets(tag, commit, distributions, files)


def _existing_release(github: _GitHub, tag: str) -> dict[str, object] | None:
    try:
        raw = github.runner(
            (github.executable, "api", f"repos/{github.repository}/releases/tags/{tag}")
        )
    except ReleasePreflightError as error:
        if "HTTP 404" in str(error):
            return None
        raise
    return _mapping(json.loads(raw))


def publish_github(github: _GitHub, assets: _Assets) -> None:
    """Create the release, or verify existing immutable distribution files."""
    existing = _existing_release(github, assets.tag)
    if existing is None:
        github.command(
            "release",
            "create",
            assets.tag,
            *(str(path) for path in assets.files),
            "--target",
            assets.commit,
            "--generate-notes",
            "--title",
            f"yakbox {assets.tag}",
        )
        return
    _verify_github(github, assets, existing)


def _verify_github(
    github: _GitHub, assets: _Assets, existing: dict[str, object]
) -> None:
    tagged = github.runner(
        (
            github.executable,
            "api",
            f"repos/{github.repository}/commits/{assets.tag}",
            "--jq",
            ".sha",
        )
    ).strip()
    if tagged != assets.commit or existing.get("draft") is not False:
        raise ReleasePreflightError(
            "Existing GitHub release does not match the tested commit"
        )
    files = existing.get("assets")
    if not isinstance(files, list):
        raise ReleasePreflightError("Invalid existing GitHub release assets")
    digests = {
        _mapping(item).get("name"): _mapping(item).get("digest") for item in files
    }
    for path in assets.distributions:
        if digests.get(path.name) != f"sha256:{sha256_file(path)}":
            raise ReleasePreflightError(
                "Existing GitHub distribution differs from the tested file"
            )


def _pypi_files(version: str) -> dict[str, str]:
    try:
        with urlopen(
            f"https://pypi.org/pypi/yakbox/{version}/json", timeout=15
        ) as response:
            raw = _mapping(json.load(response))
    except HTTPError as error:
        if error.code == HTTPStatus.NOT_FOUND:
            return {}
        raise ReleasePreflightError("Cannot read PyPI release metadata") from error
    except (OSError, URLError, json.JSONDecodeError) as error:
        raise ReleasePreflightError("Cannot read PyPI release metadata") from error
    if _mapping(raw.get("info")).get("version") != version:
        raise ReleasePreflightError("PyPI returned a different release version")
    urls = raw.get("urls")
    if not isinstance(urls, list):
        raise ReleasePreflightError("PyPI returned invalid distribution metadata")
    result: dict[str, str] = {}
    for item in urls:
        value = _mapping(item)
        filename = value.get("filename")
        digest = _mapping(value.get("digests")).get("sha256")
        if not isinstance(filename, str) or not isinstance(digest, str):
            raise ReleasePreflightError("PyPI returned invalid file digests")
        result[filename] = digest
    return result


def verify_pypi(assets: _Assets, *, published: bool = False) -> None:
    """Permit retries only for identical files; wait for both published files."""
    expected = {path.name: sha256_file(path) for path in assets.distributions}
    deadline = time.monotonic() + 120
    while True:
        existing = _pypi_files(assets.tag[1:])
        for name, digest in existing.items():
            if name not in expected or expected[name] != digest:
                raise ReleasePreflightError(
                    "Existing PyPI distribution differs from the tested file"
                )
        if not published or existing == expected:
            return
        if time.monotonic() >= deadline:
            raise ReleasePreflightError(
                "PyPI has not published every tested distribution"
            )
        time.sleep(5)


def finish(github: _GitHub, assets: _Assets, release_pr: str) -> None:
    """Mark the release complete only after both destinations are verified."""
    verify_pypi(assets, published=True)
    existing = _existing_release(github, assets.tag)
    if existing is None:
        raise ReleasePreflightError("GitHub publication is missing")
    _verify_github(github, assets, existing)
    if release_pr:
        if re.fullmatch(r"[1-9][0-9]*", release_pr) is None:
            raise ReleasePreflightError("Invalid release PR number")
        pr = _pull_request(github.json("pr", "view", release_pr, "--json", _PR_FIELDS))
    else:
        pr = github.pending("main", "merged")
    if pr is not None:
        if pr.merged is None or github.manifest_version(pr.merged) != assets.tag[1:]:
            raise ReleasePreflightError(
                "Pending release PR does not match the published version"
            )
        github.command(
            "label", "create", "autorelease: tagged", "--color", "0e8a16", "--force"
        )
        github.command(
            "pr",
            "edit",
            str(pr.number),
            "--remove-label",
            "autorelease: pending",
            "--add-label",
            "autorelease: tagged",
        )
    github.dispatch("release-please.yml", "main")


def main(arguments: Sequence[str] | None = None) -> None:
    """Execute one publication step with the same checks at each boundary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step", choices=("verify-assets", "publish-github", "check-pypi", "finish")
    )
    options = parser.parse_args(arguments)
    try:
        assets = _assets(
            Path.cwd(), os.environ["RELEASE_TAG"], os.environ["RELEASE_COMMIT"]
        )
        if options.step == "check-pypi":
            verify_pypi(assets)
        elif options.step != "verify-assets":
            github = _GitHub(os.environ["GH_REPO"], "gh", _run)
            if options.step == "publish-github":
                publish_github(github, assets)
            else:
                finish(github, assets, os.environ.get("RELEASE_PR", ""))
    except (ReleasePreflightError, OSError, ValueError) as error:
        parser.exit(1, f"release publication failed: {error}\n")


if __name__ == "__main__":
    main()
