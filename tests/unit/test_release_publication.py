from __future__ import annotations

import io
import json
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from yakbox import release_publication as publication
from yakbox._files import sha256_file
from yakbox.release_automation import _GitHub
from yakbox.release_preflight import ReleasePreflightError

COMMIT = "a" * 40
TAG = "v0.3.2"


@pytest.fixture
def assets(tmp_path: Path) -> publication._Assets:
    dist = tmp_path / "dist"
    metadata = tmp_path / "release-metadata"
    dist.mkdir()
    metadata.mkdir()
    (dist / "yakbox-0.3.2-py3-none-any.whl").write_bytes(b"tested wheel")
    (dist / "yakbox-0.3.2.tar.gz").write_bytes(b"tested source")
    (metadata / "release-preflight.json").write_text(
        json.dumps({"status": "ok", "version": "0.3.2", "tag": TAG, "head_sha": COMMIT})
    )
    (metadata / "yakbox.cdx.json").write_text("{}")
    for family in ("whisper", "parakeet", "qwen"):
        (metadata / f"yakbox-worker-{family}.cdx.json").write_text("{}")
    files = (*dist.iterdir(), *metadata.iterdir())
    (metadata / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files)
    )
    return publication._assets(tmp_path, TAG, COMMIT)


@pytest.mark.parametrize(
    "change",
    [
        "distribution",
        "metadata",
        "missing",
        "extra",
        "symlink",
        "wrong-version",
        "wrong-commit",
        "invalid-manifest",
        "duplicate-manifest",
        "unsafe-path",
    ],
)
def test_publication_rejects_tampered_or_unbound_assets(
    assets: publication._Assets,
    tmp_path: Path,
    change: str,
) -> None:
    wheel = assets.distributions[0]
    manifest = tmp_path / "release-metadata/SHA256SUMS"
    if change in {"distribution", "metadata"}:
        path = (
            wheel
            if change == "distribution"
            else tmp_path / "release-metadata/yakbox.cdx.json"
        )
        path.write_bytes(b"tampered")
    elif change == "missing":
        wheel.unlink()
    elif change == "extra":
        (tmp_path / "release-metadata/unreviewed.txt").write_text("unreviewed")
    elif change == "symlink":
        original = wheel.read_bytes()
        wheel.unlink()
        (tmp_path / "external").write_bytes(original)
        wheel.symlink_to(tmp_path / "external")
    elif change == "invalid-manifest":
        manifest.write_text("invalid")
    elif change == "duplicate-manifest":
        manifest.write_text(manifest.read_text() * 2)
    elif change == "unsafe-path":
        manifest.write_text(f"{'a' * 64}  ../external\n")
    with pytest.raises((ReleasePreflightError, ValueError)):
        publication._assets(
            tmp_path,
            "v0.3.3" if change == "wrong-version" else TAG,
            "b" * 40 if change == "wrong-commit" else COMMIT,
        )


def _github(
    assets: publication._Assets,
    *,
    missing: bool = False,
    conflict: bool = False,
    wrong_commit: bool = False,
    unavailable: bool = False,
    version: str = "0.3.2",
) -> tuple[_GitHub, list[tuple[str, ...]]]:
    commands: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        commands.append(command)
        if command[1] != "api":
            return ""
        endpoint = command[2]
        if "POST" in command:
            return '{"workflow_run_id":123}'
        if "/releases/tags/" in endpoint:
            if missing or unavailable:
                raise ReleasePreflightError("HTTP 503" if unavailable else "HTTP 404")
            return json.dumps(
                {
                    "draft": False,
                    "assets": [
                        {
                            "name": path.name,
                            "digest": "sha256:wrong"
                            if conflict
                            else f"sha256:{sha256_file(path)}",
                        }
                        for path in assets.distributions
                    ],
                }
            )
        if "/commits/" in endpoint:
            return "b" * 40 if wrong_commit else COMMIT
        if "/contents/" in endpoint:
            return json.dumps({".": version})
        if "--paginate" in command:
            return json.dumps(
                [
                    {
                        "number": 24,
                        "headRefName": "release-please--branches--main",
                        "headRefOid": COMMIT,
                        "mergeCommit": {"oid": COMMIT},
                        "isCrossRepository": False,
                    }
                ]
            )
        raise AssertionError(command)

    return _GitHub("example/yakbox", "gh", runner), commands


def test_github_uploads_every_tested_asset_to_the_exact_commit(
    assets: publication._Assets,
) -> None:
    github, commands = _github(assets, missing=True)
    publication.publish_github(github, assets)
    create = commands[-1]
    assert create[1:4] == ("release", "create", TAG)
    assert create[create.index("--target") + 1] == COMMIT
    assert all(str(path) in create for path in assets.files)


def test_retry_accepts_identical_existing_github_distributions(
    assets: publication._Assets,
) -> None:
    github, commands = _github(assets)
    publication.publish_github(github, assets)
    assert not any(command[1:3] == ("release", "create") for command in commands)


@pytest.mark.parametrize("failure", ["conflict", "wrong_commit", "unavailable"])
def test_retry_rejects_conflicting_or_unavailable_github_release(
    assets: publication._Assets,
    failure: str,
) -> None:
    github, commands = _github(
        assets,
        missing=failure == "missing",
        conflict=failure == "conflict",
        wrong_commit=failure == "wrong_commit",
        unavailable=failure == "unavailable",
    )
    with pytest.raises(ReleasePreflightError):
        publication.publish_github(github, assets)
    assert not any(command[1:3] == ("release", "create") for command in commands)


@pytest.mark.parametrize("published", [False, True])
def test_pypi_retry_accepts_only_identical_files(
    assets: publication._Assets,
    monkeypatch: pytest.MonkeyPatch,
    published: bool,
) -> None:
    monkeypatch.setattr(
        publication,
        "_pypi_files",
        lambda _: {path.name: sha256_file(path) for path in assets.distributions},
    )
    publication.verify_pypi(assets, published=published)


@pytest.mark.parametrize(
    "existing",
    [{}, {"yakbox-0.3.2.tar.gz": "conflicting"}, {"unexpected.whl": "other"}],
)
def test_pypi_precheck_rejects_conflicts_but_allows_a_new_version(
    assets: publication._Assets,
    monkeypatch: pytest.MonkeyPatch,
    existing: dict[str, str],
) -> None:
    monkeypatch.setattr(publication, "_pypi_files", lambda _: existing)
    if existing:
        with pytest.raises(ReleasePreflightError, match="differs"):
            publication.verify_pypi(assets)
    else:
        publication.verify_pypi(assets)


def test_pypi_retry_allows_a_partial_upload_then_waits_for_both_files(
    assets: publication._Assets,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = {path.name: sha256_file(path) for path in assets.distributions}
    partial = dict([next(iter(complete.items()))])
    responses = iter([partial, partial, complete])
    monkeypatch.setattr(publication, "_pypi_files", lambda _: next(responses))
    monkeypatch.setattr(publication.time, "sleep", lambda _: None)
    publication.verify_pypi(assets)
    publication.verify_pypi(assets, published=True)


def test_incomplete_pypi_publication_cannot_mark_release_complete(
    assets: publication._Assets,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    github, commands = _github(assets)
    times = iter([0, 121])
    monkeypatch.setattr(publication.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(publication, "_pypi_files", lambda _: {})
    with pytest.raises(ReleasePreflightError, match="every tested distribution"):
        publication.finish(github, assets, "")
    assert commands == []


def test_finish_marks_tagged_and_prepares_next_pr_after_both_publications(
    assets: publication._Assets,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        publication,
        "_pypi_files",
        lambda _: {path.name: sha256_file(path) for path in assets.distributions},
    )
    github, commands = _github(assets)
    publication.finish(github, assets, "")
    label = next(
        i for i, command in enumerate(commands) if command[1:3] == ("pr", "edit")
    )
    assert "autorelease: pending" in commands[label]
    assert "autorelease: tagged" in commands[label]
    assert commands[-1][1:4] == ("api", "--method", "POST")
    assert "workflows/release-please.yml/dispatches" in commands[-1][4]


@pytest.mark.parametrize("failure", ["missing", "conflict", "unavailable"])
def test_failed_github_publication_cannot_mark_release_complete(
    assets: publication._Assets,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    monkeypatch.setattr(
        publication,
        "_pypi_files",
        lambda _: {path.name: sha256_file(path) for path in assets.distributions},
    )
    github, commands = _github(
        assets,
        missing=failure == "missing",
        conflict=failure == "conflict",
        wrong_commit=failure == "wrong_commit",
        unavailable=failure == "unavailable",
    )
    with pytest.raises(ReleasePreflightError):
        publication.finish(github, assets, "")
    assert not any(command[1:3] == ("pr", "edit") for command in commands)


def test_finish_cannot_label_another_release_version(
    assets: publication._Assets,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        publication,
        "_pypi_files",
        lambda _: {path.name: sha256_file(path) for path in assets.distributions},
    )
    github, commands = _github(assets, version="0.4.0")
    with pytest.raises(ReleasePreflightError, match="does not match"):
        publication.finish(github, assets, "")
    assert not any(command[1:3] == ("pr", "edit") for command in commands)


def test_pypi_metadata_is_read_from_the_exact_version_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def response(url: str, *, timeout: int) -> io.BytesIO:
        assert url == "https://pypi.org/pypi/yakbox/0.3.2/json"
        assert timeout == 15
        return io.BytesIO(
            json.dumps(
                {
                    "info": {"version": "0.3.2"},
                    "urls": [{"filename": "wheel", "digests": {"sha256": "digest"}}],
                }
            ).encode()
        )

    monkeypatch.setattr(publication, "urlopen", response)
    assert publication._pypi_files("0.3.2") == {"wheel": "digest"}


@pytest.mark.parametrize("status", [404, 403, 503])
def test_pypi_metadata_lookup_distinguishes_a_new_version_from_errors(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    def error(*_args: object, **_kwargs: object) -> None:
        raise HTTPError("https://pypi.org", status, "failure", Message(), None)

    monkeypatch.setattr(publication, "urlopen", error)
    if status == 404:
        assert publication._pypi_files("0.3.2") == {}
    else:
        with pytest.raises(ReleasePreflightError):
            publication._pypi_files("0.3.2")


def test_pypi_network_errors_are_expected_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def error(*_args: object, **_kwargs: object) -> None:
        raise URLError("offline")

    monkeypatch.setattr(publication, "urlopen", error)
    with pytest.raises(ReleasePreflightError):
        publication._pypi_files("0.3.2")


def test_publication_cli_verifies_assets_without_a_github_token(
    assets: publication._Assets,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RELEASE_TAG", assets.tag)
    monkeypatch.setenv("RELEASE_COMMIT", assets.commit)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    publication.main(["verify-assets"])


def test_publication_cli_reports_conflicts_with_failure_exit_code(
    assets: publication._Assets,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RELEASE_TAG", assets.tag)
    monkeypatch.setenv("RELEASE_COMMIT", assets.commit)
    assets.distributions[0].write_bytes(b"modified")
    with pytest.raises(SystemExit) as error:
        publication.main(["verify-assets"])
    assert error.value.code == 1
    assert "checksums do not match" in capsys.readouterr().err
