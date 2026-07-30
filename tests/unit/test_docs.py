from __future__ import annotations

import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from yakbox.audiobook.manifest import load_manifest
from yakbox.cli import main


@pytest.mark.parametrize(
    "arguments",
    [
        ("--help",),
        ("init", "--help"),
        ("plan", "--help"),
        ("audition", "--help"),
        ("build", "--help"),
        ("release", "check", "--help"),
        ("artifacts", "clean", "--help"),
        ("artifacts", "trash", "restore", "--help"),
        ("cloud", "batch", "--help"),
        ("cloud", "voices", "recordings", "create", "--help"),
        ("doctor", "--help"),
    ],
)
def test_documented_command_surfaces_have_configuration_free_help(
    arguments: tuple[str, ...],
) -> None:
    result = CliRunner().invoke(main, list(arguments))
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_documentation_index_links_resolve() -> None:
    root = Path(__file__).parents[2]
    index = root / "docs" / "README.md"
    links = re.findall(r"\[[^]]+]\(([^)]+\.md)\)", index.read_text(encoding="utf-8"))

    assert links
    assert all((index.parent / link).is_file() for link in links)


def test_readme_local_links_resolve() -> None:
    root = Path(__file__).parents[2]
    readme = root / "README.md"
    links = re.findall(
        r"\[[^]]+]\((?!https?://)([^)]+)\)", readme.read_text(encoding="utf-8")
    )

    assert links
    assert all((root / link).exists() for link in links)


def test_example_index_links_resolve() -> None:
    root = Path(__file__).parents[2]
    index = root / "examples" / "README.md"
    links = re.findall(r"\[[^]]+]\(([^)]+\.md)\)", index.read_text(encoding="utf-8"))

    assert len(links) == 7
    assert all((index.parent / link).is_file() for link in links)


def test_every_packaged_example_validates_and_plans_every_target() -> None:
    root = Path(__file__).parents[2]
    manifests = tuple(sorted((root / "examples").glob("*/yakbox.toml")))
    runner = CliRunner()

    assert len(manifests) == 7
    for manifest_path in manifests:
        validated = runner.invoke(main, ["validate", str(manifest_path)])
        assert validated.exit_code == 0, f"{manifest_path}: {validated.output}"
        loaded = load_manifest(manifest_path)
        for target in loaded.targets:
            planned = runner.invoke(
                main,
                ["plan", str(manifest_path), "--target", target.name],
            )
            assert planned.exit_code == 0, (
                f"{manifest_path} target {target.name}: {planned.output}"
            )
            assert "nodes; plan" in planned.output


def test_actions_are_commit_pinned_and_release_attests_artifacts() -> None:
    root = Path(__file__).parents[2]
    workflows = tuple((root / ".github" / "workflows").glob("*.yml"))
    assert workflows
    for workflow in workflows:
        content = workflow.read_text(encoding="utf-8")
        actions = re.findall(r"uses:\s+[^@\s]+@([^\s#]+)", content)
        assert actions, workflow
        assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in actions), (
            workflow
        )

    release = (root / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert release.count("actions/attest@") == 2
    assert "yakbox-release-preflight" in release
    assert "sbom-path: release-metadata/yakbox.cdx.json" in release
    assert "attestations: write" in release
    assert "id-token: write" in release


def test_ci_installs_ffmpeg_on_every_supported_runner_and_allows_only_local_ipc() -> (
    None
):
    root = Path(__file__).parents[2]
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    project = (root / "pyproject.toml").read_text(encoding="utf-8")

    for runner in ("Linux", "macOS", "Windows"):
        assert f"runner.os == '{runner}'" in workflow
    assert "--disable-socket" not in project
    assert "localhost,127.0.0.0/8,::1/128" in project
    assert "--allow-unix-socket" in project


def test_live_workflow_and_canaries_are_explicitly_bounded() -> None:
    root = Path(__file__).parents[2]
    workflow = (root / ".github" / "workflows" / "live-canary.yml").read_text(
        encoding="utf-8"
    )
    resemble = (root / "tests" / "live" / "test_resemble_live.py").read_text(
        encoding="utf-8"
    )
    local = (root / "tests" / "live" / "test_local_chatterbox_live.py").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow
    assert "schedule:" not in workflow
    assert 'text = "Hi."' in resemble
    assert "max_provider_requests=1" in resemble
    assert "max_connections=1" in resemble
    assert "max_attempts=1" in resemble
    assert 'text="Hi."' in local
    assert "threads_per_process=1" in local
    assert "self-hosted" in workflow
