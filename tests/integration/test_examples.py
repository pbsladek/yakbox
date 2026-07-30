from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from yakbox.audiobook import (
    assemble_release,
    build_audiobook,
    check_release,
    load_manifest,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "targets", "assemble"),
    [
        ("tiny-book", ("default",), False),
        (
            "multiple-voices",
            ("narrator-edition", "character-edition"),
            False,
        ),
        ("pronunciation-heavy", ("default",), False),
        ("selective-rebuild", ("default",), False),
        ("m4b-release", ("default",), True),
    ],
)
async def test_offline_examples_build_and_release_from_a_copy(
    tmp_path: Path,
    name: str,
    targets: tuple[str, ...],
    assemble: bool,
) -> None:
    repository = Path(__file__).parents[2]
    workspace = tmp_path / name
    shutil.copytree(repository / "examples" / name, workspace)
    manifest = load_manifest(workspace / "yakbox.toml")

    for target in targets:
        result = await build_audiobook(manifest, target_name=target)
        assert result.status == "complete"
        release = check_release(manifest, target_name=target, write_manifest=True)
        assert release.complete
        assert release.release_manifest is not None
    if assemble:
        output = assemble_release(manifest)
        assert output.is_file()
        assert output.suffix == ".m4b"


@pytest.mark.asyncio
async def test_selective_rebuild_example_changes_only_one_chapter(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[2]
    workspace = tmp_path / "selective-rebuild"
    shutil.copytree(repository / "examples" / "selective-rebuild", workspace)
    manifest = load_manifest(workspace / "yakbox.toml")
    first = await build_audiobook(manifest)
    assert len(first.artifacts) == 12

    source = workspace / "source" / "book.md"
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "The conductor checked the platform.",
            "The conductor carefully checked the empty platform.",
        ),
        encoding="utf-8",
    )
    second = await build_audiobook(load_manifest(manifest.path))

    assert len(second.reused_nodes) == 8
    assert len(second.artifacts) == 12
    assert {node.split(":")[0] for node in second.reused_nodes} == {
        "0001-first-stop",
        "0003-third-stop",
    }
