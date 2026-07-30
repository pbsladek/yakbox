from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

import yakbox.audiobook.build as build_module
from yakbox.audiobook import build_audiobook, check_release, load_manifest
from yakbox.errors import ArtifactError, BuildError
from yakbox.speech import (
    FakeSpeechService,
    SpeechArtifact,
    SpeechSynthesisRequest,
    TextToSpeechService,
)


class _TerminatedSynthesisService:
    capabilities = FakeSpeechService.capabilities

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        del request, destination, overwrite
        raise ProcessLookupError("simulated synthesis worker termination")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "patched_name"),
    [
        ("synthesize", None),
        ("master", "master_wav"),
        ("encode_mp3", "encode_mp3"),
        ("inspect", "_write_inspection_report"),
    ],
)
async def test_failure_at_every_build_stage_is_journaled_and_resumable(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    patched_name: str | None,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    service: TextToSpeechService = (
        _TerminatedSynthesisService() if stage == "synthesize" else FakeSpeechService()
    )

    @asynccontextmanager
    async def backend(
        _name: str, **_options: object
    ) -> AsyncIterator[TextToSpeechService]:
        yield service

    monkeypatch.setattr(build_module, "open_speech_backend", backend)
    original: Callable[..., object] | None = None
    if patched_name is not None:
        original = getattr(build_module, patched_name)

        def terminated(*_args: object, **_kwargs: object) -> None:
            raise ProcessLookupError(f"simulated {stage} process termination")

        monkeypatch.setattr(build_module, patched_name, terminated)

    with pytest.raises(BuildError, match="Build failed"):
        await build_audiobook(manifest)

    run_directory = next((book_workspace / ".yakbox" / "runs").iterdir())
    events = [
        json.loads(line)
        for line in (run_directory / "journal.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    failed = [event for event in events if event["event"] == "node_failed"]
    assert len(failed) == 1
    assert str(failed[0]["node_id"]).endswith(f":{stage}")
    summary = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert not (book_workspace / ".yakbox" / "locks" / "default.lock").exists()
    assert not tuple(book_workspace.rglob("*.part"))
    assert not tuple(book_workspace.rglob(".*.chunk-*.wav"))

    @asynccontextmanager
    async def healthy_backend(
        _name: str, **_options: object
    ) -> AsyncIterator[TextToSpeechService]:
        yield FakeSpeechService()

    monkeypatch.setattr(build_module, "open_speech_backend", healthy_backend)
    if patched_name is not None:
        assert original is not None
        monkeypatch.setattr(build_module, patched_name, original)

    resumed = await build_audiobook(manifest)
    assert resumed.status == "complete"
    assert resumed.resumed
    assert resumed.run_directory == run_directory
    assert check_release(manifest).complete


@pytest.mark.asyncio
async def test_corrupted_master_is_rejected_then_rebuilt(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    await build_audiobook(manifest)
    master = next(manifest.target("default").output_root.glob("mastered/*.wav"))
    master.write_bytes(b"corrupted")

    broken = check_release(manifest)
    assert not broken.complete
    assert any("digest mismatch" in issue for issue in broken.issues)

    rebuilt = await build_audiobook(manifest)
    assert rebuilt.status == "complete"
    assert check_release(manifest).complete


def test_release_snapshot_disk_exhaustion_fails_before_copy(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    target = manifest.target("default")
    master = target.output_root / "mastered" / "chapter.wav"
    delivery = target.output_root / "release" / "mp3" / "chapter.mp3"
    master.parent.mkdir(parents=True)
    delivery.parent.mkdir(parents=True)
    master.write_bytes(b"master")
    delivery.write_bytes(b"delivery")

    class _Usage:
        free = 0

    monkeypatch.setattr(build_module.shutil, "disk_usage", lambda _path: _Usage())
    with pytest.raises(ArtifactError, match="only 0 are free"):
        build_module._validate_release_storage(target, (master, delivery))
    release_root = target.output_root / "release"
    assert {path.name for path in release_root.iterdir() if path.is_dir()} == {"mp3"}
