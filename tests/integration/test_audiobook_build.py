from __future__ import annotations

import asyncio
import json
import tomllib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from tests.schema_helpers import validate_contract

import yakbox.audiobook.build as build_module
from yakbox.audiobook import (
    apply_cache_cleanup,
    apply_cleanup,
    assemble_release,
    audition_audiobook,
    build_audiobook,
    check_release,
    diff_releases,
    inventory_artifacts,
    inventory_synthesis_cache,
    load_manifest,
    plan_cache_cleanup,
    plan_cleanup,
    preflight_audiobook_build,
    preview_audiobook,
    restore_trash,
    select_build_chapters,
)
from yakbox.audiobook.artifacts import ArtifactKind
from yakbox.audiobook.planner import plan_audiobook
from yakbox.audiobook.shards import (
    export_shard_manifests,
    verify_shard_manifests,
)
from yakbox.audiobook.sources import normalize_sources
from yakbox.cloud.usage import HostedUsageGate
from yakbox.errors import BuildError, ValidationError
from yakbox.speech import (
    BackendCapabilities,
    FakeSpeechService,
    HostedUsageBudget,
    HostedUsageRecorder,
    HostedUsageSnapshot,
    SpeechArtifact,
    SpeechSynthesisRequest,
    TextToSpeechService,
)


@pytest.mark.asyncio
async def test_fake_build_release_resume_and_cleanup(book_workspace: Path) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    first = await build_audiobook(manifest)
    second = await build_audiobook(manifest)
    auditions = await audition_audiobook(manifest, profiles=("default",), text="Hi.")

    assert first.status == "complete"
    assert first.preflight.pending_nodes == 4
    assert first.preflight.storage_sufficient
    assert len(second.reused_nodes) == 4
    assert second.preflight.pending_nodes == 0
    assert second.preflight.change_summary.previous_run_id == first.run_id
    assert len(second.preflight.change_summary.unchanged_nodes) == 4
    release = check_release(manifest, write_manifest=True)
    assert release.complete
    assert len(release.master_wavs) == 1
    assert len(release.delivery_mp3s) == 1
    assert release.release_manifest is not None
    m4b = assemble_release(manifest)
    assert m4b.is_file()
    validate_contract("audiobook-release-check", release.to_dict())
    validate_contract(
        "audiobook-manifest",
        tomllib.loads(manifest.path.read_text(encoding="utf-8")),
    )
    validate_contract(
        "audiobook-plan",
        json.loads((first.run_directory / "plan.json").read_text(encoding="utf-8")),
    )
    assert "A very short opening" not in (first.run_directory / "plan.json").read_text(
        encoding="utf-8"
    )
    validate_contract(
        "audiobook-run",
        json.loads((first.run_directory / "run.json").read_text(encoding="utf-8")),
    )
    for line in (
        (first.run_directory / "journal.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        validate_contract("audiobook-journal", json.loads(line))
    validate_contract(
        "audiobook-release",
        json.loads(release.release_manifest.read_text(encoding="utf-8")),
    )
    audition_report = auditions[0].path.parent / "audition.json"
    validate_contract(
        "audiobook-audition",
        json.loads(audition_report.read_text(encoding="utf-8")),
    )

    inventory = inventory_artifacts(manifest.target("default").output_root)
    assert {item.kind for item in inventory.records} >= {
        ArtifactKind.RAW,
        ArtifactKind.MASTER,
        ArtifactKind.DELIVERY,
        ArtifactKind.RELEASE,
    }
    validate_contract("audiobook-inventory", inventory.to_dict(workspace=manifest.root))
    for metadata in manifest.target("default").output_root.rglob("*.artifact.json"):
        validate_contract(
            "audiobook-artifact",
            json.loads(metadata.read_text(encoding="utf-8")),
        )
    for report_path in manifest.target("default").output_root.glob(
        "reports/*.inspection.json"
    ):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validate_contract("audiobook-chapter-inspection", report)
        for inspection in report["inspections"]:
            validate_contract("audio-inspection", inspection)

    cleanup = plan_cleanup(
        manifest.root,
        manifest.target("default").output_root,
        kind=ArtifactKind.RAW,
    )
    assert len(cleanup.candidates) == 1
    validate_contract("audiobook-cleanup-plan", cleanup.to_dict())
    raw = cleanup.candidates[0]
    trash = apply_cleanup(cleanup)
    assert trash.is_dir()
    validate_contract(
        "audiobook-quarantine",
        json.loads((trash / "cleanup.json").read_text(encoding="utf-8")),
    )
    assert not raw.path.exists()
    assert restore_trash(manifest.root, cleanup.cleanup_id) == 1
    validate_contract(
        "audiobook-cleanup-report",
        json.loads((trash / "restored.json").read_text(encoding="utf-8")),
    )
    assert raw.path.is_file()

    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
    )
    plan = plan_audiobook(manifest, document)
    shards = export_shard_manifests(
        plan,
        manifest.root / ".yakbox" / "shards",
        count=1,
        root=manifest.root,
    )
    assert len(verify_shard_manifests(shards, root=manifest.root)) == 1
    for shard in shards:
        validate_contract(
            "audiobook-shard",
            json.loads(shard.read_text(encoding="utf-8")),
        )


@pytest.mark.asyncio
async def test_build_reports_monotonic_application_progress(
    book_workspace: Path,
) -> None:
    events: list[build_module.BuildProgress] = []

    result = await build_audiobook(
        load_manifest(book_workspace / "yakbox.toml"),
        through_stage="synthesize",
        progress=events.append,
    )

    assert result.status == "complete"
    assert [event.event for event in events] == ["started", "completed"]
    assert [event.completed for event in events] == [0, 1]
    assert all(event.total == 1 for event in events)


@pytest.mark.asyncio
async def test_synthesis_cache_has_safe_inventory_and_explicit_cleanup(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    result = await build_audiobook(manifest, through_stage="synthesize")
    inventory = inventory_synthesis_cache(manifest.root)

    assert result.artifacts[0].path.is_file()
    assert len(inventory.entries) == 2
    assert inventory.invalid_entries == 0
    cleanup = plan_cache_cleanup(manifest.root, max_bytes=0)
    assert len(cleanup.candidates) == 2
    assert apply_cache_cleanup(cleanup) == 2
    assert inventory_synthesis_cache(manifest.root).entries == ()
    assert result.artifacts[0].path.is_file()


@pytest.mark.asyncio
async def test_release_snapshots_survive_changed_source_rebuild(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    await build_audiobook(manifest)
    release = check_release(manifest, write_manifest=True)
    assert release.release_manifest is not None
    snapshot_master = release.master_wavs[0]
    snapshot_mp3 = release.delivery_mp3s[0]
    snapshot_master_bytes = snapshot_master.read_bytes()
    snapshot_mp3_bytes = snapshot_mp3.read_bytes()
    assert snapshot_master.parent.name == "wav"
    assert snapshot_mp3.parent.name == "mp3"
    assert snapshot_master.is_relative_to(release.release_manifest.parent)
    assert snapshot_mp3.is_relative_to(release.release_manifest.parent)

    source = book_workspace / "source" / "book.md"
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "A very short opening.",
            "A distinctly changed opening.",
        ),
        encoding="utf-8",
    )
    changed_manifest = load_manifest(manifest.path)
    assert select_build_chapters(
        changed_manifest,
        selection="changed",
        since_release=release.release_manifest,
    ) == ("0001-chapter-one",)
    rebuilt = await build_audiobook(changed_manifest)

    assert rebuilt.status == "complete"
    assert snapshot_master.read_bytes() == snapshot_master_bytes
    assert snapshot_mp3.read_bytes() == snapshot_mp3_bytes
    release_document = json.loads(release.release_manifest.read_text(encoding="utf-8"))
    assert release_document["master_wavs"][0]["path"].startswith(
        f"release/{release.release_manifest.parent.name}/wav/"
    )
    next_release = check_release(changed_manifest, write_manifest=True)
    assert next_release.release_manifest is not None
    difference = diff_releases(
        release.release_manifest,
        next_release.release_manifest,
    )
    assert difference.changed_artifacts == (
        "delivery_mp3s/0001-chapter-one.mp3",
        "master_wavs/0001-chapter-one.wav",
    )
    assert "document_sha256" in difference.metadata_changes


@pytest.mark.asyncio
async def test_cleanup_can_select_removed_chapter_master_but_not_current_master(
    book_workspace: Path,
) -> None:
    source = book_workspace / "source" / "book.md"
    source.write_text(
        "# Chapter One\n\nFirst chapter.\n\n# Chapter Two\n\nSecond chapter.\n",
        encoding="utf-8",
    )
    manifest = load_manifest(book_workspace / "yakbox.toml")
    await build_audiobook(manifest)

    source.write_text("# Chapter One\n\nFirst chapter.\n", encoding="utf-8")
    current_manifest = load_manifest(manifest.path)
    current_document = normalize_sources(
        current_manifest.sources,
        pronunciations=current_manifest.pronunciations,
    )
    current_plan = plan_audiobook(current_manifest, current_document)
    await build_audiobook(current_manifest)
    current_masters = tuple(
        node.output for node in current_plan.nodes if node.stage.value == "master"
    )
    cleanup = plan_cleanup(
        current_manifest.root,
        current_manifest.target("default").output_root,
        kind=ArtifactKind.MASTER,
        raw_until_release=False,
        current_paths=current_masters,
    )

    assert len(cleanup.candidates) == 1
    assert cleanup.candidates[0].path.stem.startswith("0002-")


@pytest.mark.asyncio
async def test_preview_is_isolated_from_production(book_workspace: Path) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    build = await build_audiobook(manifest)
    preview = await preview_audiobook(manifest, text="A tiny preview.")

    assert preview.kind is ArtifactKind.PREVIEW
    assert "previews" in preview.path.parts
    assert preview.path not in {record.path for record in build.artifacts}
    inventory = inventory_artifacts(manifest.target("default").output_root)
    assert preview in inventory.records
    validate_contract(
        "audiobook-artifact",
        json.loads(
            preview.path.with_suffix(f"{preview.path.suffix}.artifact.json").read_text(
                encoding="utf-8"
            )
        ),
    )


@pytest.mark.asyncio
async def test_audition_matrix_is_deterministic_and_reports_persistable_settings(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    original_open = build_module.open_speech_backend
    opened = 0

    @asynccontextmanager
    async def tracked_open(
        name: str,
        *,
        api_key: str | None = None,
        device: str | None = None,
    ) -> AsyncIterator[TextToSpeechService]:
        nonlocal opened
        opened += 1
        async with original_open(name, api_key=api_key, device=device) as service:
            yield service

    monkeypatch.setattr(build_module, "open_speech_backend", tracked_open)
    records = await audition_audiobook(
        manifest,
        profiles=("default",),
        text="Hi.",
        matrix=("sample_rate=8000,16000",),
    )

    assert [record.path.name for record in records] == [
        "default--sample_rate-8000.wav",
        "default--sample_rate-16000.wav",
    ]
    report = json.loads(
        (records[0].path.parent / "audition.json").read_text(encoding="utf-8")
    )
    assert [
        comparison["resolved_settings"]["matrix_overrides"]["sample_rate"]
        for comparison in report["comparisons"]
    ] == [8000, 16000]
    assert opened == 1
    assert "profiles.default" in report["comparisons"][0]["persist_as"]
    validate_contract("audiobook-audition", report)

    with pytest.raises(ValidationError, match="not supported"):
        await audition_audiobook(
            manifest,
            profiles=("default",),
            text="Hi.",
            matrix=("use_hd=true,false",),
        )


@pytest.mark.asyncio
async def test_stage_bounded_build_requires_and_reuses_verified_prerequisites(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    with pytest.raises(BuildError, match="prerequisites"):
        await build_audiobook(
            manifest,
            from_stage="master",
            dry_run=True,
        )

    synthesis = await build_audiobook(manifest, through_stage="synthesize")
    assert synthesis.preflight.from_stage == "synthesize"
    assert synthesis.preflight.through_stage == "synthesize"
    assert len(synthesis.artifacts) == 1
    assert synthesis.artifacts[0].kind is ArtifactKind.RAW

    downstream = await build_audiobook(manifest, from_stage="master")
    assert downstream.preflight.planned_nodes == 3
    assert {record.kind for record in downstream.artifacts} == {
        ArtifactKind.MASTER,
        ArtifactKind.DELIVERY,
        ArtifactKind.REPORT,
    }
    assert check_release(manifest).complete

    with pytest.raises(ValidationError, match="comes after"):
        await build_audiobook(
            manifest,
            from_stage="inspect",
            through_stage="master",
            dry_run=True,
        )


@pytest.mark.asyncio
async def test_failed_build_resumes_same_run_and_reuses_verified_nodes(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    original = build_module.master_wav
    calls = 0

    def fail_once(
        source: Path,
        destination: Path,
        *,
        sample_rate: int,
        overwrite: bool = False,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise BuildError("injected mastering failure")
        original(
            source,
            destination,
            sample_rate=sample_rate,
            overwrite=overwrite,
        )

    monkeypatch.setattr(build_module, "master_wav", fail_once)
    with pytest.raises(BuildError, match="Build failed"):
        await build_audiobook(manifest)
    failed_runs = tuple((book_workspace / ".yakbox" / "runs").iterdir())
    assert len(failed_runs) == 1

    resumed = await build_audiobook(manifest)
    assert resumed.resumed
    assert resumed.run_directory == failed_runs[0]
    assert any(node.endswith(":synthesize") for node in resumed.reused_nodes)
    events = [
        json.loads(line)
        for line in (resumed.run_directory / "journal.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert "run_resumed" in {event["event"] for event in events}

    fresh = await build_audiobook(manifest, resume=False)
    assert not fresh.resumed
    assert fresh.run_directory != resumed.run_directory


@pytest.mark.asyncio
async def test_preflight_explains_source_change_against_last_success(
    book_workspace: Path,
) -> None:
    manifest = load_manifest(book_workspace / "yakbox.toml")
    completed = await build_audiobook(manifest)
    source = book_workspace / "source" / "book.md"
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "A short ending.",
            "A revised short ending.",
        ),
        encoding="utf-8",
    )

    preflight = preflight_audiobook_build(manifest)

    assert preflight.change_summary.previous_run_id == completed.run_id
    assert len(preflight.change_summary.changed_nodes) == 4
    assert preflight.pending_nodes == 4
    reasons = dict(preflight.change_summary.reasons)
    assert (
        reasons["0001-chapter-one:synthesize"]
        == "normalized source or pronunciation input changed"
    )


@pytest.mark.asyncio
async def test_storage_budget_fails_before_build_state_or_backend(
    book_workspace: Path,
) -> None:
    manifest_path = book_workspace / "yakbox.toml"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            "m4b = true",
            "m4b = true\nstorage_budget_bytes = 1",
        ),
        encoding="utf-8",
    )

    with pytest.raises(BuildError, match="storage budget"):
        await build_audiobook(load_manifest(manifest_path))

    assert not (book_workspace / ".yakbox").exists()
    assert not (book_workspace / "build").exists()


class _HostedTrackingService:
    capabilities = BackendCapabilities(
        name="hosted-test",
        synthesis=True,
        transformation=False,
        streaming=False,
        hosted=True,
        output_formats=("wav",),
    )

    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0
        self.delegate = FakeSpeechService()

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            await asyncio.sleep(0.02)
            return await self.delegate.synthesize_to_file(
                request,
                destination,
                overwrite=overwrite,
            )
        finally:
            self.active -= 1


class _BlockingSpeechService:
    capabilities = BackendCapabilities(
        name="blocking",
        synthesis=True,
        transformation=False,
        streaming=False,
        hosted=False,
        output_formats=("wav",),
    )

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        del request, destination, overwrite
        self.started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")


class _BatchingFakeService(FakeSpeechService):
    def __init__(self) -> None:
        self.batch_calls = 0
        self.requests = 0
        self.delegate = FakeSpeechService()

    async def synthesize_many_to_files(
        self,
        requests: tuple[tuple[SpeechSynthesisRequest, Path], ...],
        *,
        overwrite: bool = False,
    ) -> tuple[SpeechArtifact, ...]:
        self.batch_calls += 1
        self.requests += len(requests)
        return tuple(
            [
                await self.delegate.synthesize_to_file(
                    request,
                    destination,
                    overwrite=overwrite,
                )
                for request, destination in requests
            ]
        )


class _FailingFakeService(FakeSpeechService):
    def __init__(self, *, fail_after: int | None) -> None:
        self.fail_after = fail_after
        self.calls = 0
        self.delegate = FakeSpeechService()

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise BuildError("injected chunk failure")
        return await self.delegate.synthesize_to_file(
            request,
            destination,
            overwrite=overwrite,
        )


class _JournaledHostedService(_HostedTrackingService):
    def __init__(self) -> None:
        super().__init__()
        self.gate = HostedUsageGate(HostedUsageBudget(max_provider_requests=1))
        self.provider_sends = 0

    def set_usage_recorder(self, recorder: HostedUsageRecorder | None) -> None:
        self.gate.set_recorder(recorder)

    async def restore_usage(self, snapshot: HostedUsageSnapshot) -> None:
        await self.gate.restore_prior_usage(
            logical_items=snapshot.logical_items,
            provider_attempts=snapshot.provider_attempts,
            submitted_characters=snapshot.submitted_characters,
            ambiguous_attempts=snapshot.ambiguous_attempts,
        )

    async def usage_snapshot(self) -> HostedUsageSnapshot:
        return await self.gate.snapshot()

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        await self.gate.add_logical_item()
        await self.gate.reserve_attempt(len(request.text))
        self.provider_sends += 1
        return await super().synthesize_to_file(
            request,
            destination,
            overwrite=overwrite,
        )


@pytest.mark.asyncio
async def test_hosted_audiobook_scheduling_is_bounded_and_overlapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "book.md").write_text(
        "\n".join(f"# Chapter {index}\n\nHi {index}." for index in range(1, 5)),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "yakbox.toml"
    manifest_path.write_text(
        '"$schema" = '
        '"https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"\n'
        'schema_version = 1\nsources = ["book.md"]\n'
        '[book]\ntitle = "Hosted"\n'
        '[voices.narrator]\ndisplay_name = "Narrator"\n'
        '[profiles.remote]\nbackend = "resemble"\nvoice = "narrator"\n'
        'voice_uuid = "provider-voice"\n'
        '[targets.default]\nprofile = "remote"\nmastering = false\n'
        "provider_concurrency = 3\n",
        encoding="utf-8",
    )
    service = _HostedTrackingService()

    @asynccontextmanager
    async def backend(
        _name: str, **_options: object
    ) -> AsyncIterator[TextToSpeechService]:
        yield service

    monkeypatch.setattr(build_module, "open_speech_backend", backend)
    result = await build_audiobook(load_manifest(manifest_path), api_key="test")

    assert result.status == "complete"
    assert service.maximum_active == 3


@pytest.mark.asyncio
async def test_cancelled_audiobook_run_is_resumable_and_releases_lock(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _BlockingSpeechService()

    @asynccontextmanager
    async def blocking_backend(
        _name: str, **_options: object
    ) -> AsyncIterator[TextToSpeechService]:
        yield service

    monkeypatch.setattr(build_module, "open_speech_backend", blocking_backend)
    manifest = load_manifest(book_workspace / "yakbox.toml")
    task = asyncio.create_task(build_audiobook(manifest))
    await service.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    run_directory = next((book_workspace / ".yakbox" / "runs").iterdir())
    events = [
        json.loads(line)
        for line in (run_directory / "journal.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[-1]["event"] == "run_interrupted"
    assert not (book_workspace / ".yakbox" / "locks" / "default.lock").exists()

    @asynccontextmanager
    async def fake_backend(
        _name: str, **_options: object
    ) -> AsyncIterator[TextToSpeechService]:
        yield FakeSpeechService()

    monkeypatch.setattr(build_module, "open_speech_backend", fake_backend)
    resumed = await build_audiobook(manifest)
    assert resumed.resumed
    assert resumed.run_directory == run_directory
    assert resumed.status == "complete"


@pytest.mark.asyncio
async def test_local_chapter_chunks_share_one_isolated_worker_request(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = book_workspace / "yakbox.toml"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            "chunk_chars = 100",
            "chunk_chars = 10",
        ),
        encoding="utf-8",
    )
    service = _BatchingFakeService()

    @asynccontextmanager
    async def backend(
        _name: str, **_options: object
    ) -> AsyncIterator[TextToSpeechService]:
        yield service

    monkeypatch.setattr(build_module, "open_speech_backend", backend)
    result = await build_audiobook(
        load_manifest(manifest_path),
        through_stage="synthesize",
    )

    assert result.status == "complete"
    assert service.requests > 1
    assert service.batch_calls == 1


@pytest.mark.asyncio
async def test_failed_chapter_resumes_from_verified_file_backed_chunk_cache(
    book_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = book_workspace / "yakbox.toml"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            "chunk_chars = 100",
            "chunk_chars = 10",
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    plan = plan_audiobook(
        manifest,
        normalize_sources(
            manifest.sources,
            pronunciations=manifest.pronunciations,
        ),
    )
    synthesis = next(node for node in plan.nodes if node.stage.value == "synthesize")
    expected_requests = sum(
        not chunk.startswith("__YAKBOX_PAUSE_MS=") for chunk in synthesis.chunks
    )
    failed = _FailingFakeService(fail_after=1)
    resumed = _FailingFakeService(fail_after=None)
    services = iter((failed, resumed))

    @asynccontextmanager
    async def backend(
        _name: str, **_options: object
    ) -> AsyncIterator[TextToSpeechService]:
        yield next(services)

    monkeypatch.setattr(build_module, "open_speech_backend", backend)

    with pytest.raises(BuildError, match=r"book\.md:\d+-\d+.*injected chunk failure"):
        await build_audiobook(manifest, through_stage="synthesize")
    cached = tuple((book_workspace / ".yakbox" / "cache" / "synthesis").rglob("*.wav"))
    assert len(cached) == 1

    result = await build_audiobook(manifest, through_stage="synthesize")

    assert result.status == "complete"
    assert resumed.calls == expected_requests - 1


@pytest.mark.asyncio
async def test_hosted_audiobook_reservation_survives_failed_run_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "book.md").write_text("# One\n\nTiny text.", encoding="utf-8")
    manifest_path = tmp_path / "yakbox.toml"
    manifest_path.write_text(
        '"$schema" = '
        '"https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"\n'
        'schema_version = 1\nsources = ["book.md"]\n'
        '[book]\ntitle = "Hosted resume"\n'
        '[voices.narrator]\ndisplay_name = "Narrator"\n'
        '[profiles.remote]\nbackend = "resemble"\nvoice = "narrator"\n'
        'voice_uuid = "provider-voice"\n'
        '[targets.default]\nprofile = "remote"\nmastering = true\n'
        "max_provider_requests = 1\n",
        encoding="utf-8",
    )
    services: list[_JournaledHostedService] = []

    @asynccontextmanager
    async def backend(
        _name: str, **_options: object
    ) -> AsyncIterator[TextToSpeechService]:
        service = _JournaledHostedService()
        services.append(service)
        yield service

    monkeypatch.setattr(build_module, "open_speech_backend", backend)
    original_master = build_module.master_wav

    def fail_master(_source: Path, _destination: Path, *, sample_rate: int) -> None:
        del sample_rate
        raise BuildError("injected mastering failure")

    monkeypatch.setattr(build_module, "master_wav", fail_master)
    manifest = load_manifest(manifest_path)
    with pytest.raises(BuildError, match="Build failed"):
        await build_audiobook(manifest, api_key="test")

    run_directory = next((tmp_path / ".yakbox" / "runs").iterdir())
    events = [
        json.loads(line)
        for line in (run_directory / "journal.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    event_names = [event["event"] for event in events]
    assert event_names.index("usage_reserved") < event_names.index("node_failed")
    assert services[0].provider_sends == 1

    raw = next((tmp_path / "build" / "yakbox" / "raw").glob("*.wav"))
    raw.unlink()
    raw.with_suffix(".wav.artifact.json").unlink()
    monkeypatch.setattr(build_module, "master_wav", original_master)
    resumed = await build_audiobook(manifest, api_key="test")

    assert resumed.status == "complete"
    assert resumed.preflight.hosted_work is not None
    assert resumed.preflight.hosted_work.logical_items == 0
    assert len(services) == 2
    assert services[1].provider_sends == 0
    restored = await services[1].usage_snapshot()
    assert restored.provider_attempts == 1


@pytest.mark.asyncio
async def test_twenty_one_chapter_audiobook_dogfood_pipeline(
    tmp_path: Path,
) -> None:
    chapters = [
        (
            f"# Chapter {index}\n\nAnima Cara term {index}.\n\n"
            + (
                "<!-- yakbox:speech:exclude:start -->\n"
                "Printed production note.\n"
                "<!-- yakbox:speech:exclude:end -->\n\n"
                "<!-- yakbox:speech:only:start -->Spoken subtitle."
                "<!-- yakbox:speech:only:end -->\n\n"
                "<!-- yakbox:speech:pause ms=10 -->\n"
                if index == 1
                else ""
            )
        )
        for index in range(1, 22)
    ]
    (tmp_path / "book.md").write_text("\n".join(chapters), encoding="utf-8")
    (tmp_path / "pronunciations.toml").write_text(
        "schema_version = 1\n"
        "[[terms]]\n"
        'written = "Anima Cara"\n'
        'spoken = "Ah-nee-mah Kah-rah"\n'
        'status = "approved"\n'
        "enabled = true\n"
        "priority = 100\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "yakbox.toml"
    manifest_path.write_text(
        '"$schema" = '
        '"https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"\n'
        'schema_version = 1\nsources = ["book.md"]\n'
        'pronunciations = "pronunciations.toml"\n'
        '[book]\ntitle = "Anima Cara Shape"\nauthor = "Author"\n'
        '[voices.narrator]\ndisplay_name = "Narrator"\n'
        '[profiles.default]\nbackend = "fake"\nvoice = "narrator"\n'
        "sample_rate = 16000\n"
        '[profiles.alternate]\nbackend = "fake"\nvoice = "narrator"\n'
        "sample_rate = 22050\n"
        '[targets.default]\nprofile = "default"\nchunk_chars = 100\n'
        "mastering = true\nwav_sample_rate = 44100\n"
        'mp3_bitrate = "64k"\nm4b = true\n',
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    auditions = await audition_audiobook(
        manifest,
        profiles=("default", "alternate"),
        text="A short audition.",
    )
    first = await build_audiobook(manifest)
    resumed = await build_audiobook(manifest)
    release = check_release(manifest, write_manifest=True)
    plan = plan_audiobook(
        manifest,
        normalize_sources(
            manifest.sources,
            pronunciations=manifest.pronunciations,
        ),
    )
    shard_paths = export_shard_manifests(
        plan,
        tmp_path / ".yakbox" / "dogfood-shards",
        count=4,
        root=tmp_path,
    )
    verified = verify_shard_manifests(shard_paths, root=tmp_path)
    m4b = assemble_release(manifest)

    assert len(auditions) == 2
    assert len(first.artifacts) == 84
    assert len(resumed.reused_nodes) == 84
    assert release.complete
    assert len(release.master_wavs) == 21
    assert len(release.delivery_mp3s) == 21
    assert len(verified) == 4
    assert m4b.is_file()
