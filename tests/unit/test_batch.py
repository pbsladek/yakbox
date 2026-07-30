from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import pytest
from tests.schema_helpers import validate_contract

from yakbox.cloud.batch import BatchStatus, run_cloud_batch
from yakbox.cloud.journal import read_journal
from yakbox.cloud.usage import HostedUsageGate
from yakbox.speech import AudioFormat, BackendCapabilities, HostedUsageBudget
from yakbox.speech.models import SpeechArtifact, SpeechSynthesisRequest
from yakbox.speech.services import FakeSpeechService
from yakbox.textutils import BatchRow


class TrackingService:
    capabilities = BackendCapabilities(
        name="tracking",
        synthesis=True,
        transformation=False,
        streaming=False,
        hosted=True,
        output_formats=("wav",),
        max_text_characters=3_000,
    )

    def __init__(self) -> None:
        self.calls = 0
        self.active = 0
        self.maximum_active = 0
        self._delegate = FakeSpeechService()

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        self.calls += 1
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            await asyncio.sleep(0.02)
            return await self._delegate.synthesize_to_file(
                SpeechSynthesisRequest(
                    text=request.text,
                    voice=request.voice,
                    output_format=AudioFormat.WAV,
                ),
                destination,
                overwrite=overwrite,
            )
        finally:
            self.active -= 1


class BudgetTrackingService(TrackingService):
    def __init__(self, gate: HostedUsageGate) -> None:
        super().__init__()
        self.gate = gate

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        await self.gate.add_logical_item()
        await self.gate.reserve_attempt(len(request.text))
        return await super().synthesize_to_file(
            request,
            destination,
            overwrite=overwrite,
        )


class BlockingService(TrackingService):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        del request, destination, overwrite
        self.active += 1
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.active -= 1
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_batch_parallelizes_and_isolates_validation(tmp_path: Path) -> None:
    service = TrackingService()
    rows = (
        BatchRow(index=1, text="one"),
        BatchRow(index=2, text="x" * 3_001),
        BatchRow(index=3, text="three"),
        BatchRow(index=4, text="four"),
    )
    report = await run_cloud_batch(
        rows,
        service,
        default_voice="voice",
        project_uuid=None,
        out_dir=tmp_path,
        concurrency=3,
    )

    assert report.ok == 3
    assert report.failed == 1
    assert report.results[1].status is BatchStatus.ERROR
    assert "3000" in (report.results[1].error_message or "")
    assert service.calls == 3
    assert service.maximum_active == 3
    assert (tmp_path / "batch-journal.ndjson").is_file()
    assert (tmp_path / "batch-report.json").is_file()
    validate_contract("batch-report", report.to_dict())
    records = read_journal(report.journal_path)
    for record in records:
        validate_contract("batch-journal", record)
    assert [record["record_type"] for record in records].count("row") == 4


@pytest.mark.asyncio
async def test_batch_closes_without_leaking_on_error(tmp_path: Path) -> None:
    service = TrackingService()
    report = await run_cloud_batch(
        (BatchRow(index=1, text="", validation_error="bad row"),),
        service,
        default_voice="voice",
        project_uuid=None,
        out_dir=tmp_path,
        concurrency=1,
    )
    assert report.failed == 1
    assert service.active == 0


@pytest.mark.asyncio
async def test_batch_resume_verifies_and_skips_completed_output(tmp_path: Path) -> None:
    rows = (BatchRow(index=1, text="resume me"),)
    first_service = TrackingService()
    first = await run_cloud_batch(
        rows,
        first_service,
        default_voice="voice",
        project_uuid=None,
        out_dir=tmp_path,
        concurrency=1,
    )
    second_service = TrackingService()
    second = await run_cloud_batch(
        rows,
        second_service,
        default_voice="voice",
        project_uuid=None,
        out_dir=tmp_path,
        concurrency=1,
        resume_path=first.journal_path,
    )

    assert second.results[0].status is BatchStatus.SKIPPED
    assert second_service.calls == 0


@pytest.mark.asyncio
async def test_batch_usage_gate_is_atomic_and_survives_resume(tmp_path: Path) -> None:
    rows = (
        BatchRow(index=1, text="one"),
        BatchRow(index=2, text="two"),
    )
    first_gate = HostedUsageGate(HostedUsageBudget(max_provider_requests=2))
    first = await run_cloud_batch(
        rows,
        BudgetTrackingService(first_gate),
        default_voice="voice",
        project_uuid=None,
        out_dir=tmp_path,
        concurrency=2,
        usage_gate=first_gate,
    )
    assert first.usage is not None
    assert first.usage.provider_attempts == 2
    records = read_journal(first.journal_path)
    reservations = [
        record for record in records if record["record_type"] == "usage_reserved"
    ]
    assert len(reservations) == 2
    terminal_reservation = reservations[-1]["usage"]
    assert isinstance(terminal_reservation, dict)
    assert cast(dict[str, object], terminal_reservation)["provider_attempts"] == 2
    assert records.index(reservations[0]) < next(
        index for index, record in enumerate(records) if record["record_type"] == "row"
    )

    first_path = first.results[0].path
    assert first_path is not None
    first_path.unlink()
    resumed_gate = HostedUsageGate(HostedUsageBudget(max_provider_requests=2))
    resumed_service = BudgetTrackingService(resumed_gate)
    resumed = await run_cloud_batch(
        rows,
        resumed_service,
        default_voice="voice",
        project_uuid=None,
        out_dir=tmp_path,
        concurrency=2,
        resume_path=first.journal_path,
        usage_gate=resumed_gate,
    )

    assert resumed.aborted
    assert resumed.usage is not None
    assert resumed.usage.provider_attempts == 2
    assert resumed_service.calls == 0


@pytest.mark.asyncio
async def test_batch_dry_run_has_no_filesystem_or_service_side_effect(
    tmp_path: Path,
) -> None:
    output = tmp_path / "does-not-exist"
    service = TrackingService()
    report = await run_cloud_batch(
        (BatchRow(index=1, text="hello"),),
        service,
        default_voice="voice",
        project_uuid=None,
        out_dir=output,
        dry_run=True,
    )

    assert report.results[0].status is BatchStatus.NOT_RUN
    assert service.calls == 0
    assert not output.exists()


@pytest.mark.asyncio
async def test_batch_cancellation_is_journaled_and_leaves_no_worker(
    tmp_path: Path,
) -> None:
    service = BlockingService()
    task = asyncio.create_task(
        run_cloud_batch(
            (BatchRow(index=1, text="cancel me"),),
            service,
            default_voice="voice",
            project_uuid=None,
            out_dir=tmp_path,
            concurrency=1,
        )
    )
    await service.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert service.active == 0
    records = read_journal(tmp_path / "batch-journal.ndjson")
    assert [record["record_type"] for record in records] == [
        "header",
        "row",
        "interrupted",
    ]
    report = json.loads((tmp_path / "batch-report.json").read_text(encoding="utf-8"))
    assert report["aborted"] is True
    assert report["summary"]["not_run"] == 1
    validate_contract("batch-report", report)
