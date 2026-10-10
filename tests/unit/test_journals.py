from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import BinaryIO

import pytest
from tests.schema_helpers import validate_contract

from yakbox.audiobook.journal import RunJournal, target_lock
from yakbox.cloud.errors import BatchJournalError
from yakbox.cloud.journal import BatchJournalWriter, read_journal
from yakbox.cloud.journal import _durable_append as durable_append
from yakbox.contracts import runtime_metadata
from yakbox.errors import BuildError


@pytest.mark.asyncio
async def test_cancelled_journal_append_does_not_stop_the_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    def delayed_append(stream: BinaryIO, payload: bytes) -> None:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        durable_append(stream, payload)

    monkeypatch.setattr("yakbox.cloud.journal._durable_append", delayed_append)
    path = tmp_path / "batch.ndjson"
    try:
        async with BatchJournalWriter(path) as writer:
            append = asyncio.create_task(writer.append_record(_batch_record("header")))
            await asyncio.wait_for(started.wait(), timeout=2)
            append.cancel()
            with pytest.raises(asyncio.CancelledError):
                await append
            release.set()
            await asyncio.wait_for(
                writer.append_record(_batch_record("interrupted")), timeout=2
            )
    finally:
        release.set()

    assert [record["record_type"] for record in read_journal(path)] == [
        "header",
        "interrupted",
    ]


@pytest.mark.asyncio
async def test_journal_failure_unblocks_producers_waiting_for_queue_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    def failing_append(_stream: object, _payload: bytes) -> None:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        raise OSError("disk full")

    monkeypatch.setattr("yakbox.cloud.journal._durable_append", failing_append)
    results: list[BaseException | None] | None = None
    try:
        with pytest.raises(BatchJournalError, match="disk full"):
            async with BatchJournalWriter(tmp_path / "batch.ndjson") as writer:
                appends = [
                    asyncio.create_task(writer.append_record(_batch_record("header")))
                ]
                await asyncio.wait_for(started.wait(), timeout=2)
                appends.extend(
                    asyncio.create_task(writer.append_record(_batch_record("row")))
                    for _ in range(102)
                )
                await asyncio.sleep(0)
                release.set()
                results = await asyncio.wait_for(
                    asyncio.gather(*appends, return_exceptions=True), timeout=2
                )
    finally:
        release.set()

    assert results is not None
    assert len(results) == 103
    assert all(isinstance(result, (OSError, BatchJournalError)) for result in results)


def test_audiobook_journal_recovers_only_a_torn_final_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "journal.ndjson"
    journal = RunJournal(path, "run-1")
    journal.append("run_started", fingerprint="abc")
    with path.open("ab") as stream:
        stream.write(b'{"event":"partial"')

    events = journal.events()

    assert [event["event"] for event in events] == [
        "run_started",
        "journal_recovered",
    ]
    assert path.with_suffix(".torn").read_bytes() == b'{"event":"partial"'
    assert path.read_bytes().endswith(b"\n")
    for event in events:
        validate_contract("audiobook-journal", event)
    journal.append("run_complete")
    assert len(journal.events()) == 3


def test_audiobook_journal_rejects_midstream_corruption_and_wrong_run(
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "corrupt.ndjson"
    corrupt.write_bytes(b"{not-json}\n" + b'{"still":"data"}\n')
    with pytest.raises(BuildError, match="Corrupt journal record 1"):
        RunJournal(corrupt, "run-1").events()

    wrong = tmp_path / "wrong.ndjson"
    writer = RunJournal(wrong, "run-other")
    writer.append("started")
    with pytest.raises(BuildError, match="identity mismatch"):
        RunJournal(wrong, "run-1").events()


def test_target_lock_is_exclusive_and_released_after_failure(tmp_path: Path) -> None:
    state = tmp_path / ".yakbox"
    with (
        target_lock(state, "default"),
        pytest.raises(BuildError, match="already locked"),
        target_lock(state, "default"),
    ):
        pytest.fail("second target lock must not be acquired")

    with pytest.raises(RuntimeError, match="boom"), target_lock(state, "default"):
        raise RuntimeError("boom")
    with target_lock(state, "default"):
        pass


def _batch_record(record_type: str) -> dict[str, object]:
    return {
        **runtime_metadata("batch-journal"),
        "record_type": record_type,
        "run_id": "batch-1",
    }


@pytest.mark.asyncio
async def test_batch_journal_torn_tail_is_preserved_truncated_and_appendable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "batch.ndjson"
    async with BatchJournalWriter(path) as writer:
        await writer.append_record(_batch_record("header"))
    with path.open("ab") as stream:
        stream.write(b'{"record_type":"row"')

    records = read_journal(path)
    assert len(records) == 1
    assert path.with_suffix(".torn").read_bytes() == b'{"record_type":"row"'
    async with BatchJournalWriter(path, append=True) as writer:
        await writer.append_record(_batch_record("resumed"))

    recovered = read_journal(path)
    assert [item["record_type"] for item in recovered] == ["header", "resumed"]
    for record in recovered:
        validate_contract("batch-journal", record)


@pytest.mark.asyncio
async def test_batch_journal_refuses_unsafe_append_and_future_schema(
    tmp_path: Path,
) -> None:
    torn = tmp_path / "torn.ndjson"
    torn.write_bytes(b"partial")
    with pytest.raises(BatchJournalError, match="torn final"):
        async with BatchJournalWriter(torn, append=True):
            pytest.fail("unsafe append must not open")

    future = tmp_path / "future.ndjson"
    record = _batch_record("header")
    record["schema_version"] = 2
    future.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(BatchJournalError, match="Unsupported"):
        read_journal(future)
