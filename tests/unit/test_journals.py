from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.schema_helpers import validate_contract

from yakbox.audiobook.journal import RunJournal, target_lock
from yakbox.cloud.errors import BatchJournalError
from yakbox.cloud.journal import BatchJournalWriter, read_journal
from yakbox.contracts import runtime_metadata
from yakbox.errors import BuildError


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
