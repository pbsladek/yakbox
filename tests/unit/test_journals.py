from __future__ import annotations

from pathlib import Path

import pytest
from tests.schema_helpers import validate_contract

from yakbox.audiobook.journal import RunJournal, target_lock
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
