from pathlib import Path

import pytest

from yakbox.errors import ValidationError
from yakbox.textutils import BatchRow, read_batch_rows


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_txt_uses_nonempty_trimmed_lines(tmp_path: Path) -> None:
    path = _write(tmp_path, "script.txt", "\ufeff First line \n\nSecond line\n")

    assert read_batch_rows(path) == (
        BatchRow(index=1, text="First line"),
        BatchRow(index=3, text="Second line"),
    )


def test_csv_reads_supported_per_row_fields(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "script.csv",
        "text,id,voice_uuid,title,output\n"
        "Hello,opening,voice-1,Opening,opening.wav\n"
        "World,,,,\n",
    )

    assert read_batch_rows(path) == (
        BatchRow(
            index=2,
            text="Hello",
            row_id="opening",
            voice_uuid="voice-1",
            title="Opening",
            output="opening.wav",
        ),
        BatchRow(index=3, text="World"),
    )


def test_csv_rejects_unknown_headers(tmp_path: Path) -> None:
    path = _write(tmp_path, "script.csv", "text,speaker\nHello,Narrator\n")

    with pytest.raises(ValidationError, match="Unknown CSV columns: speaker"):
        read_batch_rows(path)


def test_csv_marks_extra_columns_as_a_row_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "script.csv", "text\nHello,unexpected\n")

    assert read_batch_rows(path) == (
        BatchRow(index=2, text="", validation_error="extra columns"),
    )


def test_jsonl_reads_fields_and_preserves_row_local_errors(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "script.jsonl",
        '{"text":" Hello ","id":"opening","voice_uuid":"voice-1",'
        '"title":"Opening","output":"opening.mp3"}\n'
        "not-json\n"
        '["not", "an", "object"]\n'
        '{"text":"World","speaker":"Narrator"}\n',
    )

    rows = read_batch_rows(path)

    assert rows[0] == BatchRow(
        index=1,
        text="Hello",
        row_id="opening",
        voice_uuid="voice-1",
        title="Opening",
        output="opening.mp3",
    )
    assert rows[1].validation_error == "invalid JSON: Expecting value"
    assert rows[2].validation_error == "record must be an object"
    assert rows[3].validation_error == "unknown keys: speaker"


@pytest.mark.parametrize("name", ["script.yaml", "script"])
def test_unsupported_batch_extension_is_rejected(tmp_path: Path, name: str) -> None:
    path = _write(tmp_path, name, "Hello")

    with pytest.raises(
        ValidationError, match=r"Batch input must use \.txt, \.csv, or \.jsonl"
    ):
        read_batch_rows(path)
