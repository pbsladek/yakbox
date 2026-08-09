from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import cast

import pytest

from tests.narration_review import narration_review_issues
from yakbox._files import sha256_file

pytestmark = pytest.mark.live


def test_completed_narration_review_is_approved() -> None:
    configured = os.environ.get("YAKBOX_NARRATION_REVIEW")
    if configured is None:
        pytest.skip("set YAKBOX_NARRATION_REVIEW to a listening-review.toml file")
    review_path = Path(configured).expanduser().resolve()
    if not review_path.is_file():
        pytest.fail(f"Narration review does not exist: {review_path}")
    qa_directory = review_path.parent
    workspace = qa_directory.parent
    report_path = qa_directory / "report.json"
    qa_path = workspace / "qa.toml"
    if not report_path.is_file() or not qa_path.is_file():
        pytest.fail("Review must be beside report.json in a generated QA workspace")

    review = _load_toml(review_path)
    qa = _load_toml(qa_path)
    issues = narration_review_issues(
        review,
        qa,
        report_sha256=sha256_file(report_path),
        require_approved=True,
    )
    assert not issues, "Narration review failed:\n- " + "\n- ".join(issues)


def _load_toml(path: Path) -> dict[str, object]:
    return cast(dict[str, object], tomllib.loads(path.read_text(encoding="utf-8")))
