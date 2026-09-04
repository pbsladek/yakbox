from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from yakbox.errors import ValidationError
from yakbox.schemas import load_schema
from yakbox.speech.analysis_attestation import (
    EvidenceSubject,
    attest_pytest_junit,
    load_independent_evidence,
    write_independent_evidence,
)
from yakbox.speech.analysis_cutover import CutoverEvidenceKind

SHA_A = "a" * 64
SHA_B = "b" * 64


def _junit(path: Path, *, failures: int = 0, skipped: int = 0) -> None:
    path.write_text(
        f'<testsuites><testsuite tests="1" failures="{failures}" '
        f'errors="0" skipped="{skipped}"/></testsuites>',
        encoding="utf-8",
    )


def test_junit_attestation_is_schema_valid_round_trippable_and_cutover_ready(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.xml"
    output = tmp_path / "evidence.json"
    _junit(report)

    evidence = attest_pytest_junit(
        report,
        kind=CutoverEvidenceKind.AUTOMATED_TESTS,
        execution_class_fingerprint=SHA_A,
        subjects=(EvidenceSubject("runtime.whisper", SHA_B),),
    )
    write_independent_evidence(output, evidence)
    loaded = load_independent_evidence(output)

    assert loaded == evidence
    assert loaded.to_cutover_evidence().passed
    Draft202012Validator(load_schema("speech-analysis-independent-evidence")).validate(
        json.loads(output.read_text(encoding="utf-8"))
    )


@pytest.mark.parametrize(
    ("failures", "skipped", "issue"),
    ((1, 0, "junit_failures"), (0, 1, "junit_skipped")),
)
def test_junit_attestation_fails_closed(
    tmp_path: Path,
    failures: int,
    skipped: int,
    issue: str,
) -> None:
    report = tmp_path / "report.xml"
    _junit(report, failures=failures, skipped=skipped)

    evidence = attest_pytest_junit(
        report,
        kind=CutoverEvidenceKind.AUTOMATED_TESTS,
        execution_class_fingerprint=SHA_A,
        subjects=(EvidenceSubject("source.tree", SHA_B),),
    )

    assert not evidence.passed
    assert issue in evidence.issues


def test_loaded_attestation_rejects_unknown_fields_and_changed_fingerprint(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.xml"
    output = tmp_path / "evidence.json"
    _junit(report)
    evidence = attest_pytest_junit(
        report,
        kind=CutoverEvidenceKind.AUTOMATED_TESTS,
        execution_class_fingerprint=SHA_A,
        subjects=(EvidenceSubject("source.tree", SHA_B),),
    )
    write_independent_evidence(output, evidence)
    raw = json.loads(output.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    output.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValidationError, match="fields"):
        load_independent_evidence(output)
