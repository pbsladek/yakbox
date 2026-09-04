"""Digest-bound attestations for independent speech-analysis release evidence."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_json, sha256_file
from yakbox.contracts import runtime_metadata
from yakbox.errors import ValidationError
from yakbox.speech.analysis_cutover import CutoverEvidence, CutoverEvidenceKind
from yakbox.speech.analysis_fingerprints import semantic_fingerprint

_MAXIMUM_JUNIT_BYTES = 16 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.-]{1,127}")
_SCHEMA = (
    "https://yakbox.dev/schemas/speech-analysis-independent-evidence-v1.schema.json"
)
_MINIMUM_TESTS = {
    CutoverEvidenceKind.APPLE_SILICON_REAL_MODELS: 8,
    CutoverEvidenceKind.REPEATED_CALL_ENDURANCE: 1,
}
_REQUIRED_SUBJECTS = {
    CutoverEvidenceKind.APPLE_SILICON_REAL_MODELS: frozenset(
        {
            "development.lock",
            "test.real-models",
            "worker.artifact",
            "runtime.whisper",
            "runtime.parakeet",
            "runtime.qwen",
            "model.whisper",
            "model.parakeet",
            "model.qwen",
            "model.qwen-forced",
            "candidate.qwen-8bit",
            "candidate.qwen-forced-8bit",
        }
    ),
    CutoverEvidenceKind.REPEATED_CALL_ENDURANCE: frozenset(
        {
            "development.lock",
            "test.endurance",
            "worker.artifact",
            "runtime.whisper",
            "runtime.parakeet",
            "runtime.qwen",
            "model.whisper",
            "model.parakeet",
            "model.qwen",
            "model.qwen-forced",
        }
    ),
}


@dataclass(frozen=True, slots=True)
class EvidenceSubject:
    """One immutable input bound to an independent evidence result."""

    name: str
    fingerprint: str

    def __post_init__(self) -> None:
        if _IDENTIFIER.fullmatch(self.name) is None:
            raise ValidationError("Evidence subject name is invalid")
        _require_sha256(self.fingerprint, "evidence subject fingerprint")


@dataclass(frozen=True, slots=True)
class IndependentEvidenceAttestation:
    """A test result bound to its report, execution class, and exact subjects."""

    kind: CutoverEvidenceKind
    passed: bool
    producer: str
    report_digest: str
    execution_class_fingerprint: str
    subjects: tuple[EvidenceSubject, ...]
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        if _IDENTIFIER.fullmatch(self.producer) is None:
            raise ValidationError("Evidence producer is invalid")
        _require_sha256(self.report_digest, "evidence report digest")
        _require_sha256(
            self.execution_class_fingerprint,
            "evidence execution class fingerprint",
        )
        names = tuple(item.name for item in self.subjects)
        required_subjects = _REQUIRED_SUBJECTS.get(self.kind, frozenset())
        if (
            not self.subjects
            or names != tuple(sorted(set(names)))
            or not required_subjects.issubset(names)
            or self.issues != tuple(sorted(set(self.issues)))
            or any(_IDENTIFIER.fullmatch(item) is None for item in self.issues)
            or self.passed == bool(self.issues)
        ):
            raise ValidationError("Independent evidence attestation is inconsistent")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "speech-analysis-independent-evidence-v1",
            self,
        )

    def to_cutover_evidence(self) -> CutoverEvidence:
        return CutoverEvidence(self.kind, self.fingerprint, self.passed)

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("speech-analysis-independent-evidence"),
            "fingerprint": self.fingerprint,
            "kind": self.kind.value,
            "passed": self.passed,
            "producer": self.producer,
            "report_digest": self.report_digest,
            "execution_class_fingerprint": self.execution_class_fingerprint,
            "subjects": [
                {"name": item.name, "fingerprint": item.fingerprint}
                for item in self.subjects
            ],
            "issues": list(self.issues),
        }


def attest_pytest_junit(
    report: Path,
    *,
    kind: CutoverEvidenceKind,
    execution_class_fingerprint: str,
    subjects: tuple[EvidenceSubject, ...],
) -> IndependentEvidenceAttestation:
    """Parse a bounded JUnit report and reject failures, errors, or skips."""
    try:
        if report.stat().st_size > _MAXIMUM_JUNIT_BYTES:
            raise ValidationError("JUnit evidence report exceeds 16 MiB")
        payload = report.read_bytes()
        if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
            raise ValidationError("JUnit evidence cannot contain XML declarations")
        root = ET.fromstring(  # noqa: S314 - bounded local XML rejects DTD/entities
            payload
        )
    except (OSError, ET.ParseError) as error:
        raise ValidationError("Cannot read JUnit evidence report") from error
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValidationError("JUnit evidence report has an invalid root")
    suites = (root,) if root.tag == "testsuite" else tuple(root.findall("testsuite"))
    if not suites:
        raise ValidationError("JUnit evidence report contains no test suites")
    tests = sum(_junit_count(item, "tests") for item in suites)
    failures = sum(_junit_count(item, "failures") for item in suites)
    errors = sum(_junit_count(item, "errors") for item in suites)
    skipped = sum(_junit_count(item, "skipped") for item in suites)
    minimum_tests = _MINIMUM_TESTS.get(kind, 1)
    issues = tuple(
        name
        for name, present in (
            ("junit_insufficient_tests", tests < minimum_tests),
            ("junit_failures", failures > 0),
            ("junit_errors", errors > 0),
            ("junit_skipped", skipped > 0),
        )
        if present
    )
    return IndependentEvidenceAttestation(
        kind,
        not issues,
        "pytest-junit",
        sha256_file(report),
        execution_class_fingerprint,
        tuple(sorted(subjects, key=lambda item: item.name)),
        tuple(sorted(issues)),
    )


def write_independent_evidence(
    path: Path,
    attestation: IndependentEvidenceAttestation,
) -> None:
    atomic_write_json(path, attestation.to_dict())


def load_independent_evidence(path: Path) -> IndependentEvidenceAttestation:
    """Load a strict attestation and verify its semantic fingerprint."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError("Cannot read independent evidence") from error
    if not isinstance(raw, dict):
        raise ValidationError("Independent evidence must be a JSON object")
    expected = {
        "$schema",
        "schema_version",
        "yakbox_version",
        "timestamp",
        "fingerprint",
        "kind",
        "passed",
        "producer",
        "report_digest",
        "execution_class_fingerprint",
        "subjects",
        "issues",
    }
    if set(raw) != expected or raw.get("$schema") != _SCHEMA:
        raise ValidationError("Independent evidence fields are invalid")
    if raw.get("schema_version") != 1:
        raise ValidationError("Independent evidence schema version is unsupported")
    if not isinstance(raw.get("yakbox_version"), str) or not isinstance(
        raw.get("timestamp"), str
    ):
        raise ValidationError("Independent evidence metadata is invalid")
    try:
        subjects_raw = cast(list[object], raw["subjects"])
        subjects = tuple(
            EvidenceSubject(
                _mapping_string(item, "name"),
                _mapping_string(item, "fingerprint"),
            )
            for item in subjects_raw
        )
        issues_raw = cast(list[object], raw["issues"])
        attestation = IndependentEvidenceAttestation(
            CutoverEvidenceKind(_string(raw, "kind")),
            _boolean(raw, "passed"),
            _string(raw, "producer"),
            _string(raw, "report_digest"),
            _string(raw, "execution_class_fingerprint"),
            subjects,
            tuple(_text(item, "evidence issue") for item in issues_raw),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValidationError("Independent evidence value is invalid") from error
    if _string(raw, "fingerprint") != attestation.fingerprint:
        raise ValidationError("Independent evidence fingerprint differs")
    return attestation


def _junit_count(element: ET.Element, name: str) -> int:
    raw = element.get(name, "0")
    try:
        value = int(raw)
    except ValueError as error:
        raise ValidationError("JUnit evidence count is invalid") from error
    if value < 0:
        raise ValidationError("JUnit evidence count cannot be negative")
    return value


def _mapping_string(value: object, key: str) -> str:
    if not isinstance(value, dict) or set(value) != {"name", "fingerprint"}:
        raise ValidationError("Evidence subject fields are invalid")
    return _string(cast(dict[str, object], value), key)


def _string(raw: dict[str, object], key: str) -> str:
    return _text(raw.get(key), key)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Independent evidence {label} must be text")
    return value


def _boolean(raw: dict[str, object], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValidationError(f"Independent evidence {key} must be boolean")
    return value


def _require_sha256(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label.capitalize()} must be a SHA-256")


__all__ = [
    "EvidenceSubject",
    "IndependentEvidenceAttestation",
    "attest_pytest_junit",
    "load_independent_evidence",
    "write_independent_evidence",
]
