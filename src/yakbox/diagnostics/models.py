from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum

from yakbox.contracts import runtime_metadata


class DiagnosticStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class DiagnosticSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    id: str
    status: DiagnosticStatus
    severity: DiagnosticSeverity
    summary: str
    detail: str | None = None
    action: str | None = None
    elapsed_seconds: float = 0.0
    skipped_by_policy: bool = False
    evidence: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class DoctorReport:
    schema_version: int
    diagnostics: tuple[Diagnostic, ...]

    @property
    def healthy(self) -> bool:
        return not any(
            item.status is DiagnosticStatus.FAIL for item in self.diagnostics
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("doctor-report"),
            "healthy": self.healthy,
            "diagnostics": [
                {
                    **asdict(item),
                    "status": item.status.value,
                    "severity": item.severity.value,
                }
                for item in self.diagnostics
            ],
        }
