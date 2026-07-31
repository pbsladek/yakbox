from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum

from yakbox.contracts import runtime_metadata


class DiagnosticStatus(StrEnum):
    PASS = "pass"  # noqa: S105 - diagnostic status, not a credential
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class DiagnosticSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One health check result with severity, remediation, timing, and evidence."""

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
    """Versioned collection of environment and backend health diagnostics."""

    schema_version: int
    diagnostics: tuple[Diagnostic, ...]

    @property
    def healthy(self) -> bool:
        """Return whether no diagnostic has an error-severity failure."""
        return not any(
            item.status is DiagnosticStatus.FAIL for item in self.diagnostics
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize diagnostics using the versioned doctor-report contract."""
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
