"""Typed yakbox installation and workspace diagnostics."""

from yakbox.diagnostics.checks import run_doctor
from yakbox.diagnostics.models import (
    Diagnostic,
    DiagnosticSeverity,
    DiagnosticStatus,
    DoctorReport,
)

__all__ = [
    "Diagnostic",
    "DiagnosticSeverity",
    "DiagnosticStatus",
    "DoctorReport",
    "run_doctor",
]
