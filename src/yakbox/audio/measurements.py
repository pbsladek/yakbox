"""Streaming loudness summaries and edge-silence measurements."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from yakbox.errors import ArtifactError

type QualityMeasurements = tuple[float | None, float | None, float, float, float]

_METRICS = {
    "integrated": re.compile(r"\bI:\s*(-?(?:inf|\d+(?:\.\d+)?))\s+LUFS", re.I),
    "peak": re.compile(r"\bPeak:\s*(-?(?:inf|\d+(?:\.\d+)?))\s+dBFS", re.I),
    "range": re.compile(r"\bLRA:\s*(\d+(?:\.\d+)?)\s+LU", re.I),
}
_SILENCE = re.compile(r"silence_(start|end):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
_EDGE_TOLERANCE = 0.05


@dataclass
class QualityCapture:
    """Retain summary values and silence edges without retaining frame logs."""

    metrics: dict[str, float | None] = field(default_factory=dict)
    first_start: float | None = None
    first_end: float | None = None
    last_start: float | None = None
    last_end: float | None = None

    def consume(self, line: str) -> None:
        for name, pattern in _METRICS.items():
            if match := pattern.search(line):
                raw = match.group(1).casefold()
                self.metrics[name] = None if raw in {"-inf", "inf"} else float(raw)
        for match in _SILENCE.finditer(line):
            instant = float(match.group(2))
            if match.group(1) == "start":
                if self.first_start is None:
                    self.first_start = instant
                self.last_start = instant
            else:
                if self.first_end is None:
                    self.first_end = instant
                self.last_end = instant

    def result(self, duration: float) -> QualityMeasurements:
        """Return the final summary and edge durations, rejecting missing metrics."""
        if self.metrics.keys() != _METRICS.keys() or self.metrics["range"] is None:
            raise ArtifactError("FFmpeg quality analysis omitted a required metric")
        leading = 0.0
        trailing = 0.0
        if self.first_start is not None and self.first_start <= _EDGE_TOLERANCE:
            end = self.first_end if self.first_end is not None else duration
            leading = min(duration, max(0.0, end))
        if self.last_start is not None and (
            self.last_end is None
            or self.last_end < self.last_start
            or self.last_end >= duration - _EDGE_TOLERANCE
        ):
            trailing = max(0.0, duration - self.last_start)
        return (
            self.metrics["integrated"],
            self.metrics["peak"],
            self.metrics["range"],
            leading,
            trailing,
        )
