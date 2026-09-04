"""Process-local coordination for shared accelerator operations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from yakbox.errors import ValidationError


class AcceleratorLease:
    """Serialize TTS and analysis operations that share unified memory."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: str | None = None

    @property
    def owner(self) -> str | None:
        """Return the current operation label, or ``None`` while idle."""
        return self._owner

    @asynccontextmanager
    async def hold(self, owner: str) -> AsyncIterator[None]:
        """Hold the process-local accelerator lease for one named operation."""
        if not owner:
            raise ValidationError("Accelerator lease owner is required")
        async with self._lock:
            self._owner = owner
            try:
                yield
            finally:
                self._owner = None


@asynccontextmanager
async def accelerator_operation(
    lease: AcceleratorLease | None,
    *,
    owner: str,
    enabled: bool,
) -> AsyncIterator[None]:
    """Hold the shared lease only when an operation can use an accelerator."""
    if lease is None or not enabled:
        yield
        return
    async with lease.hold(owner):
        yield


__all__ = ["AcceleratorLease", "accelerator_operation"]
