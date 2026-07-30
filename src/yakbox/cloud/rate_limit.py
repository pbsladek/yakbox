from __future__ import annotations

import asyncio
import time


class RateLimitGate:
    def __init__(self) -> None:
        self._deadline = 0.0
        self._lock = asyncio.Lock()

    async def extend(self, delay: float) -> None:
        if delay <= 0:
            return
        async with self._lock:
            self._deadline = max(self._deadline, time.monotonic() + delay)

    async def wait(self) -> None:
        while True:
            async with self._lock:
                delay = self._deadline - time.monotonic()
            if delay <= 0:
                return
            await asyncio.sleep(delay)
