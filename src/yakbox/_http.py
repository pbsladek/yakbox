"""Internal HTTP response helpers for bounded, decoded bodies."""

from __future__ import annotations

import httpx


def buffered_response(response: httpx.Response, content: bytes) -> httpx.Response:
    """Retain response metadata without decoding the buffered body twice."""
    headers = httpx.Headers(response.headers)
    for name in ("content-encoding", "content-length", "transfer-encoding"):
        headers.pop(name, None)
    return httpx.Response(
        response.status_code,
        headers=headers,
        content=content,
        request=response.request,
        extensions=response.extensions,
        history=response.history,
    )
