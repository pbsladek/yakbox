from __future__ import annotations

from pathlib import Path

import pytest

from yakbox.errors import ValidationError
from yakbox.local_runtime import (
    LocalRuntimeOptions,
    _RuntimeServer,
)


def test_runtime_starts_without_loading_models(tmp_path: Path) -> None:
    server = _RuntimeServer(
        tmp_path,
        LocalRuntimeOptions(idle_timeout_seconds=30),
    )

    status = server.status()

    assert status["running"] is True
    assert status["tts_model_loaded"] is False
    assert status["whisper_models"] == 0
    assert status["conditioning_cache_entries"] == 0


def test_runtime_options_reject_unbounded_values() -> None:
    with pytest.raises(ValidationError, match="idle timeout"):
        LocalRuntimeOptions(idle_timeout_seconds=0)
    with pytest.raises(ValidationError, match="cache size"):
        LocalRuntimeOptions(conditioning_cache_size=0)
