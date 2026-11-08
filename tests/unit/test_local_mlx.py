from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from yakbox.errors import BackendUnavailableError
from yakbox.local_mlx import _local_downloads


def test_local_mlx_download_bridge_resolves_auxiliary_weights_and_restores_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Mock(side_effect=AssertionError("Network callback must not run"))
    hub = SimpleNamespace(hf_hub_download=original)
    monkeypatch.setattr("yakbox.local_mlx.importlib.import_module", lambda _name: hub)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with _local_downloads(tmp_path):
        assert hub.hf_hub_download(
            repo_id="mlx-community/S3TokenizerV2", filename="model.safetensors"
        ) == str(tmp_path / "s3tokenizer/model.safetensors")
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        with pytest.raises(BackendUnavailableError, match="complete local snapshot"):
            hub.hf_hub_download(repo_id="other/model", filename="weights.safetensors")
    assert hub.hf_hub_download is original
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    original.assert_not_called()


def test_local_mlx_download_bridge_restores_state_after_failed_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Mock()
    hub = SimpleNamespace(hf_hub_download=original)
    monkeypatch.setattr("yakbox.local_mlx.importlib.import_module", lambda _name: hub)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    with pytest.raises(RuntimeError, match="failed load"), _local_downloads(tmp_path):
        raise RuntimeError("failed load")
    assert hub.hf_hub_download is original
    assert "HF_HUB_OFFLINE" not in os.environ
    assert os.environ["TRANSFORMERS_OFFLINE"] == "0"
