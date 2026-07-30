from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def book_workspace(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "book.md").write_text(
        "# Chapter One\n\nA very short opening.\n\n"
        "<!-- yakbox:speech:pause ms=50 -->\n\n"
        "A short ending.\n",
        encoding="utf-8",
    )
    (tmp_path / "pronunciations.toml").write_text(
        "schema_version = 1\n\n"
        "[[terms]]\n"
        'written = "short"\n'
        'spoken = "brief"\n'
        'status = "approved"\n'
        "enabled = true\n",
        encoding="utf-8",
    )
    (tmp_path / "yakbox.toml").write_text(
        '"$schema" = '
        '"https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"\n'
        'schema_version = 1\nsources = ["source/book.md"]\n'
        'pronunciations = "pronunciations.toml"\n\n'
        "[book]\n"
        'title = "Test Book"\n'
        'author = "Test Author"\n'
        'narrator = "Test Narrator"\n\n'
        "[voices.narrator]\n"
        'display_name = "Narrator"\n'
        'rights_basis = "not_applicable"\n\n'
        "[profiles.default]\n"
        'backend = "fake"\n'
        'voice = "narrator"\n'
        "sample_rate = 16000\n\n"
        "[targets.default]\n"
        'profile = "default"\n'
        'output_root = "build/yakbox"\n'
        "chunk_chars = 100\n"
        "mastering = true\n"
        "wav_sample_rate = 44100\n"
        'mp3_bitrate = "96k"\n'
        "m4b = true\n",
        encoding="utf-8",
    )
    return tmp_path
