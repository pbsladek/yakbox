from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
from typing import cast

import pytest
import yaml

from yakbox.audiobook import load_manifest, normalize_sources, plan_audiobook
from yakbox.textutils import iter_batch_rows

pytestmark = pytest.mark.performance

_BASELINES = cast(
    dict[str, dict[str, float]],
    yaml.safe_load(
        (Path(__file__).parent / "baselines.yaml").read_text(encoding="utf-8")
    ),
)


def test_planner_300_chapter_time_and_peak_memory(tmp_path: Path) -> None:
    source = tmp_path / "book.md"
    source.write_text(
        "\n".join(
            f"# Chapter {index}\n\nA short sentence for chapter {index}.\n"
            for index in range(1, 301)
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "yakbox.toml"
    manifest_path.write_text(
        '"$schema" = "https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"\n'
        'schema_version = 1\nsources = ["book.md"]\n\n'
        '[book]\ntitle = "Benchmark"\n\n'
        '[voices.narrator]\ndisplay_name = "Narrator"\n\n'
        '[profiles.default]\nbackend = "fake"\nvoice = "narrator"\n\n'
        '[targets.default]\nprofile = "default"\noutput_root = "build/yakbox"\n',
        encoding="utf-8",
    )
    baseline = _BASELINES["planner_300_chapters"]

    tracemalloc.start()
    started = time.perf_counter()
    manifest = load_manifest(manifest_path)
    document = normalize_sources(manifest.sources)
    plan = plan_audiobook(manifest, document)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(plan.nodes) == 1_200
    assert elapsed <= baseline["max_seconds"]
    assert peak <= baseline["max_peak_bytes"]


def test_streaming_batch_reader_time_and_peak_memory(tmp_path: Path) -> None:
    script = tmp_path / "large.txt"
    script.write_text(
        "".join(f"{index:05d} " + ("x" * 1_900) + "\n" for index in range(10_000)),
        encoding="utf-8",
    )
    baseline = _BASELINES["stream_10000_rows"]

    tracemalloc.start()
    started = time.perf_counter()
    characters = sum(len(row.text) for row in iter_batch_rows(script))
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert characters > 19_000_000
    assert elapsed <= baseline["max_seconds"]
    assert peak <= baseline["max_peak_bytes"]
