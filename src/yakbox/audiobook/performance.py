"""Durable, privacy-safe performance summaries for audiobook builds."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from yakbox._files import atomic_write_json
from yakbox.audiobook.journal import RunJournal
from yakbox.contracts import runtime_metadata

if TYPE_CHECKING:
    from yakbox.audiobook.planner import BuildPlan
    from yakbox.speech.analysis_scheduler import AnalysisPerformanceCollector


def write_performance_report(
    run_directory: Path,
    *,
    journal: RunJournal,
    plan: BuildPlan,
    whisper_cache_directory: Path,
    pending_synthesis_chunks: int,
    reusable_synthesis_chunks: int,
    estimated_model_loads: int,
    speech_analysis: AnalysisPerformanceCollector | None = None,
) -> Path:
    """Summarize wall time, node stages, and planned cache reuse for one run."""
    events = journal.events()
    node_stages = {node.id: node.stage.value for node in plan.nodes}
    starts: dict[str, datetime] = {}
    stage_seconds: defaultdict[str, float] = defaultdict(float)
    node_seconds: dict[str, float] = {}
    for event in events:
        node_id = event.get("node_id")
        timestamp = event.get("timestamp")
        if not isinstance(node_id, str) or not isinstance(timestamp, str):
            continue
        instant = datetime.fromisoformat(timestamp)
        if event.get("event") == "node_started":
            starts[node_id] = instant
        elif event.get("event") in {"node_completed", "node_failed"}:
            started = starts.get(node_id)
            if started is not None:
                duration = max(0.0, (instant - started).total_seconds())
                node_seconds[node_id] = duration
                stage_seconds[node_stages.get(node_id, "unknown")] += duration
    timestamps = tuple(
        datetime.fromisoformat(value)
        for event in events
        if isinstance((value := event.get("timestamp")), str)
    )
    reused_nodes = sum(event.get("event") == "node_reused" for event in events)
    run_started_at = timestamps[0] if timestamps else None
    candidate_counts = _candidate_counts(
        plan,
        since=run_started_at,
    )
    whisper_hits, whisper_misses = _whisper_counts(plan, since=run_started_at)
    whisper_entries = _json_entry_count(whisper_cache_directory)
    path = run_directory / "performance.json"
    atomic_write_json(
        path,
        {
            **runtime_metadata("audiobook-performance"),
            "run_id": journal.run_id,
            "wall_seconds": (
                max(0.0, (timestamps[-1] - timestamps[0]).total_seconds())
                if len(timestamps) > 1
                else 0.0
            ),
            "node_seconds": node_seconds,
            "stage_seconds": dict(sorted(stage_seconds.items())),
            "nodes": {
                "executed": len(node_seconds),
                "reused": reused_nodes,
                "failed": sum(event.get("event") == "node_failed" for event in events),
            },
            "synthesis_cache": {
                "pending_chunks": pending_synthesis_chunks,
                "reusable_chunks": reusable_synthesis_chunks,
                "estimated_model_loads": estimated_model_loads,
            },
            "short_utterance_candidates": {
                **candidate_counts,
            },
            "whisper_cache": {
                "hits": whisper_hits,
                "misses": whisper_misses,
                "entries": whisper_entries,
            },
            "speech_analysis": (
                speech_analysis.to_dict()
                if speech_analysis is not None
                else {"engines": {}}
            ),
        },
    )
    return path


def _candidate_counts(
    plan: BuildPlan,
    *,
    since: datetime | None,
) -> dict[str, int]:
    counts = {
        "generated": 0,
        "generation_reused": 0,
        "extraction_reused": 0,
        "evaluation_reused": 0,
    }
    roots = {
        node.output.parents[1] / "qa" / "short-utterances" / node.chapter_id
        for node in plan.nodes
        if node.stage.value == "synthesize"
    }
    for root in roots:
        preflight = _recent_json(root / "preflight.json", since=since)
        results = preflight.get("results") if preflight is not None else None
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict) or not isinstance(
                (report_value := result.get("qa_report")), str
            ):
                continue
            raw = _recent_json(Path(report_value), since=since)
            candidates = raw.get("candidates") if raw is not None else None
            if not isinstance(candidates, list):
                continue
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                counts[
                    "generation_reused"
                    if item.get("generation_cache_hit") is True
                    else "generated"
                ] += 1
                counts["extraction_reused"] += item.get("extraction_cache_hit") is True
                counts["evaluation_reused"] += item.get("evaluation_cache_hit") is True
    return counts


def _whisper_counts(
    plan: BuildPlan,
    *,
    since: datetime | None,
) -> tuple[int, int]:
    reports = {
        node.output for node in plan.nodes if node.stage.value == "verify_manuscript"
    }
    reports.update(
        node.output.parents[1]
        / "qa"
        / "changed-regions"
        / f"{node.chapter_id}.chunks.json"
        for node in plan.nodes
        if node.stage.value == "synthesize"
    )
    hits = 0
    misses = 0
    for path in reports:
        raw = _recent_json(path, since=since)
        if raw is None:
            continue
        caches: list[object] = [raw.get("alignment_cache")]
        results = raw.get("results")
        if isinstance(results, list):
            caches.extend(
                item.get("alignment_cache")
                for item in results
                if isinstance(item, dict)
            )
        for cache in caches:
            if isinstance(cache, dict):
                hit_value = cache.get("hits", 0)
                miss_value = cache.get("misses", 0)
                if isinstance(hit_value, int) and not isinstance(hit_value, bool):
                    hits += hit_value
                if isinstance(miss_value, int) and not isinstance(miss_value, bool):
                    misses += miss_value
    return hits, misses


def _recent_json(path: Path, *, since: datetime | None) -> dict[str, object] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    timestamp = raw.get("timestamp")
    if since is not None and (
        not isinstance(timestamp, str) or datetime.fromisoformat(timestamp) < since
    ):
        return None
    return raw


def _json_entry_count(root: Path) -> int:
    return sum(1 for _ in root.glob("*/*.json")) if root.is_dir() else 0


__all__ = ["write_performance_report"]
