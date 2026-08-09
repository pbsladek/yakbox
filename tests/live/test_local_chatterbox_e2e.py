from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tomllib
import uuid
import warnings
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

import yakbox.audiobook.build as build_module
from tests.narration_review import (
    narration_review_issues,
    narration_review_template,
)
from yakbox._files import sha256_file
from yakbox.audio.inspect import AudioQualityPolicy, inspect_audio
from yakbox.audiobook.manifest import load_manifest
from yakbox.audiobook.planner import plan_audiobook
from yakbox.audiobook.sources import (
    SpeechSegment,
    apply_pronunciations,
    normalize_sources,
)
from yakbox.speech.waves import boundary_pause_milliseconds

pytestmark = pytest.mark.live

ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "local-chatterbox-e2e"
VOICE_ASSETS = ROOT / "examples" / "local-chatterbox" / "voices"


@dataclass(frozen=True, slots=True)
class AuditionPassage:
    id: str
    text_file: str


@dataclass(frozen=True, slots=True)
class LocalE2EResult:
    workspace: Path
    plan: dict[str, object]
    auditions: dict[str, dict[str, object]]
    build: dict[str, object]
    resumed_build: dict[str, object]
    report_path: Path
    review_path: Path
    structured_review_path: Path


@pytest.fixture(scope="module")
def local_e2e(tmp_path_factory: pytest.TempPathFactory) -> LocalE2EResult:
    _require_opt_in()
    workspace = _prepare_workspace(tmp_path_factory)
    timeout = _bounded_timeout()

    validate = _run_cli(workspace, timeout, "validate")
    plan = _run_cli(workspace, timeout, "plan")
    profiles = _qa_profiles(workspace)
    profile_arguments = tuple(
        argument for profile in profiles for argument in ("--profile", profile)
    )
    auditions = {
        passage.id: _run_cli(
            workspace,
            timeout,
            "audition",
            *profile_arguments,
            "--text-file",
            passage.text_file,
        )
        for passage in _qa_audition_passages(workspace)
    }
    build = _run_cli(workspace, timeout, "build", "--no-progress")
    resumed = _run_cli(workspace, timeout, "build", "--no-progress")

    report_path, review_path, structured_review_path = _write_qa_package(
        workspace,
        validate=validate,
        plan=plan,
        auditions=auditions,
        build=build,
        resumed=resumed,
    )
    return LocalE2EResult(
        workspace=workspace,
        plan=plan,
        auditions=auditions,
        build=build,
        resumed_build=resumed,
        report_path=report_path,
        review_path=review_path,
        structured_review_path=structured_review_path,
    )


def test_cli_plan_exposes_narration_boundaries(local_e2e: LocalE2EResult) -> None:
    qa = _qa_config(local_e2e.workspace)
    chunks = _synthesis_chunks(local_e2e.plan)
    boundaries = {str(chunk["boundary"]) for chunk in chunks}

    assert set(cast(list[str], qa["required_boundaries"])) <= boundaries
    assert all(
        _integer(chunk["characters"]) <= _integer(qa["max_chunk_characters"])
        for chunk in chunks
    )
    assert all(
        cast(dict[str, object], chunk["source"])["start_line"] for chunk in chunks
    )


def test_cli_audition_renders_every_configured_voice(
    local_e2e: LocalE2EResult,
) -> None:
    qa = _qa_config(local_e2e.workspace)
    expected_profiles = set(cast(list[str], qa["profiles"]))
    passage_ids = {item.id for item in _qa_audition_passages(local_e2e.workspace)}

    assert set(local_e2e.auditions) == passage_ids
    all_hashes: list[str] = []
    for audition in local_e2e.auditions.values():
        artifacts = _audition_artifacts(local_e2e.workspace, audition)
        assert {path.stem for path in artifacts} == expected_profiles
        assert all(path.is_file() and path.stat().st_size > 44 for path in artifacts)
        all_hashes.extend(sha256_file(path) for path in artifacts)
    assert len(set(all_hashes)) == len(all_hashes)


def test_cli_build_completes_full_pipeline_and_reuses_it(
    local_e2e: LocalE2EResult,
) -> None:
    first = _command_data(local_e2e.build)
    resumed = _command_data(local_e2e.resumed_build)
    output = local_e2e.workspace / "artifacts"

    assert first["status"] == "complete"
    assert resumed["status"] == "complete"
    assert len(cast(list[str], resumed["reused_nodes"])) == 4
    assert tuple((output / "raw").glob("*.wav"))
    assert tuple((output / "mastered").glob("*.wav"))
    assert tuple((output / "release" / "mp3").glob("*.mp3"))
    inspections = tuple((output / "reports").glob("*.inspection.json"))
    assert len(inspections) == 1
    assert all(
        item["valid"]
        for item in cast(
            list[dict[str, object]],
            json.loads(inspections[0].read_text(encoding="utf-8"))["inspections"],
        )
    )


def test_qa_package_preserves_audio_metrics_and_human_review_contract(
    local_e2e: LocalE2EResult,
) -> None:
    report = cast(
        dict[str, object],
        json.loads(local_e2e.report_path.read_text(encoding="utf-8")),
    )
    qa = _qa_config(local_e2e.workspace)
    audio = cast(list[dict[str, object]], report["audio"])
    automated = cast(dict[str, object], qa["automated"])
    manual = cast(dict[str, object], qa["manual"])
    dimensions = (
        *cast(list[dict[str, object]], manual["voice_dimensions"]),
        *cast(list[dict[str, object]], manual["dialogue_dimensions"]),
        *cast(list[dict[str, object]], manual["chapter_dimensions"]),
    )
    passages = _qa_audition_passages(local_e2e.workspace)

    assert report["schema_version"] == 1
    assert report["manual_review_status"] == "not_reviewed"
    assert len(audio) == len(cast(list[str], qa["profiles"])) * len(passages) + 3
    assert all(
        _number(automated["minimum_clip_seconds"])
        <= _number(item["duration_seconds"])
        <= _number(automated["maximum_clip_seconds"])
        for item in audio
    )
    assert all(
        item["integrated_loudness_lufs"] is not None
        and item["true_peak_dbfs"] is not None
        and _number(item["true_peak_dbfs"])
        <= _number(automated["maximum_true_peak_dbfs"])
        for item in audio
    )
    assert all(
        _number(automated["minimum_words_per_minute"])
        <= _number(item["estimated_words_per_minute"])
        <= _number(automated["maximum_words_per_minute"])
        for item in audio
    )
    assert all(
        _number(automated["minimum_loudness_lufs"])
        <= _number(item["integrated_loudness_lufs"])
        <= _number(automated["maximum_loudness_lufs"])
        for item in audio
    )
    assert all(
        _number(item[edge]) <= _number(automated["maximum_edge_silence_seconds"])
        for item in audio
        for edge in ("leading_silence_seconds", "trailing_silence_seconds")
    )
    timeline = cast(list[dict[str, object]], report["chunk_timeline"])
    assembly = cast(dict[str, object], report["assembly"])
    silence_windows = cast(list[dict[str, object]], report["silence_windows"])
    audition_inputs = cast(dict[str, dict[str, object]], report["audition_inputs"])
    assert len(timeline) == len(_synthesis_chunks(local_e2e.plan))
    assert _number(assembly["duration_delta_ms"]) <= 1.0
    assert any(
        item["boundary"] == "sentence" and item["pause_after_ms"] == 100
        for item in timeline
    )
    assert any(
        item["boundary"] == "paragraph" and item["pause_after_ms"] == 250
        for item in timeline
    )
    assert any(
        item["boundary"] == "explicit_pause"
        and round(_number(item["audio_duration_seconds"]) * 1_000) == 450
        for item in timeline
    )
    assert {item["boundary"] for item in silence_windows} == {
        "sentence",
        "paragraph",
        "explicit_pause",
    }
    assert all(item["all_samples_silent"] is True for item in silence_windows)
    assert all(
        _number(item[edge]) <= _number(automated["maximum_splice_step_dbfs"])
        for item in silence_windows
        for edge in ("start_step_dbfs", "end_step_dbfs")
    )
    speech_timeline = [item for item in timeline if item["kind"] == "speech"]
    assert all(
        _number(item["estimated_words_per_minute"])
        <= _number(automated["maximum_words_per_minute"])
        and (
            _integer(item["spoken_words"])
            < _integer(automated["minimum_chunk_words_for_pace"])
            or _number(item["estimated_words_per_minute"])
            >= _number(automated["minimum_words_per_minute"])
        )
        for item in speech_timeline
    )
    assert set(audition_inputs) == {item.id for item in passages}
    assert all(
        item["pronunciation_rules_changed_text"] is False
        and _integer(item["normalized_spoken_words"]) == _integer(item["written_words"])
        for item in audition_inputs.values()
    )
    assert local_e2e.review_path.is_file()
    review = local_e2e.review_path.read_text(encoding="utf-8")
    assert all(str(dimension["id"]) in review for dimension in dimensions)
    assert local_e2e.structured_review_path.is_file()
    structured = cast(
        dict[str, object],
        tomllib.loads(local_e2e.structured_review_path.read_text(encoding="utf-8")),
    )
    assert not narration_review_issues(
        structured,
        qa,
        report_sha256=sha256_file(local_e2e.report_path),
    )
    assert structured["status"] == "pending"


def _require_opt_in() -> None:
    if os.environ.get("YAKBOX_RUN_LOCAL_E2E") != "1":
        pytest.skip("set YAKBOX_RUN_LOCAL_E2E=1 to run local narration E2E QA")
    if importlib.util.find_spec("chatterbox") is None:
        pytest.fail('local Chatterbox is not installed; install "yakbox[local]"')
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            pytest.fail(f"{executable} is required for local narration E2E QA")


def _prepare_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    configured = os.environ.get("YAKBOX_LOCAL_E2E_OUTPUT")
    base = (
        Path(configured).expanduser().resolve()
        if configured
        else tmp_path_factory.mktemp("local-chatterbox-e2e")
    )
    base.mkdir(parents=True, exist_ok=True)
    workspace = base / f"run-{uuid.uuid4().hex[:12]}"
    shutil.copytree(FIXTURE, workspace)
    shutil.copytree(VOICE_ASSETS, workspace / "voices")
    device = os.environ.get("YAKBOX_LIVE_LOCAL_DEVICE", "auto").strip().casefold()
    if re.fullmatch(r"[a-z0-9_.:-]+", device) is None:
        pytest.fail("YAKBOX_LIVE_LOCAL_DEVICE contains unsupported characters")
    manifest = workspace / "yakbox.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            'device = "auto"', f'device = "{device}"'
        ),
        encoding="utf-8",
    )
    return workspace


def _bounded_timeout() -> float:
    raw = os.environ.get("YAKBOX_LOCAL_E2E_TIMEOUT_SECONDS", "1800")
    try:
        timeout = float(raw)
    except ValueError:
        pytest.fail("YAKBOX_LOCAL_E2E_TIMEOUT_SECONDS must be a number")
    if not 60 <= timeout <= 3_600:
        pytest.fail("YAKBOX_LOCAL_E2E_TIMEOUT_SECONDS must be between 60 and 3600")
    return timeout


def _run_cli(workspace: Path, timeout: float, *arguments: str) -> dict[str, object]:
    command = [sys.executable, "-m", "yakbox", "--json", *arguments]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed current-interpreter E2E argv
            command,
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"Yakbox CLI timed out: {' '.join(command)}\n{error.stderr or ''}")
    if completed.returncode != 0:
        pytest.fail(
            f"Yakbox CLI failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    try:
        payload = cast(dict[str, object], json.loads(completed.stdout.strip()))
    except json.JSONDecodeError:
        pytest.fail(f"Yakbox CLI returned invalid JSON: {completed.stdout[-4000:]}")
    assert payload["status"] == "ok"
    assert payload["exit_code"] == 0
    return payload


def _command_data(payload: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], payload["data"])


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        pytest.fail(f"Expected integer QA value, received {value!r}")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        pytest.fail(f"Expected numeric QA value, received {value!r}")
    return float(value)


def _qa_config(workspace: Path) -> dict[str, object]:
    return cast(
        dict[str, object],
        tomllib.loads((workspace / "qa.toml").read_text(encoding="utf-8")),
    )


def _qa_profiles(workspace: Path) -> tuple[str, ...]:
    value = _qa_config(workspace).get("profiles")
    if not isinstance(value, list) or not value:
        pytest.fail("qa.toml profiles must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        pytest.fail("qa.toml profiles must contain non-empty strings")
    profiles = cast(tuple[str, ...], tuple(value))
    if len(set(profiles)) != len(profiles):
        pytest.fail("qa.toml profiles must not contain duplicates")
    return profiles


def _qa_audition_passages(workspace: Path) -> tuple[AuditionPassage, ...]:
    value = _qa_config(workspace).get("audition_passages")
    if not isinstance(value, list) or not value:
        pytest.fail("qa.toml audition_passages must be a non-empty list")
    passages: list[AuditionPassage] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            pytest.fail(f"qa.toml audition_passages[{index}] must be a table")
        identifier = item.get("id")
        text_file = item.get("text_file")
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"[a-z0-9-]+", identifier) is None
        ):
            pytest.fail(f"qa.toml audition_passages[{index}].id is invalid")
        if not isinstance(text_file, str) or Path(text_file).name != text_file:
            pytest.fail(f"qa.toml audition_passages[{index}].text_file is invalid")
        if not (workspace / text_file).is_file():
            pytest.fail(f"qa.toml audition passage does not exist: {text_file}")
        passages.append(AuditionPassage(identifier, text_file))
    if len({item.id for item in passages}) != len(passages):
        pytest.fail("qa.toml audition passage ids must not contain duplicates")
    return tuple(passages)


def _synthesis_chunks(plan: dict[str, object]) -> list[dict[str, object]]:
    nodes = cast(list[dict[str, object]], _command_data(plan)["nodes"])
    synthesis = next(node for node in nodes if node["stage"] == "synthesize")
    return cast(list[dict[str, object]], synthesis["chunks"])


def _audition_artifacts(
    workspace: Path, audition: dict[str, object]
) -> tuple[Path, ...]:
    records = cast(list[dict[str, object]], _command_data(audition)["artifacts"])
    return tuple(workspace / str(record["path"]) for record in records)


def _write_qa_package(
    workspace: Path,
    *,
    validate: dict[str, object],
    plan: dict[str, object],
    auditions: dict[str, dict[str, object]],
    build: dict[str, object],
    resumed: dict[str, object],
) -> tuple[Path, Path, Path]:
    qa = _qa_config(workspace)
    manual = cast(dict[str, object], qa["manual"])
    chapter_profile = str(manual["chapter_profile"])
    passages = _qa_audition_passages(workspace)
    audio_paths = (
        *(
            path
            for audition in auditions.values()
            for path in _audition_artifacts(workspace, audition)
        ),
        *sorted((workspace / "artifacts" / "raw").glob("*.wav")),
        *sorted((workspace / "artifacts" / "mastered").glob("*.wav")),
        *sorted((workspace / "artifacts" / "release" / "mp3").glob("*.mp3")),
    )
    spoken_words = _spoken_word_count(workspace)
    audition_contexts = _audition_contexts(workspace, auditions, passages)
    audio = [
        _audio_evidence(
            workspace,
            path,
            spoken_words=spoken_words,
            audition_contexts=audition_contexts,
            chapter_profile=chapter_profile,
        )
        for path in audio_paths
    ]
    chunk_timeline, assembly = _chunk_timeline(workspace)
    silence_windows = _silence_windows(workspace, chunk_timeline)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "python_version": sys.version.split()[0],
        "manual_review_status": "not_reviewed",
        "qa_config": qa,
        "commands": {
            "validate": validate,
            "plan": plan,
            "auditions": auditions,
            "build": build,
            "resumed_build": resumed,
        },
        "chunks": _synthesis_chunks(plan),
        "chunk_timeline": chunk_timeline,
        "assembly": assembly,
        "silence_windows": silence_windows,
        "audition_inputs": {
            passage.id: _audition_input_evidence(workspace, passage.text_file)
            for passage in passages
        },
        "audio": audio,
    }
    package = workspace / "qa"
    package.mkdir(parents=True, exist_ok=True)
    report_path = package / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    review_path = package / "listening-review.md"
    review_path.write_text(
        _listening_review(workspace, qa, audio, chunk_timeline, silence_windows),
        encoding="utf-8",
    )
    structured_review_path = package / "listening-review.toml"
    structured_review_path.write_text(
        narration_review_template(qa, report_sha256=sha256_file(report_path)),
        encoding="utf-8",
    )
    warnings.warn(
        f"Local Chatterbox E2E QA package: {package}",
        UserWarning,
        stacklevel=2,
    )
    return report_path, review_path, structured_review_path


def _audition_contexts(
    workspace: Path,
    auditions: dict[str, dict[str, object]],
    passages: tuple[AuditionPassage, ...],
) -> dict[Path, tuple[str, str, int]]:
    contexts: dict[Path, tuple[str, str, int]] = {}
    for passage in passages:
        words = _audition_word_count(workspace, passage.text_file)
        records = cast(
            list[dict[str, object]],
            _command_data(auditions[passage.id])["artifacts"],
        )
        for record in records:
            path = workspace / str(record["path"])
            contexts[path] = (passage.id, str(record["logical_voice"]), words)
    return contexts


def _spoken_word_count(workspace: Path) -> int:
    document = normalize_sources(
        (workspace / "source" / "book.md",),
        pronunciations=workspace / "pronunciations.toml",
    )
    return sum(
        len(item.text.split())
        for chapter in document.chapters
        for item in chapter.segments
        if isinstance(item, SpeechSegment)
    )


def _audio_evidence(
    workspace: Path,
    path: Path,
    *,
    spoken_words: int,
    audition_contexts: dict[Path, tuple[str, str, int]],
    chapter_profile: str,
) -> dict[str, object]:
    inspection = inspect_audio(path, quality=AudioQualityPolicy())
    value = inspection.to_dict(root=workspace)
    context = audition_contexts.get(path)
    words = context[2] if context is not None else spoken_words
    value.update(
        {
            "kind": _audio_kind(path),
            "sha256": sha256_file(path),
            "spoken_words": words,
            "estimated_words_per_minute": round(
                words * 60 / inspection.duration_seconds, 2
            ),
            "logical_voice": context[1] if context is not None else chapter_profile,
        }
    )
    if context is not None:
        value["audition_passage"] = context[0]
    return value


def _audition_word_count(workspace: Path, text_file: str) -> int:
    raw = (workspace / text_file).read_text(encoding="utf-8")
    normalized = apply_pronunciations(raw, workspace / "pronunciations.toml")
    return len(normalized.split())


def _audition_input_evidence(workspace: Path, text_file: str) -> dict[str, object]:
    raw = (workspace / text_file).read_text(encoding="utf-8").strip()
    normalized = apply_pronunciations(raw, workspace / "pronunciations.toml")
    return {
        "written_characters": len(raw),
        "written_words": len(raw.split()),
        "written_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "normalized_characters": len(normalized),
        "normalized_spoken_words": len(normalized.split()),
        "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "pronunciation_rules_changed_text": normalized != raw,
    }


def _audio_kind(path: Path) -> str:
    for kind in ("auditions", "raw", "mastered", "mp3"):
        if kind in path.parts:
            return "delivery_mp3" if kind == "mp3" else kind.removesuffix("s")
    return "unknown"


def _chunk_timeline(
    workspace: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest = load_manifest(workspace / "yakbox.toml")
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
    )
    plan = plan_audiobook(manifest, document)
    node = next(item for item in plan.nodes if item.stage == "synthesize")
    default_profile = manifest.profile(plan.profile)
    timeline: list[dict[str, object]] = []
    cursor = 0.0
    for index, (text, source, boundary, route) in enumerate(
        zip(
            node.chunks,
            node.chunk_sources,
            node.chunk_boundaries,
            node.chunk_routes,
            strict=True,
        ),
        start=1,
    ):
        explicit_pause_ms = _explicit_pause_milliseconds(text)
        cache_sha256: str | None = None
        if explicit_pause_ms is not None:
            duration = explicit_pause_ms / 1_000
        else:
            profile = build_module._profile_from_route(
                manifest,
                route,
                default_profile,
            )
            (
                voice,
                sample_rate,
                project,
                use_hd,
                reference_audio,
                chatterbox,
            ) = build_module._resolved_speech(profile, manifest)
            request = build_module._new_speech_request(
                text,
                profile=profile,
                voice=voice,
                sample_rate=sample_rate,
                project=project,
                use_hd=use_hd,
                reference_audio=reference_audio,
                chatterbox=build_module._chunk_chatterbox(
                    chatterbox,
                    chapter_id=node.chapter_id,
                    chunk_index=index,
                    text=text,
                ),
            )
            fingerprint = build_module._speech_request_fingerprint(request)
            cached = build_module._cached_chunk(workspace, fingerprint)
            if cached is None:
                pytest.fail(f"Built speech chunk {index} is missing from the cache")
            duration = _wav_duration(cached)
            cache_sha256 = sha256_file(cached)
        next_is_explicit = index < len(node.chunks) and (
            _explicit_pause_milliseconds(node.chunks[index]) is not None
        )
        pause_after_ms = (
            0
            if index == len(node.chunks)
            or explicit_pause_ms is not None
            or next_is_explicit
            else boundary_pause_milliseconds(boundary)
        )
        end = cursor + duration
        timeline.append(
            {
                "index": index,
                "kind": "explicit_pause" if explicit_pause_ms is not None else "speech",
                "speaker": route.speaker,
                "profile": route.profile,
                "characters": len(text),
                "boundary": boundary,
                "source": {
                    "path": source.path.relative_to(workspace).as_posix(),
                    "start_line": source.start_line,
                    "end_line": source.end_line,
                },
                "start_seconds": round(cursor, 6),
                "end_seconds": round(end, 6),
                "audio_duration_seconds": round(duration, 6),
                "spoken_words": 0
                if explicit_pause_ms is not None
                else len(text.split()),
                "estimated_words_per_minute": None
                if explicit_pause_ms is not None
                else round(len(text.split()) * 60 / duration, 2),
                "pause_after_ms": pause_after_ms,
                "next_start_seconds": round(end + pause_after_ms / 1_000, 6),
                "cache_sha256": cache_sha256,
            }
        )
        cursor = end + pause_after_ms / 1_000
    raw = next((workspace / "artifacts" / "raw").glob("*.wav"))
    actual_duration = _wav_duration(raw)
    return timeline, {
        "raw_path": raw.relative_to(workspace).as_posix(),
        "planned_duration_seconds": round(cursor, 6),
        "actual_duration_seconds": round(actual_duration, 6),
        "duration_delta_ms": round(abs(actual_duration - cursor) * 1_000, 6),
    }


def _explicit_pause_milliseconds(text: str) -> int | None:
    match = re.fullmatch(r"__YAKBOX_PAUSE_MS=(\d+)__", text)
    return int(match.group(1)) if match is not None else None


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def _silence_windows(
    workspace: Path, chunk_timeline: list[dict[str, object]]
) -> list[dict[str, object]]:
    raw = next((workspace / "artifacts" / "raw").glob("*.wav"))
    evidence: list[dict[str, object]] = []
    with wave.open(str(raw), "rb") as audio:
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        silent_sample = b"\x80" if sample_width == 1 else b"\0" * sample_width
        for item in chunk_timeline:
            explicit = item["kind"] == "explicit_pause"
            milliseconds = (
                round(_number(item["audio_duration_seconds"]) * 1_000)
                if explicit
                else _integer(item["pause_after_ms"])
            )
            if milliseconds == 0:
                continue
            start_seconds = (
                _number(item["start_seconds"])
                if explicit
                else _number(item["end_seconds"])
            )
            start_frame = round(start_seconds * sample_rate)
            frames = round(milliseconds * sample_rate / 1_000)
            audio.setpos(start_frame)
            content = audio.readframes(frames)
            expected = silent_sample * channels * frames
            evidence.append(
                {
                    "chunk_index": item["index"],
                    "boundary": item["boundary"],
                    "start_seconds": round(start_seconds, 6),
                    "milliseconds": milliseconds,
                    "frames": frames,
                    "all_samples_silent": content == expected,
                    "start_step_dbfs": _splice_step_dbfs(audio, start_frame),
                    "end_step_dbfs": _splice_step_dbfs(audio, start_frame + frames),
                }
            )
    return evidence


def _splice_step_dbfs(audio: wave.Wave_read, frame: int) -> float:
    if frame <= 0 or frame >= audio.getnframes():
        return -120.0
    audio.setpos(frame - 1)
    content = audio.readframes(2)
    width = audio.getsampwidth()
    channels = audio.getnchannels()
    samples = _pcm_samples(content, width)
    difference = max(
        abs(samples[channel] - samples[channel + channels])
        for channel in range(channels)
    )
    if difference == 0:
        return -120.0
    full_scale = (1 << (width * 8 - 1)) - 1
    return round(20 * math.log10(difference / full_scale), 2)


def _pcm_samples(content: bytes, width: int) -> tuple[int, ...]:
    signed = width != 1
    offset = 128 if width == 1 else 0
    return tuple(
        int.from_bytes(content[index : index + width], "little", signed=signed) - offset
        for index in range(0, len(content), width)
    )


def _listening_review(
    workspace: Path,
    qa: dict[str, object],
    audio: list[dict[str, object]],
    chunk_timeline: list[dict[str, object]],
    silence_windows: list[dict[str, object]],
) -> str:
    manual = cast(dict[str, object], qa["manual"])
    voice_dimensions = cast(list[dict[str, object]], manual["voice_dimensions"])
    dialogue_dimensions = cast(list[dict[str, object]], manual["dialogue_dimensions"])
    chapter_dimensions = cast(list[dict[str, object]], manual["chapter_dimensions"])
    profiles = cast(list[str], qa["profiles"])
    lines = [
        "# Local Chatterbox narration review",
        "",
        "Reviewer: **TODO**  ",
        "Reviewed at: **TODO**  ",
        "Status: **not reviewed**",
        "",
        "Use headphones. Listen once without reading the corpus, then once while",
        "following `source/book.md`. Score each dimension from",
        f"{manual['scale_min']} (unusable) to {manual['scale_max']} (excellent).",
        f"The configured passing score is {manual['passing_score']} per dimension.",
        "Use this Markdown file as the listening guide, then record the durable",
        "result in `listening-review.toml`. The TOML file is bound to this run's",
        "technical report and is the artifact checked by the approval test.",
        "",
        "## Audio",
        "",
    ]
    lines.extend(
        f"- [{_audio_review_label(item)}](../{item['path']})" for item in audio
    )
    lines.extend(["", "## Narration audition dimensions", ""])
    lines.extend(f"- `{item['id']}`: {item['prompt']}" for item in voice_dimensions)
    lines.extend(["", "## Narration audition scores", ""])
    for profile in profiles:
        lines.extend([f"### {profile}", ""])
        lines.extend(
            f"- `{item['id']}`: TODO / {manual['scale_max']}"
            for item in voice_dimensions
        )
        lines.extend(["- Notes: TODO", ""])
    lines.extend(["## Dialogue audition dimensions", ""])
    lines.extend(f"- `{item['id']}`: {item['prompt']}" for item in dialogue_dimensions)
    lines.extend(["", "## Dialogue audition scores", ""])
    for profile in profiles:
        lines.extend([f"### {profile}", ""])
        lines.extend(
            f"- `{item['id']}`: TODO / {manual['scale_max']}"
            for item in dialogue_dimensions
        )
        lines.extend(["- Notes: TODO", ""])
    lines.extend(
        [
            f"## Joined chapter scores — {manual['chapter_profile']}",
            "",
        ]
    )
    lines.extend(
        f"- `{item['id']}`: TODO / {manual['scale_max']} — {item['prompt']}"
        for item in chapter_dimensions
    )
    lines.extend(
        [
            "",
            "## Join timeline",
            "",
            "Use the mastered chapter and seek to each timestamp before scoring",
            "pacing and continuity.",
            "",
            "| Chunk | Timestamp | Boundary | Words | Speech WPM | "
            "Planned silence | Source |",
            "| ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    lines.extend(_join_review_row(item) for item in chunk_timeline)
    lines.extend(
        [
            "",
            "## Automated splice evidence",
            "",
            "The PCM inside every planned window is silent. Start/end values are",
            "the sample discontinuity at each edge; more negative is smoother.",
            "",
            "| Boundary | Timestamp | Silence | Start step | End step | PCM silent |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    lines.extend(_splice_review_row(item) for item in silence_windows)
    lines.extend(
        [
            "",
            "## Required join observations",
            "",
            "- Sentence joins: TODO",
            "- Paragraph joins: TODO",
            f"- Explicit {qa['explicit_pause_ms']} ms pause: TODO",
            "",
            "## Decision",
            "",
            "Preferred profile: TODO  ",
            "Approved settings: TODO  ",
            "Blocking defects: TODO",
            "",
            "After completing `listening-review.toml`, validate it from the repo",
            "root with:",
            "",
            "```console",
            'YAKBOX_NARRATION_REVIEW="/path/to/qa/listening-review.toml" \\',
            "uv run pytest -m live tests/live/test_narration_review.py",
            "```",
            "",
            f"QA workspace: `{workspace}`",
            "",
        ]
    )
    return "\n".join(lines)


def _audio_review_label(item: dict[str, object]) -> str:
    passage = item.get("audition_passage")
    kind = f"{passage} audition" if isinstance(passage, str) else str(item["kind"])
    return f"{item['logical_voice']} — {kind}"


def _join_review_row(item: dict[str, object]) -> str:
    explicit = item["kind"] == "explicit_pause"
    timestamp = (
        _number(item["start_seconds"]) if explicit else _number(item["end_seconds"])
    )
    silence_ms = (
        round(_number(item["audio_duration_seconds"]) * 1_000)
        if explicit
        else _integer(item["pause_after_ms"])
    )
    source = cast(dict[str, object], item["source"])
    words = _integer(item["spoken_words"])
    words_per_minute = item["estimated_words_per_minute"]
    cadence = "—" if words_per_minute is None else f"{_number(words_per_minute):.2f}"
    return (
        f"| {_integer(item['index'])} | {timestamp:.2f} s | {item['boundary']} | "
        f"{words} | {cadence} | {silence_ms} ms | "
        f"{source['path']}:{source['start_line']} |"
    )


def _splice_review_row(item: dict[str, object]) -> str:
    silent = "yes" if item["all_samples_silent"] is True else "no"
    return (
        f"| {item['boundary']} | {_number(item['start_seconds']):.2f} s | "
        f"{_integer(item['milliseconds'])} ms | "
        f"{_number(item['start_step_dbfs']):.1f} dBFS | "
        f"{_number(item['end_step_dbfs']):.1f} dBFS | {silent} |"
    )
