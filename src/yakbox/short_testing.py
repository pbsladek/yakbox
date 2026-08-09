"""Focused short-utterance generation and listening-review workflows."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_bytes, atomic_write_json, sha256_file
from yakbox.audiobook.build import (
    _is_local_backend,
    _profile_device,
    _profile_threads,
    _profile_worker_timeout,
    _resolved_speech,
)
from yakbox.audiobook.journal import new_run_id
from yakbox.audiobook.manifest import load_manifest
from yakbox.contracts import runtime_metadata
from yakbox.errors import BuildError, ValidationError
from yakbox.local_alignment import open_local_aligner
from yakbox.speech.models import AudioFormat, SpeechSynthesisRequest
from yakbox.speech.services import open_speech_backend
from yakbox.speech.short_synthesis import synthesize_short_utterance
from yakbox.speech.short_utterances import (
    ShortUtteranceStrategy,
    carrier_recipes,
)


async def run_short_test(
    manifest_path: Path,
    *,
    profile_name: str,
    text: str,
    output_root: Path,
    previous_context: str | None = None,
    next_context: str | None = None,
) -> Path:
    """Generate a complete direct/carrier/crop/refinement diagnostic package."""
    manifest = load_manifest(manifest_path)
    profile = manifest.profile(profile_name)
    policy = replace(
        manifest.short_utterances,
        strategy=ShortUtteranceStrategy.CONTEXT_EXTRACT,
        keep_candidates=True,
        require_review_for_one_word=False,
    )
    run_directory = output_root.resolve() / new_run_id()
    run_directory.mkdir(parents=True, exist_ok=False)
    voice, sample_rate, project, use_hd, reference_audio, chatterbox = _resolved_speech(
        profile, manifest
    )
    request = SpeechSynthesisRequest(
        text=text,
        voice=voice,
        backend=profile.backend,
        profile=profile.name,
        output_format=AudioFormat.WAV,
        sample_rate=sample_rate,
        project=project,
        use_hd=use_hd,
        reference_audio=reference_audio,
        chatterbox=chatterbox,
    )
    recipes = carrier_recipes(
        text,
        policy,
        seed_material=f"short-test:{profile_name}:{text}",
        previous_context=previous_context,
        next_context=next_context,
    )
    aligner = open_local_aligner(
        policy.alignment_backend,
        model=policy.alignment_model,
        revision=policy.alignment_revision,
        timeout_seconds=policy.alignment_timeout_seconds,
        prompted_timing=policy.prompted_timing,
        decode_consensus=policy.decode_consensus,
        prompt_sensitivity=policy.prompt_sensitivity,
        maximum_consensus_timing_delta_ms=(policy.maximum_consensus_timing_delta_ms),
        hallucination_silence_threshold=policy.hallucination_silence_threshold,
    )
    destination = run_directory / "selected.wav"
    async with open_speech_backend(
        profile.backend,
        isolated_local=_is_local_backend(profile.backend),
        device=_profile_device(profile),
        local_worker_timeout_seconds=_profile_worker_timeout(profile),
        local_threads_per_process=_profile_threads(profile),
        local_worker_log_path=run_directory / "logs" / "local-worker.log",
    ) as service:
        selection = await synthesize_short_utterance(
            service=service,
            aligner=aligner,
            request=request,
            destination=destination,
            recipes=recipes,
            policy=policy,
            language=manifest.book.language,
            qa_directory=run_directory,
        )
    atomic_write_json(
        run_directory / "short-test.json",
        {
            **runtime_metadata("short-utterance-test"),
            "kind": "short_utterance_test",
            "profile": profile_name,
            "text": text,
            "previous_context": previous_context,
            "next_context": next_context,
            "selected_candidate": selection.selected.recipe.candidate_index,
            "selected_audio": destination.name,
            "selected_audio_sha256": sha256_file(destination),
            "report": selection.report.name if selection.report else None,
            "recipes": [
                {
                    "candidate_index": recipe.candidate_index,
                    "template_id": recipe.template_id,
                    "position": recipe.position.value,
                    "text": recipe.text,
                }
                for recipe in recipes
            ],
        },
    )
    return run_directory


def list_short_reviews(root: Path) -> tuple[dict[str, object], ...]:
    """Discover valid short-utterance reports and their bound review state."""
    if not root.exists():
        return ()
    results: list[dict[str, object]] = []
    for report in sorted(root.rglob("report.json")):
        try:
            raw = _load_report(report)
        except BuildError:
            continue
        selected = raw.get("selected_candidate")
        candidate = _selected_candidate(raw, selected)
        review_path = report.with_name("listening-review.toml")
        results.append(
            {
                "report": str(report.resolve()),
                "report_sha256": sha256_file(report),
                "selected_candidate": selected,
                "selected_audio": str(
                    (report.parent / str(candidate["extracted_audio"])).resolve()
                ),
                "selected_audio_sha256": candidate.get("extracted_audio_sha256"),
                "review": str(review_path.resolve()),
                "status": _review_status(review_path),
            }
        )
    return tuple(results)


def review_audio_path(report: Path) -> Path:
    """Resolve and integrity-check the selected candidate audio."""
    raw = _load_report(report)
    candidate = _selected_candidate(raw, raw.get("selected_candidate"))
    relative = candidate.get("extracted_audio")
    digest = candidate.get("extracted_audio_sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise BuildError("Selected short-review candidate has no audio evidence")
    audio = (report.resolve().parent / relative).resolve()
    if not audio.is_relative_to(report.resolve().parent) or not audio.is_file():
        raise BuildError("Selected short-review audio path is unsafe or missing")
    if sha256_file(audio) != digest:
        raise BuildError("Selected short-review audio checksum does not match")
    return audio


def play_short_review(report: Path) -> Path:
    """Open the integrity-checked selected take in the operating-system player."""
    audio = review_audio_path(report)
    if sys.platform == "darwin":
        command = ["open", str(audio)]
    elif sys.platform == "win32":
        command = ["cmd", "/c", "start", "", str(audio)]
    else:
        command = ["xdg-open", str(audio)]
    try:
        subprocess.run(command, check=True)  # noqa: S603 - fixed platform opener.
    except (OSError, subprocess.CalledProcessError) as error:
        raise BuildError(f"Could not open selected review audio: {audio}") from error
    return audio


def write_short_review(report: Path, *, approved: bool, notes: str) -> Path:
    """Write a decision bound to both report bytes and selected audio bytes."""
    raw = _load_report(report)
    selected = raw.get("selected_candidate")
    if not isinstance(selected, int):
        raise BuildError("Short-review report has no selected candidate")
    audio = review_audio_path(report)
    review = report.resolve().with_name("listening-review.toml")
    escaped_notes = json.dumps(notes, ensure_ascii=False)
    content = (
        "schema_version = 1\n"
        f"status = {json.dumps('pass' if approved else 'fail')}\n"
        f'report_sha256 = "{sha256_file(report)}"\n'
        f"selected_candidate = {selected}\n"
        f'selected_audio_sha256 = "{sha256_file(audio)}"\n'
        f"notes = {escaped_notes}\n"
    )
    atomic_write_bytes(review, content.encode(), overwrite=True)
    return review


def _load_report(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildError(f"Cannot read short-utterance report: {path}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValidationError(f"Invalid short-utterance report: {path}")
    if raw.get("kind") != "short_utterance_qa" or not isinstance(
        raw.get("candidates"), list
    ):
        raise ValidationError(f"Not a short-utterance QA report: {path}")
    return raw


def _selected_candidate(raw: dict[str, object], selected: object) -> dict[str, object]:
    if not isinstance(selected, int):
        raise BuildError("Short-review report has no selected candidate")
    candidates = raw.get("candidates")
    if not isinstance(candidates, list):
        raise BuildError("Short-review report has no candidates")
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("candidate_index") == selected:
            return cast(dict[str, object], candidate)
    raise BuildError("Short-review selected candidate is missing")


def _review_status(path: Path) -> str:
    if not path.is_file():
        return "unreviewed"
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("status = "):
                return line.partition("=")[2].strip().strip('"')
    except OSError, UnicodeDecodeError:
        return "invalid"
    return "invalid"
