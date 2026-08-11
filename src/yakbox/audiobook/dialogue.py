"""Reviewed dialogue routing suggestions and sidecar serialization."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from yakbox.audiobook.manifest import AudiobookManifest
from yakbox.audiobook.sources import (
    AttributionContext,
    NormalizedDocument,
    SpeechSegment,
    normalize_sources,
    transform_dialogue_attributions,
)
from yakbox.contracts import runtime_metadata, schema_uri

_QUOTED_SPEECH = re.compile(r'"[^"\n]+"|“[^”\n]+”|«[^»\n]+»')
_ATTRIBUTION_WORDS = re.compile(
    r"\b(?:said|says|asked|asks|replied|replies|answered|answers|"
    r"added|adds|continued|continues|snapped|snaps|whispered|whispers|"
    r"shouted|shouts|muttered|mutters|murmured|murmurs|warned|warns|"
    r"explained|explains|quipped|quips|responded|responds)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DialogueRouteSuggestion:
    """One source-located speaker route proposed for human review."""

    source: Path
    line: int
    speaker: str
    reason: str


@dataclass(frozen=True, slots=True)
class DialogueRouteSuggestions:
    """Suggested routes plus the number of unresolved dialogue paragraphs."""

    routes: tuple[DialogueRouteSuggestion, ...]
    unresolved: int


def suggest_dialogue_routes(manifest: AudiobookManifest) -> DialogueRouteSuggestions:
    """Infer review-only routes from explicit names near quoted dialogue."""
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
    )
    aliases = _speaker_aliases(manifest)
    suggestions: list[DialogueRouteSuggestion] = []
    unresolved = 0
    for chapter in document.chapters:
        for item in chapter.segments:
            if not isinstance(item, SpeechSegment) or item.speaker_explicit:
                continue
            if _QUOTED_SPEECH.search(item.text) is None:
                continue
            speaker, reason = _suggested_speaker(item.text, aliases)
            if speaker is None:
                unresolved += 1
                continue
            suggestions.append(
                DialogueRouteSuggestion(
                    source=item.source.path,
                    line=item.source.start_line,
                    speaker=speaker,
                    reason=reason,
                )
            )
    return DialogueRouteSuggestions(tuple(suggestions), unresolved)


def render_dialogue_routes(
    suggestions: DialogueRouteSuggestions,
    *,
    relative_to: Path,
) -> str:
    """Serialize suggestions as a TOML sidecar requiring explicit review."""
    lines = [
        f'"$schema" = {json.dumps(schema_uri("dialogue-routes"))}',
        "schema_version = 1",
        "",
        "# Review every route. Change status to approved or rejected before use.",
    ]
    for route in suggestions.routes:
        source = Path(os.path.relpath(route.source, relative_to)).as_posix()
        lines.extend(
            [
                "",
                "[[routes]]",
                f"source = {json.dumps(source)}",
                f"line = {route.line}",
                f"speaker = {json.dumps(route.speaker)}",
                'status = "suggested"',
                f"notes = {json.dumps(route.reason)}",
            ]
        )
    return "\n".join(lines) + "\n"


def dialogue_transformation_report(manifest: AudiobookManifest) -> dict[str, object]:
    """Return an explicit-text local preview of routed dialogue transformations."""
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
        strip_attribution_tags=manifest.dialogue.strip_attribution_tags,
        dialogue_routes=manifest.dialogue.routes,
        expressive_tag_handling=manifest.dialogue.expressive_tag_handling,
        retain_first_attribution_per_scene=(
            manifest.dialogue.retain_first_attribution_per_scene
        ),
    )
    original = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
        dialogue_routes=manifest.dialogue.routes,
    )
    grouped = _group_speech_segments(document)
    original_grouped = _group_speech_segments(original)
    routed_groups = [
        (source, start, end, segments)
        for (source, start, end), segments in grouped.items()
        if any(item.speaker_explicit for item in segments)
    ]
    paragraphs = [
        _dialogue_paragraph(
            manifest,
            source,
            start=start,
            end=end,
            segments=segments,
            original_segments=original_grouped[(source, start, end)],
        )
        for source, start, end, segments in routed_groups
    ]
    stripped_tags = sum(
        len(cast(list[object], paragraph["stripped_tags"])) for paragraph in paragraphs
    )
    return {
        **runtime_metadata("dialogue-transformation"),
        "manifest": manifest.path.relative_to(manifest.root).as_posix(),
        "strip_attribution_tags": manifest.dialogue.strip_attribution_tags,
        "expressive_tag_handling": manifest.dialogue.expressive_tag_handling,
        "retain_first_attribution_per_scene": (
            manifest.dialogue.retain_first_attribution_per_scene
        ),
        "paragraph_count": len(paragraphs),
        "stripped_tag_count": stripped_tags,
        "paragraphs": paragraphs,
    }


def _dialogue_paragraph(
    manifest: AudiobookManifest,
    source: Path,
    *,
    start: int,
    end: int,
    segments: list[SpeechSegment],
    original_segments: list[SpeechSegment],
) -> dict[str, object]:
    tags = _stripped_attributions(
        original_segments,
        segments,
        expressive_tag_handling=manifest.dialogue.expressive_tag_handling,
    )
    return {
        "source": {
            "path": source.relative_to(manifest.root).as_posix(),
            "start_line": start,
            "end_line": end,
        },
        "spans": [
            {
                "speaker": segment.speaker,
                "spoken": segment.text,
                "speaker_explicit": segment.speaker_explicit,
            }
            for segment in segments
        ],
        "stripped_tags": [
            {"text": context.text, "kind": context.kind.value} for context in tags
        ],
    }


def _group_speech_segments(
    document: NormalizedDocument,
) -> dict[tuple[Path, int, int], list[SpeechSegment]]:
    grouped: dict[tuple[Path, int, int], list[SpeechSegment]] = {}
    for chapter in document.chapters:
        for item in chapter.segments:
            if not isinstance(item, SpeechSegment):
                continue
            key = (item.source.path, item.source.start_line, item.source.end_line)
            grouped.setdefault(key, []).append(item)
    return grouped


def _stripped_attributions(
    original: list[SpeechSegment],
    transformed: list[SpeechSegment],
    *,
    expressive_tag_handling: str,
) -> tuple[AttributionContext, ...]:
    result: dict[tuple[str, object], AttributionContext] = {}
    retained_narration = {
        segment.text for segment in transformed if not segment.speaker_explicit
    }
    for segment in original:
        if segment.speaker_explicit or segment.text in retained_narration:
            continue
        _remaining, contexts = transform_dialogue_attributions(
            segment.text,
            expressive_tag_handling=expressive_tag_handling,
        )
        for context in contexts:
            result.setdefault((context.text, context.kind), context)
    return tuple(result.values())


def _speaker_aliases(manifest: AudiobookManifest) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for character in manifest.characters:
        if character.name == "narrator":
            continue
        values = {
            character.name.replace("-", " ").casefold(),
            character.display_name.casefold(),
        }
        result[character.name] = tuple(sorted(values, key=len, reverse=True))
    return result


def _suggested_speaker(
    text: str,
    aliases: dict[str, tuple[str, ...]],
) -> tuple[str | None, str]:
    surrounding = " ".join(_QUOTED_SPEECH.split(text))
    named = {
        speaker
        for speaker, values in aliases.items()
        if any(_contains_name(surrounding, value) for value in values)
    }
    if len(named) != 1:
        return None, "ambiguous or unnamed dialogue"
    speaker = next(iter(named))
    reason = (
        "explicit speaker name with attribution verb"
        if _ATTRIBUTION_WORDS.search(surrounding)
        else "single explicit character name beside dialogue"
    )
    return speaker, reason


def _contains_name(text: str, name: str) -> bool:
    return (
        re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE) is not None
    )


__all__ = [
    "DialogueRouteSuggestion",
    "DialogueRouteSuggestions",
    "dialogue_transformation_report",
    "render_dialogue_routes",
    "suggest_dialogue_routes",
]
