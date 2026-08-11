from __future__ import annotations

import hashlib
import re
import tomllib
import unicodedata
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import cast

from markdown_it import MarkdownIt
from markdown_it.token import Token

from yakbox.contracts import schema_uri
from yakbox.errors import ValidationError
from yakbox.speech.chunking import (
    CHATTERBOX_CHUNK_CHARACTERS,
    ChunkBoundary,
    TextChunk,
    chunk_text,
    plan_text_chunks,
)

__all__ = [
    "CHATTERBOX_CHUNK_CHARACTERS",
    "ChunkBoundary",
    "TextChunk",
    "chunk_text",
    "plan_text_chunks",
]

_DIRECTIVE = re.compile(
    r"<!--\s*yakbox:speech:"
    r"(?P<kind>exclude:start|exclude:end|only:start|only:end|pause(?:\s+ms=(?P<ms>-?\d+))?)"
    r"\s*-->"
)
_SPEAKER_DIRECTIVE = re.compile(
    r"<!--\s*yakbox:speech:speaker\s+name=(?P<name>[a-z][a-z0-9_-]*)"
    r"(?:\s+profile=(?P<profile>[a-z][a-z0-9_-]*))?"
    r"(?:\s+narrator_profile=(?P<narrator_profile>[a-z][a-z0-9_-]*))?\s*-->"
)
_PAUSE_SENTINEL = "YAKBOXPAUSE"
_PAUSE_PATTERN = re.compile(rf"(?:<!--\s*)?{_PAUSE_SENTINEL}(\d+)(?:\s*-->)?")
_SPEAKER_SENTINEL = "YAKBOXSPEAKER"
_SPEAKER_PATTERN = re.compile(
    rf"(?:<!--\s*)?{_SPEAKER_SENTINEL}(?P<name>[a-z][a-z0-9_-]*)"
    rf"(?:PROFILE(?P<profile>[a-z][a-z0-9_-]*))?"
    rf"(?:NARRATORPROFILE(?P<narrator_profile>[a-z][a-z0-9_-]*))?"
    rf"(?:\s*-->)?"
)
_DIALOGUE_QUOTE_PATTERN = re.compile(r'"[^"\n]+"|“[^”\n]+”|«[^»\n]+»')
_ATTRIBUTION_VERBS = frozenset(
    {
        "added",
        "adds",
        "admitted",
        "admits",
        "agreed",
        "agrees",
        "announced",
        "announces",
        "answered",
        "answers",
        "asked",
        "asks",
        "begged",
        "begs",
        "called",
        "calls",
        "confessed",
        "confesses",
        "continued",
        "continues",
        "countered",
        "counters",
        "cried",
        "cries",
        "declared",
        "declares",
        "demanded",
        "demands",
        "explained",
        "explains",
        "growled",
        "growls",
        "hissed",
        "hisses",
        "insisted",
        "insists",
        "interrupted",
        "interrupts",
        "joked",
        "jokes",
        "murmured",
        "murmurs",
        "muttered",
        "mutters",
        "noted",
        "notes",
        "objected",
        "objects",
        "observed",
        "observes",
        "pleaded",
        "pleads",
        "protested",
        "protests",
        "quipped",
        "quips",
        "remarked",
        "remarks",
        "repeated",
        "repeats",
        "replied",
        "replies",
        "responded",
        "responds",
        "said",
        "says",
        "shouted",
        "shouts",
        "snapped",
        "snaps",
        "stammered",
        "stammers",
        "stated",
        "states",
        "stuttered",
        "stutters",
        "suggested",
        "suggests",
        "warned",
        "warns",
        "whispered",
        "whispers",
        "yelled",
        "yells",
        "sounded",
        "sounds",
    }
)
_EXPRESSIVE_ATTRIBUTION_VERBS = frozenset(
    {
        "cried",
        "cries",
        "growled",
        "growls",
        "hissed",
        "hisses",
        "murmured",
        "murmurs",
        "muttered",
        "mutters",
        "shouted",
        "shouts",
        "snapped",
        "snaps",
        "sounded",
        "sounds",
        "stammered",
        "stammers",
        "stuttered",
        "stutters",
        "whispered",
        "whispers",
        "yelled",
        "yells",
    }
)
_EXPRESSIVE_TAG_PATTERN = re.compile(
    r"\b(?:with|flat|flatly|amused|angrily|bitterly|cheerfully|coldly|"
    r"contempt|disgust|dryly|gently|harshly|quietly|sharply|softly)\b",
    re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)?", re.UNICODE)
_SENTENCE_GAP_PATTERN = re.compile(r"(?<=[.!?])\s+")
_DIRECTIVE_GAP = "\ue000"


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """Inclusive line range identifying content in one source file."""

    path: Path
    start_line: int
    end_line: int


class AttributionTagKind(StrEnum):
    """Semantic class of a recognized dialogue attribution tag."""

    PURE = "pure"
    EXPRESSIVE = "expressive"


class AttributionContextPosition(StrEnum):
    """Position of a stripped attribution relative to its dialogue."""

    BEFORE = "before"
    AFTER = "after"


@dataclass(frozen=True, slots=True)
class AttributionContext:
    """A stripped tag retained as non-spoken synthesis context."""

    text: str
    kind: AttributionTagKind
    position: AttributionContextPosition


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """Normalized spoken text with stable identity and source provenance."""

    id: str
    chapter_id: str
    text: str
    source: SourceLocation
    sha256: str
    speaker: str = "narrator"
    speaker_explicit: bool = False
    profile_override: str | None = None
    boundary_after: ChunkBoundary = ChunkBoundary.PARAGRAPH
    attribution_context: tuple[AttributionContext, ...] = ()


@dataclass(frozen=True, slots=True)
class _SourceEvent:
    kind: str
    text: str
    start: int
    end: int
    speaker: str = "narrator"
    speaker_explicit: bool = False
    profile_override: str | None = None
    narrator_profile_override: str | None = None
    boundary_after: ChunkBoundary = ChunkBoundary.PARAGRAPH
    attribution_context: tuple[AttributionContext, ...] = ()


@dataclass(frozen=True, slots=True)
class _PendingSpeaker:
    name: str
    profile: str | None
    narrator_profile: str | None
    line: int


@dataclass(frozen=True, slots=True)
class _ReviewedDialogueRoute:
    source: Path
    line: int
    speaker: str


@dataclass(frozen=True, slots=True)
class _DialogueRouteEntry:
    source: Path
    source_value: str
    line: int
    speaker: str
    status: str


@dataclass(frozen=True, slots=True)
class Pause:
    """Explicit silence event attached to a source location and chapter."""

    chapter_id: str
    milliseconds: int
    source: SourceLocation


@dataclass(frozen=True, slots=True)
class Chapter:
    """Ordered chapter containing normalized speech and pause events."""

    id: str
    title: str
    order: int
    source_path: Path
    segments: tuple[SpeechSegment | Pause, ...]


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    """Deterministic chapter sequence and digest derived from source files."""

    chapters: tuple[Chapter, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class Pronunciation:
    written: str
    spoken: str
    match: str
    case: str
    priority: int
    language: str | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class PronunciationRuleAudit:
    """Match, application, and shadowing evidence for one pronunciation rule."""

    written: str
    spoken: str
    priority: int
    matches: int
    applied: int
    shadowed: int
    locations: tuple[SourceLocation, ...]

    @property
    def unused(self) -> bool:
        """Return whether the rule had no source matches."""
        return self.matches == 0

    def to_dict(self, *, root: Path | None = None) -> dict[str, object]:
        """Serialize one pronunciation-rule audit."""

        def location_value(location: SourceLocation) -> dict[str, object]:
            path = location.path
            path_value = (
                path.relative_to(root).as_posix()
                if root is not None and path.is_relative_to(root)
                else path.as_posix()
            )
            return {
                "path": path_value,
                "start_line": location.start_line,
                "end_line": location.end_line,
            }

        return {
            "written": self.written,
            "spoken": self.spoken,
            "priority": self.priority,
            "matches": self.matches,
            "applied": self.applied,
            "shadowed": self.shadowed,
            "unused": self.unused,
            "locations": [location_value(item) for item in self.locations],
        }


@dataclass(frozen=True, slots=True)
class PronunciationAudit:
    """Aggregate usage findings for all configured pronunciation rules."""

    rules: tuple[PronunciationRuleAudit, ...]

    @property
    def unused_rules(self) -> int:
        """Return the number of rules with no source matches."""
        return sum(rule.unused for rule in self.rules)

    @property
    def shadowed_matches(self) -> int:
        """Return the number of matches hidden by higher-priority rules."""
        return sum(rule.shadowed for rule in self.rules)

    def to_dict(self, *, root: Path | None = None) -> dict[str, object]:
        """Serialize the complete pronunciation audit."""
        return {
            "schema_version": 1,
            "rule_count": len(self.rules),
            "unused_rules": self.unused_rules,
            "shadowed_matches": self.shadowed_matches,
            "rules": [rule.to_dict(root=root) for rule in self.rules],
        }


def normalize_sources(
    paths: tuple[Path, ...],
    *,
    pronunciations: Path | None = None,
    max_pause_ms: int = 30_000,
    strip_attribution_tags: bool = False,
    dialogue_routes: Path | None = None,
    expressive_tag_handling: str = "context",
    retain_first_attribution_per_scene: bool = False,
) -> NormalizedDocument:
    """Parse source files into deterministic chapters, speech, and pause events."""
    if expressive_tag_handling not in {"context", "narrate", "strip"}:
        raise ValidationError(
            "expressive_tag_handling must be context, narrate, or strip"
        )
    rules = _load_pronunciations(pronunciations)
    routes = _load_dialogue_routes(dialogue_routes, paths)
    used_routes: set[tuple[Path, int]] = set()
    chapters: list[Chapter] = []
    for path in paths:
        chapters.extend(
            _normalize_one(
                path,
                start_order=len(chapters) + 1,
                rules=rules,
                max_pause_ms=max_pause_ms,
                strip_attribution_tags=strip_attribution_tags,
                dialogue_routes=routes,
                used_dialogue_routes=used_routes,
                expressive_tag_handling=expressive_tag_handling,
                retain_first_attribution_per_scene=retain_first_attribution_per_scene,
            )
        )
    unused_routes = set(routes) - used_routes
    if unused_routes:
        source, line = min(unused_routes, key=lambda item: (str(item[0]), item[1]))
        raise ValidationError(
            f"Dialogue route does not match a spoken paragraph: {source}:{line}"
        )
    if not chapters:
        raise ValidationError("No speakable content was found")
    payload = "\n".join(
        _identity_line(chapter.id, item)
        for chapter in chapters
        for item in chapter.segments
    )
    return NormalizedDocument(
        chapters=tuple(chapters),
        sha256=hashlib.sha256(payload.encode()).hexdigest(),
    )


def apply_pronunciations(text: str, pronunciations: Path | None = None) -> str:
    """Apply the approved manifest pronunciation rules to explicit speech text."""
    return _apply_pronunciations(text, _load_pronunciations(pronunciations))


def audit_pronunciations(
    paths: tuple[Path, ...],
    pronunciations: Path | None,
    *,
    max_pause_ms: int = 30_000,
    strip_attribution_tags: bool = False,
    dialogue_routes: Path | None = None,
    expressive_tag_handling: str = "context",
    retain_first_attribution_per_scene: bool = False,
) -> PronunciationAudit:
    """Report which approved pronunciation rules affect speakable source text."""
    rules = _load_pronunciations(pronunciations)
    document = normalize_sources(
        paths,
        max_pause_ms=max_pause_ms,
        strip_attribution_tags=strip_attribution_tags,
        dialogue_routes=dialogue_routes,
        expressive_tag_handling=expressive_tag_handling,
        retain_first_attribution_per_scene=retain_first_attribution_per_scene,
    )
    matches = [0] * len(rules)
    applied = [0] * len(rules)
    locations: list[list[SourceLocation]] = [[] for _rule in rules]
    for chapter in document.chapters:
        for segment in chapter.segments:
            if not isinstance(segment, SpeechSegment):
                continue
            occupied = [False] * len(segment.text)
            for index, rule in enumerate(rules):
                candidates = tuple(_pronunciation_pattern(rule).finditer(segment.text))
                matches[index] += len(candidates)
                for candidate in candidates:
                    if any(occupied[candidate.start() : candidate.end()]):
                        continue
                    occupied[candidate.start() : candidate.end()] = [True] * (
                        candidate.end() - candidate.start()
                    )
                    applied[index] += 1
                    if not locations[index] or locations[index][-1] != segment.source:
                        locations[index].append(segment.source)
    return PronunciationAudit(
        rules=tuple(
            PronunciationRuleAudit(
                written=rule.written,
                spoken=rule.spoken,
                priority=rule.priority,
                matches=matches[index],
                applied=applied[index],
                shadowed=matches[index] - applied[index],
                locations=tuple(locations[index]),
            )
            for index, rule in enumerate(rules)
        )
    )


def _normalize_one(
    path: Path,
    *,
    start_order: int,
    rules: tuple[Pronunciation, ...],
    max_pause_ms: int,
    strip_attribution_tags: bool,
    dialogue_routes: dict[tuple[Path, int], _ReviewedDialogueRoute],
    used_dialogue_routes: set[tuple[Path, int]],
    expressive_tag_handling: str,
    retain_first_attribution_per_scene: bool,
) -> list[Chapter]:
    source = _read_source(path)
    prepared = _apply_directives(source, path=path, max_pause_ms=max_pause_ms)
    tokens = MarkdownIt("commonmark", {"html": True}).parse(prepared)
    chapters: list[Chapter] = []
    order = start_order
    for title, blocks in _chapter_blocks(
        tokens,
        _default_chapter_title(path),
        path,
        strip_attribution_tags=strip_attribution_tags,
        dialogue_routes=dialogue_routes,
        used_dialogue_routes=used_dialogue_routes,
        expressive_tag_handling=expressive_tag_handling,
        retain_first_attribution_per_scene=retain_first_attribution_per_scene,
    ):
        chapter = _build_chapter(path, title, order, blocks, rules)
        if chapter is not None:
            chapters.append(chapter)
            order += 1
    return chapters


def _read_source(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValidationError(
            f"Cannot read audiobook source {path}: {error}"
        ) from error


def _default_chapter_title(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").title()


def _chapter_blocks(
    tokens: list[Token],
    default_title: str,
    path: Path,
    *,
    strip_attribution_tags: bool,
    dialogue_routes: dict[tuple[Path, int], _ReviewedDialogueRoute],
    used_dialogue_routes: set[tuple[Path, int]],
    expressive_tag_handling: str,
    retain_first_attribution_per_scene: bool,
) -> list[tuple[str, tuple[_SourceEvent, ...]]]:
    chapters: list[tuple[str, tuple[_SourceEvent, ...]]] = []
    current_title = default_title
    current: list[_SourceEvent] = []
    for event in _source_events(
        tokens,
        path,
        strip_attribution_tags=strip_attribution_tags,
        dialogue_routes=dialogue_routes,
        used_dialogue_routes=used_dialogue_routes,
        expressive_tag_handling=expressive_tag_handling,
        retain_first_attribution_per_scene=retain_first_attribution_per_scene,
    ):
        if event.kind == "heading":
            if current:
                chapters.append((current_title, tuple(current)))
                current = []
            if event.text:
                current_title = event.text
            continue
        current.append(event)
    if current:
        chapters.append((current_title, tuple(current)))
    return chapters


def _source_events(
    tokens: list[Token],
    path: Path,
    *,
    strip_attribution_tags: bool,
    dialogue_routes: dict[tuple[Path, int], _ReviewedDialogueRoute],
    used_dialogue_routes: set[tuple[Path, int]],
    expressive_tag_handling: str,
    retain_first_attribution_per_scene: bool,
) -> list[_SourceEvent]:
    events: list[_SourceEvent] = []
    pending_speaker: _PendingSpeaker | None = None
    attributed_speakers: set[str] = set()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == "heading_open" and token.tag in {"h1", "h2"}:
            _require_used_speaker(pending_speaker, path)
            attributed_speakers.clear()
            inline = tokens[index + 1] if index + 1 < len(tokens) else None
            heading = _inline_text(inline.children or []) if inline else ""
            events.append(_SourceEvent("heading", heading, 0, 0))
            index += 3
            continue
        if token.type == "hr":
            _require_used_speaker(pending_speaker, path)
            attributed_speakers.clear()
            index += 1
            continue
        if token.type == "paragraph_open":
            event = _paragraph_source_event(tokens, index, pending_speaker)
            if event is not None:
                event = _apply_dialogue_route(
                    event,
                    pending_speaker=pending_speaker,
                    path=path,
                    dialogue_routes=dialogue_routes,
                    used_dialogue_routes=used_dialogue_routes,
                )
                retain_attribution = (
                    retain_first_attribution_per_scene
                    and strip_attribution_tags
                    and event.speaker_explicit
                    and event.speaker != "narrator"
                    and event.speaker not in attributed_speakers
                    and _contains_dialogue_attribution(event.text)
                )
                events.extend(
                    _split_routed_dialogue(
                        event,
                        strip_attribution_tags=(
                            strip_attribution_tags and not retain_attribution
                        ),
                        expressive_tag_handling=expressive_tag_handling,
                    )
                )
                if retain_attribution:
                    attributed_speakers.add(event.speaker)
                pending_speaker = None
        elif token.type == "html_block":
            event, pending_speaker = _html_source_event(token, pending_speaker, path)
            if event is not None:
                events.append(event)
        index += 1
    _require_used_speaker(pending_speaker, path)
    return events


def _apply_dialogue_route(
    event: _SourceEvent,
    *,
    pending_speaker: _PendingSpeaker | None,
    path: Path,
    dialogue_routes: dict[tuple[Path, int], _ReviewedDialogueRoute],
    used_dialogue_routes: set[tuple[Path, int]],
) -> _SourceEvent:
    route_key = (path.resolve(), event.start)
    route = dialogue_routes.get(route_key)
    if route is None:
        return event
    if pending_speaker is not None:
        raise ValidationError(
            f"{path}:{event.start}: dialogue route conflicts with a speaker directive"
        )
    used_dialogue_routes.add(route_key)
    return replace(event, speaker=route.speaker, speaker_explicit=True)


def _contains_dialogue_attribution(text: str) -> bool:
    """Return whether prose outside quote delimiters contains a known tag."""
    cursor = 0
    for match in _DIALOGUE_QUOTE_PATTERN.finditer(text):
        if _dialogue_attribution_kind(text[cursor : match.start()]) is not None:
            return True
        cursor = match.end()
    return _dialogue_attribution_kind(text[cursor:]) is not None


def _paragraph_source_event(
    tokens: list[Token],
    index: int,
    pending_speaker: _PendingSpeaker | None,
) -> _SourceEvent | None:
    token = tokens[index]
    inline = _next_inline(tokens, index)
    if inline is None:
        return None
    text = _inline_text(inline.children or []).strip()
    if not text:
        return None
    start, end = _token_lines(token, inline)
    return _SourceEvent(
        "block",
        text,
        start,
        end,
        speaker=pending_speaker.name if pending_speaker else "narrator",
        speaker_explicit=pending_speaker is not None,
        profile_override=pending_speaker.profile if pending_speaker else None,
        narrator_profile_override=(
            pending_speaker.narrator_profile if pending_speaker else None
        ),
    )


def _split_routed_dialogue(
    event: _SourceEvent,
    *,
    strip_attribution_tags: bool,
    expressive_tag_handling: str,
) -> tuple[_SourceEvent, ...]:
    """Route quoted speech to a character and surrounding prose to narration."""
    if not event.speaker_explicit or event.speaker == "narrator":
        return (event,)
    matches = tuple(_DIALOGUE_QUOTE_PATTERN.finditer(event.text))
    if not matches:
        return (event,)

    result: list[_SourceEvent] = []
    cursor = 0
    pending_context: tuple[AttributionContext, ...] = ()
    for match in matches:
        surrounding = event.text[cursor : match.start()]
        pending_context = _append_routed_surrounding(
            result,
            event,
            surrounding,
            strip_attribution_tags=strip_attribution_tags,
            expressive_tag_handling=expressive_tag_handling,
        )
        _append_dialogue_event(
            result,
            event,
            match.group()[1:-1],
            True,
            attribution_context=pending_context,
        )
        cursor = match.end()
    surrounding = event.text[cursor:]
    _append_routed_surrounding(
        result,
        event,
        surrounding,
        strip_attribution_tags=strip_attribution_tags,
        expressive_tag_handling=expressive_tag_handling,
    )
    if not result:
        return (event,)
    return tuple(
        replace(
            item,
            boundary_after=(
                event.boundary_after
                if index == len(result) - 1
                else _internal_dialogue_boundary(item.text)
            ),
        )
        for index, item in enumerate(result)
    )


def _append_routed_surrounding(
    result: list[_SourceEvent],
    event: _SourceEvent,
    text: str,
    *,
    strip_attribution_tags: bool,
    expressive_tag_handling: str,
) -> tuple[AttributionContext, ...]:
    """Append narrator prose while optionally removing attribution sentences."""
    if not strip_attribution_tags:
        _append_dialogue_event(result, event, text, False)
        return ()
    remaining, stripped_contexts = transform_dialogue_attributions(
        text,
        expressive_tag_handling=expressive_tag_handling,
    )
    contexts = tuple(
        context
        for context in stripped_contexts
        if not (
            context.kind is AttributionTagKind.EXPRESSIVE
            and expressive_tag_handling == "strip"
        )
    )
    if stripped_contexts:
        _close_dialogue_before_stripped_attribution(
            result,
            continues_sentence=stripped_contexts[-1].text.rstrip().endswith(","),
        )
        if result and result[-1].speaker_explicit:
            previous = result[-1]
            result[-1] = replace(
                previous,
                attribution_context=_unique_attribution_context(
                    (
                        *previous.attribution_context,
                        *(
                            replace(
                                context,
                                position=AttributionContextPosition.AFTER,
                            )
                            for context in contexts
                        ),
                    )
                ),
            )
    _append_dialogue_event(result, event, remaining, False)
    return tuple(
        replace(context, position=AttributionContextPosition.BEFORE)
        for context in contexts
    )


def transform_dialogue_attributions(
    text: str,
    *,
    expressive_tag_handling: str,
) -> tuple[str, tuple[AttributionContext, ...]]:
    """Remove canonical attribution sentences and retain adjacent action prose."""
    fragments = _SENTENCE_GAP_PATTERN.split(text.strip())
    remaining: list[str] = []
    contexts: list[AttributionContext] = []
    for fragment in fragments:
        kind = _dialogue_attribution_kind(fragment)
        if kind is None or (
            kind is AttributionTagKind.EXPRESSIVE
            and expressive_tag_handling == "narrate"
        ):
            remaining.append(fragment)
            continue
        contexts.append(
            AttributionContext(
                fragment.strip(),
                kind,
                AttributionContextPosition.AFTER,
            )
        )
    return " ".join(remaining), tuple(contexts)


def _dialogue_attribution_kind(text: str) -> AttributionTagKind | None:
    """Classify an unquoted routed fragment as pure or expressive attribution."""
    words = {match.group().casefold() for match in _WORD_PATTERN.finditer(text)}
    verbs = words & _ATTRIBUTION_VERBS
    if not verbs:
        return None
    if verbs & _EXPRESSIVE_ATTRIBUTION_VERBS or _EXPRESSIVE_TAG_PATTERN.search(text):
        return AttributionTagKind.EXPRESSIVE
    return AttributionTagKind.PURE


def _unique_attribution_context(
    values: tuple[AttributionContext, ...],
) -> tuple[AttributionContext, ...]:
    return tuple(dict.fromkeys(values))


def _close_dialogue_before_stripped_attribution(
    result: list[_SourceEvent],
    *,
    continues_sentence: bool,
) -> None:
    """Make comma-ended dialogue grammatical after its attribution is removed."""
    if not result or not result[-1].speaker_explicit:
        return
    previous = result[-1]
    replacement = "" if continues_sentence else "."
    closed = re.sub(r"[,;:]\s*$", replacement, previous.text)
    if closed != previous.text:
        result[-1] = replace(previous, text=closed)


def _internal_dialogue_boundary(text: str) -> ChunkBoundary:
    """Infer the pause at an intra-paragraph speaker handoff."""
    tail = text.rstrip().rstrip("\"”'\u2019»)]}")
    if tail.endswith((".", "!", "?", "…")):
        return ChunkBoundary.SENTENCE
    return ChunkBoundary.CLAUSE


def _append_dialogue_event(
    result: list[_SourceEvent],
    event: _SourceEvent,
    text: str,
    character_speech: bool,
    *,
    attribution_context: tuple[AttributionContext, ...] = (),
) -> None:
    spoken = text.strip()
    if not spoken:
        return
    speaker = event.speaker if character_speech else "narrator"
    speaker_explicit = character_speech
    candidate = _SourceEvent(
        kind=event.kind,
        text=spoken,
        start=event.start,
        end=event.end,
        speaker=speaker,
        speaker_explicit=speaker_explicit,
        profile_override=(
            event.profile_override
            if character_speech
            else event.narrator_profile_override
        ),
        attribution_context=attribution_context,
    )
    if result and (
        result[-1].speaker,
        result[-1].speaker_explicit,
        result[-1].profile_override,
    ) == (speaker, speaker_explicit, candidate.profile_override):
        previous = result[-1]
        result[-1] = _SourceEvent(
            kind=previous.kind,
            text=f"{previous.text} {spoken}",
            start=previous.start,
            end=previous.end,
            speaker=speaker,
            speaker_explicit=speaker_explicit,
            profile_override=candidate.profile_override,
            attribution_context=_unique_attribution_context(
                (*previous.attribution_context, *candidate.attribution_context)
            ),
        )
        return
    result.append(candidate)


def _html_source_event(
    token: Token,
    pending_speaker: _PendingSpeaker | None,
    path: Path,
) -> tuple[_SourceEvent | None, _PendingSpeaker | None]:
    value = token.content.strip()
    speaker = _SPEAKER_PATTERN.fullmatch(value)
    if speaker:
        start, _ = _token_lines(token, token)
        if pending_speaker is not None:
            raise ValidationError(
                f"{path}:{start}: speaker directive replaces an unused directive"
            )
        return None, _PendingSpeaker(
            speaker.group("name"),
            speaker.group("profile"),
            speaker.group("narrator_profile"),
            start,
        )
    if _PAUSE_PATTERN.fullmatch(value):
        _require_used_speaker(pending_speaker, path)
        start, end = _token_lines(token, token)
        return _SourceEvent("block", value, start, end), None
    return None, pending_speaker


def _require_used_speaker(
    pending_speaker: _PendingSpeaker | None,
    path: Path,
) -> None:
    if pending_speaker is not None:
        raise ValidationError(
            f"{path}:{pending_speaker.line}: speaker directive must precede speech"
        )


def _build_chapter(
    path: Path,
    title: str,
    order: int,
    blocks: tuple[_SourceEvent, ...],
    rules: tuple[Pronunciation, ...],
) -> Chapter | None:
    chapter_id = f"{order:04d}-{_slug(title)}"
    items = tuple(
        item
        for item_index, block in enumerate(blocks, 1)
        if (item := _build_chapter_item(path, chapter_id, item_index, block, rules))
        is not None
    )
    if not items:
        return None
    return Chapter(
        id=chapter_id,
        title=title,
        order=order,
        source_path=path,
        segments=items,
    )


def _build_chapter_item(
    path: Path,
    chapter_id: str,
    item_index: int,
    block: _SourceEvent,
    rules: tuple[Pronunciation, ...],
) -> SpeechSegment | Pause | None:
    text = block.text
    location = SourceLocation(path=path, start_line=block.start, end_line=block.end)
    pause = _PAUSE_PATTERN.fullmatch(text)
    if pause:
        return Pause(
            chapter_id=chapter_id,
            milliseconds=int(pause.group(1)),
            source=location,
        )
    spoken = _apply_pronunciations(text, rules).strip()
    if not spoken:
        return None
    context_identity = [
        (item.kind.value, item.position.value, item.text)
        for item in block.attribution_context
    ]
    digest = hashlib.sha256(
        f"{block.speaker}\0{block.profile_override or ''}\0"
        f"{block.boundary_after.value}\0{spoken}\0"
        f"{context_identity}".encode()
    ).hexdigest()
    return SpeechSegment(
        id=f"{chapter_id}-{item_index:04d}-{digest[:10]}",
        chapter_id=chapter_id,
        text=spoken,
        source=location,
        sha256=digest,
        speaker=block.speaker,
        speaker_explicit=block.speaker_explicit,
        profile_override=block.profile_override,
        boundary_after=block.boundary_after,
        attribution_context=block.attribution_context,
    )


def _apply_directives(source: str, *, path: Path, max_pause_ms: int) -> str:
    if any(marker in source for marker in (_DIRECTIVE_GAP, _SPEAKER_SENTINEL)):
        raise ValidationError(f"{path}: source contains a reserved directive marker")
    _validate_directive_tokens(source, path)
    source = _SPEAKER_DIRECTIVE.sub(
        lambda match: _speaker_replacement(source, match, path=path),
        source,
    )
    result: list[str] = []
    cursor = 0
    active: str | None = None
    for match in _DIRECTIVE.finditer(source):
        kind = match.group("kind")
        content = source[cursor : match.start()]
        if active != "exclude":
            result.append(content)
        else:
            result.append(_removed_text_gap(content))
        line = source.count("\n", 0, match.start()) + 1
        active, replacement = _directive_transition(
            source,
            match,
            kind,
            active=active,
            path=path,
            line=line,
            max_pause_ms=max_pause_ms,
        )
        result.append(replacement)
        cursor = match.end()
    tail = source[cursor:]
    if active != "exclude":
        result.append(tail)
    else:
        result.append(_removed_text_gap(tail))
    if active is not None:
        raise ValidationError(f"{path}: unclosed speech {active} directive")
    return re.sub(f"{_DIRECTIVE_GAP}+", " ", "".join(result))


def _validate_directive_tokens(source: str, path: Path) -> None:
    recognized = {
        *(match.start() for match in _DIRECTIVE.finditer(source)),
        *(match.start() for match in _SPEAKER_DIRECTIVE.finditer(source)),
    }
    for generic in re.finditer(r"<!--\s*yakbox:speech:", source):
        if generic.start() not in recognized:
            line = source.count("\n", 0, generic.start()) + 1
            raise ValidationError(f"{path}:{line}: malformed yakbox speech directive")


def _speaker_replacement(
    source: str,
    match: re.Match[str],
    *,
    path: Path,
) -> str:
    line = source.count("\n", 0, match.start()) + 1
    line_start = source.rfind("\n", 0, match.start()) + 1
    line_end = source.find("\n", match.end())
    line_end = len(source) if line_end < 0 else line_end
    if (
        source[line_start : match.start()].strip()
        or source[match.end() : line_end].strip()
    ):
        raise ValidationError(
            f"{path}:{line}: speaker directive must occupy its own line"
        )
    profile = match.group("profile")
    profile_marker = f"PROFILE{profile}" if profile else ""
    narrator_profile = match.group("narrator_profile")
    narrator_marker = f"NARRATORPROFILE{narrator_profile}" if narrator_profile else ""
    return (
        f"<!--{_SPEAKER_SENTINEL}{match.group('name')}"
        f"{profile_marker}{narrator_marker}-->" + ("\n" * match.group(0).count("\n"))
    )


def _directive_transition(
    source: str,
    match: re.Match[str],
    kind: str,
    *,
    active: str | None,
    path: Path,
    line: int,
    max_pause_ms: int,
) -> tuple[str | None, str]:
    if kind in {"exclude:start", "only:start", "exclude:end", "only:end"}:
        return _region_transition(kind, active=active, path=path, line=line)
    return (
        active,
        _pause_replacement(
            source,
            match,
            active=active,
            path=path,
            line=line,
            max_pause_ms=max_pause_ms,
        ),
    )


def _region_transition(
    kind: str,
    *,
    active: str | None,
    path: Path,
    line: int,
) -> tuple[str | None, str]:
    region, transition = kind.split(":", 1)
    if transition == "start":
        if active is not None:
            raise ValidationError(
                f"{path}:{line}: nested or overlapping speech directive"
            )
        return region, _DIRECTIVE_GAP
    if active != region:
        raise ValidationError(f"{path}:{line}: unmatched speech directive {kind}")
    return None, _DIRECTIVE_GAP


def _pause_replacement(
    source: str,
    match: re.Match[str],
    *,
    active: str | None,
    path: Path,
    line: int,
    max_pause_ms: int,
) -> str:
    if active is not None:
        raise ValidationError(
            f"{path}:{line}: pause cannot appear inside a speech region"
        )
    milliseconds = int(match.group("ms") or "-1")
    if not 0 <= milliseconds <= max_pause_ms:
        raise ValidationError(
            f"{path}:{line}: pause must be between 0 and {max_pause_ms} ms"
        )
    line_start = source.rfind("\n", 0, match.start()) + 1
    line_end = source.find("\n", match.end())
    line_end = len(source) if line_end < 0 else line_end
    if (
        source[line_start : match.start()].strip()
        or source[match.end() : line_end].strip()
    ):
        raise ValidationError(
            f"{path}:{line}: pause directive must occupy its own line"
        )
    return f"<!--{_PAUSE_SENTINEL}{milliseconds}-->" + (
        "\n" * match.group(0).count("\n")
    )


def _inline_text(children: list[Token]) -> str:
    pieces: list[str] = []
    for child in children:
        if child.type == "text":
            pieces.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            pieces.append(" ")
        elif child.type == "html_inline":
            value = child.content.strip()
            pause = _PAUSE_PATTERN.fullmatch(value)
            if pause:
                pieces.append(f" {_PAUSE_SENTINEL}{pause.group(1)} ")
    return unicodedata.normalize("NFC", "".join(pieces))


def _next_inline(tokens: list[Token], index: int) -> Token | None:
    for candidate in tokens[index + 1 : min(len(tokens), index + 4)]:
        if candidate.type == "inline":
            return candidate
        if candidate.type.endswith("_close"):
            break
    return None


def _token_lines(*tokens: Token) -> tuple[int, int]:
    mappings = [token.map for token in tokens if token.map]
    if not mappings:
        return (1, 1)
    return min(item[0] for item in mappings) + 1, max(item[1] for item in mappings)


def _load_pronunciations(path: Path | None) -> tuple[Pronunciation, ...]:
    if path is None:
        return ()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValidationError(
            f"Cannot read pronunciation file {path}: {error}"
        ) from error
    if raw.get("schema_version") != 1:
        raise ValidationError("Pronunciation file requires schema_version = 1")
    unknown_root = set(raw) - {"schema_version", "terms"}
    if unknown_root:
        raise ValidationError(
            f"Unknown pronunciation keys: {', '.join(sorted(unknown_root))}"
        )
    terms = raw.get("terms", [])
    if not isinstance(terms, list):
        raise ValidationError("Pronunciation terms must use [[terms]] records")
    result: list[Pronunciation] = []
    identities: set[tuple[str, str, str, int]] = set()
    for index, term in enumerate(terms, 1):
        parsed = _parse_pronunciation(term, index)
        if parsed is None:
            continue
        identity = (parsed.written, parsed.match, parsed.case, parsed.priority)
        if identity in identities:
            raise ValidationError(
                f"Pronunciation term {index} duplicates an equal-priority rule"
            )
        identities.add(identity)
        result.append(parsed)
    result.sort(key=lambda item: (-item.priority, -len(item.written), item.written))
    return tuple(result)


def _load_dialogue_routes(
    path: Path | None,
    sources: tuple[Path, ...],
) -> dict[tuple[Path, int], _ReviewedDialogueRoute]:
    if path is None:
        return {}
    raw = _read_dialogue_routes(path)
    entries = raw.get("routes")
    if not isinstance(entries, list) or not entries:
        raise ValidationError("Dialogue routes require [[routes]] records")
    allowed_sources = {source.resolve() for source in sources}
    parsed = tuple(
        _parse_dialogue_route(value, index, path, allowed_sources)
        for index, value in enumerate(entries, 1)
    )
    pending = [
        f"{item.source_value}:{item.line}"
        for item in parsed
        if item.status == "suggested"
    ]
    if pending:
        raise ValidationError(
            "Dialogue routes require review before use; change status to approved "
            "or rejected: " + ", ".join(pending)
        )
    result: dict[tuple[Path, int], _ReviewedDialogueRoute] = {}
    for index, item in enumerate(parsed, 1):
        if item.status == "rejected":
            continue
        key = (item.source, item.line)
        if key in result:
            raise ValidationError(
                f"Dialogue route {index} duplicates {item.source_value}:{item.line}"
            )
        result[key] = _ReviewedDialogueRoute(
            item.source,
            item.line,
            item.speaker,
        )
    return result


def _read_dialogue_routes(path: Path) -> dict[str, object]:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValidationError(f"Cannot read dialogue routes {path}: {error}") from error
    if raw.get("$schema") != schema_uri("dialogue-routes"):
        raise ValidationError(
            f'dialogue routes require "$schema" = "{schema_uri("dialogue-routes")}"'
        )
    if raw.get("schema_version") != 1:
        raise ValidationError("Dialogue routes require schema_version = 1")
    unknown = set(raw) - {"$schema", "schema_version", "routes"}
    if unknown:
        raise ValidationError(
            f"Unknown dialogue route keys: {', '.join(sorted(unknown))}"
        )
    return raw


def _parse_dialogue_route(
    value: object,
    index: int,
    path: Path,
    allowed_sources: set[Path],
) -> _DialogueRouteEntry:
    if not isinstance(value, dict):
        raise ValidationError(f"Dialogue route {index} must be a table")
    item = cast(dict[str, object], value)
    unknown = set(item) - {
        "source",
        "line",
        "speaker",
        "status",
        "enabled",
        "notes",
    }
    if unknown:
        raise ValidationError(
            f"Unknown dialogue route {index} keys: {', '.join(sorted(unknown))}"
        )
    enabled = item.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValidationError(f"Dialogue route {index} enabled must be boolean")
    status = item.get("status") if enabled else "rejected"
    if status not in {"suggested", "approved", "rejected"}:
        raise ValidationError(f"Dialogue route {index} has invalid status")
    source_value = item.get("source")
    line = item.get("line")
    speaker = item.get("speaker")
    if not isinstance(source_value, str) or not source_value.strip():
        raise ValidationError(f"Dialogue route {index} needs source")
    if not isinstance(line, int) or isinstance(line, bool) or line < 1:
        raise ValidationError(f"Dialogue route {index} line must be positive")
    if (
        not isinstance(speaker, str)
        or re.fullmatch(r"[a-z][a-z0-9_-]*", speaker) is None
    ):
        raise ValidationError(f"Dialogue route {index} has invalid speaker")
    source = (path.parent / source_value).resolve()
    if source not in allowed_sources:
        raise ValidationError(
            f"Dialogue route {index} source is not declared in the manifest: "
            f"{source_value}"
        )
    return _DialogueRouteEntry(source, source_value, line, speaker, str(status))


def _parse_pronunciation(value: object, index: int) -> Pronunciation | None:
    table = _pronunciation_table(value, index)
    if not _pronunciation_is_approved(table, index):
        return None
    written = _required_pronunciation_text(table, "written", index)
    spoken = _required_pronunciation_text(table, "spoken", index)
    language = _optional_pronunciation_text(table, "language", index)
    notes = _optional_pronunciation_text(table, "notes", index, strip=False)
    priority = _pronunciation_priority(table, index)
    match_mode = table.get("match", "whole_word")
    case_mode = table.get("case", "sensitive")
    _validate_pronunciation_modes(match_mode, case_mode, index)
    return Pronunciation(
        written=unicodedata.normalize("NFC", written),
        spoken=unicodedata.normalize("NFC", spoken),
        match=str(match_mode),
        case=str(case_mode),
        priority=priority,
        language=unicodedata.normalize("NFC", language) if language else None,
        notes=unicodedata.normalize("NFC", notes) if notes is not None else None,
    )


def _pronunciation_table(value: object, index: int) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValidationError(f"Pronunciation term {index} must be a table")
    table = cast(dict[str, object], value)
    unknown = set(table) - {
        "written",
        "spoken",
        "language",
        "match",
        "case",
        "priority",
        "status",
        "enabled",
        "notes",
    }
    if unknown:
        raise ValidationError(
            f"Unknown pronunciation term {index} keys: {', '.join(sorted(unknown))}"
        )
    return table


def _pronunciation_is_approved(value: dict[str, object], index: int) -> bool:
    enabled = value.get("enabled", True)
    status = value.get("status")
    if not isinstance(enabled, bool):
        raise ValidationError(f"Pronunciation term {index} enabled must be boolean")
    if status not in {"approved", "draft", "rejected"}:
        raise ValidationError(f"Pronunciation term {index} has invalid status")
    return enabled and status == "approved"


def _required_pronunciation_text(value: dict[str, object], key: str, index: int) -> str:
    text = value.get(key)
    if not isinstance(text, str) or not text.strip():
        raise ValidationError(f"Pronunciation term {index} needs {key}")
    return text.strip()


def _optional_pronunciation_text(
    value: dict[str, object],
    key: str,
    index: int,
    *,
    strip: bool = True,
) -> str | None:
    text = value.get(key)
    if text is None:
        return None
    if not isinstance(text, str) or (key == "language" and not text.strip()):
        qualifier = "a non-empty string" if key == "language" else "a string"
        raise ValidationError(f"Pronunciation term {index} {key} must be {qualifier}")
    return text.strip() if strip else text


def _pronunciation_priority(value: dict[str, object], index: int) -> int:
    priority = value.get("priority", 0)
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise ValidationError(f"Pronunciation term {index} priority must be integer")
    return priority


def _validate_pronunciation_modes(
    match_mode: object,
    case_mode: object,
    index: int,
) -> None:
    if match_mode not in {"whole_word", "substring"}:
        raise ValidationError(
            f"Pronunciation term {index} match must be whole_word or substring"
        )
    if case_mode not in {"sensitive", "insensitive"}:
        raise ValidationError(
            f"Pronunciation term {index} case must be sensitive or insensitive"
        )


def _apply_pronunciations(text: str, rules: tuple[Pronunciation, ...]) -> str:
    occupied = [False] * len(text)
    replacements: list[tuple[int, int, str]] = []
    for rule in rules:
        for match in _pronunciation_pattern(rule).finditer(text):
            if any(occupied[match.start() : match.end()]):
                continue
            occupied[match.start() : match.end()] = [True] * (
                match.end() - match.start()
            )
            replacements.append((match.start(), match.end(), rule.spoken))
    for start, end, spoken in sorted(replacements, reverse=True):
        text = f"{text[:start]}{spoken}{text[end:]}"
    return text


def _pronunciation_pattern(rule: Pronunciation) -> re.Pattern[str]:
    flags = re.IGNORECASE if rule.case == "insensitive" else 0
    escaped = re.escape(rule.written)
    pattern = rf"(?<!\w){escaped}(?!\w)" if rule.match == "whole_word" else escaped
    return re.compile(pattern, flags)


def _slug(value: str) -> str:
    slug = re.sub(r"[^\w]+", "-", value.casefold(), flags=re.UNICODE).strip("-")
    return slug or "chapter"


def _identity_line(chapter_id: str, item: SpeechSegment | Pause) -> str:
    value = (
        f"{item.speaker}:{int(item.speaker_explicit)}:"
        f"{item.boundary_after.value}:{item.text}"
        if isinstance(item, SpeechSegment)
        else str(item.milliseconds)
    )
    return f"{chapter_id}:{value}"


def _removed_text_gap(value: str) -> str:
    newlines = "\n" * value.count("\n")
    return newlines if newlines else _DIRECTIVE_GAP
