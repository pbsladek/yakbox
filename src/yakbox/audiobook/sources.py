from __future__ import annotations

import hashlib
import re
import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from markdown_it import MarkdownIt
from markdown_it.token import Token

from yakbox.errors import ValidationError

_DIRECTIVE = re.compile(
    r"<!--\s*yakbox:speech:"
    r"(?P<kind>exclude:start|exclude:end|only:start|only:end|pause(?:\s+ms=(?P<ms>-?\d+))?)"
    r"\s*-->"
)
_PAUSE_SENTINEL = "YAKBOXPAUSE"
_PAUSE_PATTERN = re.compile(rf"(?:<!--\s*)?{_PAUSE_SENTINEL}(\d+)(?:\s*-->)?")
_DIRECTIVE_GAP = "\ue000"


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """Inclusive line range identifying content in one source file."""

    path: Path
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """Normalized spoken text with stable identity and source provenance."""

    id: str
    chapter_id: str
    text: str
    source: SourceLocation
    sha256: str


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
) -> NormalizedDocument:
    """Parse source files into deterministic chapters, speech, and pause events."""
    rules = _load_pronunciations(pronunciations)
    chapters: list[Chapter] = []
    for path in paths:
        chapters.extend(
            _normalize_one(
                path,
                start_order=len(chapters) + 1,
                rules=rules,
                max_pause_ms=max_pause_ms,
            )
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


def audit_pronunciations(
    paths: tuple[Path, ...],
    pronunciations: Path | None,
    *,
    max_pause_ms: int = 30_000,
) -> PronunciationAudit:
    """Report which approved pronunciation rules affect speakable source text."""
    rules = _load_pronunciations(pronunciations)
    document = normalize_sources(paths, max_pause_ms=max_pause_ms)
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


def chunk_text(text: str, maximum: int) -> tuple[str, ...]:
    if maximum < 1:
        raise ValidationError("Chunk size must be positive")
    text = text.strip()
    if len(text) <= maximum:
        return (text,) if text else ()
    chunks: list[str] = []
    remainder = text
    boundaries = ("\n\n", ". ", "? ", "! ", "; ", ", ", " ")
    while len(remainder) > maximum:
        point = -1
        boundary_length = 0
        window = remainder[:maximum]
        for boundary in boundaries:
            candidate = window.rfind(boundary)
            candidate_end = candidate + len(boundary)
            if candidate >= 0 and candidate_end <= maximum and candidate > point:
                point = candidate
                boundary_length = len(boundary)
        if point < max(1, maximum // 3):
            point = _unicode_safe_boundary(remainder, maximum)
            boundary_length = 0
        end = point + boundary_length
        chunks.append(remainder[:end].strip())
        remainder = remainder[end:].strip()
    if remainder:
        chunks.append(remainder)
    return tuple(chunk for chunk in chunks if chunk)


def _normalize_one(
    path: Path,
    *,
    start_order: int,
    rules: tuple[Pronunciation, ...],
    max_pause_ms: int,
) -> list[Chapter]:
    source = _read_source(path)
    prepared = _apply_directives(source, path=path, max_pause_ms=max_pause_ms)
    tokens = MarkdownIt("commonmark", {"html": True}).parse(prepared)
    chapters: list[Chapter] = []
    order = start_order
    for title, blocks in _chapter_blocks(tokens, _default_chapter_title(path)):
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
) -> list[tuple[str, tuple[tuple[str, int, int], ...]]]:
    chapters: list[tuple[str, tuple[tuple[str, int, int], ...]]] = []
    current_title = default_title
    current: list[tuple[str, int, int]] = []
    for kind, text, start, end in _source_events(tokens):
        if kind == "heading":
            if current:
                chapters.append((current_title, tuple(current)))
                current = []
            if text:
                current_title = text
            continue
        current.append((text, start, end))
    if current:
        chapters.append((current_title, tuple(current)))
    return chapters


def _source_events(tokens: list[Token]) -> list[tuple[str, str, int, int]]:
    events: list[tuple[str, str, int, int]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == "heading_open" and token.tag in {"h1", "h2"}:
            inline = tokens[index + 1] if index + 1 < len(tokens) else None
            heading = _inline_text(inline.children or []) if inline else ""
            events.append(("heading", heading, 0, 0))
            index += 3
            continue
        if token.type == "paragraph_open":
            inline = _next_inline(tokens, index)
            if inline is not None:
                text = _inline_text(inline.children or []).strip()
                if text:
                    start, end = _token_lines(token, inline)
                    events.append(("block", text, start, end))
        if token.type == "html_block":
            value = token.content.strip()
            if _PAUSE_PATTERN.fullmatch(value):
                start, end = _token_lines(token, token)
                events.append(("block", value, start, end))
        index += 1
    return events


def _build_chapter(
    path: Path,
    title: str,
    order: int,
    blocks: tuple[tuple[str, int, int], ...],
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
    block: tuple[str, int, int],
    rules: tuple[Pronunciation, ...],
) -> SpeechSegment | Pause | None:
    text, start, end = block
    location = SourceLocation(path=path, start_line=start, end_line=end)
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
    digest = hashlib.sha256(spoken.encode()).hexdigest()
    return SpeechSegment(
        id=f"{chapter_id}-{item_index:04d}-{digest[:10]}",
        chapter_id=chapter_id,
        text=spoken,
        source=location,
        sha256=digest,
    )


def _apply_directives(source: str, *, path: Path, max_pause_ms: int) -> str:
    if _DIRECTIVE_GAP in source:
        raise ValidationError(f"{path}: source contains a reserved directive marker")
    _validate_directive_tokens(source, path)
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
    recognized = {match.start() for match in _DIRECTIVE.finditer(source)}
    for generic in re.finditer(r"<!--\s*yakbox:speech:", source):
        if generic.start() not in recognized:
            line = source.count("\n", 0, generic.start()) + 1
            raise ValidationError(f"{path}:{line}: malformed yakbox speech directive")


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
    value = item.text if isinstance(item, SpeechSegment) else str(item.milliseconds)
    return f"{chapter_id}:{value}"


def _removed_text_gap(value: str) -> str:
    newlines = "\n" * value.count("\n")
    return newlines if newlines else _DIRECTIVE_GAP


def _unicode_safe_boundary(text: str, maximum: int) -> int:
    point = maximum
    while point > 0 and (
        unicodedata.combining(text[point]) != 0
        or text[point] in {"\ufe0e", "\ufe0f"}
        or text[point] == "\u200d"
        or text[point - 1] == "\u200d"
    ):
        point -= 1
    if point == 0:
        raise ValidationError(
            "A single Unicode grapheme exceeds the configured chunk size"
        )
    return point
