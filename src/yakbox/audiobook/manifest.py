from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import cast

from yakbox.contracts import schema_uri
from yakbox.errors import ValidationError
from yakbox.speech.alignment import lexical_tokens
from yakbox.speech.short_utterances import (
    CarrierPosition,
    ShortUtteranceFailure,
    ShortUtterancePolicy,
    ShortUtteranceStrategy,
)

_MAXIMUM_REPAIR_TAKES = 20


@dataclass(frozen=True, slots=True)
class BookMetadata:
    """Descriptive and publishing metadata embedded in audiobook outputs."""

    title: str
    subtitle: str | None = None
    author: str | None = None
    narrator: str | None = None
    language: str = "en"
    copyright: str | None = None
    publisher: str | None = None
    genre: str | None = None
    series: str | None = None
    series_position: str | None = None
    isbn: str | None = None
    publication_date: str | None = None
    cover: Path | None = None


@dataclass(frozen=True, slots=True)
class LogicalVoice:
    """Manifest voice name mapped to rights and optional reference audio."""

    name: str
    display_name: str
    rights_basis: str = "not_applicable"
    reference_audio: Path | None = None


@dataclass(frozen=True, slots=True)
class FakeOptions:
    """Configuration for deterministic fake-backend audio generation."""

    sample_rate: int = 16_000


@dataclass(frozen=True, slots=True)
class ChatterboxOptions:
    """Runtime and synthesis controls for the local Chatterbox backend."""

    device: str = "cpu"
    cfg_weight: float | None = None
    exaggeration: float | None = None
    seed: int | None = 0
    max_processes: int = 1
    threads_per_process: int = 1
    worker_timeout_seconds: float = 3_600
    estimated_model_memory_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ResembleOptions:
    """Voice, project, and output controls for the Resemble backend."""

    voice_uuid: str
    project_uuid: str | None = None
    use_hd: bool = False
    sample_rate: int | None = None


type BackendOptions = FakeOptions | ChatterboxOptions | ResembleOptions
MAX_PROVIDER_CONCURRENCY = 100
MAX_MEDIA_CONCURRENCY = 32


@dataclass(frozen=True, slots=True)
class BackendProfile:
    """Named backend, voice, executor, and options used by build targets."""

    name: str
    backend: str
    voice: str
    executor: str
    options: BackendOptions


@dataclass(frozen=True, slots=True)
class CharacterRole:
    """Role metadata, profile mapping, and optional performance controls."""

    name: str
    display_name: str
    profile: str
    cfg_weight: float | None = None
    exaggeration: float | None = None
    seed: int | None = None
    gender: str = "unspecified"


@dataclass(frozen=True, slots=True)
class DialoguePolicy:
    """Controls routed dialogue, tags, and ambiguous-attribution assistance."""

    attribution_assistance: str = "warn"
    short_utterance_words: int = 3
    strip_attribution_tags: bool = False
    routes: Path | None = None
    expressive_tag_handling: str = "context"
    retain_first_attribution_per_scene: bool = False


@dataclass(frozen=True, slots=True)
class WhisperQaPolicy:
    """Managed chapter, join-cache, and phoneme-alignment policy."""

    chapter_verification: bool = False
    cache_enabled: bool = True
    cache_directory: Path = Path(".yakbox/cache/whisper")
    join_coalesce_gap_ms: int = 100
    phoneme_alignment: bool = False
    phoneme_backend: str = "wav2vec2-ctc"
    phoneme_model: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    phoneme_revision: str | None = "c43348bbaa5a77692c8e7bf3409d683474fdf2a4"
    phoneme_language: str = "en-us"
    phoneme_timeout_seconds: float = 180.0
    minimum_phoneme_confidence: float = 0.20
    manuscript_aliases: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if self.join_coalesce_gap_ms < 0:
            raise ValidationError("whisper_qa.join_coalesce_gap_ms cannot be negative")
        if self.phoneme_timeout_seconds <= 0 or not math.isfinite(
            self.phoneme_timeout_seconds
        ):
            raise ValidationError("whisper_qa.phoneme_timeout_seconds must be positive")
        if not 0 <= self.minimum_phoneme_confidence <= 1:
            raise ValidationError(
                "whisper_qa.minimum_phoneme_confidence must be between 0 and 1"
            )

    @property
    def manuscript_alias_map(self) -> dict[str, tuple[str, ...]]:
        """Return reviewed chapter-ASR spellings keyed by manuscript spelling."""
        return dict(self.manuscript_aliases)


@dataclass(frozen=True, slots=True)
class BuildTarget:
    """Named output pipeline with stage, quality, concurrency, and budget policy."""

    name: str
    profile: str
    output_root: Path
    chunk_chars: int = 2_800
    mastering: bool = True
    wav_sample_rate: int = 44_100
    mp3_bitrate: str = "192k"
    m4b: bool = False
    m4b_bitrate: str = "192k"
    provider_concurrency: int = 5
    media_concurrency: int = 2
    from_stage: str = "synthesize"
    through_stage: str = "inspect"
    quality_min_lufs: float | None = None
    quality_max_lufs: float | None = None
    quality_max_true_peak_dbfs: float | None = None
    quality_max_leading_silence_seconds: float | None = None
    quality_max_trailing_silence_seconds: float | None = None
    max_submitted_characters: int | None = None
    max_provider_requests: int | None = None
    max_estimated_spend: Decimal | None = None
    currency: str | None = None
    pricing_source: str | None = None
    price_per_character: Decimal | None = None
    confirm_above_characters: int | None = None
    confirm_above_requests: int | None = None
    storage_budget_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Workspace retention rules used when planning artifact cleanup."""

    keep_successful_runs: int = 3
    audition_days: int | None = 30
    preview_days: int | None = 7
    raw_until_release: bool = True


@dataclass(frozen=True, slots=True)
class RepairPolicy:
    """Configurable defaults for localized regeneration and approval."""

    mode: str = "context"
    takes: int = 4
    whisper_qa: bool = True
    rebuild_on_approval: bool = True
    minimum_passing_takes: int = 2


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    """Opt-in persistent local-model runtime controls."""

    enabled: bool = False
    idle_timeout_seconds: float = 900.0
    conditioning_cache_size: int = 8
    maximum_memory_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class AudiobookManifest:
    """Validated, path-resolved configuration for an audiobook workspace."""

    path: Path
    schema_version: int
    book: BookMetadata
    sources: tuple[Path, ...]
    pronunciations: Path | None
    voices: tuple[LogicalVoice, ...]
    profiles: tuple[BackendProfile, ...]
    characters: tuple[CharacterRole, ...]
    dialogue: DialoguePolicy
    whisper_qa: WhisperQaPolicy
    short_utterances: ShortUtterancePolicy
    targets: tuple[BuildTarget, ...]
    retention: RetentionPolicy
    repairs: RepairPolicy = RepairPolicy()
    runtime: RuntimePolicy = RuntimePolicy()
    max_pause_ms: int = 30_000

    @property
    def root(self) -> Path:
        """Return the directory containing the manifest."""
        return self.path.parent

    def profile(self, name: str) -> BackendProfile:
        """Return a named backend profile or raise `ValidationError`."""
        for profile in self.profiles:
            if profile.name == name:
                return profile
        raise ValidationError(f"Unknown profile: {name}")

    def target(self, name: str) -> BuildTarget:
        """Return a named build target or raise `ValidationError`."""
        for target in self.targets:
            if target.name == name:
                return target
        raise ValidationError(f"Unknown target: {name}")

    def voice(self, name: str) -> LogicalVoice:
        """Return a named logical voice or raise `ValidationError`."""
        for voice in self.voices:
            if voice.name == name:
                return voice
        raise ValidationError(f"Unknown logical voice: {name}")

    def character(self, name: str) -> CharacterRole:
        """Return a configured narrator or character role."""
        for character in self.characters:
            if character.name == name:
                return character
        raise ValidationError(f"Unknown character: {name}")

    def profile_for_speaker(
        self,
        speaker: str,
        *,
        fallback_profile: str,
    ) -> BackendProfile:
        """Resolve a speaker profile with character-level performance overrides."""
        if not self.characters:
            return self.profile(fallback_profile)
        character = self.character(speaker)
        profile = self.profile(character.profile)
        overrides = (
            character.cfg_weight,
            character.exaggeration,
            character.seed,
        )
        if not any(value is not None for value in overrides):
            return profile
        if not isinstance(profile.options, ChatterboxOptions):
            raise ValidationError(
                f"characters.{speaker} performance settings require a "
                "Chatterbox profile"
            )
        options = profile.options
        return replace(
            profile,
            options=replace(
                options,
                cfg_weight=(
                    character.cfg_weight
                    if character.cfg_weight is not None
                    else options.cfg_weight
                ),
                exaggeration=(
                    character.exaggeration
                    if character.exaggeration is not None
                    else options.exaggeration
                ),
                seed=character.seed if character.seed is not None else options.seed,
            ),
        )


_ROOT_KEYS = {
    "$schema",
    "schema_version",
    "book",
    "sources",
    "pronunciations",
    "voices",
    "profiles",
    "characters",
    "dialogue",
    "whisper_qa",
    "short_utterances",
    "repairs",
    "runtime",
    "targets",
    "source",
    "retention",
}


def load_manifest(path: Path) -> AudiobookManifest:
    """Load, validate, and resolve an audiobook TOML manifest."""
    resolved = path.expanduser().resolve()
    raw = _read_manifest(resolved)
    _validate_manifest_header(raw)
    root = resolved.parent
    book_raw = _table(raw, "book")
    book = _parse_book(book_raw, root)
    sources = _parse_sources(raw, book_raw, root)
    voices = _parse_voices(raw.get("voices"), root)
    profiles = _parse_profiles(raw.get("profiles"))
    characters = _parse_characters(raw.get("characters"))
    dialogue = _parse_dialogue(raw.get("dialogue"), root)
    whisper_qa = _parse_whisper_qa(raw.get("whisper_qa"), root)
    short_utterances = _parse_short_utterances(raw.get("short_utterances"))
    _validate_whisper_configuration(whisper_qa, short_utterances)
    targets = _parse_targets(raw.get("targets"), root)
    _validate_manifest_references(root, sources, voices, profiles, targets)
    _validate_character_references(characters, profiles)
    return AudiobookManifest(
        path=resolved,
        schema_version=1,
        book=book,
        sources=sources,
        pronunciations=_parse_pronunciation_path(raw, root),
        voices=voices,
        profiles=profiles,
        characters=characters,
        dialogue=dialogue,
        whisper_qa=whisper_qa,
        short_utterances=short_utterances,
        targets=targets,
        retention=_parse_retention(raw.get("retention")),
        repairs=_parse_repairs(raw.get("repairs")),
        runtime=_parse_runtime(raw.get("runtime")),
        max_pause_ms=_parse_source_options(raw),
    )


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValidationError(f"Manifest does not exist: {path}") from error
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValidationError(f"Cannot read manifest {path}: {error}") from error


def _validate_manifest_header(raw: dict[str, object]) -> None:
    _reject_unknown(raw, _ROOT_KEYS, "manifest")
    if raw.get("$schema") != schema_uri("audiobook-manifest"):
        raise ValidationError(
            f'yakbox.toml requires "$schema" = "{schema_uri("audiobook-manifest")}"'
        )
    if raw.get("schema_version") != 1:
        raise ValidationError("yakbox.toml requires schema_version = 1")


def _parse_book(raw: dict[str, object], root: Path) -> BookMetadata:
    _reject_unknown(
        raw,
        {
            "title",
            "subtitle",
            "author",
            "narrator",
            "language",
            "copyright",
            "publisher",
            "genre",
            "series",
            "series_position",
            "isbn",
            "publication_date",
            "cover",
        },
        "book",
    )
    title = _required_string(raw, "title", "book")
    cover_value = raw.get("cover")
    if cover_value is not None and not isinstance(cover_value, str):
        raise ValidationError("book.cover must be a relative path string")
    return BookMetadata(
        title=title,
        subtitle=_optional_string(raw.get("subtitle"), "book.subtitle"),
        author=_optional_string(raw.get("author"), "book.author"),
        narrator=_optional_string(raw.get("narrator"), "book.narrator"),
        language=_string_or_default(raw.get("language"), "book.language", "en"),
        copyright=_optional_string(raw.get("copyright"), "book.copyright"),
        publisher=_optional_string(raw.get("publisher"), "book.publisher"),
        genre=_optional_string(raw.get("genre"), "book.genre"),
        series=_optional_string(raw.get("series"), "book.series"),
        series_position=_string_or_number_or_none(
            raw.get("series_position"),
            "book.series_position",
        ),
        isbn=_optional_string(raw.get("isbn"), "book.isbn"),
        publication_date=_optional_string(
            raw.get("publication_date"),
            "book.publication_date",
        ),
        cover=(
            _workspace_path(root, cover_value, "book.cover", must_exist=True)
            if isinstance(cover_value, str)
            else None
        ),
    )


def _parse_sources(
    raw: dict[str, object],
    book: dict[str, object],
    root: Path,
) -> tuple[Path, ...]:
    sources_raw = raw.get("sources", book.get("sources"))
    if sources_raw is None:
        source_table = raw.get("source")
        if isinstance(source_table, dict):
            sources_raw = source_table.get("paths")
    return _paths(sources_raw, root)


def _validate_manifest_references(
    root: Path,
    sources: tuple[Path, ...],
    voices: tuple[LogicalVoice, ...],
    profiles: tuple[BackendProfile, ...],
    targets: tuple[BuildTarget, ...],
) -> None:
    for target in targets:
        if target.output_root == root:
            raise ValidationError("A target output_root cannot be the workspace root")
        for source in sources:
            if source.is_relative_to(target.output_root):
                raise ValidationError(
                    f"Target {target.name!r} output_root contains source file {source}"
                )
    voice_names = {voice.name for voice in voices}
    for profile in profiles:
        if profile.voice not in voice_names:
            raise ValidationError(
                f"Profile {profile.name!r} references unknown voice {profile.voice!r}"
            )
    profile_names = {profile.name for profile in profiles}
    for target in targets:
        if target.profile not in profile_names:
            raise ValidationError(
                f"Target {target.name!r} references unknown profile {target.profile!r}"
            )


def _validate_character_references(
    characters: tuple[CharacterRole, ...],
    profiles: tuple[BackendProfile, ...],
) -> None:
    profiles_by_name = {profile.name: profile for profile in profiles}
    if characters and not any(character.name == "narrator" for character in characters):
        raise ValidationError("characters must define a narrator role")
    for character in characters:
        profile = profiles_by_name.get(character.profile)
        if profile is None:
            raise ValidationError(
                f"Character {character.name!r} references unknown profile "
                f"{character.profile!r}"
            )
        if (
            character.cfg_weight is not None
            or character.exaggeration is not None
            or character.seed is not None
        ) and not isinstance(profile.options, ChatterboxOptions):
            raise ValidationError(
                f"characters.{character.name} performance settings require a "
                "Chatterbox profile"
            )
    _validate_character_profile_compatibility(characters, profiles_by_name)


def _validate_character_profile_compatibility(
    characters: tuple[CharacterRole, ...],
    profiles: dict[str, BackendProfile],
) -> None:
    if not characters:
        return
    narrator = next(item for item in characters if item.name == "narrator")
    baseline = profiles[narrator.profile]
    for character in characters:
        profile = profiles[character.profile]
        if profile.backend != baseline.backend or profile.executor != baseline.executor:
            raise ValidationError(
                "Character profiles must use the narrator backend and executor; "
                f"characters.{character.name} uses {profile.backend}/{profile.executor}"
            )
        if isinstance(profile.options, ChatterboxOptions) and isinstance(
            baseline.options, ChatterboxOptions
        ):
            runtime = (
                profile.options.device,
                profile.options.max_processes,
                profile.options.threads_per_process,
                profile.options.worker_timeout_seconds,
                profile.options.estimated_model_memory_bytes,
            )
            baseline_runtime = (
                baseline.options.device,
                baseline.options.max_processes,
                baseline.options.threads_per_process,
                baseline.options.worker_timeout_seconds,
                baseline.options.estimated_model_memory_bytes,
            )
            if runtime != baseline_runtime:
                raise ValidationError(
                    "Character Chatterbox profiles must use the narrator runtime "
                    f"settings; characters.{character.name} differs"
                )


def _parse_pronunciation_path(raw: dict[str, object], root: Path) -> Path | None:
    pronunciation_value = raw.get("pronunciations")
    if pronunciation_value is not None and not isinstance(pronunciation_value, str):
        raise ValidationError("pronunciations must be a relative path string")
    return (
        _workspace_path(root, pronunciation_value, "pronunciations", must_exist=True)
        if isinstance(pronunciation_value, str)
        else None
    )


def _parse_source_options(raw: dict[str, object]) -> int:
    source_table = raw.get("source")
    if source_table is not None and not isinstance(source_table, dict):
        raise ValidationError("source must be a TOML table")
    if isinstance(source_table, dict):
        source_table = cast(dict[str, object], source_table)
        _reject_unknown(source_table, {"paths", "max_pause_ms"}, "source")
        return _positive_int(
            source_table.get("max_pause_ms", 30_000), "source.max_pause_ms"
        )
    return 30_000


def _parse_voices(value: object, root: Path) -> tuple[LogicalVoice, ...]:
    table = _named_tables(
        value,
        "voices",
        {"narrator": {"display_name": "Narrator"}},
    )
    result: list[LogicalVoice] = []
    for name, item in table.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(item, dict):
            raise ValidationError("voices must be named TOML tables")
        item = cast(dict[str, object], item)
        _reject_unknown(
            item, {"display_name", "rights_basis", "reference_audio"}, f"voices.{name}"
        )
        reference = item.get("reference_audio")
        display_name = _string_or_default(
            item.get("display_name"), f"voices.{name}.display_name", name
        )
        rights_basis = _string_or_default(
            item.get("rights_basis"),
            f"voices.{name}.rights_basis",
            "not_applicable",
        )
        if rights_basis not in {
            "not_applicable",
            "owned",
            "licensed",
            "consented",
            "public_domain",
            "restricted",
            "unknown",
        }:
            raise ValidationError(f"voices.{name}.rights_basis is invalid")
        if reference is not None and not isinstance(reference, str):
            raise ValidationError(
                f"voices.{name}.reference_audio must be a relative path string"
            )
        result.append(
            LogicalVoice(
                name=name,
                display_name=display_name,
                rights_basis=rights_basis,
                reference_audio=_workspace_path(
                    root,
                    reference,
                    f"voices.{name}.reference_audio",
                    must_exist=True,
                )
                if isinstance(reference, str)
                else None,
            )
        )
    return tuple(result)


def _parse_profiles(value: object) -> tuple[BackendProfile, ...]:
    table = _named_tables(
        value,
        "profiles",
        {"default": {"backend": "fake", "voice": "narrator"}},
    )
    result: list[BackendProfile] = []
    for name, item in table.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(item, dict):
            raise ValidationError("profiles must be named TOML tables")
        item = cast(dict[str, object], item)
        backend = _required_string(item, "backend", f"profiles.{name}").casefold()
        voice = _string_or_default(
            item.get("voice"), f"profiles.{name}.voice", "narrator"
        )
        default_executor = (
            "local-process"
            if backend in {"local", "chatterbox", "chatterbox-local"}
            else "async"
        )
        executor = _string_or_default(
            item.get("executor"),
            f"profiles.{name}.executor",
            default_executor,
        )
        allowed_executors = (
            {"in-process", "local-process"}
            if backend in {"local", "chatterbox", "chatterbox-local"}
            else {"async"}
        )
        if executor not in allowed_executors:
            raise ValidationError(
                f"profiles.{name}.executor {executor!r} is incompatible with {backend}"
            )
        common = {"backend", "voice", "executor"}
        if backend == "fake":
            _reject_unknown(item, common | {"sample_rate"}, f"profiles.{name}")
            options: BackendOptions = FakeOptions(
                sample_rate=_positive_int(
                    item.get("sample_rate", 16_000), f"profiles.{name}.sample_rate"
                )
            )
        elif backend in {"local", "chatterbox", "chatterbox-local"}:
            _reject_unknown(
                item,
                common
                | {
                    "device",
                    "cfg_weight",
                    "exaggeration",
                    "seed",
                    "max_processes",
                    "threads_per_process",
                    "worker_timeout_seconds",
                    "estimated_model_memory_bytes",
                },
                f"profiles.{name}",
            )
            max_processes = _positive_int(
                item.get("max_processes", 1),
                f"profiles.{name}.max_processes",
            )
            if max_processes != 1:
                raise ValidationError(
                    f"profiles.{name}.max_processes must be 1 because a local "
                    "model instance renders sequentially"
                )
            options = ChatterboxOptions(
                device=_device(item.get("device"), f"profiles.{name}.device"),
                cfg_weight=_float_or_none(
                    item.get("cfg_weight"), f"profiles.{name}.cfg_weight"
                ),
                exaggeration=_float_or_none(
                    item.get("exaggeration"), f"profiles.{name}.exaggeration"
                ),
                seed=_int_or_none(item.get("seed", 0), f"profiles.{name}.seed"),
                max_processes=max_processes,
                threads_per_process=_positive_int(
                    item.get("threads_per_process", 1),
                    f"profiles.{name}.threads_per_process",
                ),
                worker_timeout_seconds=_positive_float(
                    item.get("worker_timeout_seconds", 3_600),
                    f"profiles.{name}.worker_timeout_seconds",
                ),
                estimated_model_memory_bytes=_positive_int_or_none(
                    item.get("estimated_model_memory_bytes"),
                    f"profiles.{name}.estimated_model_memory_bytes",
                ),
            )
        elif backend in {"resemble", "cloud"}:
            _reject_unknown(
                item,
                common | {"voice_uuid", "project_uuid", "use_hd", "sample_rate"},
                f"profiles.{name}",
            )
            options = ResembleOptions(
                voice_uuid=_required_string(item, "voice_uuid", f"profiles.{name}"),
                project_uuid=_optional_string(
                    item.get("project_uuid"), f"profiles.{name}.project_uuid"
                ),
                use_hd=_boolean(item.get("use_hd", False), f"profiles.{name}.use_hd"),
                sample_rate=_positive_int_or_none(
                    item.get("sample_rate"), f"profiles.{name}.sample_rate"
                ),
            )
        elif backend in {"remote", "chatterbox-remote"}:
            raise ValidationError(
                "Remote Chatterbox cannot be configured until its API "
                "contract is defined"
            )
        else:
            raise ValidationError(f"Unsupported backend in profile {name!r}: {backend}")
        result.append(
            BackendProfile(
                name=name,
                backend=backend,
                voice=voice,
                executor=executor,
                options=options,
            )
        )
    return tuple(result)


def _parse_characters(value: object) -> tuple[CharacterRole, ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValidationError("characters must be named TOML tables")
    table = cast(dict[str, object], value)
    result: list[CharacterRole] = []
    for name, raw in table.items():
        context = f"characters.{name}"
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[a-z][a-z0-9_-]*", name) is None
            or not isinstance(raw, dict)
        ):
            raise ValidationError("characters must use lowercase named TOML tables")
        item = cast(dict[str, object], raw)
        _reject_unknown(
            item,
            {
                "display_name",
                "profile",
                "gender",
                "cfg_weight",
                "exaggeration",
                "seed",
            },
            context,
        )
        result.append(
            CharacterRole(
                name=name,
                display_name=_string_or_default(
                    item.get("display_name"), f"{context}.display_name", name
                ),
                profile=_required_string(item, "profile", context),
                gender=_character_gender(item.get("gender"), f"{context}.gender"),
                cfg_weight=_float_or_none(
                    item.get("cfg_weight"), f"{context}.cfg_weight"
                ),
                exaggeration=_float_or_none(
                    item.get("exaggeration"), f"{context}.exaggeration"
                ),
                seed=_int_or_none(item.get("seed"), f"{context}.seed"),
            )
        )
    return tuple(result)


def _parse_dialogue(value: object, root: Path) -> DialoguePolicy:
    if value is None:
        return DialoguePolicy()
    if not isinstance(value, dict):
        raise ValidationError("dialogue must be a TOML table")
    table = cast(dict[str, object], value)
    _reject_unknown(
        table,
        {
            "attribution_assistance",
            "short_utterance_words",
            "strip_attribution_tags",
            "routes",
            "expressive_tag_handling",
            "retain_first_attribution_per_scene",
        },
        "dialogue",
    )
    assistance = _string_or_default(
        table.get("attribution_assistance"),
        "dialogue.attribution_assistance",
        "warn",
    )
    if assistance not in {"off", "warn", "error"}:
        raise ValidationError(
            "dialogue.attribution_assistance must be off, warn, or error"
        )
    expressive_tag_handling = _string_or_default(
        table.get("expressive_tag_handling"),
        "dialogue.expressive_tag_handling",
        "context",
    )
    if expressive_tag_handling not in {"context", "narrate", "strip"}:
        raise ValidationError(
            "dialogue.expressive_tag_handling must be context, narrate, or strip"
        )
    return DialoguePolicy(
        attribution_assistance=assistance,
        short_utterance_words=_positive_int(
            table.get("short_utterance_words", 3),
            "dialogue.short_utterance_words",
        ),
        strip_attribution_tags=_boolean(
            table.get("strip_attribution_tags", False),
            "dialogue.strip_attribution_tags",
        ),
        routes=_dialogue_routes_path(table.get("routes"), root),
        expressive_tag_handling=expressive_tag_handling,
        retain_first_attribution_per_scene=_boolean(
            table.get("retain_first_attribution_per_scene", False),
            "dialogue.retain_first_attribution_per_scene",
        ),
    )


def _dialogue_routes_path(value: object, root: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError("dialogue.routes must be a relative path string")
    return _workspace_path(
        root,
        value,
        "dialogue.routes",
        must_exist=True,
    )


def _parse_whisper_qa(value: object, root: Path) -> WhisperQaPolicy:
    if value is None:
        return WhisperQaPolicy(
            cache_directory=(root / ".yakbox/cache/whisper").resolve()
        )
    if not isinstance(value, dict):
        raise ValidationError("whisper_qa must be a TOML table")
    table = cast(dict[str, object], value)
    allowed = {
        "chapter_verification",
        "cache_enabled",
        "cache_directory",
        "join_coalesce_gap_ms",
        "phoneme_alignment",
        "phoneme_backend",
        "phoneme_model",
        "phoneme_revision",
        "phoneme_language",
        "phoneme_timeout_seconds",
        "minimum_phoneme_confidence",
        "manuscript_aliases",
    }
    _reject_unknown(table, allowed, "whisper_qa")
    cache_value = table.get("cache_directory", ".yakbox/cache/whisper")
    if not isinstance(cache_value, str) or not cache_value.strip():
        raise ValidationError("whisper_qa.cache_directory must be a relative path")
    backend = _string_or_default(
        table.get("phoneme_backend"),
        "whisper_qa.phoneme_backend",
        "wav2vec2-ctc",
    )
    if backend != "wav2vec2-ctc":
        raise ValidationError("whisper_qa.phoneme_backend must be wav2vec2-ctc")
    default_phoneme_model = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    phoneme_model = _string_or_default(
        table.get("phoneme_model"),
        "whisper_qa.phoneme_model",
        default_phoneme_model,
    )
    phoneme_revision = (
        _optional_string(
            table.get("phoneme_revision"),
            "whisper_qa.phoneme_revision",
        )
        if "phoneme_revision" in table
        else (
            "c43348bbaa5a77692c8e7bf3409d683474fdf2a4"
            if phoneme_model == default_phoneme_model
            else None
        )
    )
    return WhisperQaPolicy(
        chapter_verification=_boolean(
            table.get("chapter_verification", False),
            "whisper_qa.chapter_verification",
        ),
        cache_enabled=_boolean(
            table.get("cache_enabled", True), "whisper_qa.cache_enabled"
        ),
        cache_directory=_workspace_path(
            root, cache_value, "whisper_qa.cache_directory"
        ),
        join_coalesce_gap_ms=_nonnegative_int(
            table.get("join_coalesce_gap_ms", 100),
            "whisper_qa.join_coalesce_gap_ms",
        ),
        phoneme_alignment=_boolean(
            table.get("phoneme_alignment", False),
            "whisper_qa.phoneme_alignment",
        ),
        phoneme_backend=backend,
        phoneme_model=phoneme_model,
        phoneme_revision=phoneme_revision,
        phoneme_language=_string_or_default(
            table.get("phoneme_language"), "whisper_qa.phoneme_language", "en-us"
        ),
        phoneme_timeout_seconds=_number_or_default(
            table.get("phoneme_timeout_seconds"),
            "whisper_qa.phoneme_timeout_seconds",
            180.0,
        ),
        minimum_phoneme_confidence=_bounded_confidence(
            table.get("minimum_phoneme_confidence", 0.20),
            "whisper_qa.minimum_phoneme_confidence",
        ),
        manuscript_aliases=_alignment_aliases(
            table.get("manuscript_aliases"), field="whisper_qa.manuscript_aliases"
        ),
    )


def _validate_whisper_configuration(
    policy: WhisperQaPolicy,
    short_utterances: ShortUtterancePolicy,
) -> None:
    if policy.chapter_verification and short_utterances.alignment_revision is None:
        raise ValidationError(
            "whisper_qa.chapter_verification requires a pinned "
            "short_utterances.alignment_revision"
        )
    if policy.phoneme_alignment and policy.phoneme_revision is None:
        raise ValidationError(
            "whisper_qa.phoneme_alignment requires phoneme_revision for "
            "reproducible builds"
        )


def _parse_short_utterances(value: object) -> ShortUtterancePolicy:
    if value is None:
        return ShortUtterancePolicy()
    if not isinstance(value, dict):
        raise ValidationError("short_utterances must be a TOML table")
    table = cast(dict[str, object], value)
    allowed = {
        "strategy",
        "maximum_words",
        "candidate_count",
        "prefer_natural_context",
        "carrier_positions",
        "alignment_backend",
        "alignment_model",
        "alignment_revision",
        "alignment_aliases",
        "prompted_timing",
        "decode_consensus",
        "prompt_sensitivity",
        "maximum_consensus_timing_delta_ms",
        "hallucination_silence_threshold",
        "automatic_join_inspection",
        "join_inspection_window_seconds",
        "alignment_timeout_seconds",
        "minimum_alignment_confidence",
        "minimum_extracted_confidence",
        "minimum_one_word_confidence",
        "minimum_short_phrase_confidence",
        "minimum_segment_average_log_probability",
        "maximum_segment_compression_ratio",
        "maximum_segment_no_speech_probability",
        "maximum_segment_temperature",
        "candidate_confidence_tolerance",
        "maximum_extra_speech_ms",
        "maximum_internal_token_gap_ms",
        "maximum_token_duration_ms",
        "acoustic_refinement",
        "acoustic_threshold_dbfs",
        "speech_island_gap_ms",
        "minimum_edge_silence_ms",
        "maximum_edge_silence_ms",
        "maximum_clipped_sample_ratio",
        "maximum_boundary_jump_ratio",
        "maximum_vad_disagreement_ms",
        "maximum_stationary_voiced_ms",
        "minimum_pause_ms",
        "pre_roll_ms",
        "post_roll_ms",
        "fade_ms",
        "failure",
        "require_review_for_one_word",
        "keep_candidates",
    }
    _reject_unknown(table, allowed, "short_utterances")
    strategy = _enum_value(
        table.get("strategy"),
        "short_utterances.strategy",
        ShortUtteranceStrategy,
        ShortUtteranceStrategy.DIRECT,
    )
    failure = _enum_value(
        table.get("failure"),
        "short_utterances.failure",
        ShortUtteranceFailure,
        ShortUtteranceFailure.ERROR,
    )
    positions = _carrier_positions(table.get("carrier_positions"))
    backend = _string_or_default(
        table.get("alignment_backend"),
        "short_utterances.alignment_backend",
        "mlx-whisper",
    )
    if backend != "mlx-whisper":
        raise ValidationError("short_utterances.alignment_backend must be mlx-whisper")
    default_model = "mlx-community/whisper-large-v3-turbo"
    model = _string_or_default(
        table.get("alignment_model"),
        "short_utterances.alignment_model",
        default_model,
    )
    revision = (
        _optional_string(
            table.get("alignment_revision"),
            "short_utterances.alignment_revision",
        )
        if "alignment_revision" in table
        else (
            "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
            if model == default_model
            else None
        )
    )
    if strategy is ShortUtteranceStrategy.CONTEXT_EXTRACT and revision is None:
        raise ValidationError(
            "A custom short_utterances.alignment_model requires "
            "alignment_revision for reproducible builds"
        )
    return ShortUtterancePolicy(
        strategy=strategy,
        maximum_words=_positive_int(
            table.get("maximum_words", 3), "short_utterances.maximum_words"
        ),
        candidate_count=_positive_int(
            table.get("candidate_count", 5), "short_utterances.candidate_count"
        ),
        prefer_natural_context=_boolean(
            table.get("prefer_natural_context", True),
            "short_utterances.prefer_natural_context",
        ),
        carrier_positions=positions,
        alignment_backend=backend,
        alignment_model=model,
        alignment_revision=revision,
        alignment_aliases=_alignment_aliases(table.get("alignment_aliases")),
        prompted_timing=_boolean(
            table.get("prompted_timing", True),
            "short_utterances.prompted_timing",
        ),
        decode_consensus=_boolean(
            table.get("decode_consensus", True),
            "short_utterances.decode_consensus",
        ),
        prompt_sensitivity=_boolean(
            table.get("prompt_sensitivity", True),
            "short_utterances.prompt_sensitivity",
        ),
        maximum_consensus_timing_delta_ms=_nonnegative_int(
            table.get("maximum_consensus_timing_delta_ms", 180),
            "short_utterances.maximum_consensus_timing_delta_ms",
        ),
        hallucination_silence_threshold=_number_or_default(
            table.get("hallucination_silence_threshold"),
            "short_utterances.hallucination_silence_threshold",
            0.8,
        ),
        automatic_join_inspection=_boolean(
            table.get("automatic_join_inspection", True),
            "short_utterances.automatic_join_inspection",
        ),
        join_inspection_window_seconds=_number_or_default(
            table.get("join_inspection_window_seconds"),
            "short_utterances.join_inspection_window_seconds",
            1.5,
        ),
        alignment_timeout_seconds=_number_or_default(
            table.get("alignment_timeout_seconds"),
            "short_utterances.alignment_timeout_seconds",
            180.0,
        ),
        minimum_alignment_confidence=_bounded_confidence(
            table.get("minimum_alignment_confidence", 0.5),
            "short_utterances.minimum_alignment_confidence",
        ),
        minimum_extracted_confidence=_bounded_confidence(
            table.get("minimum_extracted_confidence", 0.2),
            "short_utterances.minimum_extracted_confidence",
        ),
        minimum_one_word_confidence=_bounded_confidence(
            table.get("minimum_one_word_confidence", 0.6),
            "short_utterances.minimum_one_word_confidence",
        ),
        minimum_short_phrase_confidence=_bounded_confidence(
            table.get("minimum_short_phrase_confidence", 0.5),
            "short_utterances.minimum_short_phrase_confidence",
        ),
        minimum_segment_average_log_probability=_number_or_default(
            table.get("minimum_segment_average_log_probability"),
            "short_utterances.minimum_segment_average_log_probability",
            -1.0,
        ),
        maximum_segment_compression_ratio=_number_or_default(
            table.get("maximum_segment_compression_ratio"),
            "short_utterances.maximum_segment_compression_ratio",
            2.4,
        ),
        maximum_segment_no_speech_probability=_bounded_confidence(
            table.get("maximum_segment_no_speech_probability", 0.6),
            "short_utterances.maximum_segment_no_speech_probability",
        ),
        maximum_segment_temperature=_bounded_confidence(
            table.get("maximum_segment_temperature", 0.2),
            "short_utterances.maximum_segment_temperature",
        ),
        candidate_confidence_tolerance=_bounded_confidence(
            table.get("candidate_confidence_tolerance", 0.05),
            "short_utterances.candidate_confidence_tolerance",
        ),
        maximum_extra_speech_ms=_nonnegative_int(
            table.get("maximum_extra_speech_ms", 60),
            "short_utterances.maximum_extra_speech_ms",
        ),
        maximum_internal_token_gap_ms=_nonnegative_int(
            table.get("maximum_internal_token_gap_ms", 350),
            "short_utterances.maximum_internal_token_gap_ms",
        ),
        maximum_token_duration_ms=_positive_int(
            table.get("maximum_token_duration_ms", 1_200),
            "short_utterances.maximum_token_duration_ms",
        ),
        acoustic_refinement=_boolean(
            table.get("acoustic_refinement", True),
            "short_utterances.acoustic_refinement",
        ),
        acoustic_threshold_dbfs=_number_or_default(
            table.get("acoustic_threshold_dbfs"),
            "short_utterances.acoustic_threshold_dbfs",
            -48.0,
        ),
        speech_island_gap_ms=_nonnegative_int(
            table.get("speech_island_gap_ms", 300),
            "short_utterances.speech_island_gap_ms",
        ),
        minimum_edge_silence_ms=_nonnegative_int(
            table.get("minimum_edge_silence_ms", 10),
            "short_utterances.minimum_edge_silence_ms",
        ),
        maximum_edge_silence_ms=_nonnegative_int(
            table.get("maximum_edge_silence_ms", 120),
            "short_utterances.maximum_edge_silence_ms",
        ),
        maximum_clipped_sample_ratio=_bounded_confidence(
            table.get("maximum_clipped_sample_ratio", 0.005),
            "short_utterances.maximum_clipped_sample_ratio",
        ),
        maximum_boundary_jump_ratio=_bounded_confidence(
            table.get("maximum_boundary_jump_ratio", 0.35),
            "short_utterances.maximum_boundary_jump_ratio",
        ),
        maximum_vad_disagreement_ms=_nonnegative_int(
            table.get("maximum_vad_disagreement_ms", 500),
            "short_utterances.maximum_vad_disagreement_ms",
        ),
        maximum_stationary_voiced_ms=_nonnegative_int(
            table.get("maximum_stationary_voiced_ms", 1_200),
            "short_utterances.maximum_stationary_voiced_ms",
        ),
        minimum_pause_ms=_nonnegative_int(
            table.get("minimum_pause_ms", 180),
            "short_utterances.minimum_pause_ms",
        ),
        pre_roll_ms=_nonnegative_int(
            table.get("pre_roll_ms", 30), "short_utterances.pre_roll_ms"
        ),
        post_roll_ms=_nonnegative_int(
            table.get("post_roll_ms", 40), "short_utterances.post_roll_ms"
        ),
        fade_ms=_nonnegative_int(table.get("fade_ms", 8), "short_utterances.fade_ms"),
        failure=failure,
        require_review_for_one_word=_boolean(
            table.get("require_review_for_one_word", True),
            "short_utterances.require_review_for_one_word",
        ),
        keep_candidates=_boolean(
            table.get("keep_candidates", False),
            "short_utterances.keep_candidates",
        ),
    )


def _alignment_aliases(
    value: object,
    *,
    field: str = "short_utterances.alignment_aliases",
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValidationError(f"{field} must be a table")
    result: list[tuple[str, tuple[str, ...]]] = []
    for raw_word, raw_aliases in sorted(value.items()):
        if not isinstance(raw_word, str) or lexical_tokens(raw_word) != (
            raw_word.casefold(),
        ):
            raise ValidationError(f"{field} keys must be single words")
        if not isinstance(raw_aliases, list) or not raw_aliases:
            raise ValidationError(f"{field} values must be non-empty arrays")
        aliases: list[str] = []
        for raw_alias in raw_aliases:
            if not isinstance(raw_alias, str):
                raise ValidationError(f"{field} values must contain strings")
            tokens = lexical_tokens(raw_alias)
            if len(tokens) != 1:
                raise ValidationError(f"{field} values must be single words")
            aliases.append(tokens[0])
        result.append((raw_word.casefold(), tuple(dict.fromkeys(aliases))))
    return tuple(result)


def _carrier_positions(value: object) -> tuple[CarrierPosition, ...]:
    if value is None:
        return (CarrierPosition.MIDDLE,)
    if not isinstance(value, list) or not value:
        raise ValidationError(
            "short_utterances.carrier_positions must be a non-empty array"
        )
    positions = [
        _enum_value(
            item,
            "short_utterances.carrier_positions",
            CarrierPosition,
            CarrierPosition.MIDDLE,
            allowed=(
                CarrierPosition.MIDDLE,
                CarrierPosition.INITIAL,
                CarrierPosition.FINAL,
            ),
        )
        for item in value
    ]
    if len(set(positions)) != len(positions):
        raise ValidationError(
            "short_utterances.carrier_positions must not contain duplicates"
        )
    return tuple(positions)


def _parse_targets(value: object, root: Path) -> tuple[BuildTarget, ...]:
    table = _resolve_target_inheritance(
        _named_tables(
            value,
            "targets",
            {"default": {"profile": "default"}},
        )
    )
    result: list[BuildTarget] = []
    allowed = {
        "profile",
        "output_root",
        "chunk_chars",
        "mastering",
        "wav_sample_rate",
        "mp3_bitrate",
        "m4b",
        "m4b_bitrate",
        "provider_concurrency",
        "media_concurrency",
        "from_stage",
        "through_stage",
        "quality_min_lufs",
        "quality_max_lufs",
        "quality_max_true_peak_dbfs",
        "quality_max_leading_silence_seconds",
        "quality_max_trailing_silence_seconds",
        "max_submitted_characters",
        "max_provider_requests",
        "max_estimated_spend",
        "currency",
        "pricing_source",
        "price_per_character",
        "confirm_above_characters",
        "confirm_above_requests",
        "storage_budget_bytes",
    }
    for name, item in table.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(item, dict):
            raise ValidationError("targets must be named TOML tables")
        item = cast(dict[str, object], item)
        _reject_unknown(item, allowed, f"targets.{name}")
        output_value = item.get("output_root", "build/yakbox")
        if not isinstance(output_value, str) or not output_value.strip():
            raise ValidationError(
                f"targets.{name}.output_root must be a relative path string"
            )
        output_root = _workspace_path(
            root,
            output_value,
            f"targets.{name}.output_root",
        )
        concurrency = _positive_int(
            item.get("provider_concurrency", 5),
            f"targets.{name}.provider_concurrency",
        )
        if concurrency > MAX_PROVIDER_CONCURRENCY:
            raise ValidationError(
                f"targets.{name}.provider_concurrency must be at most 100"
            )
        media_concurrency = _positive_int(
            item.get("media_concurrency", 2),
            f"targets.{name}.media_concurrency",
        )
        if media_concurrency > MAX_MEDIA_CONCURRENCY:
            raise ValidationError(
                f"targets.{name}.media_concurrency must be at most "
                f"{MAX_MEDIA_CONCURRENCY}"
            )
        from_stage = _build_stage(
            item.get("from_stage", "synthesize"),
            f"targets.{name}.from_stage",
        )
        through_stage = _build_stage(
            item.get("through_stage", "inspect"),
            f"targets.{name}.through_stage",
        )
        stages = (
            "synthesize",
            "master",
            "verify_manuscript",
            "encode_mp3",
            "inspect",
        )
        if stages.index(from_stage) > stages.index(through_stage):
            raise ValidationError(
                f"targets.{name}.from_stage must not come after through_stage"
            )
        quality_min_lufs = _float_or_none(
            item.get("quality_min_lufs"),
            f"targets.{name}.quality_min_lufs",
        )
        quality_max_lufs = _float_or_none(
            item.get("quality_max_lufs"),
            f"targets.{name}.quality_max_lufs",
        )
        if (
            quality_min_lufs is not None
            and quality_max_lufs is not None
            and quality_min_lufs > quality_max_lufs
        ):
            raise ValidationError(
                f"targets.{name}.quality_min_lufs must not exceed quality_max_lufs"
            )
        maximum_spend = _decimal_or_none(
            item.get("max_estimated_spend"),
            f"targets.{name}.max_estimated_spend",
        )
        currency = _optional_string(item.get("currency"), f"targets.{name}.currency")
        pricing_source = _optional_string(
            item.get("pricing_source"), f"targets.{name}.pricing_source"
        )
        price_per_character = _decimal_or_none(
            item.get("price_per_character"),
            f"targets.{name}.price_per_character",
        )
        if maximum_spend is not None and (
            currency is None or pricing_source is None or price_per_character is None
        ):
            raise ValidationError(
                f"targets.{name} monetary budget requires currency, "
                "pricing_source, and price_per_character"
            )
        result.append(
            BuildTarget(
                name=name,
                profile=_string_or_default(
                    item.get("profile"), f"targets.{name}.profile", "default"
                ),
                output_root=output_root,
                chunk_chars=_positive_int(
                    item.get("chunk_chars", 2_800), f"targets.{name}.chunk_chars"
                ),
                mastering=_boolean(
                    item.get("mastering", True), f"targets.{name}.mastering"
                ),
                wav_sample_rate=_positive_int(
                    item.get("wav_sample_rate", 44_100),
                    f"targets.{name}.wav_sample_rate",
                ),
                mp3_bitrate=_mp3_bitrate(
                    item.get("mp3_bitrate", "192k"),
                    f"targets.{name}.mp3_bitrate",
                ),
                m4b=_boolean(item.get("m4b", False), f"targets.{name}.m4b"),
                m4b_bitrate=_mp3_bitrate(
                    item.get("m4b_bitrate", "192k"),
                    f"targets.{name}.m4b_bitrate",
                ),
                provider_concurrency=concurrency,
                media_concurrency=media_concurrency,
                from_stage=from_stage,
                through_stage=through_stage,
                quality_min_lufs=quality_min_lufs,
                quality_max_lufs=quality_max_lufs,
                quality_max_true_peak_dbfs=_float_or_none(
                    item.get("quality_max_true_peak_dbfs"),
                    f"targets.{name}.quality_max_true_peak_dbfs",
                ),
                quality_max_leading_silence_seconds=_nonnegative_float_or_none(
                    item.get("quality_max_leading_silence_seconds"),
                    f"targets.{name}.quality_max_leading_silence_seconds",
                ),
                quality_max_trailing_silence_seconds=_nonnegative_float_or_none(
                    item.get("quality_max_trailing_silence_seconds"),
                    f"targets.{name}.quality_max_trailing_silence_seconds",
                ),
                max_submitted_characters=_nonnegative_int_or_none(
                    item.get("max_submitted_characters"),
                    f"targets.{name}.max_submitted_characters",
                ),
                max_provider_requests=_nonnegative_int_or_none(
                    item.get("max_provider_requests"),
                    f"targets.{name}.max_provider_requests",
                ),
                max_estimated_spend=maximum_spend,
                currency=currency,
                pricing_source=pricing_source,
                price_per_character=price_per_character,
                confirm_above_characters=_nonnegative_int_or_none(
                    item.get("confirm_above_characters"),
                    f"targets.{name}.confirm_above_characters",
                ),
                confirm_above_requests=_nonnegative_int_or_none(
                    item.get("confirm_above_requests"),
                    f"targets.{name}.confirm_above_requests",
                ),
                storage_budget_bytes=_nonnegative_int_or_none(
                    item.get("storage_budget_bytes"),
                    f"targets.{name}.storage_budget_bytes",
                ),
            )
        )
    return tuple(result)


def _resolve_target_inheritance(
    table: dict[str, object],
) -> dict[str, object]:
    resolved: dict[str, object] = {}

    def resolve(name: str, trail: tuple[str, ...]) -> dict[str, object]:
        existing = resolved.get(name)
        if isinstance(existing, dict):
            return cast(dict[str, object], existing)
        if name in trail:
            raise ValidationError(
                "Target inheritance cycle: " + " -> ".join((*trail, name))
            )
        raw = table.get(name)
        if not isinstance(raw, dict):
            raise ValidationError(f"Unknown inherited target: {name}")
        item = cast(dict[str, object], raw)
        parent = item.get("extends")
        if parent is not None and (not isinstance(parent, str) or not parent.strip()):
            raise ValidationError(f"targets.{name}.extends must be a target name")
        base = resolve(parent, (*trail, name)) if isinstance(parent, str) else {}
        merged = {**base, **item}
        merged.pop("extends", None)
        resolved[name] = merged
        return merged

    for name in table:
        resolve(name, ())
    return resolved


def _parse_retention(value: object) -> RetentionPolicy:
    if value is None:
        return RetentionPolicy()
    if not isinstance(value, dict):
        raise ValidationError("retention must be a TOML table")
    table = cast(dict[str, object], value)
    _reject_unknown(
        table,
        {
            "keep_successful_runs",
            "audition_days",
            "preview_days",
            "raw_until_release",
        },
        "retention",
    )
    return RetentionPolicy(
        keep_successful_runs=_nonnegative_int(
            table.get("keep_successful_runs", 3),
            "retention.keep_successful_runs",
        ),
        audition_days=_nonnegative_int_or_none(
            table.get("audition_days", 30),
            "retention.audition_days",
        ),
        preview_days=_nonnegative_int_or_none(
            table.get("preview_days", 7),
            "retention.preview_days",
        ),
        raw_until_release=_boolean(
            table.get("raw_until_release", True),
            "retention.raw_until_release",
        ),
    )


def _parse_repairs(value: object) -> RepairPolicy:
    if value is None:
        return RepairPolicy()
    if not isinstance(value, dict):
        raise ValidationError("repairs must be a TOML table")
    table = cast(dict[str, object], value)
    _reject_unknown(
        table,
        {
            "mode",
            "takes",
            "minimum_passing_takes",
            "whisper_qa",
            "rebuild_on_approval",
        },
        "repairs",
    )
    mode = _string_or_default(table.get("mode"), "repairs.mode", "context")
    if mode not in {
        "target-only",
        "context",
        "sentence",
        "clause",
        "neighbors",
        "paragraph",
        "scene",
    }:
        raise ValidationError(
            "repairs.mode must be target-only, context, sentence, clause, "
            "neighbors, paragraph, or scene"
        )
    takes = _positive_int(table.get("takes", 4), "repairs.takes")
    if takes > _MAXIMUM_REPAIR_TAKES:
        raise ValidationError("repairs.takes must not exceed 20")
    minimum_passing_takes = _positive_int(
        table.get("minimum_passing_takes", min(2, takes)),
        "repairs.minimum_passing_takes",
    )
    if minimum_passing_takes > takes:
        raise ValidationError(
            "repairs.minimum_passing_takes must not exceed repairs.takes"
        )
    return RepairPolicy(
        mode=mode,
        takes=takes,
        minimum_passing_takes=minimum_passing_takes,
        whisper_qa=_boolean(table.get("whisper_qa", True), "repairs.whisper_qa"),
        rebuild_on_approval=_boolean(
            table.get("rebuild_on_approval", True),
            "repairs.rebuild_on_approval",
        ),
    )


def _parse_runtime(value: object) -> RuntimePolicy:
    if value is None:
        return RuntimePolicy()
    if not isinstance(value, dict):
        raise ValidationError("runtime must be a TOML table")
    table = cast(dict[str, object], value)
    _reject_unknown(
        table,
        {
            "enabled",
            "idle_timeout_seconds",
            "conditioning_cache_size",
            "maximum_memory_bytes",
        },
        "runtime",
    )
    return RuntimePolicy(
        enabled=_boolean(table.get("enabled", False), "runtime.enabled"),
        idle_timeout_seconds=_positive_float(
            table.get("idle_timeout_seconds", 900),
            "runtime.idle_timeout_seconds",
        ),
        conditioning_cache_size=_positive_int(
            table.get("conditioning_cache_size", 8),
            "runtime.conditioning_cache_size",
        ),
        maximum_memory_bytes=_positive_int_or_none(
            table.get("maximum_memory_bytes"),
            "runtime.maximum_memory_bytes",
        ),
    )


def _paths(value: object, root: Path) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise ValidationError("Manifest requires a non-empty sources array")
    paths: list[Path] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ValidationError("Every source path must be a string")
        candidate = _workspace_path(root, entry, "source", must_exist=True)
        paths.append(candidate)
    return tuple(paths)


def _table(raw: dict[str, object], name: str) -> dict[str, object]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValidationError(f"Manifest requires a [{name}] table")
    return cast(dict[str, object], value)


def _workspace_path(
    root: Path,
    value: str,
    name: str,
    *,
    must_exist: bool = False,
) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        raise ValidationError(f"{name} must be relative to the audiobook workspace")
    candidate = (root / raw).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValidationError(f"{name} escapes the audiobook workspace: {value}")
    if must_exist and not candidate.is_file():
        raise ValidationError(f"{name} does not exist: {candidate}")
    return candidate


def _required_string(table: dict[str, object], key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{context}.{key} must be a non-empty string")
    return value


def _reject_unknown(table: dict[str, object], allowed: set[str], context: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValidationError(f"Unknown {context} keys: {', '.join(unknown)}")


def _named_tables(
    value: object,
    name: str,
    default: dict[str, object],
) -> dict[str, object]:
    if value is None:
        return default
    if not isinstance(value, dict) or not value:
        raise ValidationError(f"{name} must be a non-empty set of named TOML tables")
    return cast(dict[str, object], value)


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value


def _string_or_default(value: object, name: str, default: str) -> str:
    if value is None:
        return default
    result = _optional_string(value, name)
    if result is None:
        raise ValidationError(f"{name} must be a non-empty string")
    return result


def _enum_value[EnumValue: StrEnum](
    value: object,
    name: str,
    enum_type: type[EnumValue],
    default: EnumValue,
    *,
    allowed: tuple[EnumValue, ...] | None = None,
) -> EnumValue:
    raw = _string_or_default(value, name, default.value)
    try:
        result = enum_type(raw)
    except ValueError as error:
        choices = allowed if allowed is not None else tuple(enum_type)
        options = ", ".join(item.value for item in choices)
        raise ValidationError(f"{name} must be one of: {options}") from error
    if allowed is not None and result not in allowed:
        options = ", ".join(item.value for item in allowed)
        raise ValidationError(f"{name} must be one of: {options}")
    return result


def _bounded_confidence(value: object, name: str) -> float:
    result = _float_or_none(value, name)
    if result is None or not 0 <= result <= 1:
        raise ValidationError(f"{name} must be between 0 and 1")
    return result


def _number_or_default(value: object, name: str, default: float) -> float:
    if value is None:
        return default
    result = _float_or_none(value, name)
    if result is None:
        raise ValidationError(f"{name} must be a number")
    return result


def _string_or_number_or_none(value: object, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValidationError(f"{name} must be a string or number")
    result = str(value).strip()
    if not result:
        raise ValidationError(f"{name} must not be empty")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be boolean")
    return value


def _device(value: object, name: str) -> str:
    result = _string_or_default(value, name, "cpu")
    if result not in {"auto", "cpu", "cuda", "mps"}:
        raise ValidationError(f"{name} must be auto, cpu, cuda, or mps")
    return result


def _character_gender(value: object, name: str) -> str:
    result = _string_or_default(value, name, "unspecified")
    if result not in {"female", "male", "unspecified"}:
        raise ValidationError(f"{name} must be female, male, or unspecified")
    return result


def _mp3_bitrate(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[1-9]\d*k", value) is None:
        raise ValidationError(f"{name} must look like 192k")
    return value


def _build_stage(value: object, name: str) -> str:
    result = _string_or_default(value, name, "synthesize")
    allowed = {
        "synthesize",
        "master",
        "verify_manuscript",
        "encode_mp3",
        "inspect",
    }
    if result not in allowed:
        raise ValidationError(
            f"{name} must be synthesize, master, verify_manuscript, "
            "encode_mp3, or inspect"
        )
    return result


def _nonnegative_int_or_none(value: object, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{name} must be a non-negative integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    result = _nonnegative_int_or_none(value, name)
    if result is None:
        raise ValidationError(f"{name} must be a non-negative integer")
    return result


def _decimal_or_none(value: object, name: str) -> Decimal | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a quoted decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValidationError(f"{name} is not a valid decimal") from error
    if not result.is_finite() or result < 0:
        raise ValidationError(f"{name} must be a finite non-negative decimal")
    return result


def _int_or_none(value: object, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer")
    return value


def _positive_int_or_none(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _float_or_none(value: object, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValidationError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _float_or_none(value, name)
    if result is None or result <= 0:
        raise ValidationError(f"{name} must be a positive number")
    return result


def _nonnegative_float_or_none(value: object, name: str) -> float | None:
    result = _float_or_none(value, name)
    if result is not None and result < 0:
        raise ValidationError(f"{name} must be a non-negative number")
    return result


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return value
