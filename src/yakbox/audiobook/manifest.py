from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast

from yakbox.contracts import schema_uri
from yakbox.errors import ValidationError


@dataclass(frozen=True, slots=True)
class BookMetadata:
    title: str
    author: str | None = None
    narrator: str | None = None
    language: str = "en"
    copyright: str | None = None


@dataclass(frozen=True, slots=True)
class LogicalVoice:
    name: str
    display_name: str
    rights_basis: str = "not_applicable"
    reference_audio: Path | None = None


@dataclass(frozen=True, slots=True)
class FakeOptions:
    sample_rate: int = 16_000


@dataclass(frozen=True, slots=True)
class ChatterboxOptions:
    device: str = "auto"
    cfg_weight: float | None = None
    exaggeration: float | None = None
    seed: int | None = None
    max_processes: int = 1
    threads_per_process: int = 1
    worker_timeout_seconds: float = 3_600
    estimated_model_memory_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ResembleOptions:
    voice_uuid: str
    project_uuid: str | None = None
    use_hd: bool = False
    sample_rate: int | None = None


type BackendOptions = FakeOptions | ChatterboxOptions | ResembleOptions
MAX_PROVIDER_CONCURRENCY = 100


@dataclass(frozen=True, slots=True)
class BackendProfile:
    name: str
    backend: str
    voice: str
    executor: str
    options: BackendOptions


@dataclass(frozen=True, slots=True)
class BuildTarget:
    name: str
    profile: str
    output_root: Path
    chunk_chars: int = 2_800
    mastering: bool = True
    wav_sample_rate: int = 44_100
    mp3_bitrate: str = "192k"
    m4b: bool = False
    provider_concurrency: int = 5
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
    keep_successful_runs: int = 3
    audition_days: int | None = 30
    preview_days: int | None = 7
    raw_until_release: bool = True


@dataclass(frozen=True, slots=True)
class AudiobookManifest:
    path: Path
    schema_version: int
    book: BookMetadata
    sources: tuple[Path, ...]
    pronunciations: Path | None
    voices: tuple[LogicalVoice, ...]
    profiles: tuple[BackendProfile, ...]
    targets: tuple[BuildTarget, ...]
    retention: RetentionPolicy
    max_pause_ms: int = 30_000

    @property
    def root(self) -> Path:
        return self.path.parent

    def profile(self, name: str) -> BackendProfile:
        for profile in self.profiles:
            if profile.name == name:
                return profile
        raise ValidationError(f"Unknown profile: {name}")

    def target(self, name: str) -> BuildTarget:
        for target in self.targets:
            if target.name == name:
                return target
        raise ValidationError(f"Unknown target: {name}")

    def voice(self, name: str) -> LogicalVoice:
        for voice in self.voices:
            if voice.name == name:
                return voice
        raise ValidationError(f"Unknown logical voice: {name}")


_ROOT_KEYS = {
    "$schema",
    "schema_version",
    "book",
    "sources",
    "pronunciations",
    "voices",
    "profiles",
    "targets",
    "source",
    "retention",
}


def load_manifest(path: Path) -> AudiobookManifest:
    resolved = path.expanduser().resolve()
    try:
        raw = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValidationError(f"Manifest does not exist: {resolved}") from error
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValidationError(f"Cannot read manifest {resolved}: {error}") from error
    _reject_unknown(raw, _ROOT_KEYS, "manifest")
    if raw.get("$schema") != schema_uri("audiobook-manifest"):
        raise ValidationError(
            f'yakbox.toml requires "$schema" = "{schema_uri("audiobook-manifest")}"'
        )
    if raw.get("schema_version") != 1:
        raise ValidationError("yakbox.toml requires schema_version = 1")
    root = resolved.parent
    book_raw = _table(raw, "book")
    _reject_unknown(
        book_raw, {"title", "author", "narrator", "language", "copyright"}, "book"
    )
    title = _required_string(book_raw, "title", "book")
    sources_raw = raw.get("sources", book_raw.get("sources"))
    if sources_raw is None:
        source_table = raw.get("source")
        if isinstance(source_table, dict):
            sources_raw = source_table.get("paths")
    sources = _paths(sources_raw, root)
    voices = _parse_voices(raw.get("voices"), root)
    profiles = _parse_profiles(raw.get("profiles"))
    targets = _parse_targets(raw.get("targets"), root)
    retention = _parse_retention(raw.get("retention"))
    for target in targets:
        if target.output_root == root:
            raise ValidationError("A target output_root cannot be the workspace root")
        for source in sources:
            if source.is_relative_to(target.output_root):
                raise ValidationError(
                    f"Target {target.name!r} output_root contains source file {source}"
                )
    for profile in profiles:
        if profile.voice not in {voice.name for voice in voices}:
            raise ValidationError(
                f"Profile {profile.name!r} references unknown voice {profile.voice!r}"
            )
    for target in targets:
        if target.profile not in {profile.name for profile in profiles}:
            raise ValidationError(
                f"Target {target.name!r} references unknown profile {target.profile!r}"
            )
    pronunciation_value = raw.get("pronunciations")
    if pronunciation_value is not None and not isinstance(pronunciation_value, str):
        raise ValidationError("pronunciations must be a relative path string")
    pronunciation_path = (
        _workspace_path(root, pronunciation_value, "pronunciations", must_exist=True)
        if isinstance(pronunciation_value, str)
        else None
    )
    source_table = raw.get("source")
    if source_table is not None and not isinstance(source_table, dict):
        raise ValidationError("source must be a TOML table")
    max_pause_ms = 30_000
    if isinstance(source_table, dict):
        _reject_unknown(source_table, {"paths", "max_pause_ms"}, "source")
        max_pause_ms = _positive_int(
            source_table.get("max_pause_ms", 30_000), "source.max_pause_ms"
        )
    return AudiobookManifest(
        path=resolved,
        schema_version=1,
        book=BookMetadata(
            title=title,
            author=_optional_string(book_raw.get("author"), "book.author"),
            narrator=_optional_string(book_raw.get("narrator"), "book.narrator"),
            language=_string_or_default(
                book_raw.get("language"), "book.language", "en"
            ),
            copyright=_optional_string(book_raw.get("copyright"), "book.copyright"),
        ),
        sources=sources,
        pronunciations=pronunciation_path,
        voices=voices,
        profiles=profiles,
        targets=targets,
        retention=retention,
        max_pause_ms=max_pause_ms,
    )


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
                seed=_int_or_none(item.get("seed"), f"profiles.{name}.seed"),
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


def _parse_targets(value: object, root: Path) -> tuple[BuildTarget, ...]:
    table = _named_tables(
        value,
        "targets",
        {"default": {"profile": "default"}},
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
        "provider_concurrency",
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
                provider_concurrency=concurrency,
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


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be boolean")
    return value


def _device(value: object, name: str) -> str:
    result = _string_or_default(value, name, "auto")
    if result not in {"auto", "cpu", "cuda", "mps"}:
        raise ValidationError(f"{name} must be auto, cpu, cuda, or mps")
    return result


def _mp3_bitrate(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[1-9]\d*k", value) is None:
        raise ValidationError(f"{name} must look like 192k")
    return value


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


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return value
