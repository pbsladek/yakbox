"""Deterministic preview and guarded write path for manifest version 2."""

from __future__ import annotations

import copy
import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_bytes, sha256_file
from yakbox.audiobook.build import BuildResult, build_audiobook
from yakbox.audiobook.manifest import AudiobookManifest, load_manifest
from yakbox.audiobook.planner import BuildPlan, plan_audiobook
from yakbox.audiobook.repairs import load_approved_repairs
from yakbox.audiobook.sources import NormalizedDocument
from yakbox.contracts import SCHEMA_BASE, runtime_metadata
from yakbox.errors import ValidationError, YakboxError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_manifest import (
    DraftAnalysisCachePolicy,
    DraftChapterAnalysisPolicy,
    DraftJoinAnalysisPolicy,
    DraftPhonemeAnalysisPolicy,
    DraftShortUtteranceAnalysisPolicy,
    DraftSpeechAnalysisConfig,
    default_draft_speech_analysis_config,
    parse_draft_manifest_speech_analysis,
)
from yakbox.speech.normalization import normalize_english

DRAFT_MANIFEST_SCHEMA_URI = f"{SCHEMA_BASE}/audiobook-manifest-v2.schema.json"
SPEECH_ANALYSIS_CACHE_VERSION = 2
DRAFT_COMMAND_MAP = (
    ("yakbox whisper inspect", "yakbox speech inspect"),
    ("yakbox whisper reinspect", "yakbox speech reinspect"),
    ("yakbox whisper verify-manuscript", "yakbox speech verify-manuscript"),
    ("yakbox whisper inspect-joins", "yakbox speech inspect-joins"),
    ("yakbox whisper inspect-phonemes", "yakbox speech inspect-phonemes"),
    ("yakbox whisper calibrate", "yakbox speech calibrate"),
    ("yakbox whisper qualify-voices", "yakbox speech qualify-voices"),
    ("yakbox whisper models ...", "yakbox models ..."),
    ("", "yakbox runtimes ..."),
)
DRAFT_PYTHON_EXPORTS = (
    "SpeechAnalysisPolicy",
    "SpeechRecognizer",
    "ForcedAligner",
    "SpeechVerification",
    "VerificationScope",
)
_BARE_TOML_KEY = re.compile(r"[A-Za-z0-9_-]+")

_LEGACY_SHORT_ANALYSIS_FIELDS = frozenset(
    {
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
        "maximum_vad_disagreement_ms",
        "maximum_stationary_voiced_ms",
    }
)


@dataclass(frozen=True, slots=True)
class MigrationFinding:
    """One deterministic migration loss, ambiguity, or evidence transition."""

    code: str
    path: str
    detail: str
    lossy: bool
    review_required: bool


@dataclass(frozen=True, slots=True)
class DraftPronunciationTerm:
    """Unambiguous replacement for one version-1 pronunciation record."""

    index: int
    written: str
    synthesis_hint: str
    expected_lexical: tuple[str, ...]
    phonemes: tuple[str, ...]
    match: str
    case: str
    priority: int
    language: str | None
    status: str
    enabled: bool
    notes: str | None
    review_required: bool


@dataclass(frozen=True, slots=True)
class PreservedRepairApproval:
    """Approved v1 replacement retained while its old QA becomes stale."""

    target: str
    repair_id: str
    chunk_id: str
    audio_path: str
    audio_sha256: str
    approval_fingerprint: str
    analysis_evidence: str = "stale_requires_v2_verification"


@dataclass(frozen=True, slots=True)
class ManifestMigrationPreview:
    """Deterministic, serializable preview plus its validated v1 planning source."""

    source_manifest_digest: str
    draft_manifest: dict[str, object]
    speech_analysis: DraftSpeechAnalysisConfig
    pronunciations: tuple[DraftPronunciationTerm, ...]
    preserved_repairs: tuple[PreservedRepairApproval, ...]
    findings: tuple[MigrationFinding, ...]
    _legacy_manifest: AudiobookManifest

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "manifest-migration-preview-v1",
            {
                "source_manifest_digest": self.source_manifest_digest,
                "draft_manifest": self.draft_manifest,
                "pronunciations": self.pronunciations,
                "preserved_repairs": self.preserved_repairs,
                "findings": self.findings,
            },
        )

    @property
    def review_required(self) -> bool:
        return any(item.review_required for item in self.findings)

    def to_dict(self) -> dict[str, object]:
        """Return the timestamp-free preview contract used by cutover tests."""
        return {
            "preview_version": 1,
            "source_manifest_digest": self.source_manifest_digest,
            "fingerprint": self.fingerprint,
            "draft_manifest": copy.deepcopy(self.draft_manifest),
            "pronunciations": [asdict(item) for item in self.pronunciations],
            "preserved_repairs": [asdict(item) for item in self.preserved_repairs],
            "findings": [asdict(item) for item in self.findings],
            "review_required": self.review_required,
        }


@dataclass(frozen=True, slots=True)
class ManifestMigrationWriteResult:
    """Exact files and source identity committed by an approved migration."""

    source_manifest_digest: str
    preview_fingerprint: str
    manifest_path: Path
    manifest_digest: str
    backup_path: Path | None
    pronunciation_path: Path | None
    pronunciation_digest: str | None
    resolved_finding_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("audiobook-manifest-migration"),
            "source_manifest_digest": self.source_manifest_digest,
            "preview_fingerprint": self.preview_fingerprint,
            "manifest_path": self.manifest_path.as_posix(),
            "manifest_digest": self.manifest_digest,
            "backup_path": (
                self.backup_path.as_posix() if self.backup_path is not None else None
            ),
            "pronunciation_path": (
                self.pronunciation_path.as_posix()
                if self.pronunciation_path is not None
                else None
            ),
            "pronunciation_digest": self.pronunciation_digest,
            "resolved_finding_codes": list(self.resolved_finding_codes),
        }


def preview_manifest_migration(path: Path) -> ManifestMigrationPreview:
    """Read and translate a v1 project without modifying any workspace file."""
    resolved = path.expanduser().resolve()
    manifest = load_manifest(resolved)
    raw = _read_toml(resolved, "manifest")
    config, findings = _translate_analysis(manifest, raw)
    pronunciations, pronunciation_findings = _translate_pronunciations(
        manifest.pronunciations
    )
    repairs = _preserved_repairs(manifest)
    if repairs:
        findings.append(
            MigrationFinding(
                "approved_repairs_preserved_analysis_stale",
                ".yakbox/repairs",
                "Approved audio and source bindings remain; v1 analysis cannot "
                "verify a v2 release.",
                False,
                False,
            )
        )
    draft = _draft_manifest(raw, config)
    parse_draft_manifest_speech_analysis(draft)
    return ManifestMigrationPreview(
        source_manifest_digest=sha256_file(resolved),
        draft_manifest=draft,
        speech_analysis=config,
        pronunciations=pronunciations,
        preserved_repairs=repairs,
        findings=(*findings, *pronunciation_findings),
        _legacy_manifest=manifest,
    )


def write_manifest_migration(
    preview: ManifestMigrationPreview,
    *,
    destination: Path | None = None,
    backup: bool = True,
    resolved_finding_codes: Sequence[str] = (),
) -> ManifestMigrationWriteResult:
    """Atomically write a reviewed preview to a clean path or with a v1 backup."""
    source = preview._legacy_manifest.path.resolve()
    if sha256_file(source) != preview.source_manifest_digest:
        raise ValidationError("Manifest changed after migration preview")
    resolved = _resolved_migration_findings(preview, resolved_finding_codes)
    target, same_path, backup_path = _migration_destination(
        source, destination=destination, backup=backup
    )
    payload = render_manifest_toml(preview.draft_manifest)
    # Parse and revalidate the exact bytes before making any filesystem change.
    parsed = tomllib.loads(payload.decode("utf-8"))
    parse_draft_manifest_speech_analysis(parsed)
    pronunciation_path, pronunciation_bytes = _pronunciation_output(preview, target)
    original_manifest = source.read_bytes() if same_path else None
    original_pronunciation = (
        pronunciation_path.read_bytes()
        if pronunciation_path is not None and pronunciation_path.is_file()
        else None
    )
    _write_migration_backups(
        source,
        backup_path=backup_path,
        pronunciation_path=pronunciation_path,
        same_path=same_path,
        backup=backup,
    )
    try:
        if pronunciation_path is not None and pronunciation_bytes is not None:
            atomic_write_bytes(pronunciation_path, pronunciation_bytes, overwrite=True)
        atomic_write_bytes(target, payload, overwrite=same_path)
    except OSError, YakboxError:
        _rollback_migration(
            target=target,
            original_manifest=original_manifest,
            pronunciation_path=pronunciation_path,
            original_pronunciation=original_pronunciation,
        )
        raise
    return ManifestMigrationWriteResult(
        source_manifest_digest=preview.source_manifest_digest,
        preview_fingerprint=preview.fingerprint,
        manifest_path=target,
        manifest_digest=sha256_file(target),
        backup_path=backup_path,
        pronunciation_path=pronunciation_path,
        pronunciation_digest=(
            sha256_file(pronunciation_path)
            if pronunciation_path is not None and pronunciation_path.is_file()
            else None
        ),
        resolved_finding_codes=resolved,
    )


def _resolved_migration_findings(
    preview: ManifestMigrationPreview, values: Sequence[str]
) -> tuple[str, ...]:
    required = {finding.code for finding in preview.findings if finding.review_required}
    resolved = tuple(sorted(set(values)))
    missing = tuple(sorted(required - set(resolved)))
    unknown = tuple(sorted(set(resolved) - {item.code for item in preview.findings}))
    if missing:
        raise ValidationError(
            f"Migration finding requires explicit resolution: {missing[0]}"
        )
    if unknown:
        raise ValidationError(f"Unknown migration finding resolution: {unknown[0]}")
    return resolved


def _migration_destination(
    source: Path, *, destination: Path | None, backup: bool
) -> tuple[Path, bool, Path | None]:
    target = (destination or source).expanduser().resolve()
    same_path = target == source
    if target.exists() and not same_path:
        raise ValidationError("Migration destination must not already exist")
    if same_path and not backup:
        raise ValidationError("In-place manifest migration requires a backup")
    backup_path = source.with_name(f"{source.name}.v1.bak") if same_path else None
    if backup_path is not None and backup_path.exists():
        raise ValidationError("Manifest migration backup already exists")
    return target, same_path, backup_path


def _write_migration_backups(
    source: Path,
    *,
    backup_path: Path | None,
    pronunciation_path: Path | None,
    same_path: bool,
    backup: bool,
) -> None:
    pronunciation_backup: Path | None = None
    if pronunciation_path is not None and pronunciation_path.exists():
        pronunciation_backup = pronunciation_path.with_name(
            f"{pronunciation_path.name}.v1.bak"
        )
        if not same_path or not backup:
            raise ValidationError(
                "Existing pronunciation migration requires in-place backup mode"
            )
        if pronunciation_backup.exists():
            raise ValidationError("Pronunciation migration backup already exists")
    if backup_path is not None:
        atomic_write_bytes(backup_path, source.read_bytes(), overwrite=False)
    if pronunciation_backup is not None and pronunciation_path is not None:
        atomic_write_bytes(
            pronunciation_backup,
            pronunciation_path.read_bytes(),
            overwrite=False,
        )


def _rollback_migration(
    *,
    target: Path,
    original_manifest: bytes | None,
    pronunciation_path: Path | None,
    original_pronunciation: bytes | None,
) -> None:
    """Restore every changed destination after a partial multi-file commit."""
    if original_manifest is None:
        target.unlink(missing_ok=True)
    else:
        atomic_write_bytes(target, original_manifest, overwrite=True)
    if pronunciation_path is None:
        return
    if original_pronunciation is None:
        pronunciation_path.unlink(missing_ok=True)
    else:
        atomic_write_bytes(
            pronunciation_path,
            original_pronunciation,
            overwrite=True,
        )


def render_manifest_toml(value: Mapping[str, object]) -> bytes:
    """Render supported manifest values deterministically without hidden state."""
    lines: list[str] = []
    _render_toml_table(lines, (), value, emit_header=False)
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _pronunciation_output(
    preview: ManifestMigrationPreview, manifest_destination: Path
) -> tuple[Path | None, bytes | None]:
    if not preview.pronunciations:
        return None, None
    raw_path = preview.draft_manifest.get("pronunciations")
    if not isinstance(raw_path, str):
        raise ValidationError("Migrated pronunciation path is invalid")
    candidate = Path(raw_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValidationError("Migrated pronunciation path must remain relative")
    destination = (manifest_destination.parent / candidate).resolve()
    document: dict[str, object] = {
        "$schema": f"{SCHEMA_BASE}/pronunciations-v2.schema.json",
        "schema_version": 2,
        "terms": [
            {
                key: item
                for key, item in asdict(term).items()
                if key not in {"index", "review_required"} and item is not None
            }
            for term in preview.pronunciations
        ],
    }
    return destination, render_manifest_toml(document)


def _render_toml_table(
    lines: list[str],
    path: tuple[str, ...],
    value: Mapping[str, object],
    *,
    emit_header: bool,
) -> None:
    if emit_header:
        if lines and lines[-1]:
            lines.append("")
        lines.append(f"[{'.'.join(_toml_key(item) for item in path)}]")
    scalars = [
        (key, item)
        for key, item in value.items()
        if not isinstance(item, Mapping) and not _array_of_tables(item)
    ]
    tables = [
        (key, cast(Mapping[str, object], item))
        for key, item in value.items()
        if isinstance(item, Mapping)
    ]
    arrays = [
        (key, cast(Sequence[Mapping[str, object]], item))
        for key, item in value.items()
        if _array_of_tables(item)
    ]
    for key, item in scalars:
        lines.append(f"{_toml_key(key)} = {_toml_value(item)}")
    for key, table in tables:
        _render_toml_table(lines, (*path, key), table, emit_header=True)
    for key, records in arrays:
        for record in records:
            if lines and lines[-1]:
                lines.append("")
            lines.append(f"[[{'.'.join(_toml_key(item) for item in (*path, key))}]]")
            nested = {
                item_key: item_value
                for item_key, item_value in record.items()
                if not isinstance(item_value, Mapping)
                and not _array_of_tables(item_value)
            }
            for item_key, item_value in nested.items():
                lines.append(f"{_toml_key(item_key)} = {_toml_value(item_value)}")


def _array_of_tables(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes)
        and bool(value)
        and all(isinstance(item, Mapping) for item in value)
    )


def _toml_key(value: str) -> str:
    return value if _BARE_TOML_KEY.fullmatch(value) else json.dumps(value)


def _toml_value(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | Decimal) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, date | datetime):
        return value.isoformat()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return f"[{', '.join(_toml_value(item) for item in value)}]"
    raise ValidationError(
        f"Cannot serialize manifest TOML value {type(value).__name__}"
    )


def plan_migrated_manifest(
    preview: ManifestMigrationPreview,
    document: NormalizedDocument,
    *,
    target_name: str = "default",
    profile_override: str | None = None,
    chapter_selector: str | None = None,
) -> BuildPlan:
    """Exercise the real planner only after validating the internal v2 envelope."""
    parse_draft_manifest_speech_analysis(preview.draft_manifest)
    return plan_audiobook(
        preview._legacy_manifest,
        document,
        target_name=target_name,
        profile_override=profile_override,
        chapter_selector=chapter_selector,
    )


async def build_migrated_manifest(
    preview: ManifestMigrationPreview,
    *,
    target_name: str = "default",
    chapter_selector: str | None = None,
    dry_run: bool = False,
) -> BuildResult:
    """Exercise the fake backend through the validated internal v2 boundary."""
    parse_draft_manifest_speech_analysis(preview.draft_manifest)
    if any(profile.backend != "fake" for profile in preview._legacy_manifest.profiles):
        raise ValidationError(
            "Internal migrated builds are restricted to fake profiles"
        )
    return await build_audiobook(
        preview._legacy_manifest,
        target_name=target_name,
        chapter_selector=chapter_selector,
        dry_run=dry_run,
        resume=False,
    )


def _translate_analysis(
    manifest: AudiobookManifest,
    raw: dict[str, object],
) -> tuple[DraftSpeechAnalysisConfig, list[MigrationFinding]]:
    defaults = default_draft_speech_analysis_config()
    legacy = manifest.short_utterances
    whisper_engines = tuple(
        replace(engine, timeout_seconds=legacy.alignment_timeout_seconds)
        if engine.engine == "whisper"
        else engine
        for engine in defaults.policy.engines
    )
    config = replace(
        defaults,
        policy=replace(defaults.policy, engines=whisper_engines),
        cache=DraftAnalysisCachePolicy(
            version=SPEECH_ANALYSIS_CACHE_VERSION,
            enabled=manifest.whisper_qa.cache_enabled,
            directory=_safe_cache_directory(manifest),
        ),
        short_utterances=DraftShortUtteranceAnalysisPolicy(
            maximum_words=legacy.maximum_words,
            candidate_count=legacy.candidate_count,
        ),
        joins=DraftJoinAnalysisPolicy(
            automatic_inspection=legacy.automatic_join_inspection,
            window_seconds=legacy.join_inspection_window_seconds,
            coalesce_gap_ms=manifest.whisper_qa.join_coalesce_gap_ms,
        ),
        chapters=DraftChapterAnalysisPolicy(
            verification=manifest.whisper_qa.chapter_verification
        ),
        phonemes=DraftPhonemeAnalysisPolicy(
            enabled=manifest.whisper_qa.phoneme_alignment,
            backend=manifest.whisper_qa.phoneme_backend,
            model=manifest.whisper_qa.phoneme_model,
            revision=manifest.whisper_qa.phoneme_revision,
            language=manifest.whisper_qa.phoneme_language,
            timeout_seconds=manifest.whisper_qa.phoneme_timeout_seconds,
            minimum_confidence=manifest.whisper_qa.minimum_phoneme_confidence,
        ),
    )
    findings = [
        MigrationFinding(
            "legacy_analysis_cache_invalidated",
            "whisper_qa.cache_directory",
            "The v2 cache uses independent model, consensus, and verification "
            "identities.",
            False,
            False,
        ),
        MigrationFinding(
            "recognizer_thresholds_require_recalibration",
            "short_utterances",
            "Whisper-specific confidence and decode thresholds cannot be assigned "
            "to other recognizers.",
            True,
            True,
        ),
    ]
    short_raw = raw.get("short_utterances")
    configured = set(short_raw) if isinstance(short_raw, dict) else set()
    if configured & _LEGACY_SHORT_ANALYSIS_FIELDS:
        findings.append(
            MigrationFinding(
                "legacy_single_engine_controls_replaced",
                "short_utterances",
                "Configured single-engine QA controls are represented by v2 engine "
                "policy and calibration.",
                True,
                True,
            )
        )
    return config, findings


def _safe_cache_directory(manifest: AudiobookManifest) -> str:
    directory = manifest.whisper_qa.cache_directory.resolve()
    if directory.is_relative_to(manifest.root):
        return (
            directory.relative_to(manifest.root)
            .as_posix()
            .replace(".yakbox/cache/whisper", ".yakbox/speech-analysis")
        )
    return ".yakbox/speech-analysis"


def _draft_manifest(
    raw: dict[str, object], config: DraftSpeechAnalysisConfig
) -> dict[str, object]:
    draft = copy.deepcopy(raw)
    draft["$schema"] = DRAFT_MANIFEST_SCHEMA_URI
    draft["schema_version"] = 2
    draft.pop("whisper_qa", None)
    draft["speech_analysis"] = config.to_manifest_value()
    book = draft.get("book")
    if isinstance(book, dict):
        cast(dict[str, object], book).setdefault("language", "en")
    short = draft.get("short_utterances")
    if isinstance(short, dict):
        for key in _LEGACY_SHORT_ANALYSIS_FIELDS:
            short.pop(key, None)
    return draft


def _translate_pronunciations(
    path: Path | None,
) -> tuple[tuple[DraftPronunciationTerm, ...], tuple[MigrationFinding, ...]]:
    if path is None:
        return (), ()
    raw = _read_toml(path, "pronunciation sidecar")
    terms = raw.get("terms", [])
    if not isinstance(terms, list):
        raise ValidationError("Pronunciation terms must use [[terms]] records")
    translated: list[DraftPronunciationTerm] = []
    findings: list[MigrationFinding] = []
    for index, value in enumerate(terms, 1):
        if not isinstance(value, dict):
            raise ValidationError(f"Pronunciation term {index} must be a table")
        term = cast(dict[str, object], value)
        written = _required_text(term, "written", index)
        spoken = _required_text(term, "spoken", index)
        expected = tuple(token.text for token in normalize_english(written).tokens)
        synthesized = tuple(token.text for token in normalize_english(spoken).tokens)
        review = expected != synthesized
        translated.append(
            DraftPronunciationTerm(
                index=index,
                written=written,
                synthesis_hint=spoken,
                expected_lexical=expected,
                phonemes=(),
                match=str(term.get("match", "whole_word")),
                case=str(term.get("case", "sensitive")),
                priority=_integer_or_default(term.get("priority"), index),
                language=_optional_text(term.get("language"), index, "language"),
                status=str(term.get("status", "draft")),
                enabled=_boolean_or_default(term.get("enabled"), index),
                notes=_optional_text(
                    term.get("notes"), index, "notes", allow_empty=True
                ),
                review_required=review,
            )
        )
        if review:
            findings.append(
                MigrationFinding(
                    "pronunciation_tokenization_changed",
                    f"pronunciations.terms[{index}]",
                    "The synthesis hint does not define listener-visible lexical "
                    "truth; review expected_lexical.",
                    False,
                    True,
                )
            )
    return tuple(translated), tuple(findings)


def _preserved_repairs(
    manifest: AudiobookManifest,
) -> tuple[PreservedRepairApproval, ...]:
    records = []
    for target in sorted(manifest.targets, key=lambda item: item.name):
        for repair in load_approved_repairs(manifest.root, target.name):
            path = (
                repair.audio_path.relative_to(manifest.root).as_posix()
                if repair.audio_path.is_relative_to(manifest.root)
                else repair.audio_path.as_posix()
            )
            records.append(
                PreservedRepairApproval(
                    target.name,
                    repair.repair_id,
                    repair.chunk_id,
                    path,
                    repair.audio_sha256,
                    repair.fingerprint,
                )
            )
    return tuple(
        sorted(records, key=lambda item: (item.target, item.chunk_id, item.repair_id))
    )


def _read_toml(path: Path, label: str) -> dict[str, object]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValidationError(f"Cannot read {label} {path}: {error}") from error


def _required_text(raw: dict[str, object], key: str, index: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"Pronunciation term {index} needs {key}")
    return value.strip()


def _optional_text(
    value: object,
    index: int,
    key: str,
    *,
    allow_empty: bool = False,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValidationError(f"Pronunciation term {index} {key} must be text")
    return value


def _integer_or_default(value: object, index: int) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"Pronunciation term {index} priority must be integer")
    return value


def _boolean_or_default(value: object, index: int) -> bool:
    if value is None:
        return True
    if not isinstance(value, bool):
        raise ValidationError(f"Pronunciation term {index} enabled must be boolean")
    return value


__all__ = [
    "DRAFT_COMMAND_MAP",
    "DRAFT_MANIFEST_SCHEMA_URI",
    "DRAFT_PYTHON_EXPORTS",
    "SPEECH_ANALYSIS_CACHE_VERSION",
    "DraftPronunciationTerm",
    "ManifestMigrationPreview",
    "ManifestMigrationWriteResult",
    "MigrationFinding",
    "PreservedRepairApproval",
    "build_migrated_manifest",
    "plan_migrated_manifest",
    "preview_manifest_migration",
    "render_manifest_toml",
    "write_manifest_migration",
]
