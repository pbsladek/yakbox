"""Fetch digest-pinned licensed archives for corpus passage expansion."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.parse import urlparse

import httpx

from yakbox._files import (
    atomic_output_path,
    atomic_write_json,
    safe_child,
    sha256_file,
)
from yakbox.contracts import runtime_metadata
from yakbox.errors import ArtifactError, ValidationError
from yakbox.speech.analysis_corpus_sources import (
    LicensedVoiceSource,
    load_licensed_voice_sources,
)
from yakbox.speech.analysis_fingerprints import semantic_fingerprint

_MAXIMUM_SOURCE_BYTES = 512 * 1024 * 1024
_MAXIMUM_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024
_ALLOWED_SUFFIXES = frozenset({".flac", ".m4a", ".mp3", ".ogg", ".wav"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCHEMA = "https://yakbox.dev/schemas/speech-corpus-source-archives-v1.schema.json"
_TOP_FIELDS = {
    "$schema",
    "schema_version",
    "yakbox_version",
    "timestamp",
    "fingerprint",
    "voice_registry_digest",
    "archive_count",
    "total_size_bytes",
    "archives",
}
_ARCHIVE_FIELDS = {
    "voice",
    "reader",
    "relative_path",
    "source_url",
    "source_digest",
    "size_bytes",
    "rights_id",
    "rights_url",
    "fingerprint",
}


@dataclass(frozen=True, slots=True)
class CorpusSourceArchive:
    voice: str
    reader: str
    relative_path: str
    source_url: str
    source_digest: str
    size_bytes: int
    rights_id: str
    rights_url: str

    def __post_init__(self) -> None:
        candidate = Path(self.relative_path)
        if (
            not self.voice
            or not self.reader
            or candidate.is_absolute()
            or ".." in candidate.parts
            or not self.source_url.startswith("https://")
            or self.size_bytes <= 0
            or not self.rights_id
            or not self.rights_url.startswith("https://")
        ):
            raise ValidationError("Corpus source archive identity is invalid")
        _require_sha256(self.source_digest, "corpus source archive digest")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-source-archive-v1", self)


@dataclass(frozen=True, slots=True)
class CorpusSourceArchiveInventory:
    voice_registry_digest: str
    archives: tuple[CorpusSourceArchive, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.voice_registry_digest, "voice registry digest")
        voices = tuple(item.voice for item in self.archives)
        paths = tuple(item.relative_path for item in self.archives)
        if (
            not self.archives
            or voices != tuple(sorted(set(voices)))
            or len(paths) != len(set(paths))
        ):
            raise ValidationError("Corpus source archive inventory is inconsistent")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-source-archive-inventory-v1", self)

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("speech-corpus-source-archives"),
            "fingerprint": self.fingerprint,
            "voice_registry_digest": self.voice_registry_digest,
            "archive_count": len(self.archives),
            "total_size_bytes": sum(item.size_bytes for item in self.archives),
            "archives": [
                {
                    "voice": item.voice,
                    "reader": item.reader,
                    "relative_path": item.relative_path,
                    "source_url": item.source_url,
                    "source_digest": item.source_digest,
                    "size_bytes": item.size_bytes,
                    "rights_id": item.rights_id,
                    "rights_url": item.rights_url,
                    "fingerprint": item.fingerprint,
                }
                for item in self.archives
            ],
        }


def download_corpus_source_archives(
    voice_registry: Path,
    *,
    repository_root: Path,
    output_root: Path,
    client: httpx.Client | None = None,
) -> CorpusSourceArchiveInventory:
    """Download each trusted source URL once and verify its registered digest."""
    sources = load_licensed_voice_sources(
        voice_registry,
        repository_root=repository_root,
    )
    output_root = output_root.resolve()
    owned_client = client is None
    active_client = client or httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(180, connect=30),
        trust_env=False,
    )
    try:
        archives = tuple(
            _download_source(source, output_root=output_root, client=active_client)
            for source in sources
        )
    finally:
        if owned_client:
            active_client.close()
    total = sum(item.size_bytes for item in archives)
    if total > _MAXIMUM_TOTAL_BYTES:
        raise ArtifactError("Qualification source archives exceed the total size limit")
    return CorpusSourceArchiveInventory(
        sha256_file(voice_registry),
        tuple(sorted(archives, key=lambda item: item.voice)),
    )


def write_corpus_source_archive_inventory(
    path: Path,
    inventory: CorpusSourceArchiveInventory,
) -> None:
    atomic_write_json(path, inventory.to_dict())


def load_corpus_source_archive_inventory(
    path: Path,
    *,
    archive_root: Path,
) -> CorpusSourceArchiveInventory:
    """Load an archive inventory and reverify every downloaded source."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError("Cannot read corpus source archive inventory") from error
    if not isinstance(raw, dict) or set(raw) != _TOP_FIELDS:
        raise ValidationError("Corpus source archive inventory fields are invalid")
    if (
        raw.get("$schema") != _SCHEMA
        or raw.get("schema_version") != 1
        or not isinstance(raw.get("yakbox_version"), str)
        or not isinstance(raw.get("timestamp"), str)
    ):
        raise ValidationError("Corpus source archive inventory metadata is invalid")
    archives_raw = raw.get("archives")
    if not isinstance(archives_raw, list):
        raise ValidationError("Corpus source archives are invalid")
    archives = tuple(
        _load_archive(item, archive_root=archive_root) for item in archives_raw
    )
    inventory = CorpusSourceArchiveInventory(
        _json_text(raw, "voice_registry_digest"),
        archives,
    )
    if (
        raw.get("archive_count") != len(archives)
        or raw.get("total_size_bytes") != sum(item.size_bytes for item in archives)
        or raw.get("fingerprint") != inventory.fingerprint
    ):
        raise ValidationError("Corpus source archive inventory identity differs")
    return inventory


def _load_archive(value: object, *, archive_root: Path) -> CorpusSourceArchive:
    if not isinstance(value, dict) or set(value) != _ARCHIVE_FIELDS:
        raise ValidationError("Corpus source archive fields are invalid")
    raw = cast(dict[str, object], value)
    archive = CorpusSourceArchive(
        _json_text(raw, "voice"),
        _json_text(raw, "reader"),
        _json_text(raw, "relative_path"),
        _json_text(raw, "source_url"),
        _json_text(raw, "source_digest"),
        _json_integer(raw, "size_bytes"),
        _json_text(raw, "rights_id"),
        _json_text(raw, "rights_url"),
    )
    candidate = safe_child(archive_root, archive_root / archive.relative_path)
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or candidate.stat().st_size != archive.size_bytes
        or sha256_file(candidate) != archive.source_digest
        or raw.get("fingerprint") != archive.fingerprint
    ):
        raise ValidationError("Corpus source archive file identity differs")
    return archive


def _download_source(
    source: LicensedVoiceSource,
    *,
    output_root: Path,
    client: httpx.Client,
) -> CorpusSourceArchive:
    suffix = PurePosixPath(urlparse(source.source_url).path).suffix.casefold()
    if suffix not in _ALLOWED_SUFFIXES:
        raise ValidationError("Qualification source URL has an unsupported audio type")
    relative = Path("archives") / f"{source.source_digest}{suffix}"
    destination = output_root / relative
    if destination.is_symlink():
        raise ArtifactError("Qualification source archive cannot be a symlink")
    if destination.exists():
        if (
            not destination.is_file()
            or sha256_file(destination) != source.source_digest
        ):
            raise ArtifactError("Existing qualification source archive differs")
    else:
        _download_file(
            client,
            _source_urls(source.source_url),
            destination,
            expected_digest=source.source_digest,
        )
    size = destination.stat().st_size
    if not 0 < size <= _MAXIMUM_SOURCE_BYTES:
        raise ArtifactError("Qualification source archive exceeds the size limit")
    return CorpusSourceArchive(
        source.voice,
        source.reader,
        relative.as_posix(),
        source.source_url,
        source.source_digest,
        size,
        source.rights_id,
        source.rights_url,
    )


def _download_file(
    client: httpx.Client,
    urls: tuple[str, ...],
    destination: Path,
    *,
    expected_digest: str,
) -> None:
    last_error: Exception | None = None
    for url in urls:
        try:
            _download_url(
                client,
                url,
                destination,
                expected_digest=expected_digest,
            )
        except (httpx.HTTPError, OSError, ValueError) as error:
            last_error = error
            continue
        return
    raise ArtifactError("Cannot download qualification source archive") from last_error


def _download_url(
    client: httpx.Client,
    url: str,
    destination: Path,
    *,
    expected_digest: str,
) -> None:
    with client.stream("GET", url) as response:
        response.raise_for_status()
        if response.url.scheme != "https":
            raise ArtifactError("Qualification source redirect is not HTTPS")
        declared = response.headers.get("content-length")
        if declared is not None and int(declared) > _MAXIMUM_SOURCE_BYTES:
            raise ArtifactError("Qualification source archive exceeds the size limit")
        with atomic_output_path(destination) as temporary:
            size = 0
            with temporary.open("wb") as stream:
                for chunk in response.iter_bytes(_CHUNK_BYTES):
                    size += len(chunk)
                    if size > _MAXIMUM_SOURCE_BYTES:
                        raise ArtifactError(
                            "Qualification source archive exceeds the size limit"
                        )
                    stream.write(chunk)
            if sha256_file(temporary) != expected_digest:
                raise ArtifactError("Downloaded qualification source digest differs")


def _source_urls(source_url: str) -> tuple[str, ...]:
    parsed = urlparse(source_url)
    host = (parsed.hostname or "").casefold()
    parts = tuple(part for part in PurePosixPath(parsed.path).parts if part != "/")
    if host == "archive.org" or host.endswith(".archive.org"):
        try:
            item_index = parts.index("items") + 1
            item = parts[item_index]
            remainder = parts[item_index + 1 :]
        except ValueError, IndexError:
            return (source_url,)
        if item and remainder:
            canonical = "https://archive.org/download/" + "/".join((item, *remainder))
            if canonical != source_url:
                return source_url, canonical
    return (source_url,)


def _require_sha256(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label.capitalize()} must be a SHA-256")


def _json_text(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Corpus source archive {key} must be text")
    return value


def _json_integer(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"Corpus source archive {key} must be an integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download digest-pinned licensed qualification sources"
    )
    parser.add_argument("--voice-registry", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    inventory = download_corpus_source_archives(
        arguments.voice_registry,
        repository_root=arguments.repository_root,
        output_root=arguments.output_root,
    )
    write_corpus_source_archive_inventory(arguments.report_output, inventory)
    sys.stdout.write(
        json.dumps(
            {
                "archive_count": len(inventory.archives),
                "fingerprint": inventory.fingerprint,
                "total_size_bytes": sum(item.size_bytes for item in inventory.archives),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CorpusSourceArchive",
    "CorpusSourceArchiveInventory",
    "download_corpus_source_archives",
    "load_corpus_source_archive_inventory",
    "main",
    "write_corpus_source_archive_inventory",
]
