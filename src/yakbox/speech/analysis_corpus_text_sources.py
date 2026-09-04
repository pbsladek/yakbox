"""Acquire and verify public-domain source texts for corpus transcript truth."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import cast
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

from yakbox._files import atomic_write_bytes, atomic_write_json, safe_child, sha256_file
from yakbox.contracts import runtime_metadata, schema_uri
from yakbox.errors import ValidationError
from yakbox.speech.analysis_corpus_sources import load_licensed_voice_sources
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.normalization import NORMALIZATION_VERSION, normalize_english

_MAXIMUM_RESPONSE_BYTES = 64 * 1024 * 1024
_MAXIMUM_TOTAL_BYTES = 512 * 1024 * 1024
_MINIMUM_TEXT_TOKENS = 100
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VOICE = re.compile(r"[a-z0-9][a-z0-9-]{1,63}")
_INVENTORY_FIELDS = {
    "$schema",
    "schema_version",
    "yakbox_version",
    "timestamp",
    "fingerprint",
    "voice_registry_digest",
    "source_override_digest",
    "normalization_version",
    "source_count",
    "raw_byte_count",
    "sources",
}
_SOURCE_FIELDS = {
    "voice",
    "reader",
    "source_work",
    "catalog_url",
    "text_url",
    "rights_id",
    "rights_url",
    "relative_raw_path",
    "raw_digest",
    "raw_byte_count",
    "relative_plain_path",
    "plain_digest",
    "token_count",
    "fingerprint",
}


@dataclass(frozen=True, slots=True)
class CorpusTextSource:
    voice: str
    reader: str
    source_work: str
    catalog_url: str
    text_url: str
    rights_id: str
    rights_url: str
    relative_raw_path: str
    raw_digest: str
    raw_byte_count: int
    relative_plain_path: str
    plain_digest: str
    token_count: int

    def __post_init__(self) -> None:
        for value, label in (
            (self.raw_digest, "source-text raw digest"),
            (self.plain_digest, "source-text plain digest"),
        ):
            _require_sha256(value, label)
        for value, label in (
            (self.catalog_url, "source-text catalog URL"),
            (self.text_url, "source-text URL"),
            (self.rights_url, "source-text rights URL"),
        ):
            _require_https(value, label)
        if (
            _VOICE.fullmatch(self.voice) is None
            or not self.reader
            or not self.source_work
            or self.rights_id != "LicenseRef-Public-Domain-US"
            or self.raw_byte_count < 1
            or self.token_count < _MINIMUM_TEXT_TOKENS
        ):
            raise ValidationError("Corpus source-text metadata is invalid")
        for value in (self.relative_raw_path, self.relative_plain_path):
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValidationError("Corpus source-text path must be relative")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-text-source-v1", self)


@dataclass(frozen=True, slots=True)
class CorpusTextSourceInventory:
    voice_registry_digest: str
    source_override_digest: str | None
    normalization_version: int
    sources: tuple[CorpusTextSource, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.voice_registry_digest, "voice registry digest")
        if self.source_override_digest is not None:
            _require_sha256(self.source_override_digest, "source override digest")
        voices = tuple(item.voice for item in self.sources)
        if (
            self.normalization_version != NORMALIZATION_VERSION
            or not self.sources
            or voices != tuple(sorted(set(voices)))
        ):
            raise ValidationError("Corpus source-text inventory is inconsistent")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-text-source-inventory-v1", self)

    @property
    def raw_byte_count(self) -> int:
        return sum(item.raw_byte_count for item in self.sources)

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("speech-corpus-text-sources"),
            "fingerprint": self.fingerprint,
            "voice_registry_digest": self.voice_registry_digest,
            "source_override_digest": self.source_override_digest,
            "normalization_version": self.normalization_version,
            "source_count": len(self.sources),
            "raw_byte_count": self.raw_byte_count,
            "sources": [
                {
                    "voice": item.voice,
                    "reader": item.reader,
                    "source_work": item.source_work,
                    "catalog_url": item.catalog_url,
                    "text_url": item.text_url,
                    "rights_id": item.rights_id,
                    "rights_url": item.rights_url,
                    "relative_raw_path": item.relative_raw_path,
                    "raw_digest": item.raw_digest,
                    "raw_byte_count": item.raw_byte_count,
                    "relative_plain_path": item.relative_plain_path,
                    "plain_digest": item.plain_digest,
                    "token_count": item.token_count,
                    "fingerprint": item.fingerprint,
                }
                for item in self.sources
            ],
        }


@dataclass(frozen=True, slots=True)
class _TextRequest:
    voice: str
    reader: str
    source_work: str
    catalog_url: str
    rights_url: str
    text_url_override: str | None


@dataclass(frozen=True, slots=True)
class _TextOverride:
    url: str
    rights_url: str


@dataclass(frozen=True, slots=True)
class _DownloadedText:
    url: str
    payload: bytes
    plain_text: str
    suffix: str


def download_corpus_text_sources(
    voice_registry: Path,
    *,
    repository_root: Path,
    output_root: Path,
    source_overrides: Path | None = None,
    client: httpx.Client | None = None,
) -> CorpusTextSourceInventory:
    """Resolve, download, normalize, and checksum each linked source text."""
    requests, override_digest = _load_text_requests(
        voice_registry,
        repository_root=repository_root,
        source_overrides=source_overrides,
    )
    output_root = output_root.resolve()
    owned_client = client is None
    active_client = client or httpx.Client(
        timeout=httpx.Timeout(60),
        follow_redirects=True,
        trust_env=False,
        headers={"User-Agent": "yakbox-source-text-qualification/1"},
    )
    sources: list[CorpusTextSource] = []
    total_bytes = 0
    try:
        for request in requests:
            try:
                online_text = request.text_url_override or _discover_online_text(
                    active_client,
                    request.catalog_url,
                    source_work=request.source_work,
                    reader=request.reader,
                )
                downloaded = _download_resolved_text(active_client, online_text)
            except ValidationError as error:
                raise ValidationError(
                    f"Cannot qualify source text for voice {request.voice}: {error}"
                ) from error
            total_bytes += len(downloaded.payload)
            if total_bytes > _MAXIMUM_TOTAL_BYTES:
                raise ValidationError("Corpus source texts exceed total byte limit")
            raw_relative = Path("raw") / f"{request.voice}{downloaded.suffix}"
            plain_relative = Path("plain") / f"{request.voice}.txt"
            raw_path = safe_child(output_root, output_root / raw_relative)
            plain_path = safe_child(output_root, output_root / plain_relative)
            atomic_write_bytes(raw_path, downloaded.payload, overwrite=True)
            canonical_plain = unicodedata.normalize("NFKC", downloaded.plain_text)
            plain_payload = canonical_plain.encode("utf-8")
            atomic_write_bytes(plain_path, plain_payload, overwrite=True)
            token_count = len(normalize_english(canonical_plain).tokens)
            sources.append(
                CorpusTextSource(
                    request.voice,
                    request.reader,
                    request.source_work,
                    request.catalog_url,
                    downloaded.url,
                    "LicenseRef-Public-Domain-US",
                    request.rights_url,
                    raw_relative.as_posix(),
                    sha256_file(raw_path),
                    len(downloaded.payload),
                    plain_relative.as_posix(),
                    sha256_file(plain_path),
                    token_count,
                )
            )
    finally:
        if owned_client:
            active_client.close()
    return CorpusTextSourceInventory(
        sha256_file(voice_registry),
        override_digest,
        NORMALIZATION_VERSION,
        tuple(sorted(sources, key=lambda item: item.voice)),
    )


def write_corpus_text_source_inventory(
    path: Path,
    inventory: CorpusTextSourceInventory,
) -> None:
    atomic_write_json(path, inventory.to_dict())


def main(argv: Sequence[str] | None = None) -> int:
    """Download every licensed voice's discoverable public-domain source text."""
    parser = argparse.ArgumentParser(
        description="Checksum public-domain source texts for transcript qualification"
    )
    parser.add_argument("--voice-registry", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--inventory-output", type=Path, required=True)
    parser.add_argument("--source-overrides", type=Path)
    arguments = parser.parse_args(argv)
    try:
        inventory = download_corpus_text_sources(
            arguments.voice_registry,
            repository_root=arguments.repository_root,
            output_root=arguments.output_root,
            source_overrides=arguments.source_overrides,
        )
        write_corpus_text_source_inventory(arguments.inventory_output, inventory)
    except (OSError, UnicodeError, ValidationError) as error:
        sys.stderr.write(
            json.dumps({"status": "error", "message": str(error)}, sort_keys=True)
            + "\n"
        )
        return 2
    sys.stdout.write(
        json.dumps(
            {
                "status": "ok",
                "fingerprint": inventory.fingerprint,
                "source_count": len(inventory.sources),
                "raw_byte_count": inventory.raw_byte_count,
                "inventory": str(arguments.inventory_output.resolve()),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def load_corpus_text_source_inventory(
    path: Path,
    *,
    text_root: Path,
) -> CorpusTextSourceInventory:
    """Load a text inventory and reverify every raw and normalized source."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError("Cannot read corpus source-text inventory") from error
    if not isinstance(raw, dict) or set(raw) != _INVENTORY_FIELDS:
        raise ValidationError("Corpus source-text inventory fields are invalid")
    if (
        raw.get("$schema") != schema_uri("speech-corpus-text-sources")
        or raw.get("schema_version") != 1
        or not isinstance(raw.get("yakbox_version"), str)
        or not _utc_timestamp(raw.get("timestamp"))
    ):
        raise ValidationError("Corpus source-text inventory metadata is invalid")
    values = raw.get("sources")
    if not isinstance(values, list):
        raise ValidationError("Corpus source-text entries are invalid")
    sources = tuple(_load_source(item, text_root=text_root) for item in values)
    inventory = CorpusTextSourceInventory(
        _text(raw, "voice_registry_digest"),
        _optional_digest(raw, "source_override_digest"),
        _integer(raw, "normalization_version"),
        sources,
    )
    if (
        raw.get("source_count") != len(sources)
        or raw.get("raw_byte_count") != inventory.raw_byte_count
        or raw.get("fingerprint") != inventory.fingerprint
    ):
        raise ValidationError("Corpus source-text inventory identity differs")
    return inventory


def _load_text_requests(
    path: Path,
    *,
    repository_root: Path,
    source_overrides: Path | None,
) -> tuple[tuple[_TextRequest, ...], str | None]:
    validated = {
        item.voice: item
        for item in load_licensed_voice_sources(
            path,
            repository_root=repository_root,
        )
    }
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValidationError("Cannot read qualification voice registry") from error
    voices = raw.get("voices") if isinstance(raw, dict) else None
    if not isinstance(voices, dict) or set(voices) != set(validated):
        raise ValidationError("Qualification text-source voices are inconsistent")
    overrides = _load_source_overrides(source_overrides, voices=set(validated))
    requests: list[_TextRequest] = []
    for raw_voice, value in sorted(voices.items()):
        if not isinstance(raw_voice, str) or not isinstance(value, dict):
            raise ValidationError("Qualification text-source entry is invalid")
        voice = raw_voice
        entry = cast(dict[str, object], value)
        override = overrides.get(voice)
        requests.append(
            _TextRequest(
                voice,
                validated[voice].reader,
                _text(entry, "source_work"),
                _https_text(entry, "catalog_url"),
                override.rights_url if override else _https_text(entry, "rights_url"),
                override.url if override else None,
            )
        )
    return tuple(requests), sha256_file(source_overrides) if source_overrides else None


def _load_source_overrides(
    path: Path | None,
    *,
    voices: set[str],
) -> dict[str, _TextOverride]:
    if path is None:
        return {}
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValidationError("Cannot read corpus source-text overrides") from error
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "sources"}:
        raise ValidationError("Corpus source-text override fields are invalid")
    if raw.get("schema_version") != 1 or not isinstance(raw.get("sources"), dict):
        raise ValidationError("Corpus source-text override metadata is invalid")
    sources = cast(dict[object, object], raw["sources"])
    if not set(sources) <= voices:
        raise ValidationError("Corpus source-text override voice is unknown")
    overrides: dict[str, _TextOverride] = {}
    for raw_voice, raw_value in sources.items():
        if (
            not isinstance(raw_voice, str)
            or not isinstance(raw_value, dict)
            or set(raw_value) != {"url", "rights_url"}
        ):
            raise ValidationError("Corpus source-text override entry is invalid")
        entry = cast(dict[str, object], raw_value)
        value = entry.get("url")
        rights_url = entry.get("rights_url")
        if not isinstance(value, str) or not isinstance(rights_url, str):
            raise ValidationError("Corpus source-text override URL is invalid")
        overrides[raw_voice] = _TextOverride(
            _https_upgrade(value),
            _https_upgrade(rights_url),
        )
    return overrides


def _discover_online_text(
    client: httpx.Client,
    catalog_url: str,
    *,
    source_work: str,
    reader: str,
) -> str:
    if urlparse(catalog_url).netloc.casefold().endswith("archive.org"):
        return _archive_text_source(client, catalog_url)
    response = _bounded_get(client, catalog_url)
    parser = _LinkParser()
    parser.feed(_decode_text(response.content))
    matches = [
        urljoin(str(response.url), href)
        for label, href in parser.links
        if "online text" in label.casefold()
    ]
    if len(matches) == 1:
        return _https_upgrade(matches[0])
    row_parser = _TableRowParser()
    row_parser.feed(_decode_text(response.content))
    source_title = source_work.rsplit(": ", 1)[-1].rsplit(" by ", 1)[0]
    matching_rows = [
        row
        for row in row_parser.rows
        if source_title.casefold() in row.text.casefold()
        and reader.casefold() in row.text.casefold()
    ]
    row_matches = [
        urljoin(str(response.url), href)
        for row in matching_rows
        for label, href in row.links
        if "etext" in label.casefold() or "online text" in label.casefold()
    ]
    unique = tuple(dict.fromkeys(row_matches))
    if len(unique) != 1:
        raise ValidationError("LibriVox catalog must identify one source-text link")
    return _https_upgrade(unique[0])


def _archive_text_source(client: httpx.Client, catalog_url: str) -> str:
    identifier = urlparse(catalog_url).path.rstrip("/").rsplit("/", 1)[-1]
    if not identifier:
        raise ValidationError("Internet Archive catalog identifier is invalid")
    response = _bounded_get(client, f"https://archive.org/metadata/{identifier}")
    try:
        metadata = response.json()
    except json.JSONDecodeError as error:
        raise ValidationError("Internet Archive metadata is invalid") from error
    if not isinstance(metadata, dict):
        raise ValidationError("Internet Archive metadata is invalid")
    details = metadata.get("metadata")
    if isinstance(details, dict):
        text_url = details.get("url_text_source")
        if isinstance(text_url, str) and text_url:
            return _https_upgrade(text_url)
    files = metadata.get("files")
    if isinstance(files, list):
        names = sorted(
            value["name"]
            for value in files
            if isinstance(value, dict)
            and isinstance(value.get("name"), str)
            and value["name"].endswith("_djvu.txt")
        )
        if len(names) == 1:
            return f"https://archive.org/download/{identifier}/{names[0]}"
    raise ValidationError("Internet Archive item has no usable online text")


def _download_resolved_text(client: httpx.Client, online_url: str) -> _DownloadedText:
    response = _bounded_get(client, online_url)
    content_type = response.headers.get("content-type", "").casefold()
    final_url = _https_upgrade(str(response.url))
    if _is_plain_text(final_url, content_type) and not _looks_like_html(
        response.content
    ):
        return _DownloadedText(
            final_url, response.content, _decode_text(response.content), ".txt"
        )
    if "html" not in content_type and not final_url.casefold().endswith(
        (".htm", ".html")
    ):
        raise ValidationError("Online source text format is unsupported")
    html = _decode_text(response.content)
    parser = _LinkParser()
    parser.feed(html)
    downloadable = _plain_text_link(parser.links, base_url=final_url)
    if downloadable is not None:
        plain_response = _bounded_get(client, downloadable)
        if not _is_plain_text(
            str(plain_response.url),
            plain_response.headers.get("content-type", "").casefold(),
        ):
            raise ValidationError("Resolved source text is not plain text")
        return _DownloadedText(
            _https_upgrade(str(plain_response.url)),
            plain_response.content,
            _decode_text(plain_response.content),
            ".txt",
        )
    visible = _VisibleTextParser()
    visible.feed(html)
    plain = visible.text()
    return _DownloadedText(final_url, response.content, plain, ".html")


def _plain_text_link(
    links: tuple[tuple[str, str], ...],
    *,
    base_url: str,
) -> str | None:
    candidates = [
        urljoin(base_url, href)
        for label, href in links
        if "plain text utf-8" in label.casefold()
        or href.casefold().endswith((".txt", ".txt.utf-8"))
    ]
    unique = tuple(dict.fromkeys(_https_upgrade(item) for item in candidates))
    return unique[0] if unique else None


def _bounded_get(client: httpx.Client, url: str) -> httpx.Response:
    _require_https(url, "source-text download URL")
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise ValidationError("Cannot download public-domain source text") from error
    for followed in (*response.history, response):
        _require_https(str(followed.url), "source-text redirect URL")
    length = response.headers.get("content-length")
    if length is not None:
        try:
            declared = int(length)
        except ValueError as error:
            raise ValidationError("Source-text content length is invalid") from error
        if declared > _MAXIMUM_RESPONSE_BYTES:
            raise ValidationError("Source-text response exceeds byte limit")
    if len(response.content) > _MAXIMUM_RESPONSE_BYTES:
        raise ValidationError("Source-text response exceeds byte limit")
    return response


def _load_source(value: object, *, text_root: Path) -> CorpusTextSource:
    if not isinstance(value, dict) or set(value) != _SOURCE_FIELDS:
        raise ValidationError("Corpus source-text fields are invalid")
    raw = cast(dict[str, object], value)
    source = CorpusTextSource(
        _text(raw, "voice"),
        _text(raw, "reader"),
        _text(raw, "source_work"),
        _text(raw, "catalog_url"),
        _text(raw, "text_url"),
        _text(raw, "rights_id"),
        _text(raw, "rights_url"),
        _text(raw, "relative_raw_path"),
        _text(raw, "raw_digest"),
        _integer(raw, "raw_byte_count"),
        _text(raw, "relative_plain_path"),
        _text(raw, "plain_digest"),
        _integer(raw, "token_count"),
    )
    raw_path = _verified_child(text_root, source.relative_raw_path, source.raw_digest)
    plain_path = _verified_child(
        text_root,
        source.relative_plain_path,
        source.plain_digest,
    )
    if raw_path.stat().st_size != source.raw_byte_count:
        raise ValidationError("Corpus source-text raw byte count differs")
    try:
        plain = plain_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValidationError("Corpus source text is not valid UTF-8") from error
    if len(normalize_english(plain).tokens) != source.token_count:
        raise ValidationError("Corpus source-text token count differs")
    if raw.get("fingerprint") != source.fingerprint:
        raise ValidationError("Corpus source-text fingerprint differs")
    return source


def _verified_child(root: Path, relative_text: str, digest: str) -> Path:
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValidationError("Corpus source-text path must be relative")
    path = safe_child(root, root / relative)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
        raise ValidationError("Corpus source-text file identity differs")
    return path


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._href: str | None = None
        self._label: list[str] = []
        self._links: list[tuple[str, str]] = []

    @property
    def links(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._links)

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.casefold() != "a":
            return
        values = dict(attrs)
        href = values.get("href")
        if isinstance(href, str) and href:
            self._href = href
            self._label = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._label.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "a" and self._href is not None:
            self._links.append((" ".join(self._label).strip(), self._href))
            self._href = None
            self._label = []


@dataclass(frozen=True, slots=True)
class _TableRow:
    text: str
    links: tuple[tuple[str, str], ...]


class _TableRowParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._inside_row = False
        self._text: list[str] = []
        self._href: str | None = None
        self._label: list[str] = []
        self._links: list[tuple[str, str]] = []
        self._rows: list[_TableRow] = []

    @property
    def rows(self) -> tuple[_TableRow, ...]:
        return tuple(self._rows)

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        folded = tag.casefold()
        if folded == "tr":
            self._inside_row = True
            self._text = []
            self._links = []
        elif folded == "a" and self._inside_row:
            href = dict(attrs).get("href")
            if isinstance(href, str) and href:
                self._href = href
                self._label = []

    def handle_data(self, data: str) -> None:
        if self._inside_row and data.strip():
            value = data.strip()
            self._text.append(value)
            if self._href is not None:
                self._label.append(value)

    def handle_endtag(self, tag: str) -> None:
        folded = tag.casefold()
        if folded == "a" and self._href is not None:
            self._links.append((" ".join(self._label), self._href))
            self._href = None
            self._label = []
        elif folded == "tr" and self._inside_row:
            self._rows.append(_TableRow(" ".join(self._text), tuple(self._links)))
            self._inside_row = False


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_depth = 0
        self._parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        if tag.casefold() in {"script", "style", "svg"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "svg"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth and data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return "\n".join(self._parts) + "\n"


def _decode_text(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "windows-1252"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValidationError("Source text encoding is unsupported")


def _is_plain_text(url: str, content_type: str) -> bool:
    path = urlparse(url).path.casefold()
    return "text/plain" in content_type or path.endswith((".txt", ".txt.utf-8"))


def _looks_like_html(payload: bytes) -> bool:
    beginning = payload[:1_024].lstrip().lower()
    return beginning.startswith((b"<!doctype html", b"<html"))


def _https_upgrade(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme.casefold() == "http":
        parsed = parsed._replace(scheme="https")
    if parsed.netloc.casefold() in {
        "gutenberg.org",
        "www.gutenberg.org",
    } and parsed.path.startswith("/etext/"):
        parsed = parsed._replace(path=parsed.path.replace("/etext/", "/ebooks/", 1))
    gutenberg_text = re.fullmatch(r"/ebooks/([0-9]+)\.txt\.utf-8", parsed.path)
    if (
        parsed.netloc.casefold() in {"gutenberg.org", "www.gutenberg.org"}
        and gutenberg_text is not None
    ):
        book_id = gutenberg_text.group(1)
        parsed = parsed._replace(path=f"/cache/epub/{book_id}/pg{book_id}.txt")
    upgraded = urlunparse(parsed)
    _require_https(upgraded, "source-text URL")
    return upgraded


def _https_text(raw: dict[str, object], key: str) -> str:
    value = _text(raw, key)
    _require_https(value, f"qualification {key}")
    return value


def _text(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Corpus source-text {key} must be text")
    return value


def _integer(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"Corpus source-text {key} must be an integer")
    return value


def _optional_digest(raw: dict[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is not None and not isinstance(value, str):
        raise ValidationError(f"Corpus source-text {key} must be a SHA-256 or null")
    return value


def _utc_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.utcoffset() == timedelta(0)


def _require_https(value: str, label: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme.casefold() != "https" or not parsed.netloc:
        raise ValidationError(f"{label.capitalize()} must use HTTPS")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label.capitalize()} must be a SHA-256")


__all__ = [
    "CorpusTextSource",
    "CorpusTextSourceInventory",
    "download_corpus_text_sources",
    "load_corpus_text_source_inventory",
    "write_corpus_text_source_inventory",
]


if __name__ == "__main__":
    raise SystemExit(main())
