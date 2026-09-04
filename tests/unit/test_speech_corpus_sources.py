from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator

from yakbox._files import sha256_file
from yakbox.errors import ValidationError
from yakbox.schemas import load_schema
from yakbox.speech.analysis_corpus_archives import (
    download_corpus_source_archives,
    load_corpus_source_archive_inventory,
    write_corpus_source_archive_inventory,
)
from yakbox.speech.analysis_corpus_passages import (
    prepare_expanded_corpus_source_inventory,
)
from yakbox.speech.analysis_corpus_sources import (
    load_corpus_source_inventory,
    prepare_corpus_source_inventory,
    write_corpus_source_inventory,
)
from yakbox.speech.analysis_corpus_text_sources import (
    download_corpus_text_sources,
    load_corpus_text_source_inventory,
    write_corpus_text_source_inventory,
)


def _voice(path: Path, *, seconds: int = 20) -> None:
    rate = 16_000
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        for second in range(seconds):
            amplitude = 0 if second in {6, 13} else 2_000
            writer.writeframesraw(
                b"".join(
                    int(amplitude).to_bytes(2, "little", signed=True)
                    for _ in range(rate)
                )
            )


def _registry(
    path: Path,
    audio: Path,
    *,
    source_digest: str = "a" * 64,
    source_url: str = "https://example.invalid/source.wav",
) -> None:
    digest = sha256_file(audio)
    path.write_text(
        f'''schema_version = 1
rights_policy = "Public domain only"

[voices.reader-one]
file = "voice.wav"
sha256 = "{digest}"
reader = "Reader One"
reader_url = "https://example.invalid/reader"
source_work = "Example"
catalog_url = "https://example.invalid/catalog"
source_url = "{source_url}"
source_sha256 = "{source_digest}"
license_id = "LicenseRef-LibriVox-Public-Domain-US"
rights_url = "https://librivox.org/pages/public-domain/"
source_start_seconds = 0
duration_seconds = 20
sample_rate_hz = 16000
channels = 1
pcm_bits = 16
filters = []
''',
        encoding="utf-8",
    )


def test_source_inventory_is_deterministic_licensed_and_schema_valid(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "voice.wav"
    registry = tmp_path / "voices.toml"
    output = tmp_path / "corpus"
    report = output / "inventory.json"
    _voice(audio)
    _registry(registry, audio)

    first = prepare_corpus_source_inventory(
        registry,
        repository_root=tmp_path,
        output_root=output,
    )
    second = prepare_corpus_source_inventory(
        registry,
        repository_root=tmp_path,
        output_root=output,
    )
    write_corpus_source_inventory(report, first)

    assert first.fingerprint == second.fingerprint
    assert len(first.windows) == 1
    assert first.source_cluster_count == 1
    assert len({item.source_passage_group for item in first.windows}) == 1
    assert len({item.source_window_id for item in first.windows}) == 1
    assert all(item.frame_count >= 4 * 16_000 for item in first.windows)
    assert tuple(item.start_frame for item in first.windows) == tuple(
        sorted(item.start_frame for item in first.windows)
    )
    Draft202012Validator(load_schema("speech-corpus-source-inventory")).validate(
        json.loads(report.read_text(encoding="utf-8"))
    )
    assert load_corpus_source_inventory(report, audio_root=output) == first


def test_source_inventory_loader_rejects_tampered_audio(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "voice.wav"
    registry = tmp_path / "voices.toml"
    output = tmp_path / "corpus"
    report = output / "inventory.json"
    _voice(audio)
    _registry(registry, audio)
    inventory = prepare_corpus_source_inventory(
        registry,
        repository_root=tmp_path,
        output_root=output,
    )
    write_corpus_source_inventory(report, inventory)
    first_window = output / inventory.windows[0].relative_audio_path
    first_window.write_bytes(first_window.read_bytes() + b"tamper")

    with pytest.raises(ValidationError, match="audio digest differs"):
        load_corpus_source_inventory(report, audio_root=output)


def test_source_archives_download_once_and_verify_registered_digest(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "voice.wav"
    registry = tmp_path / "voices.toml"
    output = tmp_path / "sources"
    report = output / "archives.json"
    payload = b"digest-pinned public-domain source"
    source_digest = hashlib.sha256(payload).hexdigest()
    _voice(audio)
    _registry(registry, audio, source_digest=source_digest)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=payload, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = download_corpus_source_archives(
            registry,
            repository_root=tmp_path,
            output_root=output,
            client=client,
        )
        second = download_corpus_source_archives(
            registry,
            repository_root=tmp_path,
            output_root=output,
            client=client,
        )
    write_corpus_source_archive_inventory(report, first)

    assert second == first
    assert len(requests) == 1
    assert (output / first.archives[0].relative_path).read_bytes() == payload
    Draft202012Validator(load_schema("speech-corpus-source-archives")).validate(
        json.loads(report.read_text(encoding="utf-8"))
    )
    assert load_corpus_source_archive_inventory(report, archive_root=output) == first


def test_public_domain_source_texts_are_discovered_pinned_and_reverified(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "voice.wav"
    registry = tmp_path / "voices.toml"
    output = tmp_path / "texts"
    report = output / "inventory.json"
    words = ("alpha beta gamma delta epsilon " * 30).strip() + "\n"
    payload = words.encode()
    requests: list[httpx.Request] = []
    _voice(audio)
    _registry(registry, audio)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if str(request.url) == "https://example.invalid/catalog":
            content = b'<a href="https://texts.invalid/book">Online text</a>'
            return httpx.Response(
                200,
                content=content,
                headers={"content-type": "text/html"},
                request=request,
            )
        if str(request.url) == "https://texts.invalid/book":
            content = b'<a href="/book.txt">Plain Text UTF-8</a>'
            return httpx.Response(
                200,
                content=content,
                headers={"content-type": "text/html"},
                request=request,
            )
        assert str(request.url) == "https://texts.invalid/book.txt"
        return httpx.Response(
            200,
            content=payload,
            headers={"content-type": "text/plain; charset=utf-8"},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        inventory = download_corpus_text_sources(
            registry,
            repository_root=tmp_path,
            output_root=output,
            client=client,
        )
    write_corpus_text_source_inventory(report, inventory)

    assert [str(request.url) for request in requests] == [
        "https://example.invalid/catalog",
        "https://texts.invalid/book",
        "https://texts.invalid/book.txt",
    ]
    assert inventory.sources[0].text_url == "https://texts.invalid/book.txt"
    assert inventory.sources[0].token_count == 150
    assert (output / inventory.sources[0].relative_plain_path).read_bytes() == payload
    Draft202012Validator(load_schema("speech-corpus-text-sources")).validate(
        json.loads(report.read_text(encoding="utf-8"))
    )
    assert load_corpus_text_source_inventory(report, text_root=output) == inventory

    plain = output / inventory.sources[0].relative_plain_path
    plain.write_text("tampered source text", encoding="utf-8")
    with pytest.raises(ValidationError, match="file identity differs"):
        load_corpus_text_source_inventory(report, text_root=output)


def test_collection_source_text_is_selected_by_title_and_reader(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "voice.wav"
    registry = tmp_path / "voices.toml"
    output = tmp_path / "texts"
    payload = ("collection source words " * 50).encode()
    _voice(audio)
    _registry(registry, audio)
    registry.write_text(
        registry.read_text(encoding="utf-8").replace(
            'source_work = "Example"',
            'source_work = "Collection: Target Story"',
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://example.invalid/catalog":
            content = b"""
<table>
  <tr><td>Other Story</td><td>
    <a href="https://texts.invalid/other.txt">Etext</a>
  </td><td>Other Reader</td></tr>
  <tr><td>Target Story</td><td>
    <a href="https://texts.invalid/target.txt">Etext</a>
  </td><td>Reader One</td></tr>
</table>
"""
            return httpx.Response(
                200,
                content=content,
                headers={"content-type": "text/html"},
                request=request,
            )
        assert str(request.url) == "https://texts.invalid/target.txt"
        return httpx.Response(
            200,
            content=payload,
            headers={"content-type": "text/plain"},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        inventory = download_corpus_text_sources(
            registry,
            repository_root=tmp_path,
            output_root=output,
            client=client,
        )

    assert inventory.sources[0].text_url == "https://texts.invalid/target.txt"


def test_expanded_passages_are_independent_quiet_bounded_clusters(
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "voice.wav"
    source_audio = tmp_path / "source.wav"
    registry = tmp_path / "voices.toml"
    archive_root = tmp_path / "archives"
    output = tmp_path / "expanded"
    _voice(prompt)
    _voice(source_audio, seconds=40)
    payload = source_audio.read_bytes()
    source_digest = hashlib.sha256(payload).hexdigest()
    _registry(
        registry,
        prompt,
        source_digest=source_digest,
        source_url="https://example.invalid/source.wav",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        archives = download_corpus_source_archives(
            registry,
            repository_root=tmp_path,
            output_root=archive_root,
            client=client,
        )
    inventory = prepare_expanded_corpus_source_inventory(
        registry,
        archives,
        repository_root=tmp_path,
        archive_root=archive_root,
        output_root=output,
        passage_starts_seconds=(0, 12, 24),
        extraction_seconds=10,
    )
    report = output / "inventory.json"
    write_corpus_source_inventory(report, inventory)

    assert inventory.source_cluster_count == 3
    assert len(inventory.windows) == 3
    assert {item.archive_start_milliseconds for item in inventory.windows} == {
        0,
        12_000,
        24_000,
    }
    assert all(item.frame_count >= 5 * 16_000 for item in inventory.windows)
    assert load_corpus_source_inventory(report, audio_root=output) == inventory
