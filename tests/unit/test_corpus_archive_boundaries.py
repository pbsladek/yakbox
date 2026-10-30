from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
from tests.unit.test_speech_corpus_sources import _registry, _voice

import yakbox.speech.analysis_corpus_archives as archives
from yakbox.errors import ArtifactError, ValidationError


def test_archive_download_falls_back_to_canonical_url_and_checks_bytes(
    tmp_path: Path,
) -> None:
    payload = b"digest-pinned-audio"
    digest = hashlib.sha256(payload).hexdigest()
    urls = archives._source_urls(
        "https://ia800000.us.archive.org/items/book/chapter.mp3"
    )
    assert len(urls) == 2
    assert urls[1] == "https://archive.org/download/book/chapter.mp3"
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            503 if len(requests) == 1 else 200, content=payload, request=request
        )

    destination = tmp_path / "chapter.mp3"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        archives._download_file(client, urls, destination, expected_digest=digest)
    assert destination.read_bytes() == payload
    assert requests == list(urls)


@pytest.mark.parametrize("failure", ["digest", "size", "http", "redirect"])
def test_archive_download_failures_never_publish_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    destination = tmp_path / "archive.wav"
    if failure == "size":
        monkeypatch.setattr(archives, "_MAXIMUM_SOURCE_BYTES", 1)
    code = 503 if failure == "http" else 200
    url = (
        "http://archive.invalid/chapter.wav"
        if failure == "redirect"
        else "https://archive.invalid/chapter.wav"
    )
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    code, stream=httpx.ByteStream(b"payload"), request=request
                )
            )
        ) as client,
        pytest.raises(ArtifactError),
    ):
        archives._download_file(client, (url,), destination, expected_digest="0" * 64)
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.part*"))


def test_archive_download_cli_writes_an_inventory_and_closes_owned_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    audio = tmp_path / "voice.wav"
    _voice(audio)
    payload = b"registered-archive-content"
    registry = tmp_path / "voices.toml"
    _registry(registry, audio, source_digest=hashlib.sha256(payload).hexdigest())
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=payload, request=request)
        )
    )
    monkeypatch.setattr(archives.httpx, "Client", lambda **_kwargs: client)
    output = tmp_path / "sources"
    report = output / "inventory.json"
    assert (
        archives.main(
            [
                "--voice-registry",
                str(registry),
                "--repository-root",
                str(tmp_path),
                "--output-root",
                str(output),
                "--report-output",
                str(report),
            ]
        )
        == 0
    )
    assert client.is_closed
    assert json.loads(capsys.readouterr().out)["archive_count"] == 1
    inventory = archives.load_corpus_source_archive_inventory(
        report, archive_root=output
    )
    assert inventory.archives[0].size_bytes == len(payload)
    archive = output / inventory.archives[0].relative_path
    archive.write_bytes(b"changed")
    with pytest.raises(ValidationError, match="identity differs"):
        archives.load_corpus_source_archive_inventory(report, archive_root=output)


@pytest.mark.parametrize(
    "url",
    [
        "https://archive.org/no-items",
        "https://archive.org/items",
        "https://archive.org/items/book",
        "https://other.invalid/items/book/chapter.mp3",
    ],
)
def test_archive_fallback_does_not_invent_downloads_for_unrecognized_urls(
    url: str,
) -> None:
    assert archives._source_urls(url) == (url,)
