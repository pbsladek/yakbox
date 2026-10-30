from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from tests.unit.test_speech_corpus_sources import _registry, _voice

import yakbox.speech.analysis_corpus_text_sources as sources
from yakbox.errors import ValidationError


def _client(content: bytes, *, content_type: str = "text/html") -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=content,
                headers={"content-type": content_type},
                request=request,
            )
        )
    )


def test_source_discovery_matches_the_book_and_reader_in_a_catalog_table() -> None:
    html = (
        b"<table><tr><td>Other Book Reader One</td>"
        b'<td><a href="/wrong.txt">etext</a></td></tr>'
        b"<tr><td>Example Reader One</td>"
        b'<td><a href="http://texts.invalid/right.txt">etext</a></td></tr></table>'
    )
    with _client(html) as client:
        url = sources._discover_online_text(
            client,
            "https://catalog.invalid/reader",
            source_work="Example",
            reader="Reader One",
        )
    assert url == "https://texts.invalid/right.txt"
    with (
        _client(b"<table><tr><td>no match</td></tr></table>") as client,
        pytest.raises(ValidationError, match="one source-text link"),
    ):
        sources._discover_online_text(
            client,
            "https://catalog.invalid/reader",
            source_work="Example",
            reader="Reader One",
        )


@pytest.mark.parametrize(
    "metadata,expected",
    [
        (
            {"metadata": {"url_text_source": "http://texts.invalid/book.txt"}},
            "https://texts.invalid/book.txt",
        ),
        (
            {"files": [{"name": "book_djvu.txt"}, {"name": "cover.jpg"}]},
            "https://archive.org/download/book/book_djvu.txt",
        ),
    ],
)
def test_archive_source_discovery_accepts_explicit_text_or_unique_ocr_file(
    metadata: dict[str, object], expected: str
) -> None:
    with _client(
        json.dumps(metadata).encode(), content_type="application/json"
    ) as client:
        assert (
            sources._archive_text_source(client, "https://archive.org/details/book")
            == expected
        )


@pytest.mark.parametrize(
    "content",
    [
        b"not-json",
        b"[]",
        b"{}",
        b'{"files":[{"name":"one_djvu.txt"},{"name":"two_djvu.txt"}]}',
    ],
)
def test_archive_source_discovery_rejects_ambiguous_or_invalid_metadata(
    content: bytes,
) -> None:
    with _client(content) as client, pytest.raises(ValidationError):
        sources._archive_text_source(client, "https://archive.org/details/book")


def test_html_source_extraction_removes_hidden_content_and_preserves_unicode() -> None:
    html = (
        b"<html><style>hidden style</style><script>secret();</script>"
        b"<h1>Caf\xc3\xa9</h1><p>Visible &amp; readable.</p>"
        b"<svg>hidden image</svg></html>"
    )
    with _client(html) as client:
        result = sources._download_resolved_text(
            client, "https://texts.invalid/book.html"
        )
    assert result.suffix == ".html"
    assert result.payload == html
    assert result.plain_text == "Café\nVisible & readable.\n"


@pytest.mark.parametrize(
    "content_type,content",
    [("application/pdf", b"pdf"), ("text/plain", b"<html>unexpected html</html>")],
)
def test_source_text_rejects_unsupported_and_mislabeled_formats(
    content_type: str, content: bytes
) -> None:
    with (
        _client(content, content_type=content_type) as client,
        pytest.raises(ValidationError, match="format is unsupported"),
    ):
        sources._download_resolved_text(client, "https://texts.invalid/book")


@pytest.mark.parametrize("length", ["invalid", "-1", str(64 * 1024 * 1024 + 1)])
def test_source_text_download_rejects_bad_content_lengths_before_buffering(
    length: str,
) -> None:
    response = httpx.Response(200, headers={"content-length": length})
    with pytest.raises(ValidationError, match=r"content length|byte limit"):
        sources._check_content_length(response)


def test_source_text_download_enforces_streaming_limit_and_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sources, "_MAXIMUM_RESPONSE_BYTES", 4)
    with (
        _client(b"too many bytes", content_type="text/plain") as client,
        pytest.raises(ValidationError, match="byte limit"),
    ):
        sources._bounded_get(client, "https://texts.invalid/book.txt")
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(503, request=request)
            )
        ) as client,
        pytest.raises(ValidationError, match="Cannot download"),
    ):
        sources._bounded_get(client, "https://texts.invalid/book.txt")


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://gutenberg.org/etext/42", "https://gutenberg.org/ebooks/42"),
        (
            "https://www.gutenberg.org/ebooks/42.txt.utf-8",
            "https://www.gutenberg.org/cache/epub/42/pg42.txt",
        ),
    ],
)
def test_legacy_source_urls_resolve_to_current_https_downloads(
    url: str, expected: str
) -> None:
    assert sources._https_upgrade(url) == expected


def test_source_encoding_decodes_windows_text_and_rejects_unknown_bytes() -> None:
    assert sources._decode_text(b"A \x93quoted\x94 sentence.") == "A “quoted” sentence."
    with pytest.raises(ValidationError, match="encoding is unsupported"):
        sources._decode_text(b"\x81")


@pytest.mark.parametrize(
    "content",
    [
        "schema_version = 2\n[sources]",
        'schema_version = 1\n[sources.unknown]\nurl="https://texts.invalid/book"\nrights_url="https://texts.invalid/rights"',
        'schema_version = 1\n[sources.reader-one]\nurl=42\nrights_url="https://texts.invalid/rights"',
        'schema_version = 1\n[sources.reader-one]\nurl="file:///book"\nrights_url="https://texts.invalid/rights"',
        'schema_version = 1\n[sources.reader-one]\nurl="https://texts.invalid/book"\nrights_url="https://texts.invalid/rights"\nextra=true',
    ],
)
def test_source_overrides_reject_unauthorized_voices_and_unsafe_urls(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "overrides.toml"
    path.write_text(content)
    with pytest.raises(ValidationError):
        sources._load_source_overrides(path, voices={"reader-one"})


def test_source_download_cli_owns_its_client_and_writes_reverified_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    audio = tmp_path / "voice.wav"
    _voice(audio)
    registry = tmp_path / "voices.toml"
    _registry(registry, audio)
    overrides = tmp_path / "overrides.toml"
    overrides.write_text(
        'schema_version=1\n[sources.reader-one]\nurl="https://texts.invalid/book.txt"\nrights_url="https://texts.invalid/rights"\n'
    )
    client = _client(b"one two three four " * 30, content_type="text/plain")
    monkeypatch.setattr(sources.httpx, "Client", lambda **_kwargs: client)
    output = tmp_path / "output"
    inventory_path = output / "inventory.json"
    code = sources.main(
        [
            "--voice-registry",
            str(registry),
            "--repository-root",
            str(tmp_path),
            "--output-root",
            str(output),
            "--inventory-output",
            str(inventory_path),
            "--source-overrides",
            str(overrides),
        ]
    )
    assert code == 0
    assert client.is_closed
    assert json.loads(capsys.readouterr().out)["source_count"] == 1
    inventory = sources.load_corpus_text_source_inventory(
        inventory_path, text_root=output
    )
    assert inventory.sources[0].token_count == 120
    assert inventory.source_override_digest
    raw = json.loads(inventory_path.read_text())
    raw["sources"][0]["token_count"] += 1
    inventory_path.write_text(json.dumps(raw))
    with pytest.raises(ValidationError, match="token count differs"):
        sources.load_corpus_text_source_inventory(inventory_path, text_root=output)
