from __future__ import annotations

import random
import tomllib
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from yakbox._files import sha256_file
from yakbox.audio.pcm import decode_pcm, encode_pcm
from yakbox.audio.splice import FrameReplacement, splice_wav_region, splice_wav_regions
from yakbox.audiobook import load_manifest, normalize_sources, plan_audiobook
from yakbox.errors import ArtifactError
from yakbox.local_alignment import MlxWhisperAligner
from yakbox.local_alignment_audio import alignment_audio
from yakbox.speech.analysis_runtime import BUILT_IN_WORKERS
from yakbox.speech.model_registry import load_model_registry

_GOLDEN = [
    (1, 1, False, "12ed67727e4584d407a7fb66b3140b41a022b16234e4d34e5e5d5781f7c7f381"),
    (1, 2, True, "3bf205ba2d37affb848418f39a1298da1287ca6405e2fe4e94bc34f5864294c8"),
    (2, 1, False, "5b2e8f5f3589bd96c7bf7ad101b0834eb7ac22be6cc04298b8c2bb2ecd39e190"),
    (2, 2, True, "995bca255935805ad85c011b54cdd22fef1ab0c5228f1461bd2c94abd1b10d67"),
    (3, 1, False, "ac15d1278d27b16c5c8edb08f9c926b51f21f7a080473330409b5935c5759291"),
    (3, 2, True, "879f1bf45a93ef49f5fdafc73cb098bb49fac096344b6b524b7322eb49924aec"),
    (4, 1, False, "596bc35cdbde9f5e27443de48473e6640dd1ca15944708e20526bbcb2c129b01"),
    (4, 2, True, "1ba19ce58680f6190ca5bf863255ddd1b26847b40fb9e8382c6ce606938a9a6a"),
]


@pytest.mark.parametrize(("width", "channels", "multi", "digest"), _GOLDEN)
def test_streamed_splices_preserve_reviewed_pcm_bytes(
    tmp_path: Path, width: int, channels: int, multi: bool, digest: str
) -> None:
    # Golden digests were captured from the previous full-buffer implementation.
    rng = random.Random(width * 100 + channels + 200)  # noqa: S311 - deterministic PCM fixture
    source, replacement, output = (
        tmp_path / name for name in ("source.wav", "replacement.wav", "output.wav")
    )
    for path, frames in ((source, 200), (replacement, 100)):
        with wave.open(str(path), "wb") as writer:
            writer.setparams((channels, width, 8000, 0, "NONE", "not compressed"))
            writer.writeframes(
                encode_pcm(
                    tuple(rng.randint(-100, 100) for _ in range(frames * channels)),
                    width,
                )
            )
    if multi:
        splice_wav_regions(
            source,
            (
                FrameReplacement("a", 20, 40, replacement),
                FrameReplacement("b", 60, 80, replacement),
            ),
            output,
        )
    else:
        splice_wav_region(
            source, replacement, output, start_seconds=20 / 8000, end_seconds=60 / 8000
        )
    assert sha256_file(output) == digest


def _write_audio(path: Path, seconds: int) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        writer.writeframes((1000).to_bytes(2, "little") * 16000 * seconds)


def test_alignment_window_is_bounded_and_cleaned_up(tmp_path: Path) -> None:
    audio = tmp_path / "chapter.wav"
    _write_audio(audio, 120)
    with alignment_audio(audio, (60, 61)) as (path, offset, clip):
        assert offset == 59
        assert clip == (1, 2)
        with wave.open(str(path), "rb") as reader:
            assert reader.getnframes() == 3 * 16000
        assert path != audio
    assert not path.exists()
    with alignment_audio(audio, None) as (path, offset, clip):
        assert path == audio and offset == 0 and clip is None


@pytest.mark.asyncio
async def test_bounded_alignment_restores_absolute_word_segment_and_signal_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "chapter.wav"
    _write_audio(audio, 120)
    calls: list[tuple[str, dict[str, object]]] = []

    def transcribe(source: object, **options: object) -> dict[str, object]:
        path = str(source)
        with wave.open(path, "rb") as reader:
            assert reader.getnframes() == 3 * 16000
        calls.append((path, options))
        return {
            "text": " hello",
            "language": "en",
            "segments": [
                {
                    "start": 1.1,
                    "end": 1.8,
                    "words": [
                        {"word": " hello", "start": 1.1, "end": 1.8, "probability": 0.9}
                    ],
                }
            ],
        }

    aligner = MlxWhisperAligner(
        model="fixture", decode_consensus=True, prompt_sensitivity=True
    )
    monkeypatch.setattr(aligner, "_resolve_model_path", lambda: "fixture")
    monkeypatch.setattr(aligner, "_module", SimpleNamespace(transcribe=transcribe))
    result = await aligner.align_window(
        audio, "hello", language="en", start_seconds=60, end_seconds=61
    )
    assert result.tokens[0].start_seconds == pytest.approx(60.1)
    assert result.tokens[0].end_seconds == pytest.approx(60.8)
    assert result.segments[0].start_seconds == pytest.approx(60.1)
    assert result.speech_regions[0].start_seconds == 59
    assert result.clip_start_seconds == 60 and result.clip_end_seconds == 61
    assert len(calls) == 3
    assert all(options["clip_timestamps"] == [1, 2] for _, options in calls)
    assert len({path for path, _ in calls}) == 1
    assert not Path(calls[0][0]).exists()


def test_analysis_worker_locks_match_reviewed_registry_and_definitions() -> None:
    root = Path(__file__).parents[2]
    registry = load_model_registry()
    reviewed = {item.name: item.version for item in registry.packages}
    for family, definition in BUILT_IN_WORKERS.items():
        lock = tomllib.loads(
            (root / "src/yakbox/runtimes" / family / "uv.lock").read_text()
        )
        locked = {item["name"]: item["version"] for item in lock["package"]}
        for package in definition.packages:
            assert locked[package.name] == package.version == reviewed[package.name]


def test_manifest_model_variants_affect_build_plan_identity(
    book_workspace: Path,
) -> None:
    path = book_workspace / "yakbox.toml"
    text = (
        path.read_text()
        .replace('backend = "fake"', 'backend = "chatterbox-local"')
        .replace("sample_rate = 16000\n", "")
    )
    path.write_text(text)
    loaded = load_manifest(path)
    first = plan_audiobook(loaded, normalize_sources(loaded.sources))
    path.write_text(
        text.replace(
            'backend = "chatterbox-local"',
            'backend = "chatterbox-local"\nvariant = "turbo"',
        )
    )
    loaded = load_manifest(path)
    second = plan_audiobook(loaded, normalize_sources(loaded.sources))
    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize("start", [float("inf"), float("nan")])
def test_splice_rejects_nonfinite_timing_without_output(
    tmp_path: Path, start: float
) -> None:
    output = tmp_path / "out.wav"
    with pytest.raises(ArtifactError, match="finite"):
        splice_wav_region(
            tmp_path / "source.wav",
            tmp_path / "replacement.wav",
            output,
            start_seconds=start,
            end_seconds=2,
        )
    assert not output.exists()


@pytest.mark.asyncio
async def test_independent_recognition_reuse_does_not_accept_prompt_rescues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "audio.wav"
    _write_audio(audio, 2)
    calls: list[dict[str, object]] = []

    def transcribe(_audio: object, **options: object) -> dict[str, object]:
        calls.append(options)
        word = str(options.get("initial_prompt", "hello"))
        return {
            "text": word,
            "segments": [
                {
                    "start": 0.1,
                    "end": 0.5,
                    "words": [
                        {"word": word, "start": 0.1, "end": 0.5, "probability": 0.9}
                    ],
                }
            ],
        }

    aligner = MlxWhisperAligner(model="fixture", prompt_sensitivity=True)
    monkeypatch.setattr(aligner, "_resolve_model_path", lambda: "fixture")
    monkeypatch.setattr(aligner, "_module", SimpleNamespace(transcribe=transcribe))
    first = await aligner.align(audio, "hello", language="en")
    second = await aligner.align(audio, "goodbye", language="en")
    assert first.tokens[0].text == second.tokens[0].text == "hello"
    assert second.prompt_sensitivity == "prompt_rescued"
    assert len([call for call in calls if "initial_prompt" not in call]) == 1
    assert [call["initial_prompt"] for call in calls if "initial_prompt" in call] == [
        "hello",
        "goodbye",
    ]
    audio.write_bytes(audio.read_bytes() + b"changed identity")
    await aligner.align(audio, "hello", language="en")
    assert len([call for call in calls if "initial_prompt" not in call]) == 2


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_compact_pcm_encoding_clamps_samples_without_wraparound(width: int) -> None:
    limit = 1 << (width * 8 - 1)
    encoded = encode_pcm((-limit * 2, -1, 0, limit * 2), width)
    assert tuple(decode_pcm(encoded, width)) == (-limit, -1, 0, limit - 1)
