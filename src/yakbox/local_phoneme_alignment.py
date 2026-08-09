"""Lazy Wav2Vec2 CTC phoneme forced-alignment backend."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import json
import math
import subprocess
import sys
import unicodedata
import wave
from array import array
from pathlib import Path
from typing import Any

from yakbox.audio.crop import wav_duration_seconds
from yakbox.errors import BackendUnavailableError, BuildError
from yakbox.phoneme_models import require_phoneme_model_path
from yakbox.speech.alignment import alignment_fingerprint
from yakbox.speech.phonemes import PhonemeAlignmentResult, PhonemeToken

_SPECIAL_TOKENS = frozenset({"<pad>", "<s>", "</s>", "<unk>"})
_PCM16_SAMPLE_WIDTH = 2


class Wav2Vec2CtcPhonemeAligner:
    """Force eSpeak IPA pronunciations onto local Wav2Vec2 CTC emissions."""

    def __init__(
        self,
        *,
        model: str,
        revision: str | None,
        timeout_seconds: float = 180.0,
        phonemizer_executable: str = "espeak-ng",
    ) -> None:
        self.model = model
        self.revision = revision
        self.timeout_seconds = timeout_seconds
        self.phonemizer_executable = phonemizer_executable
        self._runtime: tuple[Any, Any, Any, str] | None = None
        try:
            version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            version = "unavailable"
        self._fingerprint = alignment_fingerprint(
            "wav2vec2-ctc-phoneme",
            f"{model}@{revision or 'local'}",
            version,
            settings={
                "phonemizer": phonemizer_executable,
                "phonemizer_mode": "ipa-stressless",
                "phoneme_tokenizer": "full-cover-v2",
                "ctc_path": "viterbi-v1",
            },
        )

    @property
    def fingerprint(self) -> str:
        """Return the model, runtime, and phonemizer fingerprint."""
        return self._fingerprint

    async def align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
    ) -> PhonemeAlignmentResult:
        """Force the expected phoneme sequence onto one PCM WAV file."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._align_sync, audio, expected_text, language),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as error:
            raise BuildError(
                f"Phoneme alignment exceeded the {self.timeout_seconds:g}s timeout"
            ) from error

    def _align_sync(
        self,
        audio: Path,
        expected_text: str,
        language: str,
    ) -> PhonemeAlignmentResult:
        model_path = require_phoneme_model_path(self.model, self.revision)
        vocabulary = _load_vocabulary(model_path)
        pronunciation = _phonemize(
            expected_text,
            language=language,
            executable=self.phonemizer_executable,
            timeout_seconds=self.timeout_seconds,
        )
        symbols, target_ids = _target_tokens(pronunciation, vocabulary)
        samples, sample_rate = _read_pcm16_mono(audio)
        samples = _resample(samples, sample_rate, 16_000)
        torch, processor, model, device = self._load_runtime(model_path)
        inputs = processor(samples, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            logits = model(inputs.input_values.to(device)).logits[0]
            log_probabilities = torch.log_softmax(logits, dim=-1).cpu().tolist()
        path = _ctc_forced_alignment(
            log_probabilities,
            target_ids,
            blank_id=int(model.config.pad_token_id),
        )
        frame_seconds = wav_duration_seconds(audio) / len(log_probabilities)
        phonemes = tuple(
            PhonemeToken(
                symbol=symbol,
                start_seconds=start_frame * frame_seconds,
                end_seconds=end_frame * frame_seconds,
                confidence=confidence,
            )
            for symbol, (start_frame, end_frame, confidence) in zip(
                symbols, path, strict=True
            )
        )
        return PhonemeAlignmentResult(
            phonemes=phonemes,
            backend="wav2vec2-ctc",
            model=self.model,
            fingerprint=self.fingerprint,
            language=language,
        )

    def _load_runtime(self, model_path: Path) -> tuple[Any, Any, Any, str]:
        if self._runtime is not None:
            return self._runtime
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise BackendUnavailableError(
                "Phoneme alignment is not installed; run uv sync --extra phoneme"
            ) from error
        processor = transformers.AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True
        )
        model = transformers.AutoModelForCTC.from_pretrained(
            str(model_path), local_files_only=True
        )
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        self._runtime = (torch, processor, model, device)
        return self._runtime


def open_phoneme_aligner(
    backend: str,
    *,
    model: str,
    revision: str | None,
    timeout_seconds: float = 180.0,
) -> Wav2Vec2CtcPhonemeAligner:
    """Open a supported lazy phoneme forced-alignment backend."""
    if backend.casefold() != "wav2vec2-ctc":
        raise BackendUnavailableError(
            f"Unsupported phoneme alignment backend: {backend}"
        )
    return Wav2Vec2CtcPhonemeAligner(
        model=model,
        revision=revision,
        timeout_seconds=timeout_seconds,
    )


def _phonemize(
    text: str,
    *,
    language: str,
    executable: str,
    timeout_seconds: float,
) -> str:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell expansion.
            [
                executable,
                "-q",
                "--ipa",
                "-v",
                language.strip().casefold() or "en-us",
                text,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise BackendUnavailableError(
            "Phoneme alignment requires the optional espeak-ng executable"
        ) from error
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise BuildError("eSpeak NG could not phonemize the expected text") from error
    value = unicodedata.normalize("NFC", completed.stdout.strip())
    value = value.translate(
        {
            ord("\N{MODIFIER LETTER VERTICAL LINE}"): None,
            ord("\N{MODIFIER LETTER LOW VERTICAL LINE}"): None,
            ord("\N{COMBINING DOUBLE INVERTED BREVE}"): None,
            ord("\N{ZERO WIDTH JOINER}"): None,
        }
    )
    if not value:
        raise BuildError("eSpeak NG returned no expected phonemes")
    return value


def _load_vocabulary(model_path: Path) -> dict[str, int]:
    try:
        raw = json.loads((model_path / "vocab.json").read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError
        return {str(key): int(value) for key, value in raw.items()}
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise BackendUnavailableError("Phoneme model vocabulary is invalid") from error


def _target_tokens(
    pronunciation: str,
    vocabulary: dict[str, int],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    candidates = tuple(
        sorted(
            (
                token
                for token in vocabulary
                if token not in _SPECIAL_TOKENS and token.strip()
            ),
            key=lambda value: (-len(value), value),
        )
    )
    value = pronunciation.replace("_", " ")
    paths: dict[int, tuple[tuple[str, int], ...]] = {len(value): ()}
    for index in range(len(value) - 1, -1, -1):
        if value[index].isspace():
            suffix = paths.get(index + 1)
            if suffix is not None:
                paths[index] = suffix
            continue
        for token in candidates:
            end = index + len(token)
            suffix = paths.get(end)
            if value.startswith(token, index) and suffix is not None:
                paths[index] = ((token, vocabulary[token]), *suffix)
                break
    path = paths.get(0)
    if path is None:
        codepoint = _first_unrepresentable_codepoint(value, candidates)
        raise BuildError(
            f"Expected phoneme {codepoint} cannot be represented by the pinned "
            "model vocabulary"
        )
    if not path:
        raise BuildError("Expected pronunciation contains no model phonemes")
    return tuple(item[0] for item in path), tuple(item[1] for item in path)


def _first_unrepresentable_codepoint(
    value: str,
    candidates: tuple[str, ...],
) -> str:
    reachable = {0}
    for index in range(len(value)):
        if index not in reachable:
            continue
        if value[index].isspace():
            reachable.add(index + 1)
        for token in candidates:
            if value.startswith(token, index):
                reachable.add(index + len(token))
    index = max(reachable)
    while index < len(value) and value[index].isspace():
        index += 1
    if index >= len(value):
        index = max(0, len(value) - 1)
    return f"U+{ord(value[index]):04X}"


def _read_pcm16_mono(path: Path) -> tuple[list[float], int]:
    try:
        with wave.open(str(path), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            compression = reader.getcomptype()
            content = reader.readframes(reader.getnframes())
    except (OSError, EOFError, wave.Error) as error:
        raise BuildError(f"Cannot read phoneme-alignment WAV: {path}") from error
    if compression != "NONE" or width != _PCM16_SAMPLE_WIDTH or channels < 1:
        raise BuildError("Phoneme alignment requires 16-bit PCM WAV audio")
    values = array("h")
    values.frombytes(content)
    if sys.byteorder == "big":
        values.byteswap()
    samples = [
        sum(values[index + channel] for channel in range(channels))
        / (channels * 32_768)
        for index in range(0, len(values), channels)
    ]
    return samples, sample_rate


def _resample(samples: list[float], source_rate: int, target_rate: int) -> list[float]:
    if source_rate == target_rate:
        return samples
    if source_rate <= 0 or target_rate <= 0 or not samples:
        raise BuildError("Phoneme alignment received invalid sample-rate metadata")
    target_count = max(1, round(len(samples) * target_rate / source_rate))
    scale = source_rate / target_rate
    result: list[float] = []
    for target_index in range(target_count):
        position = min(len(samples) - 1, target_index * scale)
        left = int(position)
        right = min(len(samples) - 1, left + 1)
        fraction = position - left
        result.append(samples[left] * (1 - fraction) + samples[right] * fraction)
    return result


def _ctc_forced_alignment(
    log_probabilities: list[list[float]],
    targets: tuple[int, ...],
    *,
    blank_id: int,
) -> tuple[tuple[int, int, float], ...]:
    """Return target frame spans using the standard CTC Viterbi topology."""
    if not log_probabilities or not targets:
        raise BuildError("CTC forced alignment requires emissions and targets")
    frame_count = len(log_probabilities)
    if frame_count < len(targets):
        raise BuildError("Audio is too short for the expected phoneme sequence")
    _states, scores, backpointers = _ctc_trellis(
        log_probabilities,
        targets,
        blank_id=blank_id,
    )
    state_path = _ctc_backtrack(scores, backpointers)
    return _ctc_target_spans(log_probabilities, targets, state_path)


def _ctc_trellis(
    log_probabilities: list[list[float]],
    targets: tuple[int, ...],
    *,
    blank_id: int,
) -> tuple[tuple[int, ...], list[float], list[list[int]]]:
    states = tuple(
        blank_id if index % 2 == 0 else targets[index // 2]
        for index in range(len(targets) * 2 + 1)
    )
    frame_count = len(log_probabilities)
    negative_infinity = float("-inf")
    previous = [negative_infinity] * len(states)
    previous[0] = log_probabilities[0][blank_id]
    if len(states) > 1:
        previous[1] = log_probabilities[0][states[1]]
    backpointers = [[-1] * len(states) for _ in range(frame_count)]
    for frame in range(1, frame_count):
        current = [negative_infinity] * len(states)
        for state, label in enumerate(states):
            score, source = _ctc_predecessor(
                previous,
                states,
                state=state,
                label=label,
                blank_id=blank_id,
            )
            current[state] = score + log_probabilities[frame][label]
            backpointers[frame][state] = source
        previous = current
    return states, previous, backpointers


def _ctc_predecessor(
    previous: list[float],
    states: tuple[int, ...],
    *,
    state: int,
    label: int,
    blank_id: int,
) -> tuple[float, int]:
    choices = [(previous[state], state)]
    if state > 0:
        choices.append((previous[state - 1], state - 1))
    if state > 1 and label != blank_id and label != states[state - 2]:
        choices.append((previous[state - 2], state - 2))
    return max(choices, key=lambda item: item[0])


def _ctc_backtrack(
    scores: list[float],
    backpointers: list[list[int]],
) -> tuple[int, ...]:
    state_count = len(scores)
    score, state = max(
        ((scores[-1], state_count - 1), (scores[-2], state_count - 2)),
        key=lambda item: item[0],
    )
    if not math.isfinite(score):
        raise BuildError("No valid CTC path covers the expected phonemes")
    state_path = [state]
    for frame in range(len(backpointers) - 1, 0, -1):
        state = backpointers[frame][state]
        if state < 0:
            raise BuildError("CTC forced-alignment backtracking failed")
        state_path.append(state)
    state_path.reverse()
    return tuple(state_path)


def _ctc_target_spans(
    log_probabilities: list[list[float]],
    targets: tuple[int, ...],
    state_path: tuple[int, ...],
) -> tuple[tuple[int, int, float], ...]:
    spans: list[tuple[int, int, float]] = []
    for target_index, target_id in enumerate(targets):
        target_state = target_index * 2 + 1
        frames = tuple(
            frame
            for frame, current_state in enumerate(state_path)
            if current_state == target_state
        )
        if not frames:
            raise BuildError("CTC path omitted an expected phoneme")
        confidence = math.exp(
            sum(log_probabilities[frame][target_id] for frame in frames) / len(frames)
        )
        spans.append((frames[0], frames[-1] + 1, min(1.0, confidence)))
    return tuple(spans)
