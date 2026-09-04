"""Versioned English lexical normalization with bounded equivalences."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

import regex

from yakbox.errors import ValidationError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint

NORMALIZATION_VERSION = 1
MAXIMUM_EQUIVALENCE_TOKENS = 8
MAXIMUM_EQUIVALENCE_RULES = 1_024
_TOKEN = regex.compile(
    r"[\p{L}\p{M}\p{N}]+(?:[\N{RIGHT SINGLE QUOTATION MARK}'-]"
    r"[\p{L}\p{M}\p{N}]+)*"
)


@dataclass(frozen=True, slots=True)
class NormalizedToken:
    """One token plus its source character projection."""

    text: str
    source_start: int
    source_end: int


@dataclass(frozen=True, slots=True)
class NormalizationTrace:
    """Deterministic English normalization output and source mapping."""

    version: int
    language: str
    unicode_form: str
    tokens: tuple[NormalizedToken, ...]

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("lexical-normalization-trace-v1", self)


@dataclass(frozen=True, slots=True)
class DirectionalEquivalence:
    """One reviewed expected-to-recognized lexical equivalence."""

    expected: tuple[str, ...]
    recognized: tuple[str, ...]
    reason_code: str

    def __post_init__(self) -> None:
        if not self.expected or not self.recognized or not self.reason_code:
            raise ValidationError("Lexical equivalences must be non-empty")
        if max(len(self.expected), len(self.recognized)) > MAXIMUM_EQUIVALENCE_TOKENS:
            raise ValidationError("Lexical equivalence exceeds the bounded token limit")
        for sequence in (self.expected, self.recognized):
            if any(not token or token != token.casefold() for token in sequence):
                raise ValidationError("Lexical equivalence tokens must be normalized")
        if not valid_reason_code(self.reason_code):
            raise ValidationError("Lexical equivalence reason code is invalid")


@dataclass(frozen=True, slots=True)
class EquivalenceSet:
    """Validated directional equivalences used during expected-side comparison."""

    version: int
    rules: tuple[DirectionalEquivalence, ...]

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValidationError("Lexical equivalence set version must be positive")
        if len(self.rules) > MAXIMUM_EQUIVALENCE_RULES:
            raise ValidationError("Lexical equivalence set exceeds the rule limit")
        expected = tuple(rule.expected for rule in self.rules)
        if len(expected) != len(set(expected)):
            raise ValidationError("Lexical equivalence expected forms must be unique")
        graph = {rule.expected: rule.recognized for rule in self.rules}
        if any(rule.recognized in graph for rule in self.rules):
            raise ValidationError("Lexical equivalences cannot form directional chains")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("lexical-equivalence-set-v1", self)


@dataclass(frozen=True, slots=True)
class SequenceEdit:
    """One deterministic expected/recognized sequence alignment edit."""

    operation: str
    expected_start: int
    expected_end: int
    recognized_start: int
    recognized_end: int
    equivalence_reason: str | None = None


def normalize_english(text: str) -> NormalizationTrace:
    """Normalize English text without losing original character locations."""
    if not isinstance(text, str):
        raise ValidationError("Lexical normalization requires text")
    normalized = unicodedata.normalize("NFKC", text)
    tokens = tuple(
        NormalizedToken(match.group().casefold(), match.start(), match.end())
        for match in _TOKEN.finditer(normalized)
    )
    return NormalizationTrace(NORMALIZATION_VERSION, "en", "NFKC", tokens)


def align_token_sequences(
    expected: tuple[str, ...],
    recognized: tuple[str, ...],
    equivalences: EquivalenceSet,
) -> tuple[SequenceEdit, ...]:
    """Align token sequences with stable opcodes and bounded equivalence labels."""
    applications = _equivalence_anchors(
        expected,
        recognized,
        equivalences,
    )
    edits: list[SequenceEdit] = []
    expected_cursor = 0
    recognized_cursor = 0
    for application in applications:
        edits.extend(
            _unalias_edits(
                expected,
                recognized,
                (expected_cursor, application.expected_start),
                (recognized_cursor, application.recognized_start),
            )
        )
        edits.append(application)
        expected_cursor = application.expected_end
        recognized_cursor = application.recognized_end
    edits.extend(
        _unalias_edits(
            expected,
            recognized,
            (expected_cursor, len(expected)),
            (recognized_cursor, len(recognized)),
        )
    )
    return tuple(edits)


def _equivalence_anchors(
    expected: tuple[str, ...],
    recognized: tuple[str, ...],
    equivalences: EquivalenceSet,
) -> tuple[SequenceEdit, ...]:
    """Return only unique, non-overlapping, order-preserving aliases.

    Keeping original recognized coordinates is essential: downstream token evidence
    hashes and audio projections refer to the recognizer output, not a rewritten
    intermediate sequence. Ambiguous or crossing rules deliberately do not apply and
    therefore remain ordinary mismatches.
    """
    applications: list[SequenceEdit] = []
    for rule in equivalences.rules:
        expected_positions = _positions(expected, rule.expected)
        recognized_positions = _positions(recognized, rule.recognized)
        if len(expected_positions) != 1 or len(recognized_positions) != 1:
            continue
        expected_start = expected_positions[0]
        recognized_start = recognized_positions[0]
        applications.append(
            SequenceEdit(
                "equivalent",
                expected_start,
                expected_start + len(rule.expected),
                recognized_start,
                recognized_start + len(rule.recognized),
                rule.reason_code,
            )
        )
    ordered = tuple(
        sorted(
            applications,
            key=lambda item: (item.expected_start, item.recognized_start),
        )
    )
    accepted: list[SequenceEdit] = []
    previous_expected_end = 0
    previous_recognized_end = 0
    for application in ordered:
        if (
            application.expected_start < previous_expected_end
            or application.recognized_start < previous_recognized_end
        ):
            return ()
        accepted.append(application)
        previous_expected_end = application.expected_end
        previous_recognized_end = application.recognized_end
    return tuple(accepted)


def _unalias_edits(
    expected: tuple[str, ...],
    recognized: tuple[str, ...],
    expected_bounds: tuple[int, int],
    recognized_bounds: tuple[int, int],
) -> tuple[SequenceEdit, ...]:
    expected_start, expected_end = expected_bounds
    recognized_start, recognized_end = recognized_bounds
    matcher = SequenceMatcher(
        a=expected[expected_start:expected_end],
        b=recognized[recognized_start:recognized_end],
        autojunk=False,
    )
    return tuple(
        SequenceEdit(
            operation,
            expected_start + start_a,
            expected_start + end_a,
            recognized_start + start_b,
            recognized_start + end_b,
        )
        for operation, start_a, end_a, start_b, end_b in matcher.get_opcodes()
    )


def _positions(tokens: tuple[str, ...], needle: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(
        index
        for index in range(len(tokens) - len(needle) + 1)
        if tokens[index : index + len(needle)] == needle
    )


def valid_reason_code(value: str) -> bool:
    """Return whether a configured equivalence reason is machine-safe."""
    return re.fullmatch(r"[a-z][a-z0-9_]{1,63}", value) is not None


__all__ = [
    "MAXIMUM_EQUIVALENCE_RULES",
    "MAXIMUM_EQUIVALENCE_TOKENS",
    "NORMALIZATION_VERSION",
    "DirectionalEquivalence",
    "EquivalenceSet",
    "NormalizationTrace",
    "NormalizedToken",
    "SequenceEdit",
    "align_token_sequences",
    "normalize_english",
    "valid_reason_code",
]
