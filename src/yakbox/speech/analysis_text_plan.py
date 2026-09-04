"""Build immutable spoken-text authority from explicit, traceable rewrites."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from yakbox.errors import ValidationError
from yakbox.speech.analysis_fingerprints import (
    semantic_fingerprint,
    text_fingerprint,
)
from yakbox.speech.analysis_models import (
    SourceTextSpan,
    SpokenTextPlan,
    SpokenTextSegment,
    TextTransform,
)
from yakbox.speech.normalization import NORMALIZATION_VERSION, normalize_english

_SHA256_LENGTH = 64


class TextTransformKind(StrEnum):
    """Closed transform vocabulary for the initial English text plan."""

    SYNTHESIS_IDENTITY = "synthesis:identity"
    SYNTHESIS_MARKDOWN_REMOVAL = "synthesis:markdown_removal"
    SYNTHESIS_DIALOGUE_QUOTE_REMOVAL = "synthesis:dialogue_quote_removal"
    SYNTHESIS_ATTRIBUTION_REMOVAL = "synthesis:attribution_removal"
    SYNTHESIS_PRONUNCIATION_HINT = "synthesis:pronunciation_hint"
    SYNTHESIS_NUMBER_EXPANSION = "synthesis:number_expansion"
    SYNTHESIS_BACKEND_PREPARATION = "synthesis:backend_preparation"
    LEXICAL_IDENTITY = "lexical:identity"
    LEXICAL_MARKDOWN_REMOVAL = "lexical:markdown_removal"
    LEXICAL_DIALOGUE_QUOTE_REMOVAL = "lexical:dialogue_quote_removal"
    LEXICAL_ATTRIBUTION_REMOVAL = "lexical:attribution_removal"
    LEXICAL_PRONUNCIATION = "lexical:pronunciation"
    LEXICAL_NUMBER_EXPANSION = "lexical:number_expansion"

    @property
    def stream(self) -> str:
        """Return the text stream on which this transform is valid."""
        return self.value.partition(":")[0]


@dataclass(frozen=True, slots=True)
class TextRewrite:
    """Replace one half-open range in a transform stage."""

    input_start: int
    input_end: int
    replacement: str

    def __post_init__(self) -> None:
        if self.input_start < 0 or self.input_end < self.input_start:
            raise ValidationError("Text rewrite bounds are invalid")
        _validate_text(self.replacement, label="Text rewrite replacement")


@dataclass(frozen=True, slots=True)
class TextTransformStage:
    """One versioned rewrite stage bound to its exact input text."""

    kind: TextTransformKind
    version: int
    input_fingerprint: str
    rule_fingerprint: str
    rewrites: tuple[TextRewrite, ...]

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValidationError("Text transform stage version must be positive")
        _require_sha256(self.input_fingerprint, "transform input fingerprint")
        _require_sha256(self.rule_fingerprint, "transform rule fingerprint")
        previous_end = 0
        for index, rewrite in enumerate(self.rewrites):
            if index and rewrite.input_start < previous_end:
                raise ValidationError(
                    "Text transform rewrites overlap or are unordered"
                )
            previous_end = rewrite.input_end


@dataclass(frozen=True, slots=True)
class SpokenTextSegmentDraft:
    """Private text and public identities used to build one plan segment."""

    segment_id: str
    source: SourceTextSpan
    display_text: str
    speaker: str
    profile: str
    boundary: str
    synthesis_stages: tuple[TextTransformStage, ...]
    lexical_stages: tuple[TextTransformStage, ...]
    expected_phonemes: tuple[str, ...] = ()
    language: str = "en"

    def __post_init__(self) -> None:
        if (
            not self.segment_id
            or not self.speaker
            or not self.profile
            or not self.boundary
        ):
            raise ValidationError("Spoken text draft identity is incomplete")
        if self.language != "en":
            raise ValidationError("The initial spoken-text plan supports en only")
        _validate_text(self.display_text, label="Display text")
        if not self.display_text:
            raise ValidationError("Spoken text draft display text cannot be empty")
        _require_stage_stream(self.synthesis_stages, "synthesis")
        _require_stage_stream(self.lexical_stages, "lexical")


@dataclass(frozen=True, slots=True)
class ResolvedText:
    """One final private text plus its public transform records."""

    text: str
    transforms: tuple[TextTransform, ...]


def transform_rule_fingerprint(
    kind: TextTransformKind,
    *,
    version: int,
    rule_identity: str,
) -> str:
    """Fingerprint a configuration rule without including manuscript text."""
    if version < 1 or not rule_identity.strip():
        raise ValidationError("Text transform rule identity is invalid")
    return semantic_fingerprint(
        "spoken-text-transform-rule-v1",
        {
            "kind": kind.value,
            "version": version,
            "rule_identity": rule_identity.strip(),
        },
    )


def resolve_text_stages(
    text: str,
    stages: tuple[TextTransformStage, ...],
    *,
    stream: str,
) -> ResolvedText:
    """Apply one stream's exact rewrite stages and retain every span mapping."""
    _validate_text(text, label="Transform input")
    _require_stage_stream(stages, stream)
    current = text
    records: list[TextTransform] = []
    for stage in stages:
        if text_fingerprint(current) != stage.input_fingerprint:
            raise ValidationError("Text transform stage input is stale")
        current, stage_records = _apply_stage(current, stage)
        records.extend(stage_records)
    if not current.strip():
        raise ValidationError("Text transform stream cannot resolve to empty speech")
    return ResolvedText(current, tuple(records))


def build_spoken_text_plan(
    *,
    source_digest: str,
    segments: tuple[SpokenTextSegmentDraft, ...],
) -> SpokenTextPlan:
    """Build a deterministic English plan without retaining full text in reports."""
    _require_sha256(source_digest, "spoken text source digest")
    if not segments:
        raise ValidationError("Spoken text plan requires at least one segment")
    planned: list[SpokenTextSegment] = []
    normalization_rules: list[tuple[str, int, str]] = []
    for draft in segments:
        synthesis = resolve_text_stages(
            draft.display_text,
            draft.synthesis_stages,
            stream="synthesis",
        )
        lexical = resolve_text_stages(
            draft.display_text,
            draft.lexical_stages,
            stream="lexical",
        )
        trace = normalize_english(lexical.text)
        tokens = tuple(item.text for item in trace.tokens)
        if not tokens:
            raise ValidationError("Expected lexical text requires spoken tokens")
        stages = (*draft.synthesis_stages, *draft.lexical_stages)
        normalization_rules.extend(
            (stage.kind.value, stage.version, stage.rule_fingerprint)
            for stage in stages
        )
        planned.append(
            SpokenTextSegment(
                segment_id=draft.segment_id,
                source=draft.source,
                display_text_hash=text_fingerprint(draft.display_text),
                synthesis_text_hash=text_fingerprint(synthesis.text),
                expected_lexical_tokens=tokens,
                expected_phonemes=draft.expected_phonemes,
                speaker=draft.speaker,
                profile=draft.profile,
                language=draft.language,
                boundary=draft.boundary,
                transforms=(*synthesis.transforms, *lexical.transforms),
            )
        )
    normalization_fingerprint = semantic_fingerprint(
        "spoken-text-normalization-policy-v1",
        {
            "normalization_version": NORMALIZATION_VERSION,
            "language": "en",
            "rules": tuple(normalization_rules),
        },
    )
    return SpokenTextPlan(1, source_digest, normalization_fingerprint, tuple(planned))


def _apply_stage(
    text: str,
    stage: TextTransformStage,
) -> tuple[str, tuple[TextTransform, ...]]:
    if not stage.rewrites:
        return text, (
            TextTransform(
                stage.kind.value,
                stage.version,
                0,
                len(text),
                0,
                len(text),
                stage.rule_fingerprint,
            ),
        )
    output: list[str] = []
    records: list[TextTransform] = []
    input_cursor = 0
    output_cursor = 0
    for rewrite in stage.rewrites:
        if rewrite.input_end > len(text):
            raise ValidationError("Text transform rewrite exceeds its stage input")
        unchanged = text[input_cursor : rewrite.input_start]
        output.append(unchanged)
        output_cursor += len(unchanged)
        output_start = output_cursor
        output.append(rewrite.replacement)
        output_cursor += len(rewrite.replacement)
        records.append(
            TextTransform(
                stage.kind.value,
                stage.version,
                rewrite.input_start,
                rewrite.input_end,
                output_start,
                output_cursor,
                stage.rule_fingerprint,
            )
        )
        input_cursor = rewrite.input_end
    output.append(text[input_cursor:])
    return "".join(output), tuple(records)


def _require_stage_stream(
    stages: tuple[TextTransformStage, ...],
    stream: str,
) -> None:
    if not stages or any(stage.kind.stream != stream for stage in stages):
        raise ValidationError(
            f"Spoken text requires explicit {stream} transform stages"
        )


def _validate_text(value: str, *, label: str) -> None:
    if "\x00" in value:
        raise ValidationError(f"{label} contains a null character")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValidationError(f"{label} contains invalid Unicode") from error


def _require_sha256(value: str, label: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValidationError(f"{label.capitalize()} must be a SHA-256")


__all__ = [
    "ResolvedText",
    "SpokenTextSegmentDraft",
    "TextRewrite",
    "TextTransformKind",
    "TextTransformStage",
    "build_spoken_text_plan",
    "resolve_text_stages",
    "transform_rule_fingerprint",
]
