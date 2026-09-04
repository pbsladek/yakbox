from __future__ import annotations

from dataclasses import replace

import pytest

from yakbox.errors import ValidationError
from yakbox.speech.analysis_fingerprints import text_fingerprint
from yakbox.speech.analysis_models import SourceTextSpan
from yakbox.speech.analysis_text_plan import (
    SpokenTextSegmentDraft,
    TextRewrite,
    TextTransformKind,
    TextTransformStage,
    build_spoken_text_plan,
    resolve_text_stages,
    transform_rule_fingerprint,
)

SHA_A = "a" * 64


def _stage(
    kind: TextTransformKind,
    input_text: str,
    rewrites: tuple[TextRewrite, ...],
) -> TextTransformStage:
    return TextTransformStage(
        kind,
        1,
        text_fingerprint(input_text),
        transform_rule_fingerprint(
            kind,
            version=1,
            rule_identity=f"test-{kind.value}",
        ),
        rewrites,
    )


def _rewrite_all(text: str, old: str, new: str) -> tuple[TextRewrite, ...]:
    values: list[TextRewrite] = []
    cursor = 0
    while (index := text.find(old, cursor)) >= 0:
        values.append(TextRewrite(index, index + len(old), new))
        cursor = index + len(old)
    assert values
    return tuple(values)


def _transform_stages() -> tuple[
    tuple[TextTransformStage, ...],
    tuple[TextTransformStage, ...],
    str,
    str,
]:
    display = "**“Asterion 12,”** Wren said."

    synthesis_stages: list[TextTransformStage] = []
    current = display
    stage = _stage(
        TextTransformKind.SYNTHESIS_MARKDOWN_REMOVAL,
        current,
        _rewrite_all(current, "**", ""),
    )
    synthesis_stages.append(stage)
    current = resolve_text_stages(current, (stage,), stream="synthesis").text
    stage = _stage(
        TextTransformKind.SYNTHESIS_DIALOGUE_QUOTE_REMOVAL,
        current,
        (
            TextRewrite(0, 1, ""),
            TextRewrite(current.index("”"), current.index("”") + 1, ""),
        ),
    )
    synthesis_stages.append(stage)
    current = resolve_text_stages(current, (stage,), stream="synthesis").text
    stage = _stage(
        TextTransformKind.SYNTHESIS_ATTRIBUTION_REMOVAL,
        current,
        _rewrite_all(current, " Wren said.", ""),
    )
    synthesis_stages.append(stage)
    current = resolve_text_stages(current, (stage,), stream="synthesis").text
    stage = _stage(
        TextTransformKind.SYNTHESIS_PRONUNCIATION_HINT,
        current,
        _rewrite_all(current, "Asterion", "As-tear-ee-on"),
    )
    synthesis_stages.append(stage)
    current = resolve_text_stages(current, (stage,), stream="synthesis").text
    stage = _stage(
        TextTransformKind.SYNTHESIS_NUMBER_EXPANSION,
        current,
        _rewrite_all(current, "12", "twelve"),
    )
    synthesis_stages.append(stage)
    current = resolve_text_stages(current, (stage,), stream="synthesis").text
    stage = _stage(
        TextTransformKind.SYNTHESIS_BACKEND_PREPARATION,
        current,
        _rewrite_all(current, ",", ""),
    )
    synthesis_stages.append(stage)
    synthesis_text = resolve_text_stages(current, (stage,), stream="synthesis").text

    lexical_stages: list[TextTransformStage] = []
    current = display
    for kind, old, new in (
        (TextTransformKind.LEXICAL_MARKDOWN_REMOVAL, "**", ""),
        (TextTransformKind.LEXICAL_DIALOGUE_QUOTE_REMOVAL, "“", ""),
        (TextTransformKind.LEXICAL_DIALOGUE_QUOTE_REMOVAL, "”", ""),
        (TextTransformKind.LEXICAL_ATTRIBUTION_REMOVAL, " Wren said.", ""),
    ):
        stage = _stage(kind, current, _rewrite_all(current, old, new))
        lexical_stages.append(stage)
        current = resolve_text_stages(current, (stage,), stream="lexical").text
    stage = _stage(TextTransformKind.LEXICAL_PRONUNCIATION, current, ())
    lexical_stages.append(stage)
    current = resolve_text_stages(current, (stage,), stream="lexical").text
    stage = _stage(
        TextTransformKind.LEXICAL_NUMBER_EXPANSION,
        current,
        _rewrite_all(current, "12", "twelve"),
    )
    lexical_stages.append(stage)
    lexical_text = resolve_text_stages(current, (stage,), stream="lexical").text
    return tuple(synthesis_stages), tuple(lexical_stages), synthesis_text, lexical_text


def test_plan_maps_every_text_stage_without_accepting_synthesis_respelling() -> None:
    synthesis, lexical, synthesis_text, lexical_text = _transform_stages()
    draft = SpokenTextSegmentDraft(
        segment_id="segment-1",
        source=SourceTextSpan(SHA_A, 4, 0, 4, 32),
        display_text="**“Asterion 12,”** Wren said.",
        speaker="character-1",
        profile="female-1",
        boundary="sentence",
        synthesis_stages=synthesis,
        lexical_stages=lexical,
        expected_phonemes=("æ", "s", "t", "ɪə", "r", "i", "ə", "n"),
    )

    plan = build_spoken_text_plan(source_digest=SHA_A, segments=(draft,))

    segment = plan.segments[0]
    assert synthesis_text == "As-tear-ee-on twelve"
    assert lexical_text == "Asterion twelve,"
    assert segment.synthesis_text_hash == text_fingerprint(synthesis_text)
    assert segment.expected_lexical_tokens == ("asterion", "twelve")
    assert "as-tear-ee-on" not in segment.expected_lexical_tokens
    assert segment.source == SourceTextSpan(SHA_A, 4, 0, 4, 32)
    kinds = {item.kind for item in segment.transforms}
    assert kinds == {item.value for item in TextTransformKind} - {
        TextTransformKind.SYNTHESIS_IDENTITY.value,
        TextTransformKind.LEXICAL_IDENTITY.value,
    }
    deleted = [
        item
        for item in segment.transforms
        if item.kind.endswith(("quote_removal", "attribution_removal"))
    ]
    assert deleted
    assert all(item.output_start == item.output_end for item in deleted)


def test_transform_stages_reject_stale_overlapping_and_wrong_stream_edits() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        _stage(
            TextTransformKind.SYNTHESIS_BACKEND_PREPARATION,
            "hello",
            (TextRewrite(0, 2, "a"), TextRewrite(1, 3, "b")),
        )

    valid = _stage(
        TextTransformKind.SYNTHESIS_BACKEND_PREPARATION,
        "hello",
        (TextRewrite(0, 1, "H"),),
    )
    with pytest.raises(ValidationError, match="stale"):
        resolve_text_stages("different", (valid,), stream="synthesis")
    with pytest.raises(ValidationError, match="explicit lexical"):
        resolve_text_stages("hello", (valid,), stream="lexical")


def test_plan_fingerprint_changes_for_source_mapping_rule_or_expected_words() -> None:
    synthesis, lexical, _synthesis_text, _lexical_text = _transform_stages()
    draft = SpokenTextSegmentDraft(
        segment_id="segment-1",
        source=SourceTextSpan(SHA_A, 1, 0, 1, 32),
        display_text="**“Asterion 12,”** Wren said.",
        speaker="character-1",
        profile="female-1",
        boundary="sentence",
        synthesis_stages=synthesis,
        lexical_stages=lexical,
    )
    original = build_spoken_text_plan(source_digest=SHA_A, segments=(draft,))
    moved = build_spoken_text_plan(
        source_digest=SHA_A,
        segments=(
            replace(draft, source=replace(draft.source, start_line=2, end_line=2)),
        ),
    )
    changed_rule = replace(
        lexical[-1],
        rule_fingerprint=transform_rule_fingerprint(
            lexical[-1].kind,
            version=1,
            rule_identity="different-number-rule",
        ),
    )
    repointed = build_spoken_text_plan(
        source_digest=SHA_A,
        segments=(replace(draft, lexical_stages=(*lexical[:-1], changed_rule)),),
    )
    changed_words = replace(
        lexical[-1],
        rewrites=(replace(lexical[-1].rewrites[0], replacement="twelve one"),),
    )
    lexical_change = build_spoken_text_plan(
        source_digest=SHA_A,
        segments=(replace(draft, lexical_stages=(*lexical[:-1], changed_words)),),
    )

    assert (
        len(
            {
                original.fingerprint,
                moved.fingerprint,
                repointed.fingerprint,
                lexical_change.fingerprint,
            }
        )
        == 4
    )
