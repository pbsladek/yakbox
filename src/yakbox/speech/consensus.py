"""Deterministic multi-recognizer consensus and escalation planning."""

from __future__ import annotations

from dataclasses import dataclass

from yakbox.errors import SpeechAnalysisError, ValidationError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint, text_fingerprint
from yakbox.speech.analysis_models import (
    AppliedLexicalEquivalence,
    ClipClass,
    ConsensusOutcome,
    ConsensusResult,
    LexicalSpan,
    RecognitionResult,
    TokenVote,
    VoteState,
)
from yakbox.speech.analysis_policy import (
    CalibrationTable,
    SpeechAnalysisPolicy,
    recognition_quality_issues,
)
from yakbox.speech.normalization import (
    EquivalenceSet,
    SequenceEdit,
    align_token_sequences,
)

_REQUIRED_MATCH_COUNT = 2


@dataclass(frozen=True, slots=True)
class EscalationPlan:
    """Bounded Qwen request required before consensus can become terminal."""

    engine: str
    spans: tuple[LexicalSpan, ...]
    reason_codes: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-escalation-plan-v1", self)


@dataclass(frozen=True, slots=True)
class ConsensusEvaluation:
    """Either a terminal result or an exact request for escalation evidence."""

    result: ConsensusResult | None
    escalation: EscalationPlan | None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.escalation is None):
            raise ValidationError(
                "Consensus evaluation requires exactly one result or escalation"
            )


@dataclass(frozen=True, slots=True)
class _EngineVote:
    engine: str
    fingerprint: str
    state: VoteState
    token_states: tuple[VoteState, ...]
    recognized_token_hashes: tuple[str | None, ...]
    reason_codes: tuple[str, ...]
    applied_equivalences: tuple[AppliedLexicalEquivalence, ...]


def evaluate_consensus(
    *,
    expected_tokens: tuple[str, ...],
    recognitions: tuple[RecognitionResult, ...],
    clip_class: ClipClass,
    policy: SpeechAnalysisPolicy,
    calibration: CalibrationTable,
    equivalences: EquivalenceSet,
    high_risk_spans: tuple[LexicalSpan, ...] = (),
    repair: bool = False,
) -> ConsensusEvaluation:
    """Apply quality gates, plan escalation, and evaluate the strict truth table."""
    if not expected_tokens or any(
        not token or token != token.casefold() for token in expected_tokens
    ):
        raise ValidationError("Consensus expected tokens must be normalized")
    if not all(isinstance(item, RecognitionResult) for item in recognitions):
        raise TypeError("Consensus accepts RecognitionResult evidence only")
    ordered = tuple(
        sorted(recognitions, key=lambda item: (item.engine, item.fingerprint))
    )
    if len({item.engine for item in ordered}) != len(ordered):
        raise ValidationError("Consensus received duplicate engine evidence")
    votes = tuple(
        _engine_vote(
            result,
            expected_tokens=expected_tokens,
            clip_class=clip_class,
            calibration=calibration,
            equivalences=equivalences,
        )
        for result in ordered
    )
    baseline = tuple(
        vote
        for name in policy.baseline_recognizers
        for vote in votes
        if vote.engine == name
    )
    if len(baseline) != len(policy.baseline_recognizers):
        raise SpeechAnalysisError("Consensus is missing baseline recognizer evidence")
    escalation_required, escalation_reasons = _requires_escalation(
        baseline,
        clip_class=clip_class,
        policy=policy,
        high_risk=bool(high_risk_spans),
        repair=repair,
    )
    escalation_vote = next(
        (vote for vote in votes if vote.engine == policy.escalation_recognizer),
        None,
    )
    disagreement_spans = _disagreement_spans(votes)
    if escalation_required and escalation_vote is None:
        spans = _merge_spans((*disagreement_spans, *high_risk_spans))
        if not spans:
            spans = (LexicalSpan(0, len(expected_tokens)),)
        return ConsensusEvaluation(
            result=None,
            escalation=EscalationPlan(
                policy.escalation_recognizer,
                spans,
                escalation_reasons,
            ),
        )
    if escalation_required:
        if escalation_vote is None:
            raise AssertionError("Escalation vote must exist after escalation planning")
        considered = (*baseline, escalation_vote)
    else:
        considered = baseline
    outcome, reasons = _terminal_outcome(
        considered,
        escalation_required=escalation_required,
    )
    token_votes = tuple(
        TokenVote(
            expected_index=index,
            engine=vote.engine,
            recognition_fingerprint=vote.fingerprint,
            state=state,
            recognized_token_hash=vote.recognized_token_hashes[index],
            reason_codes=_vote_reason_codes(state),
        )
        for vote in considered
        for index, state in enumerate(vote.token_states)
    )
    rejected = _rejected_spans(considered)
    accepted = () if rejected else (LexicalSpan(0, len(expected_tokens)),)
    return ConsensusEvaluation(
        result=ConsensusResult(
            policy_fingerprint=policy.fingerprint,
            expected_tokens_hash=text_fingerprint("\u001f".join(expected_tokens)),
            equivalence_set_fingerprint=equivalences.fingerprint,
            recognition_fingerprints=tuple(
                sorted(vote.fingerprint for vote in considered)
            ),
            applied_equivalences=tuple(
                sorted(
                    (
                        application
                        for vote in considered
                        for application in vote.applied_equivalences
                    ),
                    key=lambda item: (
                        item.engine,
                        item.expected_start,
                        item.recognized_start,
                        item.rule_fingerprint,
                    ),
                )
            ),
            votes=tuple(
                sorted(
                    token_votes,
                    key=lambda item: (item.expected_index, item.engine),
                )
            ),
            accepted_spans=accepted,
            rejected_spans=rejected,
            disagreement_spans=disagreement_spans,
            high_risk_spans=_merge_spans(high_risk_spans),
            escalation_reason=(
                ",".join(escalation_reasons) if escalation_required else "not_required"
            ),
            outcome=outcome,
            reason_codes=reasons,
        ),
        escalation=None,
    )


def _engine_vote(
    result: RecognitionResult,
    *,
    expected_tokens: tuple[str, ...],
    clip_class: ClipClass,
    calibration: CalibrationTable,
    equivalences: EquivalenceSet,
) -> _EngineVote:
    threshold = calibration.threshold(result.engine, clip_class)
    quality = recognition_quality_issues(result, threshold)
    calibration_mismatch = (
        result.execution.fingerprint not in calibration.execution_class_fingerprints
        or result.score_calibration_fingerprint
        != threshold.score_calibration_fingerprint
    )
    if calibration_mismatch:
        quality = (*quality, "engine_decode_invalid")
    if quality:
        return _EngineVote(
            result.engine,
            result.fingerprint,
            VoteState.INVALID,
            (VoteState.INVALID,) * len(expected_tokens),
            (None,) * len(expected_tokens),
            tuple(dict.fromkeys(quality)),
            (),
        )
    recognized = tuple(token.text for token in result.tokens)
    edits = align_token_sequences(expected_tokens, recognized, equivalences)
    applied_equivalences = _applied_equivalences(
        result, expected_tokens, recognized, edits
    )
    substantive = tuple(
        edit for edit in edits if edit.operation not in {"equal", "equivalent"}
    )
    if not substantive:
        return _EngineVote(
            result.engine,
            result.fingerprint,
            VoteState.MATCH,
            (VoteState.MATCH,) * len(expected_tokens),
            _recognized_hashes(expected_tokens, recognized, edits),
            (),
            applied_equivalences,
        )
    token_states = [VoteState.MATCH] * len(expected_tokens)
    token_hashes = list(_recognized_hashes(expected_tokens, recognized, edits))
    reason_codes: list[str] = []
    for edit in substantive:
        reason = _edit_reason(edit.operation)
        reason_codes.append(reason)
        start = min(edit.expected_start, len(expected_tokens) - 1)
        end = max(start + 1, edit.expected_end)
        for index in range(start, min(end, len(expected_tokens))):
            token_states[index] = VoteState.DISSENT
    return _EngineVote(
        result.engine,
        result.fingerprint,
        VoteState.DISSENT,
        tuple(token_states),
        tuple(token_hashes),
        tuple(dict.fromkeys(reason_codes)),
        applied_equivalences,
    )


def _applied_equivalences(
    result: RecognitionResult,
    expected: tuple[str, ...],
    recognized: tuple[str, ...],
    edits: tuple[SequenceEdit, ...],
) -> tuple[AppliedLexicalEquivalence, ...]:
    return tuple(
        AppliedLexicalEquivalence(
            engine=result.engine,
            recognition_fingerprint=result.fingerprint,
            rule_fingerprint=semantic_fingerprint(
                "directional-equivalence-v1",
                {
                    "expected": expected[edit.expected_start : edit.expected_end],
                    "recognized": recognized[
                        edit.recognized_start : edit.recognized_end
                    ],
                    "reason_code": edit.equivalence_reason,
                },
            ),
            reason_code=edit.equivalence_reason or "invalid_equivalence",
            expected_start=edit.expected_start,
            expected_end=edit.expected_end,
            recognized_start=edit.recognized_start,
            recognized_end=edit.recognized_end,
            recognized_sequence_hash=text_fingerprint(
                "\u001f".join(recognized[edit.recognized_start : edit.recognized_end])
            ),
        )
        for edit in edits
        if edit.operation == "equivalent"
    )


def _recognized_hashes(
    expected: tuple[str, ...],
    recognized: tuple[str, ...],
    edits: tuple[SequenceEdit, ...],
) -> tuple[str | None, ...]:
    """Project recognized lexical evidence onto expected token positions."""
    projected: list[str | None] = [None] * len(expected)
    for edit in edits:
        if edit.operation not in {"equal", "replace", "equivalent"}:
            continue
        recognized_slice = recognized[edit.recognized_start : edit.recognized_end]
        expected_count = edit.expected_end - edit.expected_start
        if expected_count == len(recognized_slice):
            for offset, token in enumerate(recognized_slice):
                projected[edit.expected_start + offset] = text_fingerprint(token)
        elif recognized_slice and expected_count:
            digest = text_fingerprint("\u001f".join(recognized_slice))
            for index in range(edit.expected_start, edit.expected_end):
                projected[index] = digest
    return tuple(projected)


def _requires_escalation(
    baseline: tuple[_EngineVote, ...],
    *,
    clip_class: ClipClass,
    policy: SpeechAnalysisPolicy,
    high_risk: bool,
    repair: bool,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if any(vote.state is not VoteState.MATCH for vote in baseline):
        reasons.append("baseline_disagreement")
    if clip_class in policy.always_escalate_clip_classes:
        reasons.append("high_risk_clip_class")
    if high_risk:
        reasons.append("high_risk_span")
    if repair and policy.always_escalate_repairs:
        reasons.append("repair")
    return bool(reasons), tuple(reasons)


def _terminal_outcome(
    votes: tuple[_EngineVote, ...],
    *,
    escalation_required: bool,
) -> tuple[ConsensusOutcome, tuple[str, ...]]:
    valid_dissent = any(vote.state is VoteState.DISSENT for vote in votes)
    matches = sum(vote.state is VoteState.MATCH for vote in votes)
    invalid = sum(vote.state is VoteState.INVALID for vote in votes)
    if valid_dissent:
        return ConsensusOutcome.REJECTED, ("persistent_valid_dissent",)
    if matches >= _REQUIRED_MATCH_COUNT and invalid <= 1:
        return ConsensusOutcome.ACCEPTED, ()
    if escalation_required and matches < _REQUIRED_MATCH_COUNT:
        return ConsensusOutcome.REJECTED, ("missing_required_engine",)
    return ConsensusOutcome.REJECTED, ("engine_result_missing",)


def _disagreement_spans(votes: tuple[_EngineVote, ...]) -> tuple[LexicalSpan, ...]:
    if not votes:
        return ()
    count = len(votes[0].token_states)
    disputed = tuple(
        index
        for index in range(count)
        if len({vote.token_states[index] for vote in votes}) > 1
        or any(vote.token_states[index] is VoteState.DISSENT for vote in votes)
    )
    return _indices_to_spans(disputed, "persistent_valid_dissent")


def _rejected_spans(votes: tuple[_EngineVote, ...]) -> tuple[LexicalSpan, ...]:
    if not votes:
        return ()
    rejected = tuple(
        (index, reason)
        for index in range(len(votes[0].token_states))
        if (
            reason := _token_rejection_reason(
                tuple(vote.token_states[index] for vote in votes)
            )
        )
        is not None
    )
    if not rejected:
        return ()
    spans: list[LexicalSpan] = []
    start = previous = rejected[0][0]
    reason = rejected[0][1]
    for index, current_reason in rejected[1:]:
        if index != previous + 1 or current_reason != reason:
            spans.append(LexicalSpan(start, previous + 1, (reason,), True))
            start = index
            reason = current_reason
        previous = index
    spans.append(LexicalSpan(start, previous + 1, (reason,), True))
    return tuple(spans)


def _token_rejection_reason(states: tuple[VoteState, ...]) -> str | None:
    if any(state is VoteState.DISSENT for state in states):
        return "persistent_valid_dissent"
    if sum(state is VoteState.MATCH for state in states) < _REQUIRED_MATCH_COUNT:
        return "engine_decode_invalid"
    return None


def _indices_to_spans(indices: tuple[int, ...], reason: str) -> tuple[LexicalSpan, ...]:
    if not indices:
        return ()
    spans: list[LexicalSpan] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            spans.append(LexicalSpan(start, previous + 1, (reason,), True))
            start = index
        previous = index
    spans.append(LexicalSpan(start, previous + 1, (reason,), True))
    return tuple(spans)


def _merge_spans(spans: tuple[LexicalSpan, ...]) -> tuple[LexicalSpan, ...]:
    if not spans:
        return ()
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    merged: list[LexicalSpan] = [ordered[0]]
    for span in ordered[1:]:
        previous = merged[-1]
        if span.start <= previous.end:
            merged[-1] = LexicalSpan(
                previous.start,
                max(previous.end, span.end),
                tuple(dict.fromkeys((*previous.reason_codes, *span.reason_codes))),
                previous.review_eligible or span.review_eligible,
            )
        else:
            merged.append(span)
    return tuple(merged)


def _edit_reason(operation: str) -> str:
    return {
        "delete": "lexical_deletion",
        "insert": "unexpected_speech",
        "replace": "lexical_substitution",
    }.get(operation, "lexical_substitution")


def _vote_reason_codes(state: VoteState) -> tuple[str, ...]:
    if state is VoteState.MATCH:
        return ("recognition_match",)
    if state is VoteState.INVALID:
        return ("engine_decode_invalid",)
    if state is VoteState.NOT_RUN:
        return ("recognition_not_run",)
    return ("persistent_valid_dissent",)


__all__ = [
    "ConsensusEvaluation",
    "EscalationPlan",
    "evaluate_consensus",
]
