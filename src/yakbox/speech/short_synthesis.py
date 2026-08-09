"""Candidate orchestration for verified short-utterance synthesis."""

from __future__ import annotations

import hashlib
import json
import tempfile
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median

from yakbox._files import atomic_write_bytes, atomic_write_json, sha256_file
from yakbox.audio.crop import (
    CropEvidence,
    SignalQualityEvidence,
    SpeechIslandEvidence,
    SpeechRegion,
    crop_aligned_wav,
    inspect_signal_quality,
    inspect_speech_islands,
    wav_duration_seconds,
)
from yakbox.contracts import runtime_metadata
from yakbox.errors import ArtifactError, BuildError
from yakbox.speech.alignment import (
    INTERNAL_SENTENCE_BOUNDARY_PAUSE_MS,
    AlignmentDecision,
    AlignmentResult,
    SpeechAligner,
    has_internal_sentence_boundary,
    lexical_tokens,
    validate_carrier_alignment,
    validate_extracted_alignment,
)
from yakbox.speech.models import SpeechSynthesisRequest
from yakbox.speech.phonemes import (
    PhonemeAligner,
    PhonemeAlignmentResult,
    final_phoneme_is_consonant,
    validate_phoneme_alignment,
)
from yakbox.speech.services import TextToSpeechService
from yakbox.speech.short_utterances import (
    CarrierPosition,
    CarrierRecipe,
    ShortUtterancePolicy,
)

_TARGET_PASSING_CANDIDATES = 2
_MINIMUM_PHONEME_PATH_CONFIDENCE_FOR_ASR_RESCUE = 0.60
_MINIMUM_ONE_WORD_ASR_RESCUE_CONFIDENCE = 0.50
_MINIMUM_ONE_WORD_BOUNDARY_CONFIDENCE = 0.80
_PHONEME_BOUNDARY_TOLERANCE_MS = 70
_MAXIMUM_PHONEME_WORD_TAIL_MS = 130


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """One generated candidate and its hard-gate outcome."""

    recipe: CarrierRecipe
    accepted: bool
    reason_codes: tuple[str, ...]
    full_audio: Path
    extracted_audio: Path | None
    confidence: float | None
    crop: CropEvidence | None
    initial_extracted_audio: Path | None = None
    acoustic: SpeechIslandEvidence | None = None
    acoustic_refined: bool = False
    acoustic_crop: CropEvidence | None = None
    signal_quality: SignalQualityEvidence | None = None
    phoneme_alignment: PhonemeAlignmentResult | None = None
    phoneme_confidence: float | None = None
    phoneme_path_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ShortUtteranceSelection:
    """Selected crop and privacy-safe evidence for a risky chunk."""

    selected: CandidateEvaluation
    candidates: tuple[CandidateEvaluation, ...]
    report: Path | None
    review_required: bool


async def synthesize_short_utterance(
    *,
    service: TextToSpeechService,
    aligner: SpeechAligner,
    request: SpeechSynthesisRequest,
    destination: Path,
    recipes: tuple[CarrierRecipe, ...],
    policy: ShortUtterancePolicy,
    language: str,
    qa_directory: Path | None = None,
    phoneme_aligner: PhonemeAligner | None = None,
    phoneme_language: str | None = None,
    minimum_phoneme_confidence: float = 0.20,
) -> ShortUtteranceSelection:
    """Generate, align, crop, validate, rank, and commit one short phrase."""
    if not recipes:
        raise BuildError("Short-utterance synthesis received no candidates")
    if qa_directory is not None:
        qa_directory.mkdir(parents=True, exist_ok=True)
        return await _evaluate_and_select(
            service=service,
            aligner=aligner,
            request=request,
            destination=destination,
            recipes=recipes,
            policy=policy,
            language=language,
            working_directory=qa_directory,
            report_directory=qa_directory,
            phoneme_aligner=phoneme_aligner,
            phoneme_language=phoneme_language or language,
            minimum_phoneme_confidence=minimum_phoneme_confidence,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".yakbox-short-utterance-", dir=destination.parent
    ) as temporary:
        return await _evaluate_and_select(
            service=service,
            aligner=aligner,
            request=request,
            destination=destination,
            recipes=recipes,
            policy=policy,
            language=language,
            working_directory=Path(temporary),
            report_directory=None,
            phoneme_aligner=phoneme_aligner,
            phoneme_language=phoneme_language or language,
            minimum_phoneme_confidence=minimum_phoneme_confidence,
        )


async def _evaluate_and_select(
    *,
    service: TextToSpeechService,
    aligner: SpeechAligner,
    request: SpeechSynthesisRequest,
    destination: Path,
    recipes: tuple[CarrierRecipe, ...],
    policy: ShortUtterancePolicy,
    language: str,
    working_directory: Path,
    report_directory: Path | None,
    phoneme_aligner: PhonemeAligner | None,
    phoneme_language: str,
    minimum_phoneme_confidence: float,
) -> ShortUtteranceSelection:
    evaluations: list[CandidateEvaluation] = []
    passing = 0
    for recipe in recipes:
        initial = await _evaluate_candidate(
            service=service,
            aligner=aligner,
            request=request,
            recipe=recipe,
            policy=policy,
            language=language,
            directory=working_directory,
        )
        evaluated = await _apply_phoneme_gate(
            initial,
            aligner=phoneme_aligner,
            target_text=request.text,
            language=phoneme_language,
            minimum_confidence=minimum_phoneme_confidence,
            policy=policy,
        )
        evaluations.append(evaluated)
        if evaluated.accepted:
            passing += 1
            if passing >= _TARGET_PASSING_CANDIDATES:
                break
    accepted = tuple(item for item in evaluations if item.accepted)
    report = _write_report(
        report_directory,
        request=request,
        policy=policy,
        aligner=aligner,
        evaluations=tuple(evaluations),
        selected=_rank_candidates(accepted, policy=policy) if accepted else None,
    )
    if not accepted:
        reason_codes = sorted(
            {reason for item in evaluations for reason in item.reason_codes}
        )
        suffix = f"; QA: {report}" if report is not None else ""
        raise BuildError(
            "No short-utterance candidate passed hard gates "
            f"({', '.join(reason_codes)}){suffix}"
        )
    selected = _rank_candidates(accepted, policy=policy)
    if selected.extracted_audio is None:
        raise BuildError("Selected short-utterance candidate has no extracted audio")
    review_required = (
        policy.require_review_for_one_word and len(lexical_tokens(request.text)) == 1
    )
    reviewed = review_required and _review_is_approved(report, selected)
    if review_required and not reviewed:
        suffix = f"; QA: {report}" if report is not None else ""
        raise BuildError(f"One-word synthesis requires listening review{suffix}")
    atomic_write_bytes(
        destination,
        selected.extracted_audio.read_bytes(),
        overwrite=True,
    )
    return ShortUtteranceSelection(
        selected=selected,
        candidates=tuple(evaluations),
        report=report,
        review_required=False,
    )


async def _apply_phoneme_gate(
    evaluation: CandidateEvaluation,
    *,
    aligner: PhonemeAligner | None,
    target_text: str,
    language: str,
    minimum_confidence: float,
    policy: ShortUtterancePolicy,
) -> CandidateEvaluation:
    """Apply an independent phoneme path gate to a candidate's final crop."""
    if aligner is None or evaluation.extracted_audio is None:
        return evaluation
    try:
        result = await aligner.align(
            evaluation.extracted_audio,
            target_text,
            language=language,
        )
    except BuildError:
        reasons = tuple(
            dict.fromkeys((*evaluation.reason_codes, "phoneme_alignment_failed"))
        )
        return replace(evaluation, accepted=False, reason_codes=reasons)
    decision = validate_phoneme_alignment(
        result,
        minimum_confidence=minimum_confidence,
    )
    boundary_reasons = _reconcile_phoneme_boundaries(
        evaluation,
        result,
        accepted=decision.accepted,
        final_consonant=final_phoneme_is_consonant(result),
        maximum_extra_speech_ms=policy.maximum_extra_speech_ms,
    )
    reasons = tuple(dict.fromkeys((*boundary_reasons, *decision.reason_codes)))
    path_confidence = median(item.confidence for item in result.phonemes)
    one_word_boundary_rescue = (
        len(lexical_tokens(target_text)) == 1
        and evaluation.confidence is not None
        and evaluation.confidence >= _MINIMUM_ONE_WORD_ASR_RESCUE_CONFIDENCE
        and decision.confidence is not None
        and decision.confidence >= _MINIMUM_ONE_WORD_BOUNDARY_CONFIDENCE
    )
    if (
        reasons == ("low_confidence",)
        and decision.accepted
        and (
            path_confidence >= _MINIMUM_PHONEME_PATH_CONFIDENCE_FOR_ASR_RESCUE
            or one_word_boundary_rescue
        )
    ):
        reasons = ()
    return replace(
        evaluation,
        accepted=not reasons and decision.accepted,
        reason_codes=reasons,
        phoneme_alignment=result,
        phoneme_confidence=decision.confidence,
        phoneme_path_confidence=path_confidence,
    )


def _reconcile_phoneme_boundaries(
    evaluation: CandidateEvaluation,
    result: PhonemeAlignmentResult,
    *,
    accepted: bool,
    final_consonant: bool,
    maximum_extra_speech_ms: int,
) -> tuple[str, ...]:
    """Use accepted phoneme edges when Whisper underestimates a final sound."""
    reasons = list(evaluation.reason_codes)
    if not accepted or not result.phonemes or evaluation.acoustic is None:
        return tuple(reasons)
    allowed = (
        min(
            maximum_extra_speech_ms + _PHONEME_BOUNDARY_TOLERANCE_MS,
            _MAXIMUM_PHONEME_WORD_TAIL_MS,
        )
        / 1_000
    )
    start = result.phonemes[0].start_seconds
    end = result.phonemes[-1].end_seconds
    regions = evaluation.acoustic.regions
    if not final_consonant and not evaluation.acoustic.detached_suffix:
        reasons = [reason for reason in reasons if reason != "unexpected_suffix_speech"]
    prefix_speech = _region_duration(regions, 0.0, start)
    tail_end = max((region.end_seconds for region in regions), default=end)
    suffix_speech = _region_duration(regions, end, tail_end)
    if prefix_speech <= allowed:
        reasons = [reason for reason in reasons if reason != "unexpected_prefix_speech"]
    if final_consonant and suffix_speech <= allowed:
        reasons = [reason for reason in reasons if reason != "unexpected_suffix_speech"]
    return tuple(reasons)


def _region_duration(
    regions: tuple[SpeechRegion, ...],
    start_seconds: float,
    end_seconds: float,
) -> float:
    return sum(
        max(
            0.0,
            min(region.end_seconds, end_seconds)
            - max(region.start_seconds, start_seconds),
        )
        for region in regions
    )


async def _evaluate_candidate(
    *,
    service: TextToSpeechService,
    aligner: SpeechAligner,
    request: SpeechSynthesisRequest,
    recipe: CarrierRecipe,
    policy: ShortUtterancePolicy,
    language: str,
    directory: Path,
) -> CandidateEvaluation:
    stem = f"candidate-{recipe.candidate_index:03d}"
    full = directory / f"{stem}-full.wav"
    extracted = directory / f"{stem}-extracted.wav"
    candidate_request = replace(
        request,
        text=recipe.text,
        chatterbox=(
            replace(request.chatterbox, seed=recipe.seed)
            if request.chatterbox is not None
            else None
        ),
    )
    await service.synthesize_to_file(candidate_request, full, overwrite=True)
    alignment = await aligner.align(full, recipe.text, language=language)
    if recipe.position is CarrierPosition.DIRECT:
        return await _evaluate_extracted_audio(
            aligner=aligner,
            recipe=recipe,
            full_audio=full,
            initial_audio=full,
            initial_alignment=alignment,
            crop=None,
            target_text=request.text,
            policy=policy,
            language=language,
            directory=directory,
        )
    carrier_decision = validate_carrier_alignment(
        alignment,
        expected_text=recipe.text,
        target_text=request.text,
        minimum_confidence=policy.minimum_confidence_for(
            len(lexical_tokens(request.text)),
            extracted=False,
        ),
        token_aliases=policy.alignment_alias_map,
        minimum_average_log_probability=policy.minimum_segment_average_log_probability,
        maximum_compression_ratio=policy.maximum_segment_compression_ratio,
        maximum_no_speech_probability=policy.maximum_segment_no_speech_probability,
        maximum_temperature=policy.maximum_segment_temperature,
        maximum_internal_token_gap_ms=policy.maximum_internal_token_gap_ms,
        maximum_token_duration_ms=policy.maximum_token_duration_ms,
    )
    if not carrier_decision.accepted:
        return _rejected(recipe, full, carrier_decision)
    if not _has_required_pause(alignment, carrier_decision, recipe.position, policy):
        return _rejected_code(recipe, full, "insufficient_carrier_pause")
    try:
        crop = crop_aligned_wav(
            full,
            extracted,
            start_seconds=_required_time(carrier_decision.start_seconds),
            end_seconds=_required_time(carrier_decision.end_seconds),
            pre_roll_ms=policy.pre_roll_ms,
            post_roll_ms=policy.post_roll_ms,
            fade_ms=policy.fade_ms,
            speech_regions=alignment.speech_regions,
            overwrite=True,
        )
    except ArtifactError:
        return _rejected_code(recipe, full, "unsafe_crop_boundary")
    extracted_alignment = await aligner.align(
        extracted, request.text, language=language
    )
    return await _evaluate_extracted_audio(
        aligner=aligner,
        recipe=recipe,
        full_audio=full,
        initial_audio=extracted,
        initial_alignment=extracted_alignment,
        crop=crop,
        target_text=request.text,
        policy=policy,
        language=language,
        directory=directory,
    )


async def _evaluate_extracted_audio(
    *,
    aligner: SpeechAligner,
    recipe: CarrierRecipe,
    full_audio: Path,
    initial_audio: Path,
    initial_alignment: AlignmentResult,
    crop: CropEvidence | None,
    target_text: str,
    policy: ShortUtterancePolicy,
    language: str,
    directory: Path,
) -> CandidateEvaluation:
    """Validate lexical and independent acoustic evidence, refining when safe."""
    decision = _validate_extracted(initial_alignment, target_text, policy)
    try:
        acoustic = inspect_speech_islands(
            initial_audio,
            threshold_dbfs=policy.acoustic_threshold_dbfs,
            island_gap_ms=_speech_island_gap_ms(target_text, policy),
        )
    except ArtifactError:
        return _acoustic_evaluation(
            recipe=recipe,
            full_audio=full_audio,
            initial_audio=initial_audio,
            final_audio=initial_audio,
            decision=decision,
            crop=crop,
            acoustic=None,
            policy=policy,
            reasons=(*decision.reason_codes, "acoustic_inspection_failed"),
        )
    acoustic_reasons = _acoustic_reason_codes(acoustic, policy)
    can_refine = (
        bool(acoustic_reasons)
        and policy.acoustic_refinement
        and acoustic.primary_island is not None
    )
    if not decision.accepted and not can_refine:
        return _acoustic_evaluation(
            recipe=recipe,
            full_audio=full_audio,
            initial_audio=initial_audio,
            final_audio=initial_audio,
            decision=decision,
            crop=crop,
            acoustic=acoustic,
            policy=policy,
            reasons=(*decision.reason_codes, *acoustic_reasons),
        )

    if can_refine:
        refined = directory / f"candidate-{recipe.candidate_index:03d}-refined.wav"
        primary = acoustic.primary_island
        if primary is None:  # narrowed by can_refine; keeps static analysis explicit
            raise AssertionError("Acoustic refinement requires a primary island")
        try:
            acoustic_crop = crop_aligned_wav(
                initial_audio,
                refined,
                start_seconds=primary.start_seconds,
                end_seconds=primary.end_seconds,
                pre_roll_ms=policy.pre_roll_ms,
                post_roll_ms=policy.post_roll_ms,
                fade_ms=policy.fade_ms,
                speech_regions=acoustic.regions,
                overwrite=True,
            )
            refined_alignment = await aligner.align(
                refined, target_text, language=language
            )
            refined_decision = _validate_extracted(
                refined_alignment, target_text, policy
            )
            refined_acoustic = inspect_speech_islands(
                refined,
                threshold_dbfs=policy.acoustic_threshold_dbfs,
                island_gap_ms=_speech_island_gap_ms(target_text, policy),
            )
        except ArtifactError:
            return _acoustic_evaluation(
                recipe=recipe,
                full_audio=full_audio,
                initial_audio=initial_audio,
                final_audio=initial_audio,
                decision=decision,
                crop=crop,
                acoustic=acoustic,
                policy=policy,
                reasons=("unsafe_acoustic_refinement",),
            )
        refined_reasons = (
            *refined_decision.reason_codes,
            *_acoustic_reason_codes(refined_acoustic, policy),
        )
        return _acoustic_evaluation(
            recipe=recipe,
            full_audio=full_audio,
            initial_audio=initial_audio,
            final_audio=refined,
            decision=refined_decision,
            crop=crop,
            acoustic=refined_acoustic,
            policy=policy,
            reasons=refined_reasons,
            acoustic_refined=True,
            acoustic_crop=acoustic_crop,
        )

    return _acoustic_evaluation(
        recipe=recipe,
        full_audio=full_audio,
        initial_audio=initial_audio,
        final_audio=initial_audio,
        decision=decision,
        crop=crop,
        acoustic=acoustic,
        policy=policy,
        reasons=(*decision.reason_codes, *acoustic_reasons),
    )


def _acoustic_reason_codes(
    evidence: SpeechIslandEvidence,
    policy: ShortUtterancePolicy,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if evidence.primary_island is None:
        reasons.append("speech_island_missing")
    if evidence.detached_prefix:
        reasons.append("detached_prefix_speech")
    if evidence.detached_suffix:
        reasons.append("detached_suffix_speech")
    if evidence.leading_silence_ms < policy.minimum_edge_silence_ms:
        reasons.append("insufficient_leading_silence")
    if evidence.trailing_silence_ms < policy.minimum_edge_silence_ms:
        reasons.append("insufficient_trailing_silence")
    if evidence.leading_silence_ms > policy.maximum_edge_silence_ms:
        reasons.append("excessive_leading_silence")
    if evidence.trailing_silence_ms > policy.maximum_edge_silence_ms:
        reasons.append("excessive_trailing_silence")
    return tuple(reasons)


def _speech_island_gap_ms(text: str, policy: ShortUtterancePolicy) -> int:
    if has_internal_sentence_boundary(text):
        return max(
            policy.speech_island_gap_ms,
            INTERNAL_SENTENCE_BOUNDARY_PAUSE_MS,
        )
    return policy.speech_island_gap_ms


def _acoustic_evaluation(
    *,
    recipe: CarrierRecipe,
    full_audio: Path,
    initial_audio: Path,
    final_audio: Path,
    decision: AlignmentDecision,
    crop: CropEvidence | None,
    acoustic: SpeechIslandEvidence | None,
    policy: ShortUtterancePolicy,
    reasons: tuple[str, ...],
    acoustic_refined: bool = False,
    acoustic_crop: CropEvidence | None = None,
) -> CandidateEvaluation:
    try:
        signal_quality = inspect_signal_quality(final_audio)
        signal_reasons = _signal_quality_reason_codes(signal_quality, policy)
    except ArtifactError:
        signal_quality = None
        signal_reasons = ("signal_quality_inspection_failed",)
    unique_reasons = tuple(dict.fromkeys((*reasons, *signal_reasons)))
    return CandidateEvaluation(
        recipe=recipe,
        accepted=decision.accepted and not unique_reasons,
        reason_codes=unique_reasons,
        full_audio=full_audio,
        extracted_audio=final_audio,
        confidence=decision.confidence,
        crop=crop,
        initial_extracted_audio=initial_audio,
        acoustic=acoustic,
        acoustic_refined=acoustic_refined,
        acoustic_crop=acoustic_crop,
        signal_quality=signal_quality,
    )


def _signal_quality_reason_codes(
    evidence: SignalQualityEvidence,
    policy: ShortUtterancePolicy,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if evidence.clipped_sample_ratio > policy.maximum_clipped_sample_ratio:
        reasons.append("clipping_detected")
    if (
        evidence.leading_boundary_jump_ratio > policy.maximum_boundary_jump_ratio
        or evidence.trailing_boundary_jump_ratio > policy.maximum_boundary_jump_ratio
    ):
        reasons.append("boundary_click_detected")
    if evidence.vad_disagreement_ms > policy.maximum_vad_disagreement_ms:
        reasons.append("vad_disagreement")
    if evidence.longest_stationary_voiced_ms > policy.maximum_stationary_voiced_ms:
        reasons.append("prolonged_stationary_voicing")
    return tuple(reasons)


def _validate_extracted(
    alignment: AlignmentResult,
    target: str,
    policy: ShortUtterancePolicy,
) -> AlignmentDecision:
    words = max(1, len(lexical_tokens(target)))
    return validate_extracted_alignment(
        alignment,
        target_text=target,
        minimum_confidence=policy.minimum_confidence_for(words, extracted=True),
        maximum_extra_speech_ms=policy.maximum_extra_speech_ms,
        minimum_duration_seconds=0.06 * words,
        maximum_duration_seconds=1.6 + 0.75 * (words - 1),
        token_aliases=policy.alignment_alias_map,
        minimum_average_log_probability=policy.minimum_segment_average_log_probability,
        maximum_compression_ratio=policy.maximum_segment_compression_ratio,
        maximum_no_speech_probability=policy.maximum_segment_no_speech_probability,
        maximum_temperature=policy.maximum_segment_temperature,
        maximum_internal_token_gap_ms=policy.maximum_internal_token_gap_ms,
        maximum_token_duration_ms=policy.maximum_token_duration_ms,
    )


def _has_required_pause(
    alignment: AlignmentResult,
    decision: AlignmentDecision,
    position: CarrierPosition,
    policy: ShortUtterancePolicy,
) -> bool:
    start = _required_time(decision.start_seconds)
    end = _required_time(decision.end_seconds)
    before = tuple(token for token in alignment.tokens if token.end_seconds <= start)
    after = tuple(token for token in alignment.tokens if token.start_seconds >= end)
    minimum = policy.minimum_pause_ms / 1_000
    before_ok = not before or start - before[-1].end_seconds >= minimum
    after_ok = not after or after[0].start_seconds - end >= minimum
    if position is CarrierPosition.INITIAL:
        return after_ok
    if position is CarrierPosition.FINAL:
        return before_ok
    return before_ok and after_ok


def _rejected(
    recipe: CarrierRecipe,
    full: Path,
    decision: AlignmentDecision,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        recipe=recipe,
        accepted=False,
        reason_codes=decision.reason_codes,
        full_audio=full,
        extracted_audio=None,
        confidence=decision.confidence,
        crop=None,
    )


def _rejected_code(
    recipe: CarrierRecipe, full: Path, reason: str
) -> CandidateEvaluation:
    return CandidateEvaluation(
        recipe=recipe,
        accepted=False,
        reason_codes=(reason,),
        full_audio=full,
        extracted_audio=None,
        confidence=None,
        crop=None,
    )


def _rank_candidates(
    candidates: tuple[CandidateEvaluation, ...],
    *,
    policy: ShortUtterancePolicy,
) -> CandidateEvaluation:
    best_confidence = max(item.confidence or 0.0 for item in candidates)
    confidence_floor = best_confidence - policy.candidate_confidence_tolerance
    eligible = tuple(
        item for item in candidates if (item.confidence or 0.0) >= confidence_floor
    )
    return min(
        eligible,
        key=lambda item: (
            0 if policy.prefer_natural_context and item.recipe.natural else 1,
            _acoustic_penalty(item),
            -(item.confidence or 0.0),
            item.recipe.candidate_index,
        ),
    )


def _acoustic_penalty(item: CandidateEvaluation) -> float:
    if item.acoustic is None:
        return float("inf")
    return (
        item.acoustic.detached_prefix_ms
        + item.acoustic.detached_suffix_ms
        + item.acoustic.leading_silence_ms
        + item.acoustic.trailing_silence_ms
        + _internal_pause_ms(item.acoustic)
    )


def _internal_pause_ms(evidence: SpeechIslandEvidence) -> float:
    return sum(
        max(0.0, following.start_seconds - previous.end_seconds) * 1_000
        for previous, following in zip(
            evidence.regions, evidence.regions[1:], strict=False
        )
    )


def _write_report(
    directory: Path | None,
    *,
    request: SpeechSynthesisRequest,
    policy: ShortUtterancePolicy,
    aligner: SpeechAligner,
    evaluations: tuple[CandidateEvaluation, ...],
    selected: CandidateEvaluation | None,
) -> Path | None:
    if directory is None:
        return None
    report = directory / "report.json"
    content: dict[str, object] = {
        "kind": "short_utterance_qa",
        "target_sha256": hashlib.sha256(request.text.encode()).hexdigest(),
        "target_word_count": len(lexical_tokens(request.text)),
        "profile": request.profile,
        "policy_fingerprint": policy.fingerprint,
        "aligner_fingerprint": aligner.fingerprint,
        "selected_candidate": (
            selected.recipe.candidate_index if selected is not None else None
        ),
        "candidates": [_evaluation_dict(item, root=directory) for item in evaluations],
    }
    timestamp = _matching_report_timestamp(report, content)
    atomic_write_json(
        report,
        {
            **runtime_metadata("short-utterance-qa", timestamp=timestamp),
            **content,
        },
    )
    return report


def _matching_report_timestamp(
    report: Path,
    content: dict[str, object],
) -> str | None:
    if not report.is_file():
        return None
    try:
        previous = json.loads(report.read_text(encoding="utf-8"))
    except OSError, UnicodeDecodeError, json.JSONDecodeError:
        return None
    if not isinstance(previous, dict):
        return None
    prior_content = {
        key: value
        for key, value in previous.items()
        if key not in {"$schema", "schema_version", "yakbox_version", "timestamp"}
    }
    timestamp = previous.get("timestamp")
    matches = json.dumps(prior_content, sort_keys=True) == json.dumps(
        content,
        sort_keys=True,
    )
    return timestamp if matches and isinstance(timestamp, str) else None


def _evaluation_dict(
    evaluation: CandidateEvaluation, *, root: Path
) -> dict[str, object]:
    return {
        "candidate_index": evaluation.recipe.candidate_index,
        "template_id": evaluation.recipe.template_id,
        "position": evaluation.recipe.position.value,
        "natural": evaluation.recipe.natural,
        "seed": evaluation.recipe.seed,
        "carrier_sha256": hashlib.sha256(evaluation.recipe.text.encode()).hexdigest(),
        "accepted": evaluation.accepted,
        "reason_codes": list(evaluation.reason_codes),
        "confidence": evaluation.confidence,
        "full_audio": evaluation.full_audio.relative_to(root).as_posix(),
        "full_audio_sha256": sha256_file(evaluation.full_audio),
        "extracted_audio": (
            evaluation.extracted_audio.relative_to(root).as_posix()
            if evaluation.extracted_audio is not None
            else None
        ),
        "extracted_audio_sha256": (
            sha256_file(evaluation.extracted_audio)
            if evaluation.extracted_audio is not None
            else None
        ),
        "extracted_duration_seconds": (
            wav_duration_seconds(evaluation.extracted_audio)
            if evaluation.extracted_audio is not None
            else None
        ),
        "crop": asdict(evaluation.crop) if evaluation.crop is not None else None,
        "initial_extracted_audio": (
            evaluation.initial_extracted_audio.relative_to(root).as_posix()
            if evaluation.initial_extracted_audio is not None
            else None
        ),
        "acoustic_refined": evaluation.acoustic_refined,
        "acoustic": (
            asdict(evaluation.acoustic) if evaluation.acoustic is not None else None
        ),
        "acoustic_penalty_ms": (
            _acoustic_penalty(evaluation) if evaluation.acoustic is not None else None
        ),
        "acoustic_crop": (
            asdict(evaluation.acoustic_crop)
            if evaluation.acoustic_crop is not None
            else None
        ),
        "signal_quality": (
            asdict(evaluation.signal_quality)
            if evaluation.signal_quality is not None
            else None
        ),
        "phoneme_alignment": (
            {
                "backend": evaluation.phoneme_alignment.backend,
                "model": evaluation.phoneme_alignment.model,
                "fingerprint": evaluation.phoneme_alignment.fingerprint,
                "language": evaluation.phoneme_alignment.language,
                "issues": list(evaluation.phoneme_alignment.issues),
                "phonemes": [
                    asdict(item) for item in evaluation.phoneme_alignment.phonemes
                ],
            }
            if evaluation.phoneme_alignment is not None
            else None
        ),
        "phoneme_confidence": evaluation.phoneme_confidence,
        "phoneme_path_confidence": evaluation.phoneme_path_confidence,
    }


def _review_is_approved(report: Path | None, selected: CandidateEvaluation) -> bool:
    if report is None:
        return False
    review = report.with_name("listening-review.toml")
    report_sha256 = sha256_file(report)
    if not review.exists():
        _write_review_template(
            review,
            report_sha256=report_sha256,
            selected_candidate=selected.recipe.candidate_index,
            selected_audio_sha256=_selected_audio_sha256(selected),
        )
        return False
    try:
        raw = tomllib.loads(review.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise BuildError(f"Cannot read short-utterance review: {review}") from error
    status = raw.get("status")
    if status == "pending":
        _write_review_template(
            review,
            report_sha256=report_sha256,
            selected_candidate=selected.recipe.candidate_index,
            selected_audio_sha256=_selected_audio_sha256(selected),
        )
        return False
    if status != "pass":
        raise BuildError(f"Short-utterance listening review is not approved: {review}")
    if raw.get("schema_version") != 1:
        raise BuildError(f"Short-utterance review schema is invalid: {review}")
    if raw.get("report_sha256") != report_sha256:
        raise BuildError(f"Short-utterance review does not match its report: {review}")
    if raw.get("selected_candidate") != selected.recipe.candidate_index:
        raise BuildError(
            f"Short-utterance review selects a different candidate: {review}"
        )
    if raw.get("selected_audio_sha256") != _selected_audio_sha256(selected):
        raise BuildError(
            f"Short-utterance review does not match selected audio: {review}"
        )
    return True


def _write_review_template(
    path: Path,
    *,
    report_sha256: str,
    selected_candidate: int,
    selected_audio_sha256: str,
) -> None:
    content = (
        "schema_version = 1\n"
        'status = "pending" # change to "pass" only after listening\n'
        f'report_sha256 = "{report_sha256}"\n'
        f"selected_candidate = {selected_candidate}\n"
        f'selected_audio_sha256 = "{selected_audio_sha256}"\n'
        'notes = ""\n'
    )
    atomic_write_bytes(path, content.encode(), overwrite=True)


def _selected_audio_sha256(selected: CandidateEvaluation) -> str:
    if selected.extracted_audio is None:
        raise BuildError("Selected short-utterance candidate has no extracted audio")
    return sha256_file(selected.extracted_audio)


def _required_time(value: float | None) -> float:
    if value is None:
        raise BuildError("Accepted alignment did not provide target timing")
    return value
