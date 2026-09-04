"""Candidate orchestration for verified short-utterance synthesis."""

from __future__ import annotations

import hashlib
import json
import tempfile
import tomllib
import wave
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from typing import cast

from yakbox._files import atomic_write_bytes, atomic_write_json, sha256_file
from yakbox.audio.crop import (
    CropEvidence,
    SignalQualityEvidence,
    SpeechIslandEvidence,
    SpeechRegion,
    crop_aligned_wav,
    inspect_signal_quality,
    inspect_speech_islands,
    pad_wav_silence,
    wav_duration_seconds,
)
from yakbox.contracts import runtime_metadata
from yakbox.errors import (
    ArtifactError,
    BuildError,
    SpeechAnalysisError,
    ValidationError,
)
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
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_pipeline import (
    AnalysisGateError,
    BoundaryObservation,
    SpeechAnalysisEnsemble,
)
from yakbox.speech.fingerprints import speech_request_fingerprint
from yakbox.speech.models import SpeechSynthesisRequest
from yakbox.speech.phonemes import (
    PhonemeAligner,
    PhonemeAlignmentResult,
    PhonemeToken,
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
    generation_cache_hit: bool = False
    extraction_cache_hit: bool = False
    evaluation_cache_hit: bool = False
    candidate_id: str | None = None
    terminal_reason: str | None = None
    ensemble_evidence_fingerprint: str | None = None
    ensemble_cache_hits: tuple[str, ...] = ()
    ensemble_cache_misses: tuple[str, ...] = ()
    ensemble_attempted: bool = False


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
    candidate_cache_directory: Path | None = None,
    ensemble: SpeechAnalysisEnsemble | None = None,
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
            candidate_cache_directory=candidate_cache_directory,
            ensemble=ensemble,
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
            candidate_cache_directory=candidate_cache_directory,
            ensemble=ensemble,
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
    candidate_cache_directory: Path | None,
    ensemble: SpeechAnalysisEnsemble | None,
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
            candidate_cache_directory=candidate_cache_directory,
        )
        evaluated = await _apply_phoneme_gate(
            initial,
            aligner=phoneme_aligner,
            target_text=request.text,
            language=phoneme_language,
            minimum_confidence=minimum_phoneme_confidence,
            policy=policy,
            cache_directory=candidate_cache_directory,
        )
        evaluated = await _apply_ensemble_gate(
            evaluated,
            ensemble=ensemble,
            aligner=aligner,
            target_text=request.text,
            language=language,
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


async def _apply_ensemble_gate(
    evaluation: CandidateEvaluation,
    *,
    ensemble: SpeechAnalysisEnsemble | None,
    aligner: SpeechAligner,
    target_text: str,
    language: str,
    policy: ShortUtterancePolicy,
) -> CandidateEvaluation:
    """Require independent consensus and forced timing when v2 is supplied."""
    candidate_id = semantic_fingerprint(
        "short-utterance-candidate-v2",
        {
            "recipe": evaluation.recipe,
            "generation_policy": policy.generation_fingerprint,
            "maximum_candidates": policy.candidate_count,
            "maximum_rounds": 1,
        },
    )
    if ensemble is None:
        terminal = (
            "accepted_legacy_gates"
            if evaluation.accepted
            else (evaluation.reason_codes[0] if evaluation.reason_codes else "rejected")
        )
        return replace(
            evaluation,
            candidate_id=candidate_id,
            terminal_reason=terminal,
        )
    if not evaluation.accepted or evaluation.extracted_audio is None:
        terminal = evaluation.reason_codes[0] if evaluation.reason_codes else "rejected"
        return replace(
            evaluation,
            candidate_id=candidate_id,
            terminal_reason=terminal,
        )
    carrier_tokens = lexical_tokens(evaluation.recipe.text)
    target_tokens = lexical_tokens(target_text)
    positions = tuple(
        index
        for index in range(len(carrier_tokens) - len(target_tokens) + 1)
        if carrier_tokens[index : index + len(target_tokens)] == target_tokens
    )
    if len(positions) != 1:
        return replace(
            evaluation,
            accepted=False,
            reason_codes=(*evaluation.reason_codes, "ambiguous_target_projection"),
            candidate_id=candidate_id,
            terminal_reason="ambiguous_target_projection",
            ensemble_cache_misses=("target_projection",),
            ensemble_attempted=True,
        )
    legacy = await aligner.align(
        evaluation.full_audio,
        evaluation.recipe.text,
        language=language,
    )
    location = _locate_carrier_target(
        legacy,
        expected_text=evaluation.recipe.text,
        target_text=target_text,
        policy=policy,
    )
    if not location.accepted:
        return replace(
            evaluation,
            accepted=False,
            reason_codes=(*evaluation.reason_codes, "ambiguous_target_projection"),
            candidate_id=candidate_id,
            terminal_reason="ambiguous_target_projection",
            ensemble_cache_misses=("target_projection",),
            ensemble_attempted=True,
        )
    sample_rate = _wav_sample_rate(evaluation.full_audio)
    start = int(_required_time(location.start_seconds) * sample_rate)
    end = int(_required_time(location.end_seconds) * sample_rate)
    vad_start, vad_end = _legacy_vad_bounds(
        legacy.speech_regions, start, end, sample_rate
    )
    waveform_start = start
    waveform_end = end
    if evaluation.crop is not None:
        waveform_start = int(evaluation.crop.source_start_seconds * sample_rate)
        waveform_end = int(evaluation.crop.source_end_seconds * sample_rate)
    signal_fingerprint = (
        semantic_fingerprint(
            "short-utterance-signal-evidence-v1", evaluation.signal_quality
        )
        if evaluation.signal_quality is not None
        else None
    )
    try:
        evidence = await ensemble.verify_carrier_extraction(
            candidate_id=candidate_id,
            carrier_audio=evaluation.full_audio,
            extracted_audio=evaluation.extracted_audio,
            carrier_tokens=carrier_tokens,
            target_token_start=positions[0],
            target_token_end=positions[0] + len(target_tokens),
            vad=BoundaryObservation("legacy-vad", vad_start, vad_end),
            waveform=BoundaryObservation("waveform-edge", waveform_start, waveform_end),
            tolerance_frames=round(
                policy.maximum_vad_disagreement_ms * sample_rate / 1_000
            ),
            language=language,
            signal_evidence_fingerprint=signal_fingerprint,
        )
    except (AnalysisGateError, SpeechAnalysisError, ValidationError) as error:
        if isinstance(error, AnalysisGateError):
            terminal = error.stage
            cache_hits = error.cache.hit_stages
            cache_misses = (*error.cache.miss_stages, error.stage)
        else:
            terminal = "ensemble_verification"
            cache_hits = ()
            cache_misses = ("ensemble_verification",)
        return replace(
            evaluation,
            accepted=False,
            reason_codes=(*evaluation.reason_codes, "ensemble_verification_failed"),
            candidate_id=candidate_id,
            terminal_reason=terminal,
            ensemble_cache_hits=cache_hits,
            ensemble_cache_misses=cache_misses,
            ensemble_attempted=True,
        )
    cache_hits = (
        *evidence.carrier.cache.hit_stages,
        *evidence.extracted.cache.hit_stages,
    )
    cache_misses = (
        *evidence.carrier.cache.miss_stages,
        *evidence.extracted.cache.miss_stages,
    )
    return replace(
        evaluation,
        candidate_id=candidate_id,
        terminal_reason=evidence.terminal_reason,
        ensemble_evidence_fingerprint=evidence.fingerprint,
        ensemble_cache_hits=cache_hits,
        ensemble_cache_misses=cache_misses,
        ensemble_attempted=True,
    )


def _wav_sample_rate(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as reader:
            return reader.getframerate()
    except (OSError, EOFError, wave.Error) as error:
        raise ArtifactError(f"Cannot inspect candidate WAV {path}: {error}") from error


def _legacy_vad_bounds(
    regions: tuple[SpeechRegion, ...],
    start_frame: int,
    end_frame: int,
    sample_rate: int,
) -> tuple[int, int]:
    overlapping = tuple(
        region
        for region in regions
        if region.end_seconds * sample_rate >= start_frame
        and region.start_seconds * sample_rate <= end_frame
    )
    if not overlapping:
        return start_frame, end_frame
    return (
        max(start_frame, int(overlapping[0].start_seconds * sample_rate)),
        min(end_frame, int(overlapping[-1].end_seconds * sample_rate)),
    )


async def _apply_phoneme_gate(
    evaluation: CandidateEvaluation,
    *,
    aligner: PhonemeAligner | None,
    target_text: str,
    language: str,
    minimum_confidence: float,
    policy: ShortUtterancePolicy,
    cache_directory: Path | None = None,
) -> CandidateEvaluation:
    """Apply an independent phoneme path gate to a candidate's final crop."""
    if aligner is None or evaluation.extracted_audio is None:
        return evaluation
    try:
        result = _read_phoneme_cache(
            cache_directory,
            audio=evaluation.extracted_audio,
            aligner_fingerprint=aligner.fingerprint,
            target_text=target_text,
            language=language,
        )
        if result is None:
            result = await aligner.align(
                evaluation.extracted_audio,
                target_text,
                language=language,
            )
            _write_phoneme_cache(
                cache_directory,
                audio=evaluation.extracted_audio,
                aligner_fingerprint=aligner.fingerprint,
                target_text=target_text,
                language=language,
                result=result,
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
    candidate_cache_directory: Path | None,
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
    cache_hit = _materialize_cached_candidate(
        candidate_cache_directory,
        candidate_request,
        full,
    )
    if not cache_hit:
        await service.synthesize_to_file(candidate_request, full, overwrite=True)
        _store_cached_candidate(candidate_cache_directory, candidate_request, full)
    evaluation_key = _evaluation_cache_key(
        full,
        request=request,
        recipe=recipe,
        policy=policy,
        aligner_fingerprint=aligner.fingerprint,
        language=language,
    )
    cached_evaluation = _materialize_cached_evaluation(
        candidate_cache_directory,
        evaluation_key,
        recipe=recipe,
        full=full,
        directory=directory,
    )
    if cached_evaluation is not None:
        return replace(
            cached_evaluation,
            generation_cache_hit=cache_hit,
            evaluation_cache_hit=True,
        )
    alignment = await aligner.align(full, recipe.text, language=language)
    if recipe.position is CarrierPosition.DIRECT:
        result = await _evaluate_extracted_audio(
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
        result = replace(result, generation_cache_hit=cache_hit)
        _store_cached_evaluation(
            candidate_cache_directory,
            evaluation_key,
            result,
        )
        return result
    location = _locate_carrier_target(
        alignment,
        expected_text=recipe.text,
        target_text=request.text,
        policy=policy,
    )
    if not location.accepted:
        return replace(
            _rejected(recipe, full, location),
            generation_cache_hit=cache_hit,
        )
    if not _has_required_pause(alignment, location, recipe.position, policy):
        return replace(
            _rejected_code(recipe, full, "insufficient_carrier_pause"),
            generation_cache_hit=cache_hit,
        )
    extraction_key = _extraction_cache_key(
        full,
        request=request,
        recipe=recipe,
        policy=policy,
        aligner_fingerprint=aligner.fingerprint,
        language=language,
    )
    crop = _materialize_cached_extraction(
        candidate_cache_directory,
        extraction_key,
        extracted,
    )
    extraction_cache_hit = crop is not None
    if crop is None:
        try:
            crop = crop_aligned_wav(
                full,
                extracted,
                start_seconds=_required_time(location.start_seconds),
                end_seconds=_required_time(location.end_seconds),
                pre_roll_ms=policy.pre_roll_ms,
                post_roll_ms=policy.post_roll_ms,
                fade_ms=policy.fade_ms,
                speech_regions=alignment.speech_regions,
                overwrite=True,
            )
        except ArtifactError:
            return replace(
                _rejected_code(recipe, full, "unsafe_crop_boundary"),
                generation_cache_hit=cache_hit,
            )
        _store_cached_extraction(
            candidate_cache_directory,
            extraction_key,
            extracted,
            crop,
        )
    extracted_alignment = await aligner.align(
        extracted, request.text, language=language
    )
    result = await _evaluate_extracted_audio(
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
    carrier_decision = _validate_carrier(alignment, recipe, request.text, policy)
    reasons = tuple(
        dict.fromkeys((*carrier_decision.reason_codes, *result.reason_codes))
    )
    result = replace(
        result,
        accepted=carrier_decision.accepted and result.accepted,
        reason_codes=reasons,
        generation_cache_hit=cache_hit,
        extraction_cache_hit=extraction_cache_hit,
    )
    _store_cached_evaluation(
        candidate_cache_directory,
        evaluation_key,
        result,
    )
    return result


def _locate_carrier_target(
    alignment: AlignmentResult,
    *,
    expected_text: str,
    target_text: str,
    policy: ShortUtterancePolicy,
) -> AlignmentDecision:
    """Locate one unambiguous target without coupling the crop to QA thresholds."""
    return validate_carrier_alignment(
        alignment,
        expected_text=expected_text,
        target_text=target_text,
        minimum_confidence=0.0,
        token_aliases=policy.alignment_alias_map,
        minimum_average_log_probability=float("-inf"),
        maximum_compression_ratio=float("inf"),
        maximum_no_speech_probability=1.0,
        maximum_temperature=float("inf"),
        maximum_internal_token_gap_ms=2_147_483_647,
        maximum_token_duration_ms=2_147_483_647,
    )


def _validate_carrier(
    alignment: AlignmentResult,
    recipe: CarrierRecipe,
    target_text: str,
    policy: ShortUtterancePolicy,
) -> AlignmentDecision:
    return validate_carrier_alignment(
        alignment,
        expected_text=recipe.text,
        target_text=target_text,
        minimum_confidence=policy.minimum_confidence_for(
            len(lexical_tokens(target_text)),
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
    initial_audio, acoustic = _pad_edge_deficits(
        initial_audio,
        acoustic,
        policy=policy,
        directory=directory,
        candidate_index=recipe.candidate_index,
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
            refined, refined_acoustic = _pad_edge_deficits(
                refined,
                refined_acoustic,
                policy=policy,
                directory=directory,
                candidate_index=recipe.candidate_index,
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


def _pad_edge_deficits(
    audio: Path,
    evidence: SpeechIslandEvidence,
    *,
    policy: ShortUtterancePolicy,
    directory: Path,
    candidate_index: int,
) -> tuple[Path, SpeechIslandEvidence]:
    """Pad verified speech when a crop misses the configured quiet edge floor."""
    if (
        evidence.primary_island is None
        or evidence.detached_prefix
        or evidence.detached_suffix
    ):
        return audio, evidence
    leading_ms = max(0.0, policy.minimum_edge_silence_ms - evidence.leading_silence_ms)
    trailing_ms = max(
        0.0,
        policy.minimum_edge_silence_ms - evidence.trailing_silence_ms,
    )
    if leading_ms == 0 and trailing_ms == 0:
        return audio, evidence
    padded = directory / f"candidate-{candidate_index:03d}-edge-padded.wav"
    pad_wav_silence(
        audio,
        padded,
        leading_ms=leading_ms,
        trailing_ms=trailing_ms,
        overwrite=True,
    )
    return padded, inspect_speech_islands(
        padded,
        threshold_dbfs=policy.acoustic_threshold_dbfs,
        island_gap_ms=policy.speech_island_gap_ms,
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


def _candidate_cache_paths(
    cache_directory: Path,
    request: SpeechSynthesisRequest,
) -> tuple[str, Path, Path]:
    fingerprint = speech_request_fingerprint(request)
    root = cache_directory / fingerprint[:2]
    return (
        fingerprint,
        root / f"{fingerprint}.wav",
        root / f"{fingerprint}.json",
    )


def _materialize_cached_candidate(
    cache_directory: Path | None,
    request: SpeechSynthesisRequest,
    destination: Path,
) -> bool:
    if cache_directory is None:
        return False
    fingerprint, audio, metadata = _candidate_cache_paths(cache_directory, request)
    if not audio.is_file() or not metadata.is_file():
        return False
    try:
        raw = json.loads(metadata.read_text(encoding="utf-8"))
        valid = (
            isinstance(raw, dict)
            and raw.get("schema_version") == 1
            and raw.get("kind") == "short_utterance_candidate"
            and raw.get("generation_fingerprint") == fingerprint
            and raw.get("size") == audio.stat().st_size
            and raw.get("sha256") == sha256_file(audio)
        )
        if valid:
            wav_duration_seconds(audio)
        else:
            return False
    except OSError, json.JSONDecodeError, ArtifactError:
        return False
    atomic_write_bytes(destination, audio.read_bytes(), overwrite=True)
    return True


def _store_cached_candidate(
    cache_directory: Path | None,
    request: SpeechSynthesisRequest,
    source: Path,
) -> None:
    if cache_directory is None:
        return
    fingerprint, audio, metadata = _candidate_cache_paths(cache_directory, request)
    atomic_write_bytes(audio, source.read_bytes(), overwrite=True)
    atomic_write_json(
        metadata,
        {
            "schema_version": 1,
            "kind": "short_utterance_candidate",
            "generation_fingerprint": fingerprint,
            "sha256": sha256_file(audio),
            "size": audio.stat().st_size,
            "text_sha256": hashlib.sha256(request.text.encode()).hexdigest(),
            "backend": request.backend,
            "profile": request.profile,
        },
    )


def _stage_cache_paths(
    cache_directory: Path,
    stage: str,
    key: str,
) -> tuple[Path, Path]:
    root = cache_directory / stage / key[:2]
    return root / f"{key}.wav", root / f"{key}.json"


def _extraction_cache_key(
    full_audio: Path,
    *,
    request: SpeechSynthesisRequest,
    recipe: CarrierRecipe,
    policy: ShortUtterancePolicy,
    aligner_fingerprint: str,
    language: str,
) -> str:
    return _stage_cache_key_value(
        "extraction-v1",
        audio_sha256=sha256_file(full_audio),
        target_sha256=hashlib.sha256(request.text.encode()).hexdigest(),
        carrier_sha256=hashlib.sha256(recipe.text.encode()).hexdigest(),
        position=recipe.position.value,
        extraction_fingerprint=policy.extraction_fingerprint,
        aligner_fingerprint=aligner_fingerprint,
        language=language.casefold(),
    )


def _evaluation_cache_key(
    full_audio: Path,
    *,
    request: SpeechSynthesisRequest,
    recipe: CarrierRecipe,
    policy: ShortUtterancePolicy,
    aligner_fingerprint: str,
    language: str,
) -> str:
    return _stage_cache_key_value(
        "evaluation-v1",
        extraction_key=_extraction_cache_key(
            full_audio,
            request=request,
            recipe=recipe,
            policy=policy,
            aligner_fingerprint=aligner_fingerprint,
            language=language,
        ),
        evaluation_fingerprint=policy.evaluation_fingerprint,
    )


def _stage_cache_key_value(kind: str, **values: object) -> str:
    payload = {"kind": kind, **values}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _materialize_cached_extraction(
    cache_directory: Path | None,
    key: str,
    destination: Path,
) -> CropEvidence | None:
    if cache_directory is None:
        return None
    audio, metadata = _stage_cache_paths(cache_directory, "extractions", key)
    raw = _read_stage_cache(metadata, audio, key=key, kind="short_utterance_extraction")
    if raw is None or not isinstance(raw.get("crop"), dict):
        return None
    try:
        crop = _required_crop_from_cache(raw["crop"])
        wav_duration_seconds(audio)
    except ArtifactError, TypeError:
        return None
    atomic_write_bytes(destination, audio.read_bytes(), overwrite=True)
    return crop


def _store_cached_extraction(
    cache_directory: Path | None,
    key: str,
    source: Path,
    crop: CropEvidence,
) -> None:
    if cache_directory is None:
        return
    audio, metadata = _stage_cache_paths(cache_directory, "extractions", key)
    atomic_write_bytes(audio, source.read_bytes(), overwrite=True)
    atomic_write_json(
        metadata,
        {
            "cache_version": 1,
            "kind": "short_utterance_extraction",
            "key": key,
            "sha256": sha256_file(audio),
            "size": audio.stat().st_size,
            "crop": asdict(crop),
        },
    )


def _materialize_cached_evaluation(
    cache_directory: Path | None,
    key: str,
    *,
    recipe: CarrierRecipe,
    full: Path,
    directory: Path,
) -> CandidateEvaluation | None:
    if cache_directory is None:
        return None
    audio, metadata = _stage_cache_paths(cache_directory, "evaluations", key)
    raw = _read_stage_cache(
        metadata,
        audio,
        key=key,
        kind="short_utterance_evaluation",
    )
    if raw is None:
        return None
    try:
        destination = (
            directory / f"candidate-{recipe.candidate_index:03d}-evaluated.wav"
        )
        atomic_write_bytes(destination, audio.read_bytes(), overwrite=True)
        return CandidateEvaluation(
            recipe=recipe,
            accepted=bool(raw["accepted"]),
            reason_codes=tuple(cast(list[str], raw["reason_codes"])),
            full_audio=full,
            extracted_audio=destination,
            confidence=_optional_float(raw.get("confidence")),
            crop=_crop_from_cache(raw.get("crop")),
            initial_extracted_audio=destination,
            acoustic=_speech_islands_from_cache(raw.get("acoustic")),
            acoustic_refined=bool(raw.get("acoustic_refined")),
            acoustic_crop=_crop_from_cache(raw.get("acoustic_crop")),
            signal_quality=_signal_quality_from_cache(raw.get("signal_quality")),
        )
    except ArtifactError, KeyError, TypeError, ValueError:
        return None


def _store_cached_evaluation(
    cache_directory: Path | None,
    key: str,
    evaluation: CandidateEvaluation,
) -> None:
    if cache_directory is None or evaluation.extracted_audio is None:
        return
    audio, metadata = _stage_cache_paths(cache_directory, "evaluations", key)
    atomic_write_bytes(audio, evaluation.extracted_audio.read_bytes(), overwrite=True)
    atomic_write_json(
        metadata,
        {
            "cache_version": 1,
            "kind": "short_utterance_evaluation",
            "key": key,
            "sha256": sha256_file(audio),
            "size": audio.stat().st_size,
            "accepted": evaluation.accepted,
            "reason_codes": list(evaluation.reason_codes),
            "confidence": evaluation.confidence,
            "crop": asdict(evaluation.crop) if evaluation.crop else None,
            "acoustic": asdict(evaluation.acoustic) if evaluation.acoustic else None,
            "acoustic_refined": evaluation.acoustic_refined,
            "acoustic_crop": (
                asdict(evaluation.acoustic_crop) if evaluation.acoustic_crop else None
            ),
            "signal_quality": (
                asdict(evaluation.signal_quality) if evaluation.signal_quality else None
            ),
        },
    )


def _read_stage_cache(
    metadata: Path,
    audio: Path,
    *,
    key: str,
    kind: str,
) -> dict[str, object] | None:
    if not metadata.is_file() or not audio.is_file():
        return None
    try:
        raw = json.loads(metadata.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or raw.get("cache_version") != 1
            or raw.get("kind") != kind
            or raw.get("key") != key
            or raw.get("size") != audio.stat().st_size
            or raw.get("sha256") != sha256_file(audio)
        ):
            return None
        return raw
    except OSError, json.JSONDecodeError:
        return None


def _crop_from_cache(value: object) -> CropEvidence | None:
    return _required_crop_from_cache(value) if isinstance(value, dict) else None


def _required_crop_from_cache(value: object) -> CropEvidence:
    if not isinstance(value, dict):
        raise TypeError
    raw = cast(dict[str, object], value)
    return CropEvidence(
        source_start_seconds=_required_float(raw["source_start_seconds"]),
        source_end_seconds=_required_float(raw["source_end_seconds"]),
        crop_start_seconds=_required_float(raw["crop_start_seconds"]),
        crop_end_seconds=_required_float(raw["crop_end_seconds"]),
        pre_roll_ms=_required_float(raw["pre_roll_ms"]),
        post_roll_ms=_required_float(raw["post_roll_ms"]),
        fade_ms=_required_int(raw["fade_ms"]),
        zero_crossing_start_shift_ms=_required_float(
            raw["zero_crossing_start_shift_ms"]
        ),
        zero_crossing_end_shift_ms=_required_float(raw["zero_crossing_end_shift_ms"]),
    )


def _required_float(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError
    return float(value)


def _required_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError
    return value


def _speech_islands_from_cache(value: object) -> SpeechIslandEvidence | None:
    if not isinstance(value, dict):
        return None
    data = dict(value)
    for key in ("regions", "islands", "detached_prefix", "detached_suffix"):
        data[key] = tuple(SpeechRegion(**item) for item in data.get(key, []))
    primary = data.get("primary_island")
    data["primary_island"] = (
        SpeechRegion(**primary) if isinstance(primary, dict) else None
    )
    return SpeechIslandEvidence(**data)


def _signal_quality_from_cache(value: object) -> SignalQualityEvidence | None:
    if not isinstance(value, dict):
        return None
    data = dict(value)
    data["adaptive_speech_regions"] = tuple(
        SpeechRegion(**item) for item in data.get("adaptive_speech_regions", [])
    )
    return SignalQualityEvidence(**data)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError
    return float(value)


def _phoneme_cache_key(
    *,
    audio: Path,
    aligner_fingerprint: str,
    target_text: str,
    language: str,
) -> str:
    return _stage_cache_key_value(
        "phoneme-alignment-v1",
        audio_sha256=sha256_file(audio),
        aligner_fingerprint=aligner_fingerprint,
        target_sha256=hashlib.sha256(target_text.encode()).hexdigest(),
        language=language.casefold(),
    )


def _read_phoneme_cache(
    cache_directory: Path | None,
    *,
    audio: Path,
    aligner_fingerprint: str,
    target_text: str,
    language: str,
) -> PhonemeAlignmentResult | None:
    if cache_directory is None:
        return None
    key = _phoneme_cache_key(
        audio=audio,
        aligner_fingerprint=aligner_fingerprint,
        target_text=target_text,
        language=language,
    )
    _, metadata = _stage_cache_paths(cache_directory, "phonemes", key)
    try:
        raw = json.loads(metadata.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("key") != key:
            return None
        return PhonemeAlignmentResult(
            phonemes=tuple(PhonemeToken(**item) for item in raw["phonemes"]),
            backend=str(raw["backend"]),
            model=str(raw["model"]),
            fingerprint=str(raw["fingerprint"]),
            language=str(raw["language"]),
            issues=tuple(raw.get("issues", [])),
        )
    except OSError, json.JSONDecodeError, KeyError, TypeError, ValueError:
        return None


def _write_phoneme_cache(
    cache_directory: Path | None,
    *,
    audio: Path,
    aligner_fingerprint: str,
    target_text: str,
    language: str,
    result: PhonemeAlignmentResult,
) -> None:
    if cache_directory is None:
        return
    key = _phoneme_cache_key(
        audio=audio,
        aligner_fingerprint=aligner_fingerprint,
        target_text=target_text,
        language=language,
    )
    _, metadata = _stage_cache_paths(cache_directory, "phonemes", key)
    atomic_write_json(
        metadata,
        {
            "cache_version": 1,
            "kind": "short_utterance_phoneme_alignment",
            "key": key,
            "backend": result.backend,
            "model": result.model,
            "fingerprint": result.fingerprint,
            "language": result.language,
            "issues": list(result.issues),
            "phonemes": [asdict(item) for item in result.phonemes],
        },
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
        "generation_fingerprint": policy.generation_fingerprint,
        "extraction_fingerprint": policy.extraction_fingerprint,
        "evaluation_fingerprint": policy.evaluation_fingerprint,
        "selection_fingerprint": policy.selection_fingerprint,
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
    value: dict[str, object] = {
        "candidate_index": evaluation.recipe.candidate_index,
        "template_id": evaluation.recipe.template_id,
        "position": evaluation.recipe.position.value,
        "natural": evaluation.recipe.natural,
        "seed": evaluation.recipe.seed,
        "carrier_sha256": hashlib.sha256(evaluation.recipe.text.encode()).hexdigest(),
        "accepted": evaluation.accepted,
        "generation_cache_hit": evaluation.generation_cache_hit,
        "extraction_cache_hit": evaluation.extraction_cache_hit,
        "evaluation_cache_hit": evaluation.evaluation_cache_hit,
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
    if evaluation.ensemble_attempted:
        value.update(
            {
                "candidate_id": evaluation.candidate_id,
                "terminal_reason": evaluation.terminal_reason,
                "ensemble_evidence_fingerprint": (
                    evaluation.ensemble_evidence_fingerprint
                ),
                "ensemble_cache_hits": list(evaluation.ensemble_cache_hits),
                "ensemble_cache_misses": list(evaluation.ensemble_cache_misses),
            }
        )
    return value


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
