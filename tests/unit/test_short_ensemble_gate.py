from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from tests.unit.test_short_utterances import (
    _ExpectedTextAligner,
    _LongFakeSpeechService,
    _rankable_candidate,
)

from yakbox.errors import SpeechAnalysisError
from yakbox.speech.analysis_pipeline import (
    AnalysisCacheTrace,
    AnalysisGateError,
    SpeechAnalysisEnsemble,
)
from yakbox.speech.models import SpeechSynthesisRequest
from yakbox.speech.short_synthesis import _apply_ensemble_gate
from yakbox.speech.short_utterances import CarrierPosition, ShortUtterancePolicy


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "consensus", "runtime"])
async def test_candidate_ensemble_gate_preserves_projection_and_cache_provenance(
    tmp_path: Path, failure: str | None
) -> None:
    candidate = _rankable_candidate(
        tmp_path,
        index=1,
        position=CarrierPosition.MIDDLE,
        confidence=0.99,
        leading_ms=10,
    )
    candidate = replace(
        candidate, recipe=replace(candidate.recipe, text="Before. No. After.")
    )
    await _LongFakeSpeechService().synthesize_to_file(
        SpeechSynthesisRequest(text=candidate.recipe.text, voice="narrator"),
        candidate.full_audio,
    )
    cache = AnalysisCacheTrace(("recognition",), ("alignment",))
    evidence = SimpleNamespace(
        carrier=SimpleNamespace(cache=cache),
        extracted=SimpleNamespace(cache=cache),
        terminal_reason="verified",
        fingerprint="c" * 64,
    )
    verify = AsyncMock(return_value=evidence)
    if failure == "consensus":
        verify.side_effect = AnalysisGateError(
            "carrier_consensus", "Private detail", cache
        )
    elif failure == "runtime":
        verify.side_effect = SpeechAnalysisError("Private detail")
    ensemble = Mock(spec=SpeechAnalysisEnsemble)
    ensemble.verify_carrier_extraction = verify
    result = await _apply_ensemble_gate(
        candidate,
        ensemble=ensemble,
        aligner=_ExpectedTextAligner(),
        target_text="No.",
        language="en",
        policy=ShortUtterancePolicy(),
    )
    assert result.candidate_id
    assert result.ensemble_attempted
    assert verify.call_args.kwargs["target_token_start"] == 1
    assert verify.call_args.kwargs["target_token_end"] == 2
    assert verify.call_args.kwargs["carrier_audio"] == candidate.full_audio
    if failure is None:
        assert result.accepted
        assert result.ensemble_evidence_fingerprint == evidence.fingerprint
        assert result.ensemble_cache_hits == ("recognition", "recognition")
        assert result.ensemble_cache_misses == ("alignment", "alignment")
    else:
        assert not result.accepted
        assert result.reason_codes == ("ensemble_verification_failed",)
        assert result.terminal_reason == (
            "carrier_consensus" if failure == "consensus" else "ensemble_verification"
        )
        assert "Private" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["No. Again no.", "Before. Nothing. After."])
async def test_ensemble_gate_refuses_ambiguous_or_missing_target_before_inference(
    tmp_path: Path, carrier: str
) -> None:
    candidate = _rankable_candidate(
        tmp_path,
        index=1,
        position=CarrierPosition.MIDDLE,
        confidence=0.99,
        leading_ms=10,
    )
    candidate = replace(candidate, recipe=replace(candidate.recipe, text=carrier))
    ensemble = Mock(spec=SpeechAnalysisEnsemble)
    ensemble.verify_carrier_extraction = AsyncMock()
    result = await _apply_ensemble_gate(
        candidate,
        ensemble=ensemble,
        aligner=_ExpectedTextAligner(),
        target_text="No.",
        language="en",
        policy=ShortUtterancePolicy(),
    )
    assert not result.accepted
    assert result.terminal_reason == "ambiguous_target_projection"
    ensemble.verify_carrier_extraction.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensemble_gate_never_revives_a_candidate_rejected_by_earlier_gates(
    tmp_path: Path,
) -> None:
    candidate = _rankable_candidate(
        tmp_path,
        index=1,
        position=CarrierPosition.MIDDLE,
        confidence=0.99,
        leading_ms=10,
    )
    candidate = replace(
        candidate, accepted=False, reason_codes=("unexpected_suffix_speech",)
    )
    ensemble = Mock(spec=SpeechAnalysisEnsemble)
    result = await _apply_ensemble_gate(
        candidate,
        ensemble=ensemble,
        aligner=_ExpectedTextAligner(),
        target_text="No.",
        language="en",
        policy=ShortUtterancePolicy(),
    )
    assert not result.accepted
    assert result.terminal_reason == "unexpected_suffix_speech"
    assert not result.ensemble_attempted
    ensemble.verify_carrier_extraction.assert_not_called()
