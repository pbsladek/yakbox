from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from yakbox._files import sha256_bytes
from yakbox.errors import ArtifactError, ValidationError
from yakbox.schemas import load_schema
from yakbox.speech.analysis_cache import (
    CacheLookup,
    CacheLookupState,
    ConsensusCacheIdentity,
    EvidenceReference,
    ForcedAlignmentCacheIdentity,
    LayeredEvidenceCache,
    RecognitionCacheIdentity,
    ReleaseEvidenceSnapshot,
    VerificationCacheIdentity,
)
from yakbox.speech.analysis_models import AlignmentPurpose, AudioSpan

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _key(label: str) -> str:
    return sha256_bytes(label.encode())


def _recognition(**changes: object) -> RecognitionCacheIdentity:
    baseline = RecognitionCacheIdentity(
        model_fingerprint=SHA_A,
        execution_fingerprint=SHA_B,
        canonical_audio_fingerprint=SHA_C,
        span=AudioSpan(SHA_C, 10, 20, 16_000),
        language="en",
        normalization_fingerprint="legacy-a",
        calibration_fingerprint="legacy-a",
        preprocessing_fingerprint=SHA_D,
        decode_settings_fingerprint=SHA_A,
    )
    return replace(baseline, **changes)


def test_recognition_identity_excludes_downstream_expected_policy() -> None:
    baseline = _recognition()
    changed = _recognition(
        normalization_fingerprint="different-normalization",
        calibration_fingerprint="different-calibration",
    )
    assert baseline.fingerprint == changed.fingerprint
    assert baseline.fingerprint != _recognition(language="fr").fingerprint
    assert (
        baseline.fingerprint
        != _recognition(span=AudioSpan(SHA_C, 11, 20, 16_000)).fingerprint
    )
    assert (
        baseline.fingerprint
        != _recognition(decode_settings_fingerprint=SHA_B).fingerprint
    )


def test_layered_identities_invalidate_only_their_declared_dependencies() -> None:
    recognition = _recognition()
    forced = ForcedAlignmentCacheIdentity(
        model_fingerprint=SHA_A,
        execution_fingerprint=SHA_B,
        canonical_audio_fingerprint=SHA_C,
        span=AudioSpan(SHA_C, 10, 20, 16_000),
        language="en",
        aligner_text_hash=SHA_D,
        expected_lexical_span_hash=SHA_A,
        purpose=AlignmentPurpose.VERIFIED_TARGET,
    )
    consensus = ConsensusCacheIdentity(
        recognition_fingerprints=(SHA_A, SHA_B),
        expected_tokens_hash=SHA_C,
        policy_fingerprint=SHA_D,
        equivalence_fingerprint=SHA_A,
        calibration_fingerprint=SHA_B,
    )
    verification = VerificationCacheIdentity(
        consensus_fingerprint=SHA_A,
        forced_alignment_fingerprint=SHA_B,
        signal_evidence_fingerprint=SHA_C,
        artifact_digest=SHA_D,
        policy_fingerprint=SHA_A,
        human_disposition_fingerprint=None,
        spoken_text_plan_fingerprint=SHA_B,
        assembly_map_fingerprint=SHA_C,
        artifact_identity_fingerprint=SHA_D,
        calibration_fingerprint=SHA_A,
    )

    assert replace(forced, aligner_text_hash=SHA_C).fingerprint != forced.fingerprint
    assert (
        replace(consensus, expected_tokens_hash=SHA_D).fingerprint
        != consensus.fingerprint
    )
    assert (
        replace(consensus, calibration_fingerprint=SHA_C).fingerprint
        != consensus.fingerprint
    )
    assert (
        replace(verification, spoken_text_plan_fingerprint=SHA_C).fingerprint
        != verification.fingerprint
    )
    assert (
        replace(verification, human_disposition_fingerprint=SHA_D).fingerprint
        != verification.fingerprint
    )
    assert (
        replace(verification, artifact_identity_fingerprint=SHA_C).fingerprint
        != verification.fingerprint
    )
    assert (
        recognition.fingerprint
        == _recognition(
            normalization_fingerprint="changed-downstream",
            calibration_fingerprint="changed-downstream",
        ).fingerprint
    )


@pytest.mark.asyncio
async def test_identical_requests_single_flight_and_repeat_without_inference(
    tmp_path: Path,
) -> None:
    cache = LayeredEvidenceCache(tmp_path / "cache")
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def produce() -> dict[str, object]:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"accepted": False, "reason_codes": ["lexical_mismatch"]}

    first = asyncio.create_task(
        cache.get_or_compute(
            stage="recognition",
            key=_key("same"),
            dependencies=(),
            producer=produce,
        )
    )
    await started.wait()
    second = asyncio.create_task(
        cache.get_or_compute(
            stage="recognition",
            key=_key("same"),
            dependencies=(),
            producer=produce,
        )
    )
    release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert calls == 1
    assert first_result[0].evidence == second_result[0].evidence
    assert not first_result[1].hit
    assert not second_result[1].hit

    repeated, diagnostic = await cache.get_or_compute(
        stage="recognition",
        key=_key("same"),
        dependencies=(),
        producer=produce,
    )
    assert calls == 1
    assert diagnostic.hit
    assert repeated.evidence["accepted"] is False


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_producer(tmp_path: Path) -> None:
    cache = LayeredEvidenceCache(tmp_path / "cache")
    started = asyncio.Event()
    release = asyncio.Event()

    async def produce() -> dict[str, object]:
        started.set()
        await release.wait()
        return {"tokens": ["safe"]}

    producer_waiter = asyncio.create_task(
        cache.get_or_compute(
            stage="recognition",
            key=_key("cancel"),
            dependencies=(),
            producer=produce,
        )
    )
    await started.wait()
    producer_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await producer_waiter
    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    async def forbidden() -> dict[str, object]:
        raise AssertionError("completed producer should have committed")

    entry, lookup = await cache.get_or_compute(
        stage="recognition",
        key=_key("cancel"),
        dependencies=(),
        producer=forbidden,
    )
    assert lookup.hit
    assert entry.evidence == {"tokens": ["safe"]}


@pytest.mark.asyncio
async def test_failed_or_invalid_producer_never_commits(tmp_path: Path) -> None:
    cache = LayeredEvidenceCache(tmp_path / "cache")

    async def crash() -> dict[str, object]:
        raise RuntimeError("worker crashed")

    with pytest.raises(RuntimeError, match="worker crashed"):
        await cache.get_or_compute(
            stage="recognition",
            key=_key("crash"),
            dependencies=(),
            producer=crash,
        )
    assert cache.lookup("recognition", _key("crash"))[0] is None

    def reject(_value: object) -> None:
        raise ValidationError("malformed result")

    async def malformed() -> dict[str, object]:
        return {"partial": True}

    with pytest.raises(ValidationError, match="malformed"):
        await cache.get_or_compute(
            stage="recognition",
            key=_key("invalid"),
            dependencies=(),
            producer=malformed,
            validator=reject,
        )
    assert cache.lookup("recognition", _key("invalid"))[0] is None


@pytest.mark.asyncio
async def test_corrupt_entry_is_quarantined_and_cannot_vote(tmp_path: Path) -> None:
    cache = LayeredEvidenceCache(tmp_path / "cache")
    key = _key("corrupt")

    async def produce() -> dict[str, object]:
        return {"accepted": True}

    await cache.get_or_compute(
        stage="consensus",
        key=key,
        dependencies=(),
        producer=produce,
    )
    path = tmp_path / "cache" / "entries" / "consensus" / key[:2] / f"{key}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["evidence"] = {"accepted": False}
    path.write_text(json.dumps(raw), encoding="utf-8")

    entry, lookup = cache.lookup("consensus", key)
    assert entry is None
    assert lookup.state is CacheLookupState.QUARANTINED
    assert not path.exists()
    assert (
        tmp_path / "cache" / "quarantine" / "consensus" / key[:2] / f"{key}.json"
    ).is_file()


@pytest.mark.asyncio
async def test_release_snapshot_pins_evidence_through_cleanup(tmp_path: Path) -> None:
    cache = LayeredEvidenceCache(tmp_path / "cache")

    async def produce() -> dict[str, object]:
        return {"accepted": True, "reason_codes": []}

    pinned, _ = await cache.get_or_compute(
        stage="verification",
        key=_key("pinned"),
        dependencies=(_key("recognition"),),
        producer=produce,
    )
    unpinned, _ = await cache.get_or_compute(
        stage="verification",
        key=_key("unpinned"),
        dependencies=(),
        producer=produce,
    )
    snapshot = ReleaseEvidenceSnapshot(
        release_id="chapter-one",
        release_audio_digest=SHA_A,
        text_plan_fingerprint=SHA_B,
        policy_fingerprint=SHA_C,
        calibration_fingerprint=SHA_D,
        execution_fingerprints=(SHA_A,),
        references=(
            EvidenceReference(
                "verification", _key("pinned"), pinned.evidence_fingerprint
            ),
        ),
        decision_fingerprint=SHA_B,
    )
    destination = cache.promote_release(tmp_path / "output", snapshot)
    Draft202012Validator(load_schema("speech-release-evidence-snapshot")).validate(
        json.loads(destination.read_text(encoding="utf-8"))
    )

    result = cache.cleanup(older_than_unix_ns=unpinned.created_at_unix_ns + 1)
    assert destination.is_file()
    assert result.removed == 1
    assert result.preserved_pinned == 1
    assert cache.lookup("verification", _key("pinned"))[0] is not None
    assert cache.lookup("verification", _key("unpinned"))[0] is None


def test_cache_paths_and_diagnostics_are_bounded(tmp_path: Path) -> None:
    cache = LayeredEvidenceCache(tmp_path / "cache")
    with pytest.raises(ValidationError, match="key"):
        cache.lookup("../recognition", SHA_A)
    with pytest.raises(ValidationError, match="diagnostic"):
        CacheLookup("recognition", SHA_A, CacheLookupState.MISS, "x" * 161)


def test_promotion_rejects_missing_evidence(tmp_path: Path) -> None:
    cache = LayeredEvidenceCache(tmp_path / "cache")
    snapshot = ReleaseEvidenceSnapshot(
        release_id="chapter-one",
        release_audio_digest=SHA_A,
        text_plan_fingerprint=SHA_B,
        policy_fingerprint=SHA_C,
        calibration_fingerprint=SHA_D,
        execution_fingerprints=(SHA_A,),
        references=(EvidenceReference("verification", SHA_A, SHA_B),),
        decision_fingerprint=SHA_C,
    )
    with pytest.raises(ArtifactError, match="unavailable"):
        cache.promote_release(tmp_path / "output", snapshot)
