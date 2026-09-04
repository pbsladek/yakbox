from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from yakbox.errors import ValidationError
from yakbox.schemas import load_schema
from yakbox.speech.analysis_evaluation import (
    CorpusPartition,
    CorpusTruth,
    EvaluationCorpus,
    EvaluationCorpusCase,
    EvaluationProtocol,
    ObservedCase,
    ReviewDecision,
    ReviewerDisposition,
    ReviewKind,
    approve_calibration,
    evaluate_shadow_corpus,
    load_default_evaluation_protocol,
    load_evaluation_corpus,
    load_evaluation_protocol,
    minimum_zero_failure_clusters,
)
from yakbox.speech.analysis_fingerprints import text_fingerprint
from yakbox.speech.analysis_models import (
    ClipClass,
    ConsensusOutcome,
    ConsensusResult,
)
from yakbox.speech.analysis_policy import CalibrationTable, CalibrationThreshold
from yakbox.speech.analysis_review import (
    QualificationReviewStatus,
    approve_calibration_review,
    load_qualification_review,
    qualification_review_template,
    reviewer_fingerprint,
    write_qualification_review_template,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
POLICY = "d" * 64
ROOT = Path(__file__).parents[2]


def _consensus(
    outcome: ConsensusOutcome, reasons: tuple[str, ...] = ()
) -> ConsensusResult:
    return ConsensusResult(
        policy_fingerprint=POLICY,
        expected_tokens_hash=SHA_A,
        equivalence_set_fingerprint=SHA_C,
        recognition_fingerprints=(SHA_A, SHA_B),
        applied_equivalences=(),
        votes=(),
        accepted_spans=(),
        rejected_spans=(),
        disagreement_spans=(),
        high_risk_spans=(),
        escalation_reason="not_required",
        outcome=outcome,
        reason_codes=reasons,
    )


def _case(
    identifier: str,
    truth: CorpusTruth,
    *,
    group: str,
    risk: str = "lexical",
    boundary_review: bool = False,
) -> EvaluationCorpusCase:
    expected_tokens = ("wren", "asked")
    return EvaluationCorpusCase(
        case_id=identifier,
        source_passage_group=group,
        voice=f"voice-{group}",
        partition=CorpusPartition.HELD_OUT,
        clip_class=ClipClass.SHORT_PHRASE,
        risk_class=risk,
        truth=truth,
        expected_tokens=expected_tokens,
        expected_tokens_hash=text_fingerprint("\u001f".join(expected_tokens)),
        expected_token_count=2,
        expected_character_count=9,
        audio_digest=SHA_B,
        audio_frame_count=16_000,
        sample_rate=16_000,
        rights_id="MIT-original-test-fixture",
        source_url="https://example.invalid/corpus",
        boundary_review_required=boundary_review,
    )


def _observed(
    identifier: str,
    *,
    baseline: bool,
    proposed: ConsensusOutcome,
    reasons: tuple[str, ...] = (),
    workflow_accepted: bool | None = None,
) -> ObservedCase:
    return ObservedCase(
        case_id=identifier,
        baseline_accepted=baseline,
        baseline_reason_codes=(),
        consensus=_consensus(proposed, reasons),
        insertion_count=0,
        deletion_count=0,
        substitution_count=0,
        character_error_count=0,
        exact_token_match=True,
        baseline_name_number_correct=True,
        proposed_name_number_correct=True,
        hallucinated_on_silence=False,
        baseline_boundary_errors_ms=(20.0, 30.0),
        proposed_boundary_errors_ms=(10.0, 20.0),
        baseline_crop_contaminated=False,
        proposed_crop_contaminated=False,
        baseline_clipped_word=False,
        proposed_clipped_word=False,
        workflow_accepted=(
            proposed is ConsensusOutcome.ACCEPTED
            if workflow_accepted is None
            else workflow_accepted
        ),
        workflow_evidence_fingerprint=SHA_C,
        cold_runtime_seconds=0.5,
        warm_runtime_seconds=0.25,
        peak_memory_bytes=1_000,
        model_load_count=1,
        model_switch_seconds=0.1,
        batch_size=1,
    )


def _evaluation_inputs() -> tuple[
    EvaluationCorpus, tuple[ObservedCase, ...], EvaluationProtocol
]:
    corpus = EvaluationCorpus(
        1,
        "en",
        (
            _case("clean", CorpusTruth.CLEAN, group="clean", boundary_review=True),
            _case("known-bad", CorpusTruth.DEFECTIVE, group="bad-one"),
            _case("caught-bad", CorpusTruth.DEFECTIVE, group="bad-two"),
        ),
    )
    observations = (
        _observed("clean", baseline=True, proposed=ConsensusOutcome.ACCEPTED),
        _observed("known-bad", baseline=False, proposed=ConsensusOutcome.REJECTED),
        _observed(
            "caught-bad",
            baseline=True,
            proposed=ConsensusOutcome.REJECTED,
            reasons=("persistent_valid_dissent",),
        ),
    )
    protocol = EvaluationProtocol(
        version=1,
        partition=CorpusPartition.HELD_OUT,
        confidence_level=0.95,
        maximum_false_accept_cluster_rate=0.9,
        maximum_false_rejection_increase=0,
        maximum_crop_contamination_increase=0,
        maximum_clipped_word_increase=0,
        require_forced_boundary_improvement=True,
        mandatory_rejection_risks=("lexical",),
        name_number_risks=("lexical",),
        minimum_defect_clusters_by_risk=(("lexical", 1),),
    )
    return corpus, observations, protocol


def test_evaluation_is_paired_sliced_schema_valid_and_order_stable() -> None:
    corpus, observations, protocol = _evaluation_inputs()

    first = evaluate_shadow_corpus(corpus, observations, protocol=protocol)
    second = evaluate_shadow_corpus(
        corpus, tuple(reversed(observations)), protocol=protocol
    )

    assert first == second
    assert first.passed is True
    overall = first.slices[0]
    assert overall.baseline_false_accepts == 1
    assert overall.candidate_false_accepts == 0
    assert overall.workflow_false_accepts == 0
    assert overall.additional_defects_caught == 1
    assert overall.new_false_accepts == 0
    assert overall.defect_cluster_count == 2
    assert overall.candidate_false_accept_cluster_count == 0
    assert overall.candidate_false_accept_cluster_upper_bound <= 0.9
    assert overall.workflow_false_accept_cluster_count == 0
    assert overall.workflow_false_accept_cluster_upper_bound <= 0.9
    assert overall.word_error_rate == 0
    assert overall.baseline_boundary_median_absolute_error_ms == 25
    assert overall.proposed_boundary_median_absolute_error_ms == 15
    assert overall.baseline_boundary_p95_error_ms == 30
    assert overall.proposed_boundary_p95_error_ms == 20
    assert overall.warm_real_time_factor == 0.25
    assert overall.peak_memory_bytes == 1_000
    Draft202012Validator(load_schema("speech-analysis-evaluation")).validate(
        first.to_dict()
    )


def test_semantic_evaluation_ignores_runtime_memory_and_input_report_order() -> None:
    corpus, observations, protocol = _evaluation_inputs()
    first = evaluate_shadow_corpus(corpus, observations, protocol=protocol)
    changed_observations = tuple(
        replace(
            item,
            cold_runtime_seconds=9.0,
            warm_runtime_seconds=8.0,
            peak_memory_bytes=9_999,
            model_load_count=7,
            model_switch_seconds=6.0,
            batch_size=5,
        )
        for item in reversed(observations)
    )
    reordered_corpus = replace(corpus, cases=tuple(reversed(corpus.cases)))

    second = evaluate_shadow_corpus(
        reordered_corpus,
        changed_observations,
        protocol=protocol,
    )

    assert reordered_corpus.fingerprint == corpus.fingerprint
    assert second.fingerprint == first.fingerprint
    assert second.passed is first.passed
    assert tuple(item.case.case_id for item in second.cases) == tuple(
        item.case.case_id for item in first.cases
    )
    assert tuple(item.shadow for item in second.cases) == tuple(
        item.shadow for item in first.cases
    )
    assert second.slices != first.slices


def test_new_false_accept_fails_noninferiority() -> None:
    corpus, observations, protocol = _evaluation_inputs()
    changed = tuple(
        replace(item, consensus=_consensus(ConsensusOutcome.ACCEPTED))
        if item.case_id == "known-bad"
        else item
        for item in observations
    )

    evaluation = evaluate_shadow_corpus(corpus, changed, protocol=protocol)

    assert evaluation.passed is False
    assert evaluation.candidate_false_acceptance_safe is False
    assert evaluation.slices[0].new_false_accepts == 1


def test_baseline_false_accept_cannot_hide_candidate_false_accept() -> None:
    corpus, observations, protocol = _evaluation_inputs()
    changed = tuple(
        replace(
            item,
            consensus=_consensus(ConsensusOutcome.ACCEPTED),
            workflow_accepted=False,
        )
        if item.case_id == "caught-bad"
        else item
        for item in observations
    )

    evaluation = evaluate_shadow_corpus(corpus, changed, protocol=protocol)

    assert evaluation.passed is False
    assert evaluation.candidate_false_acceptance_safe is False
    assert evaluation.workflow_false_acceptance_safe is True
    assert evaluation.slices[0].new_false_accepts == 0


def test_complete_candidate_workflow_must_meet_false_accept_bound() -> None:
    corpus, observations, protocol = _evaluation_inputs()
    changed = tuple(
        replace(item, workflow_accepted=True) if item.case_id == "caught-bad" else item
        for item in observations
    )

    evaluation = evaluate_shadow_corpus(corpus, changed, protocol=protocol)

    assert evaluation.passed is False
    assert evaluation.candidate_false_acceptance_safe is True
    assert evaluation.workflow_false_acceptance_safe is False
    assert evaluation.mandatory_defects_rejected is False


def test_name_number_accuracy_may_not_regress() -> None:
    corpus, observations, protocol = _evaluation_inputs()
    changed = tuple(
        replace(item, proposed_name_number_correct=False)
        if item.case_id == "clean"
        else item
        for item in observations
    )

    evaluation = evaluate_shadow_corpus(corpus, changed, protocol=protocol)

    assert evaluation.passed is False
    assert evaluation.name_number_noninferior is False


def test_forced_boundary_median_and_p95_must_both_improve() -> None:
    corpus, observations, protocol = _evaluation_inputs()
    changed = tuple(
        replace(item, proposed_boundary_errors_ms=(20.0, 30.0))
        if item.case_id == "clean"
        else item
        for item in observations
    )

    evaluation = evaluate_shadow_corpus(corpus, changed, protocol=protocol)

    assert evaluation.passed is False
    assert evaluation.forced_boundary_improved is False


def test_crop_contamination_and_clipped_words_may_not_increase() -> None:
    corpus, observations, protocol = _evaluation_inputs()
    changed = tuple(
        replace(item, proposed_crop_contaminated=True)
        if item.case_id == "clean"
        else item
        for item in observations
    )

    evaluation = evaluate_shadow_corpus(corpus, changed, protocol=protocol)

    assert evaluation.passed is False
    assert evaluation.crop_safety_noninferior is False


def test_required_paired_measurements_cannot_be_omitted() -> None:
    corpus, observations, protocol = _evaluation_inputs()
    missing_names = tuple(
        replace(item, baseline_name_number_correct=None)
        if item.case_id == "clean"
        else item
        for item in observations
    )

    with pytest.raises(ValidationError, match="paired baseline and proposed truth"):
        evaluate_shadow_corpus(corpus, missing_names, protocol=protocol)

    missing_boundaries = tuple(
        replace(item, baseline_boundary_errors_ms=())
        if item.case_id == "clean"
        else item
        for item in observations
    )

    with pytest.raises(ValidationError, match="paired baseline and proposed errors"):
        evaluate_shadow_corpus(corpus, missing_boundaries, protocol=protocol)


def test_risk_minimum_counts_independent_source_and_voice_clusters() -> None:
    first = _case("bad-one", CorpusTruth.DEFECTIVE, group="shared")
    second = replace(first, case_id="bad-two")
    corpus = EvaluationCorpus(1, "en", (first, second))
    observations = (
        _observed("bad-one", baseline=False, proposed=ConsensusOutcome.REJECTED),
        _observed("bad-two", baseline=False, proposed=ConsensusOutcome.REJECTED),
    )
    protocol = EvaluationProtocol(
        version=1,
        partition=CorpusPartition.HELD_OUT,
        confidence_level=0.95,
        maximum_false_accept_cluster_rate=0.9,
        maximum_false_rejection_increase=0,
        maximum_crop_contamination_increase=0,
        maximum_clipped_word_increase=0,
        require_forced_boundary_improvement=True,
        mandatory_rejection_risks=("lexical",),
        name_number_risks=("lexical",),
        minimum_defect_clusters_by_risk=(("lexical", 2),),
    )

    with pytest.raises(ValidationError, match="lacks preregistered samples"):
        evaluate_shadow_corpus(corpus, observations, protocol=protocol)


def test_protocol_must_name_every_observed_defect_risk() -> None:
    corpus, observations, protocol = _evaluation_inputs()
    incomplete = replace(
        protocol,
        mandatory_rejection_risks=("different_risk",),
        name_number_risks=("different_risk",),
        minimum_defect_clusters_by_risk=(("different_risk", 1),),
    )

    with pytest.raises(ValidationError, match="every defective risk class"):
        evaluate_shadow_corpus(corpus, observations, protocol=incomplete)


def test_default_protocol_preregisters_derived_cluster_minimums() -> None:
    protocol = load_default_evaluation_protocol()

    assert protocol.partition is CorpusPartition.HELD_OUT
    assert minimum_zero_failure_clusters(0.95, 0.05) == 52
    assert len(protocol.minimum_defect_clusters_by_risk) == 12
    assert {count for _name, count in protocol.minimum_defect_clusters_by_risk} == {52}
    assert protocol.mandatory_rejection_risks == (
        "clipped_boundary",
        "damaged_join",
        "extra_syllable",
        "isolated_word",
    )
    assert protocol.name_number_risks == ("number_code", "proper_name")
    assert protocol.require_forced_boundary_improvement is True
    assert load_default_evaluation_protocol().fingerprint == protocol.fingerprint


def test_protocol_loader_rejects_unregistered_fields(tmp_path: Path) -> None:
    path = tmp_path / "protocol.toml"
    path.write_text(
        """version = 1
partition = "held_out"
confidence_level = 0.95
maximum_false_accept_cluster_rate = 0.05
maximum_false_rejection_increase = 0
maximum_crop_contamination_increase = 0
maximum_clipped_word_increase = 0
require_forced_boundary_improvement = true
mandatory_rejection_risks = ["lexical"]
name_number_risks = ["lexical"]
unexpected = true
[minimum_defect_clusters_by_risk]
lexical = 52
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="fields are invalid"):
        load_evaluation_protocol(path)


def test_calibration_approval_requires_acceptance_dissent_and_two_boundary_passes() -> (
    None
):
    corpus, observations, protocol = _evaluation_inputs()
    evaluation = evaluate_shadow_corpus(corpus, observations, protocol=protocol)
    table = CalibrationTable(
        1,
        "en",
        corpus.fingerprint,
        False,
        None,
        (SHA_C,),
        (CalibrationThreshold("whisper", ClipClass.SHORT_PHRASE, SHA_C),),
    )
    dispositions = (
        ReviewerDisposition(
            "clean",
            POLICY,
            SHA_A,
            1,
            ReviewKind.AUTOMATIC_ACCEPTANCE,
            ReviewDecision.APPROVED,
        ),
        ReviewerDisposition(
            "clean",
            POLICY,
            SHA_A,
            1,
            ReviewKind.BOUNDARY,
            ReviewDecision.APPROVED,
        ),
        ReviewerDisposition(
            "clean",
            POLICY,
            SHA_A,
            2,
            ReviewKind.BOUNDARY,
            ReviewDecision.APPROVED,
        ),
        ReviewerDisposition(
            "caught-bad",
            POLICY,
            SHA_B,
            1,
            ReviewKind.VALID_DISSENT,
            ReviewDecision.APPROVED,
        ),
    )

    with pytest.raises(ValidationError, match="boundary review"):
        approve_calibration(table, evaluation, dispositions[:2] + dispositions[3:])

    approved = approve_calibration(table, evaluation, dispositions)

    assert approved.approved is True
    assert approved.reviewer_disposition_fingerprint is not None
    assert approved.fingerprint == table.fingerprint
    assert approved.disposition_fingerprint != table.disposition_fingerprint


def test_review_template_is_randomized_bound_and_approves_calibration(
    tmp_path: Path,
) -> None:
    corpus, observations, protocol = _evaluation_inputs()
    evaluation = evaluate_shadow_corpus(corpus, observations, protocol=protocol)
    path = tmp_path / "qualification-review.toml"
    write_qualification_review_template(
        path,
        evaluation,
        randomization_seed="private qualification seed",
        timestamp="2026-08-14T00:00:00+00:00",
    )

    pending = load_qualification_review(path, evaluation=evaluation)
    content = path.read_text(encoding="utf-8")

    assert pending.status is QualificationReviewStatus.PENDING
    assert len(pending.entries) == 4
    assert "private qualification seed" not in content
    assert "expected_text" not in content
    assert (
        qualification_review_template(
            evaluation,
            randomization_seed="private qualification seed",
            timestamp="2026-08-14T00:00:00+00:00",
        )
        == content
    )
    Draft202012Validator(load_schema("speech-qualification-review")).validate(
        pending.to_dict()
    )

    identity = reviewer_fingerprint("A. Listener")
    completed = content.replace('status = "pending"', 'status = "approved"').replace(
        'reviewer_fingerprint = ""\ndecision = "pending"',
        f'reviewer_fingerprint = "{identity}"\ndecision = "approved"',
    )
    path.write_text(completed, encoding="utf-8")
    review = load_qualification_review(path, evaluation=evaluation)
    table = CalibrationTable(
        1,
        "en",
        corpus.fingerprint,
        False,
        None,
        (SHA_C,),
        (CalibrationThreshold("whisper", ClipClass.SHORT_PHRASE, SHA_C),),
    )

    approved = approve_calibration_review(table, evaluation, review)

    assert review.status is QualificationReviewStatus.APPROVED
    assert approved.approved is True
    assert approved.reviewer_disposition_fingerprint == review.fingerprint


def test_qualification_review_fingerprint_ignores_timestamp_and_entry_order(
    tmp_path: Path,
) -> None:
    corpus, observations, protocol = _evaluation_inputs()
    evaluation = evaluate_shadow_corpus(corpus, observations, protocol=protocol)
    path = tmp_path / "qualification-review.toml"
    path.write_text(
        qualification_review_template(
            evaluation,
            randomization_seed="private seed",
            timestamp="2026-08-14T00:00:00+00:00",
        ),
        encoding="utf-8",
    )
    review = load_qualification_review(path, evaluation=evaluation)

    changed = replace(
        review,
        timestamp="2026-08-15T00:00:00+00:00",
        entries=tuple(reversed(review.entries)),
    )

    assert changed.fingerprint == review.fingerprint


def test_review_loader_rejects_stale_evaluation(tmp_path: Path) -> None:
    corpus, observations, protocol = _evaluation_inputs()
    evaluation = evaluate_shadow_corpus(corpus, observations, protocol=protocol)
    path = tmp_path / "qualification-review.toml"
    content = qualification_review_template(
        evaluation,
        randomization_seed="private qualification seed",
    ).replace(evaluation.fingerprint, "e" * 64, 1)
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValidationError, match="does not match the evaluation"):
        load_qualification_review(path, evaluation=evaluation)


def test_pending_review_cannot_approve_calibration(tmp_path: Path) -> None:
    corpus, observations, protocol = _evaluation_inputs()
    evaluation = evaluate_shadow_corpus(corpus, observations, protocol=protocol)
    path = tmp_path / "qualification-review.toml"
    path.write_text(
        qualification_review_template(evaluation, randomization_seed="seed"),
        encoding="utf-8",
    )
    pending = load_qualification_review(path, evaluation=evaluation)
    table = CalibrationTable(
        1,
        "en",
        corpus.fingerprint,
        False,
        None,
        (SHA_C,),
        (CalibrationThreshold("whisper", ClipClass.SHORT_PHRASE, SHA_C),),
    )

    with pytest.raises(ValidationError, match="not approved"):
        approve_calibration_review(table, evaluation, pending)


def test_corpus_split_is_grouped_by_passage_and_voice() -> None:
    held_out = _case("held", CorpusTruth.CLEAN, group="shared")
    calibration = replace(
        held_out,
        case_id="calibration",
        partition=CorpusPartition.CALIBRATION,
    )

    with pytest.raises(ValidationError, match="cannot cross"):
        EvaluationCorpus(1, "en", (held_out, calibration))


def test_versioned_corpus_loader_verifies_rights_audio_and_redacts_text() -> None:
    corpus = load_evaluation_corpus(
        ROOT / "tests/fixtures/speech-analysis-corpus-v1.toml",
        repository_root=ROOT,
    )

    assert corpus.language == "en"
    assert len(corpus.cases) == 1
    case = corpus.cases[0]
    assert case.expected_token_count == 68
    assert case.sample_rate == 24_000
    assert case.gender == "female"
    assert case.boundary_review_required is True
    assert "expected_text" not in {item.name for item in fields(case)}
