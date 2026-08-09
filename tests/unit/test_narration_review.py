from __future__ import annotations

import copy
import tomllib
from typing import cast

from tests.narration_review import narration_review_issues, narration_review_template

REPORT_SHA256 = "a" * 64
QA: dict[str, object] = {
    "profiles": ["voice-a", "voice-b"],
    "required_boundaries": ["sentence", "paragraph", "explicit_pause"],
    "manual": {
        "scale_min": 1,
        "scale_max": 5,
        "passing_score": 4,
        "chapter_profile": "voice-a",
        "voice_dimensions": [
            {"id": "narration_fit"},
            {"id": "clarity"},
        ],
        "dialogue_dimensions": [
            {"id": "speaker_differentiation"},
            {"id": "conflict_escalation"},
        ],
        "chapter_dimensions": [
            {"id": "pacing"},
            {"id": "continuity"},
        ],
    },
}


def test_pending_narration_review_template_is_complete_and_valid() -> None:
    content = narration_review_template(QA, report_sha256=REPORT_SHA256)
    review = cast(dict[str, object], tomllib.loads(content))

    assert review["status"] == "pending"
    assert not narration_review_issues(review, QA, report_sha256=REPORT_SHA256)
    assert set(cast(dict[str, object], review["voice_scores"])) == {
        "voice-a",
        "voice-b",
    }
    assert set(cast(dict[str, object], review["dialogue_scores"])) == {
        "voice-a",
        "voice-b",
    }


def test_approved_narration_review_passes_every_gate() -> None:
    review = _approved_review()

    assert not narration_review_issues(
        review,
        QA,
        report_sha256=REPORT_SHA256,
        require_approved=True,
    )


def test_approval_rejects_stale_report_low_score_and_failed_join() -> None:
    review = _approved_review()
    voices = cast(dict[str, dict[str, object]], review["voice_scores"])
    dialogue = cast(dict[str, dict[str, object]], review["dialogue_scores"])
    joins = cast(dict[str, object], review["join_observations"])
    voices["voice-b"]["clarity"] = 3
    dialogue["voice-b"]["speaker_differentiation"] = 3
    joins["paragraph"] = "fail"

    issues = narration_review_issues(
        review,
        QA,
        report_sha256="b" * 64,
        require_approved=True,
    )

    assert "report_sha256 does not match qa/report.json" in issues
    assert "voice_scores.voice-b.clarity must be at least 4 for approval" in issues
    assert (
        "dialogue_scores.voice-b.speaker_differentiation must be at least 4 "
        "for approval" in issues
    )
    assert "join_observations.paragraph must be pass" in issues


def test_pending_review_cannot_satisfy_approval_gate() -> None:
    review = cast(
        dict[str, object],
        tomllib.loads(narration_review_template(QA, report_sha256=REPORT_SHA256)),
    )

    issues = narration_review_issues(
        review,
        QA,
        report_sha256=REPORT_SHA256,
        require_approved=True,
    )

    assert "status must be approved" in issues


def test_pending_review_preserves_completed_scores_and_unscored_zeros() -> None:
    review = cast(
        dict[str, object],
        tomllib.loads(narration_review_template(QA, report_sha256=REPORT_SHA256)),
    )
    voices = cast(dict[str, dict[str, object]], review["voice_scores"])
    voices["voice-a"].update({"narration_fit": 5, "clarity": 0})
    voices["voice-b"].update({"narration_fit": 5, "clarity": 5})

    assert not narration_review_issues(review, QA, report_sha256=REPORT_SHA256)


def _approved_review() -> dict[str, object]:
    review = cast(
        dict[str, object],
        tomllib.loads(narration_review_template(QA, report_sha256=REPORT_SHA256)),
    )
    review.update(
        {
            "status": "approved",
            "reviewer": "A. Listener",
            "reviewed_at": "2026-08-01T06:00:00+00:00",
            "preferred_profile": "voice-a",
            "approved_settings": "Default Chatterbox settings",
        }
    )
    voices = cast(dict[str, dict[str, object]], review["voice_scores"])
    for scores in voices.values():
        scores.update({"narration_fit": 4, "clarity": 5})
    dialogue = cast(dict[str, dict[str, object]], review["dialogue_scores"])
    for scores in dialogue.values():
        scores.update({"speaker_differentiation": 4, "conflict_escalation": 5})
    chapter = cast(dict[str, object], review["chapter_scores"])
    chapter.update({"pacing": 4, "continuity": 4})
    joins = cast(dict[str, object], review["join_observations"])
    boundaries = cast(list[str], QA["required_boundaries"])
    joins.update(dict.fromkeys(boundaries, "pass"))
    return copy.deepcopy(review)
