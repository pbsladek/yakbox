from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import cast


def narration_review_template(qa: Mapping[str, object], *, report_sha256: str) -> str:
    """Return a pending TOML review bound to one technical QA report."""
    manual = cast(Mapping[str, object], qa["manual"])
    voice_dimensions = _dimension_ids(manual, "voice_dimensions")
    dialogue_dimensions = _optional_dimension_ids(manual, "dialogue_dimensions")
    chapter_dimensions = _dimension_ids(manual, "chapter_dimensions")
    profiles = cast(list[str], qa["profiles"])
    boundaries = cast(list[str], qa["required_boundaries"])
    lines = [
        "schema_version = 1",
        f"report_sha256 = {_toml_string(report_sha256)}",
        'status = "pending"',
        'reviewer = ""',
        'reviewed_at = ""',
        'preferred_profile = ""',
        'approved_settings = ""',
        'blocking_defects = ""',
    ]
    for profile in profiles:
        lines.extend(["", f"[voice_scores.{_toml_string(profile)}]"])
        lines.extend(f"{dimension} = 0" for dimension in voice_dimensions)
        lines.append('notes = ""')
    for profile in profiles if dialogue_dimensions else ():
        lines.extend(["", f"[dialogue_scores.{_toml_string(profile)}]"])
        lines.extend(f"{dimension} = 0" for dimension in dialogue_dimensions)
        lines.append('notes = ""')
    lines.extend(
        [
            "",
            "[chapter_scores]",
            f"profile = {_toml_string(str(manual['chapter_profile']))}",
        ]
    )
    lines.extend(f"{dimension} = 0" for dimension in chapter_dimensions)
    lines.extend(['notes = ""', "", "[join_observations]"])
    lines.extend(f'{boundary} = "pending"' for boundary in boundaries)
    lines.extend(['notes = ""', ""])
    return "\n".join(lines)


def narration_review_issues(
    review: Mapping[str, object],
    qa: Mapping[str, object],
    *,
    report_sha256: str,
    require_approved: bool = False,
) -> tuple[str, ...]:
    """Return all structural and approval defects in a narration review."""
    issues: list[str] = []
    status = review.get("status")
    if review.get("schema_version") != 1:
        issues.append("schema_version must be 1")
    if review.get("report_sha256") != report_sha256:
        issues.append("report_sha256 does not match qa/report.json")
    if status not in {"pending", "approved", "rejected"}:
        issues.append("status must be pending, approved, or rejected")
        return tuple(issues)
    if require_approved and status != "approved":
        issues.append("status must be approved")

    manual = _mapping(review=qa, key="manual", issues=issues)
    profiles = _string_list(qa.get("profiles"), field="profiles", issues=issues)
    voice_dimensions = _dimension_ids_checked(manual, "voice_dimensions", issues=issues)
    dialogue_dimensions = _optional_dimension_ids_checked(
        manual, "dialogue_dimensions", issues=issues
    )
    chapter_dimensions = _dimension_ids_checked(
        manual, "chapter_dimensions", issues=issues
    )
    boundaries = _string_list(
        qa.get("required_boundaries"), field="required_boundaries", issues=issues
    )
    scale_min = _integer(manual.get("scale_min"), field="scale_min", issues=issues)
    scale_max = _integer(manual.get("scale_max"), field="scale_max", issues=issues)
    passing = _integer(
        manual.get("passing_score"), field="passing_score", issues=issues
    )
    _validate_metadata(review, profiles, str(status), issues)
    _validate_profile_scores(
        review,
        profiles,
        voice_dimensions,
        score_field="voice_scores",
        status=str(status),
        scale=(scale_min, scale_max),
        passing=passing,
        issues=issues,
    )
    if dialogue_dimensions:
        _validate_profile_scores(
            review,
            profiles,
            dialogue_dimensions,
            score_field="dialogue_scores",
            status=str(status),
            scale=(scale_min, scale_max),
            passing=passing,
            issues=issues,
        )
    _validate_chapter_scores(
        review,
        chapter_profile=manual.get("chapter_profile"),
        dimensions=chapter_dimensions,
        status=str(status),
        scale=(scale_min, scale_max),
        passing=passing,
        issues=issues,
    )
    _validate_join_observations(review, boundaries, status=str(status), issues=issues)
    return tuple(issues)


def _validate_metadata(
    review: Mapping[str, object],
    profiles: tuple[str, ...],
    status: str,
    issues: list[str],
) -> None:
    reviewer = review.get("reviewer")
    reviewed_at = review.get("reviewed_at")
    preferred = review.get("preferred_profile")
    settings = review.get("approved_settings")
    defects = review.get("blocking_defects")
    for field, value in (
        ("reviewer", reviewer),
        ("reviewed_at", reviewed_at),
        ("preferred_profile", preferred),
        ("approved_settings", settings),
        ("blocking_defects", defects),
    ):
        if not isinstance(value, str):
            issues.append(f"{field} must be a string")
    if status == "pending":
        return
    _validate_completed_metadata(
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        preferred=preferred,
        settings=settings,
        defects=defects,
        profiles=profiles,
        status=status,
        issues=issues,
    )


def _validate_completed_metadata(
    *,
    reviewer: object,
    reviewed_at: object,
    preferred: object,
    settings: object,
    defects: object,
    profiles: tuple[str, ...],
    status: str,
    issues: list[str],
) -> None:
    if not _nonempty_string(reviewer):
        issues.append("reviewer is required after review")
    if not _timezone_datetime(reviewed_at):
        issues.append("reviewed_at must be an ISO 8601 timestamp with a timezone")
    if status == "approved":
        if preferred not in profiles:
            issues.append("preferred_profile must name a configured profile")
        if not _nonempty_string(settings):
            issues.append("approved_settings is required for approval")
        if _nonempty_string(defects):
            issues.append("blocking_defects must be empty for approval")
    elif not _nonempty_string(defects):
        issues.append("blocking_defects is required for rejection")


def _validate_profile_scores(
    review: Mapping[str, object],
    profiles: tuple[str, ...],
    dimensions: tuple[str, ...],
    *,
    score_field: str,
    status: str,
    scale: tuple[int, int],
    passing: int,
    issues: list[str],
) -> None:
    scores = _mapping(review=review, key=score_field, issues=issues)
    _exact_keys(scores, set(profiles), field=score_field, issues=issues)
    expected = {*dimensions, "notes"}
    for profile in profiles:
        section = _mapping(review=scores, key=profile, issues=issues)
        prefix = f"{score_field}.{profile}"
        _exact_keys(section, expected, field=prefix, issues=issues)
        _validate_scores(
            section,
            dimensions,
            field=prefix,
            status=status,
            scale=scale,
            passing=passing,
            issues=issues,
        )
        if not isinstance(section.get("notes"), str):
            issues.append(f"{prefix}.notes must be a string")


def _validate_chapter_scores(
    review: Mapping[str, object],
    *,
    chapter_profile: object,
    dimensions: tuple[str, ...],
    status: str,
    scale: tuple[int, int],
    passing: int,
    issues: list[str],
) -> None:
    section = _mapping(review=review, key="chapter_scores", issues=issues)
    _exact_keys(
        section,
        {*dimensions, "profile", "notes"},
        field="chapter_scores",
        issues=issues,
    )
    if section.get("profile") != chapter_profile:
        issues.append("chapter_scores.profile must match manual.chapter_profile")
    _validate_scores(
        section,
        dimensions,
        field="chapter_scores",
        status=status,
        scale=scale,
        passing=passing,
        issues=issues,
    )
    if not isinstance(section.get("notes"), str):
        issues.append("chapter_scores.notes must be a string")


def _validate_join_observations(
    review: Mapping[str, object],
    boundaries: tuple[str, ...],
    *,
    status: str,
    issues: list[str],
) -> None:
    section = _mapping(review=review, key="join_observations", issues=issues)
    _exact_keys(
        section,
        {*boundaries, "notes"},
        field="join_observations",
        issues=issues,
    )
    allowed = {"pending"} if status == "pending" else {"pass", "fail"}
    if status == "approved":
        allowed = {"pass"}
    for boundary in boundaries:
        if section.get(boundary) not in allowed:
            choices = " or ".join(sorted(allowed))
            issues.append(f"join_observations.{boundary} must be {choices}")
    if not isinstance(section.get("notes"), str):
        issues.append("join_observations.notes must be a string")


def _validate_scores(
    section: Mapping[str, object],
    dimensions: tuple[str, ...],
    *,
    field: str,
    status: str,
    scale: tuple[int, int],
    passing: int,
    issues: list[str],
) -> None:
    minimum, maximum = scale
    for dimension in dimensions:
        value = section.get(dimension)
        label = f"{field}.{dimension}"
        if not isinstance(value, int) or isinstance(value, bool):
            issues.append(f"{label} must be an integer")
        elif status == "pending" and value != 0 and not minimum <= value <= maximum:
            issues.append(
                f"{label} must be 0 or between {minimum} and {maximum} while pending"
            )
        elif status != "pending" and not minimum <= value <= maximum:
            issues.append(f"{label} must be between {minimum} and {maximum}")
        elif status == "approved" and value < passing:
            issues.append(f"{label} must be at least {passing} for approval")


def _dimension_ids(manual: Mapping[str, object], field: str) -> tuple[str, ...]:
    dimensions = cast(list[Mapping[str, object]], manual[field])
    return tuple(str(item["id"]) for item in dimensions)


def _optional_dimension_ids(
    manual: Mapping[str, object], field: str
) -> tuple[str, ...]:
    return _dimension_ids(manual, field) if field in manual else ()


def _dimension_ids_checked(
    manual: Mapping[str, object], field: str, *, issues: list[str]
) -> tuple[str, ...]:
    value = manual.get(field)
    if not isinstance(value, list):
        issues.append(f"manual.{field} must be a list")
        return ()
    identifiers: list[str] = []
    for index, item in enumerate(value):
        identifier = item.get("id") if isinstance(item, dict) else None
        if not _nonempty_string(identifier):
            issues.append(f"manual.{field}[{index}].id must be a non-empty string")
        else:
            identifiers.append(cast(str, identifier))
    if len(set(identifiers)) != len(identifiers):
        issues.append(f"manual.{field} contains duplicate ids")
    return tuple(identifiers)


def _optional_dimension_ids_checked(
    manual: Mapping[str, object], field: str, *, issues: list[str]
) -> tuple[str, ...]:
    if field not in manual:
        return ()
    return _dimension_ids_checked(manual, field, issues=issues)


def _mapping(
    *, review: Mapping[str, object], key: str, issues: list[str]
) -> Mapping[str, object]:
    value = review.get(key)
    if not isinstance(value, dict):
        issues.append(f"{key} must be a table")
        return {}
    return cast(Mapping[str, object], value)


def _string_list(value: object, *, field: str, issues: list[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(_nonempty_string(item) for item in value):
        issues.append(f"{field} must be a list of non-empty strings")
        return ()
    result = cast(tuple[str, ...], tuple(value))
    if len(set(result)) != len(result):
        issues.append(f"{field} must not contain duplicates")
    return result


def _integer(value: object, *, field: str, issues: list[str]) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        issues.append(f"manual.{field} must be an integer")
        return 0
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    field: str,
    issues: list[str],
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        issues.append(f"{field} is missing: {', '.join(missing)}")
    if extra:
        issues.append(f"{field} has unexpected fields: {', '.join(extra)}")


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _timezone_datetime(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)
