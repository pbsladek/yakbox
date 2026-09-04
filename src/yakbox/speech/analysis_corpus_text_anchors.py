"""Qualify corpus transcripts against checksum-pinned source text."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path

from yakbox._files import atomic_write_bytes, atomic_write_json, safe_child, sha256_file
from yakbox.contracts import runtime_metadata
from yakbox.errors import ValidationError
from yakbox.speech.analysis_corpus_sources import load_corpus_source_inventory
from yakbox.speech.analysis_corpus_text_sources import (
    CorpusTextSourceInventory,
    load_corpus_text_source_inventory,
)
from yakbox.speech.analysis_corpus_transcripts import (
    CorpusTranscriptDraft,
    CorpusTranscriptDraftCase,
    TranscriptCandidate,
)
from yakbox.speech.analysis_corpus_truth import load_corpus_transcript_review_draft
from yakbox.speech.analysis_fingerprints import semantic_fingerprint, text_fingerprint
from yakbox.speech.normalization import NormalizationTrace, normalize_english

_SHA256_LENGTH = 64
_REASON_NO_AGREEMENT = "no-two-engine-agreement"
_REASON_NOT_FOUND = "agreed-text-not-found"
_REASON_AMBIGUOUS = "agreed-text-not-unique"
_REASON_CORROBORATION = "insufficient-model-corroboration"
_MATCHING_STRATEGY = "bounded-typography-v1"
_SINGLE_ENGINE_MAXIMUM_ERROR_RATE = 0.05
_SINGLE_ENGINE_MINIMUM_EDIT_ALLOWANCE = 2
_SINGLE_ENGINE_MINIMUM_TOKENS = 20


@dataclass(frozen=True, slots=True)
class CorpusTextAnchor:
    source_window_id: str
    source_passage_group: str
    voice: str
    reader: str
    relative_audio_path: str
    audio_digest: str
    draft_case_fingerprint: str
    source_text_fingerprint: str
    source_plain_digest: str
    source_token_start: int
    source_token_end: int
    accepted_text: str
    accepted_tokens: tuple[str, ...]
    accepted_tokens_hash: str
    evidence_tier: str
    matched_engines: tuple[str, ...]
    recognition_fingerprints: tuple[str, ...]
    maximum_candidate_edits: int

    def __post_init__(self) -> None:
        if (
            not self.source_window_id
            or not self.source_passage_group
            or not self.voice
            or not self.reader
            or not self.relative_audio_path
            or self.source_token_start < 0
            or self.source_token_end <= self.source_token_start
            or not self.accepted_text.strip()
            or not self.accepted_tokens
            or self.evidence_tier not in {"source-plus-one", "source-plus-multiple"}
            or not self.matched_engines
            or self.matched_engines != tuple(sorted(set(self.matched_engines)))
            or len(self.recognition_fingerprints) != len(self.matched_engines)
            or self.maximum_candidate_edits < 0
            or (len(self.matched_engines) == 1)
            != (self.evidence_tier == "source-plus-one")
        ):
            raise ValidationError("Corpus source-text anchor is inconsistent")
        for value in (
            self.audio_digest,
            self.draft_case_fingerprint,
            self.source_text_fingerprint,
            self.source_plain_digest,
            self.accepted_tokens_hash,
            *self.recognition_fingerprints,
        ):
            if len(value) != _SHA256_LENGTH or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValidationError("Corpus source-text anchor identity is invalid")
        if (
            self.source_token_end - self.source_token_start != len(self.accepted_tokens)
            or text_fingerprint("\u001f".join(self.accepted_tokens))
            != self.accepted_tokens_hash
        ):
            raise ValidationError("Corpus source-text anchor token identity differs")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-text-anchor-v2", self)


@dataclass(frozen=True, slots=True)
class UnresolvedTextAnchor:
    source_window_id: str
    source_passage_group: str
    voice: str
    reason: str
    draft_case_fingerprint: str

    def __post_init__(self) -> None:
        if self.reason not in {
            _REASON_NO_AGREEMENT,
            _REASON_NOT_FOUND,
            _REASON_AMBIGUOUS,
            _REASON_CORROBORATION,
        }:
            raise ValidationError("Corpus source-text anchor reason is invalid")


@dataclass(frozen=True, slots=True)
class CorpusTextAnchoring:
    source_inventory_fingerprint: str
    draft_fingerprint: str
    text_source_inventory_fingerprint: str
    anchors: tuple[CorpusTextAnchor, ...]
    unresolved: tuple[UnresolvedTextAnchor, ...]
    audit_case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        anchor_ids = tuple(item.source_window_id for item in self.anchors)
        unresolved_ids = tuple(item.source_window_id for item in self.unresolved)
        if (
            not (self.anchors or self.unresolved)
            or anchor_ids != tuple(sorted(set(anchor_ids)))
            or unresolved_ids != tuple(sorted(set(unresolved_ids)))
            or set(anchor_ids) & set(unresolved_ids)
            or not set(self.audit_case_ids) <= set(anchor_ids)
            or len(self.audit_case_ids) != len(set(self.audit_case_ids))
        ):
            raise ValidationError("Corpus source-text anchoring is inconsistent")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-text-anchoring-v2", self)

    @property
    def automatic_case_ids(self) -> tuple[str, ...]:
        audit = set(self.audit_case_ids)
        return tuple(
            item.source_window_id
            for item in self.anchors
            if item.source_window_id not in audit
        )

    def to_report_dict(self) -> dict[str, object]:
        """Serialize evidence without copying public-domain transcript text."""
        audit = set(self.audit_case_ids)
        return {
            **runtime_metadata("speech-corpus-text-anchors"),
            "fingerprint": self.fingerprint,
            "source_inventory_fingerprint": self.source_inventory_fingerprint,
            "draft_fingerprint": self.draft_fingerprint,
            "text_source_inventory_fingerprint": (
                self.text_source_inventory_fingerprint
            ),
            "matching_strategy": _MATCHING_STRATEGY,
            "case_count": len(self.anchors) + len(self.unresolved),
            "anchored_count": len(self.anchors),
            "automatic_count": len(self.automatic_case_ids),
            "audit_count": len(self.audit_case_ids),
            "unresolved_count": len(self.unresolved),
            "audit_case_ids": list(self.audit_case_ids),
            "anchors": [
                {
                    "source_window_id": item.source_window_id,
                    "source_passage_group": item.source_passage_group,
                    "voice": item.voice,
                    "audio_digest": item.audio_digest,
                    "draft_case_fingerprint": item.draft_case_fingerprint,
                    "source_text_fingerprint": item.source_text_fingerprint,
                    "source_plain_digest": item.source_plain_digest,
                    "source_token_start": item.source_token_start,
                    "source_token_end": item.source_token_end,
                    "accepted_tokens_hash": item.accepted_tokens_hash,
                    "accepted_token_count": len(item.accepted_tokens),
                    "evidence_tier": item.evidence_tier,
                    "matched_engines": list(item.matched_engines),
                    "recognition_fingerprints": list(item.recognition_fingerprints),
                    "audit_required": item.source_window_id in audit,
                    "maximum_candidate_edits": item.maximum_candidate_edits,
                    "fingerprint": item.fingerprint,
                }
                for item in self.anchors
            ],
            "unresolved": [
                {
                    "source_window_id": item.source_window_id,
                    "source_passage_group": item.source_passage_group,
                    "voice": item.voice,
                    "reason": item.reason,
                    "draft_case_fingerprint": item.draft_case_fingerprint,
                }
                for item in self.unresolved
            ],
        }


def build_corpus_text_anchoring(
    draft: CorpusTranscriptDraft,
    text_sources: CorpusTextSourceInventory,
    *,
    text_root: Path,
    audit_size: int = 10,
) -> CorpusTextAnchoring:
    """Require two recognizers to match one unique source-text span exactly."""
    if audit_size < 0:
        raise ValidationError("Corpus source-text audit size cannot be negative")
    sources = {item.voice: item for item in text_sources.sources}
    if not {item.voice for item in draft.cases} <= set(sources):
        raise ValidationError("Corpus source texts do not cover every draft voice")
    traces: dict[str, tuple[str, NormalizationTrace]] = {}
    anchors: list[CorpusTextAnchor] = []
    unresolved: list[UnresolvedTextAnchor] = []
    for case in draft.cases:
        source = sources[case.voice]
        cached = traces.get(case.voice)
        if cached is None:
            plain_path = safe_child(text_root, text_root / source.relative_plain_path)
            if (
                plain_path.is_symlink()
                or not plain_path.is_file()
                or sha256_file(plain_path) != source.plain_digest
            ):
                raise ValidationError("Corpus source-text file identity differs")
            try:
                plain = plain_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                raise ValidationError(
                    "Corpus source text is not valid UTF-8"
                ) from error
            trace = normalize_english(plain)
            cached = (plain, trace)
            traces[case.voice] = cached
        plain, trace_value = cached
        trace = trace_value
        anchor, reason = _anchor_case(
            case,
            source.fingerprint,
            source.plain_digest,
            plain,
            trace,
        )
        if anchor is None:
            unresolved.append(
                UnresolvedTextAnchor(
                    case.source_window_id,
                    case.source_passage_group,
                    case.voice,
                    reason,
                    case.fingerprint,
                )
            )
        else:
            anchors.append(anchor)
    ordered = tuple(sorted(anchors, key=lambda item: item.source_window_id))
    audits = _select_audit_cases(
        ordered,
        size=min(audit_size, len(ordered)),
        seed=semantic_fingerprint(
            "speech-corpus-text-anchor-audit-seed-v1",
            (draft.fingerprint, text_sources.fingerprint),
        ),
    )
    return CorpusTextAnchoring(
        draft.source_inventory_fingerprint,
        draft.fingerprint,
        text_sources.fingerprint,
        ordered,
        tuple(sorted(unresolved, key=lambda item: item.source_window_id)),
        audits,
    )


def write_corpus_text_anchor_report(path: Path, anchoring: CorpusTextAnchoring) -> None:
    atomic_write_json(path, anchoring.to_report_dict())


def corpus_text_anchor_review_markdown(
    draft: CorpusTranscriptDraft,
    anchoring: CorpusTextAnchoring,
    *,
    audio_prefix: str,
) -> str:
    """Render only unresolved cases and a bounded source-anchor audit."""
    if draft.fingerprint != anchoring.draft_fingerprint:
        raise ValidationError("Corpus source-text review draft identity differs")
    prefix = audio_prefix.strip("/")
    if not prefix or ".." in Path(prefix).parts:
        raise ValidationError("Corpus source-text review audio prefix is invalid")
    cases = {item.source_window_id: item for item in draft.cases}
    anchors = {item.source_window_id: item for item in anchoring.anchors}
    review_ids = (
        *anchoring.audit_case_ids,
        *(item.source_window_id for item in anchoring.unresolved),
    )
    by_voice: dict[str, list[str]] = defaultdict(list)
    for case_id in review_ids:
        by_voice[cases[case_id].voice].append(case_id)
    lines = [
        "# Source-anchored transcript review",
        "",
        f"Anchoring fingerprint: `{anchoring.fingerprint}`",
        "",
        (
            f"Yakbox uniquely anchored {len(anchoring.anchors)} of "
            f"{len(draft.cases)} clips to checksum-pinned source text. "
            f"{len(anchoring.automatic_case_ids)} require no listening."
        ),
        "",
        (
            f"Review only these {len(review_ids)} clips: "
            f"{len(anchoring.unresolved)} exceptions and "
            f"{len(anchoring.audit_case_ids)} deterministic audit samples."
        ),
        "",
        (
            "For an audit clip, confirm that the source proposal exactly matches "
            "the audio."
        ),
        "For an exception, write the exact spoken words or mark it `reselect`.",
    ]
    unresolved_by_id = {item.source_window_id: item for item in anchoring.unresolved}
    for voice in sorted(by_voice):
        voice_cases = by_voice[voice]
        lines.extend(
            (
                "",
                f"## {cases[voice_cases[0]].reader}",
                "",
                f"Voice key: `{voice}`",
            )
        )
        for case_id in sorted(voice_cases):
            case = cases[case_id]
            audio = f"{prefix}/{case.relative_audio_path}"
            if case_id in anchors:
                kind = "source-anchor audit"
                proposal = anchors[case_id].accepted_text
                detail = (
                    "Two or more recognizers matched this one unique span in the "
                    "pinned source text. Confirm the proposal is exact."
                )
            else:
                kind = "manual exception"
                proposal = case.proposed_text or "[none]"
                reason = unresolved_by_id[case_id].reason
                detail = f"Automatic qualification stopped: `{reason}`."
            lines.extend(
                (
                    "",
                    f"### {case.source_passage_group}",
                    "",
                    f"Clip ID: `{case_id}`",
                    "",
                    f"Review type: **{kind}**",
                    "",
                    f"[Play {case.reader}]({audio})",
                    "",
                    detail,
                    "",
                    f"Proposed transcript: `{proposal}`",
                    "",
                    "Model transcripts:",
                    "",
                )
            )
            lines.extend(
                f"- {candidate.engine}: `{candidate.text or '[no tokens]'}`"
                for candidate in case.candidates
            )
            lines.extend(
                (
                    "",
                    (f"Decision: `{case_id}: approve`, corrected text, or `reselect`."),
                )
            )
    return "\n".join(lines) + "\n"


def write_corpus_text_anchor_review(
    path: Path,
    draft: CorpusTranscriptDraft,
    anchoring: CorpusTextAnchoring,
    *,
    audio_prefix: str,
) -> None:
    content = corpus_text_anchor_review_markdown(
        draft,
        anchoring,
        audio_prefix=audio_prefix,
    )
    atomic_write_bytes(path, content.encode(), overwrite=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Build source anchors and the reduced human-review packet."""
    parser = argparse.ArgumentParser(
        description="Anchor corpus transcripts to checksum-pinned source text"
    )
    parser.add_argument("--authoring", type=Path, required=True)
    parser.add_argument("--audio-inventory", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--text-inventory", type=Path, required=True)
    parser.add_argument("--text-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--audio-prefix", required=True)
    parser.add_argument("--audit-size", type=int, default=10)
    arguments = parser.parse_args(argv)
    try:
        inventory = load_corpus_source_inventory(
            arguments.audio_inventory,
            audio_root=arguments.audio_root,
        )
        draft = load_corpus_transcript_review_draft(
            arguments.authoring,
            inventory=inventory,
        )
        text_sources = load_corpus_text_source_inventory(
            arguments.text_inventory,
            text_root=arguments.text_root,
        )
        anchoring = build_corpus_text_anchoring(
            draft,
            text_sources,
            text_root=arguments.text_root,
            audit_size=arguments.audit_size,
        )
        write_corpus_text_anchor_report(arguments.report_output, anchoring)
        write_corpus_text_anchor_review(
            arguments.review_output,
            draft,
            anchoring,
            audio_prefix=arguments.audio_prefix,
        )
    except (OSError, UnicodeError, ValidationError) as error:
        sys.stderr.write(
            json.dumps({"status": "error", "message": str(error)}, sort_keys=True)
            + "\n"
        )
        return 2
    sys.stdout.write(
        json.dumps(
            {
                "status": "ok",
                "fingerprint": anchoring.fingerprint,
                "case_count": len(draft.cases),
                "anchored_count": len(anchoring.anchors),
                "automatic_count": len(anchoring.automatic_case_ids),
                "audit_count": len(anchoring.audit_case_ids),
                "unresolved_count": len(anchoring.unresolved),
                "review_count": (
                    len(anchoring.audit_case_ids) + len(anchoring.unresolved)
                ),
                "report": str(arguments.report_output.resolve()),
                "review_packet": str(arguments.review_output.resolve()),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def _anchor_case(
    case: CorpusTranscriptDraftCase,
    source_text_fingerprint: str,
    source_plain_digest: str,
    plain: str,
    trace: NormalizationTrace,
) -> tuple[CorpusTextAnchor | None, str]:
    source_tokens, source_projection = _typographic_tokens(
        tuple(item.text for item in trace.tokens)
    )
    grouped: dict[tuple[str, ...], list[TranscriptCandidate]] = defaultdict(list)
    for candidate in case.candidates:
        if candidate.eligible:
            match_tokens, _projection = _typographic_tokens(candidate.tokens)
            grouped[match_tokens].append(candidate)
    source_match, reason = _unique_source_match(
        grouped,
        source_tokens=source_tokens,
        source_projection=source_projection,
    )
    if source_match is None:
        return None, reason
    tokens, candidates, match_start = source_match
    match_end = match_start + len(tokens)
    token_start = source_projection[match_start]
    token_end = source_projection[match_end - 1] + 1
    accepted_tokens = tuple(item.text for item in trace.tokens[token_start:token_end])
    source_start = trace.tokens[token_start].source_start
    source_end = trace.tokens[token_end - 1].source_end
    matched = tuple(sorted(candidates, key=lambda item: item.engine))
    candidate_edits = tuple(
        _edit_distance(tokens, _typographic_tokens(candidate.tokens)[0])
        for candidate in case.candidates
        if candidate.eligible
    )
    maximum_edits = max(candidate_edits, default=0)
    if len(matched) == 1 and not _single_engine_is_corroborated(
        tokens,
        candidate_edits=candidate_edits,
        candidate_count=len(case.candidates),
    ):
        return None, _REASON_CORROBORATION
    return (
        CorpusTextAnchor(
            case.source_window_id,
            case.source_passage_group,
            case.voice,
            case.reader,
            case.relative_audio_path,
            case.audio_digest,
            case.fingerprint,
            source_text_fingerprint,
            source_plain_digest,
            token_start,
            token_end,
            plain[source_start:source_end],
            accepted_tokens,
            text_fingerprint("\u001f".join(accepted_tokens)),
            "source-plus-one" if len(matched) == 1 else "source-plus-multiple",
            tuple(item.engine for item in matched),
            tuple(item.recognition_fingerprint for item in matched),
            maximum_edits,
        ),
        "",
    )


def _unique_source_match(
    grouped: dict[tuple[str, ...], list[TranscriptCandidate]],
    *,
    source_tokens: tuple[str, ...],
    source_projection: tuple[int, ...],
) -> tuple[
    tuple[tuple[str, ...], list[TranscriptCandidate], int] | None,
    str,
]:
    if not grouped:
        return None, _REASON_NO_AGREEMENT
    source_matches: list[tuple[tuple[str, ...], list[TranscriptCandidate], int]] = []
    ambiguous = False
    for tokens, candidates in grouped.items():
        positions = _subsequence_positions(
            source_tokens,
            tokens,
            projection=source_projection,
            limit=2,
        )
        if len(positions) > 1:
            ambiguous = True
        elif len(positions) == 1:
            source_matches.append((tokens, candidates, positions[0]))
    if not source_matches:
        if ambiguous:
            return None, _REASON_AMBIGUOUS
        return None, _REASON_NOT_FOUND
    spans = {
        (position, position + len(tokens))
        for tokens, _candidates, position in source_matches
    }
    if len(spans) != 1 or ambiguous:
        return None, _REASON_AMBIGUOUS
    return (
        max(
            source_matches,
            key=lambda item: (len(item[1]), item[0]),
        ),
        "",
    )


def _single_engine_is_corroborated(
    tokens: tuple[str, ...],
    *,
    candidate_edits: tuple[int, ...],
    candidate_count: int,
) -> bool:
    if len(tokens) < _SINGLE_ENGINE_MINIMUM_TOKENS:
        return False
    allowance = max(
        _SINGLE_ENGINE_MINIMUM_EDIT_ALLOWANCE,
        ceil(len(tokens) * _SINGLE_ENGINE_MAXIMUM_ERROR_RATE),
    )
    return (
        len(candidate_edits) == candidate_count
        and max(candidate_edits, default=0) <= allowance
    )


def _subsequence_positions(
    source: tuple[str, ...],
    sought: tuple[str, ...],
    *,
    projection: tuple[int, ...],
    limit: int,
) -> tuple[int, ...]:
    if not sought or len(sought) > len(source):
        return ()
    positions: list[int] = []
    first = sought[0]
    maximum = len(source) - len(sought) + 1
    for index in range(maximum):
        end = index + len(sought)
        starts_at_token = index == 0 or projection[index - 1] != projection[index]
        ends_at_token = end == len(source) or projection[end - 1] != projection[end]
        if (
            starts_at_token
            and ends_at_token
            and source[index] == first
            and source[index:end] == sought
        ):
            positions.append(index)
            if len(positions) >= limit:
                break
    return tuple(positions)


def _typographic_tokens(
    tokens: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    values: list[str] = []
    projection: list[int] = []
    for index, token in enumerate(tokens):
        pieces = tuple(piece for piece in token.split("-") if piece)
        values.extend(pieces)
        projection.extend([index] * len(pieces))
    return tuple(values), tuple(projection)


def _edit_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def _select_audit_cases(
    anchors: tuple[CorpusTextAnchor, ...],
    *,
    size: int,
    seed: str,
) -> tuple[str, ...]:
    ranked = sorted(
        anchors,
        key=lambda item: semantic_fingerprint(
            "speech-corpus-text-anchor-audit-rank-v1",
            (seed, item.source_window_id),
        ),
    )
    chosen: list[CorpusTextAnchor] = []
    tiers: set[str] = set()
    for item in ranked:
        if item.evidence_tier not in tiers:
            chosen.append(item)
            tiers.add(item.evidence_tier)
            if len(chosen) == size:
                return tuple(sorted(value.source_window_id for value in chosen))
    voices: set[str] = set()
    voices.update(item.voice for item in chosen)
    for item in ranked:
        if item not in chosen and item.voice not in voices:
            chosen.append(item)
            voices.add(item.voice)
            if len(chosen) == size:
                break
    if len(chosen) < size:
        selected = {item.source_window_id for item in chosen}
        chosen.extend(item for item in ranked if item.source_window_id not in selected)
    return tuple(sorted(item.source_window_id for item in chosen[:size]))


__all__ = [
    "CorpusTextAnchor",
    "CorpusTextAnchoring",
    "UnresolvedTextAnchor",
    "build_corpus_text_anchoring",
    "corpus_text_anchor_review_markdown",
    "write_corpus_text_anchor_report",
    "write_corpus_text_anchor_review",
]


if __name__ == "__main__":
    raise SystemExit(main())
