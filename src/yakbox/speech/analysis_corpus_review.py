"""Internal, text-safe CLI for qualification-corpus transcript review."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from yakbox._files import sha256_file
from yakbox.errors import ValidationError
from yakbox.speech.analysis_corpus_sources import load_corpus_source_inventory
from yakbox.speech.analysis_corpus_transcripts import write_corpus_transcript_review
from yakbox.speech.analysis_corpus_truth import (
    approve_corpus_transcript_case,
    corpus_transcript_review_progress,
    load_corpus_transcript_review_draft,
)

_MAXIMUM_REVIEWER_LABEL_BYTES = 4_096
_MAXIMUM_TRANSCRIPT_BYTES = 65_536


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review qualification transcripts without editing JSON"
    )
    parser.add_argument("--authoring", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show text-free review progress")
    packet = commands.add_parser(
        "packet",
        help="Write a reviewer-friendly Markdown packet",
    )
    packet.add_argument("--output", type=Path, required=True)
    packet.add_argument("--audio-prefix", required=True)
    approve = commands.add_parser(
        "approve",
        help="Approve one transcript after listening to its linked WAV",
    )
    approve.add_argument("source_window_id")
    approve.add_argument("--reviewer-label-file", type=Path, required=True)
    approve.add_argument(
        "--accepted-text-file",
        type=Path,
        help="Required for dissent; omit to approve the prefilled proposal",
    )
    approve.add_argument("--expected-authoring-digest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded transcript-review operation."""
    arguments = _parser().parse_args(argv)
    try:
        inventory = load_corpus_source_inventory(
            arguments.inventory,
            audio_root=arguments.audio_root,
        )
        packet_path: Path | None = None
        if arguments.command == "status":
            progress = corpus_transcript_review_progress(
                arguments.authoring,
                inventory=inventory,
            )
        elif arguments.command == "packet":
            draft = load_corpus_transcript_review_draft(
                arguments.authoring,
                inventory=inventory,
            )
            write_corpus_transcript_review(
                arguments.output,
                draft,
                audio_prefix=arguments.audio_prefix,
            )
            packet_path = arguments.output.resolve()
            progress = corpus_transcript_review_progress(
                arguments.authoring,
                inventory=inventory,
            )
        else:
            accepted_text = (
                _read_private_text(arguments.accepted_text_file)
                if arguments.accepted_text_file is not None
                else None
            )
            progress = approve_corpus_transcript_case(
                arguments.authoring,
                inventory=inventory,
                source_window_id=arguments.source_window_id,
                reviewer_label=_read_reviewer_label(arguments.reviewer_label_file),
                accepted_text=accepted_text,
                expected_authoring_digest=arguments.expected_authoring_digest,
            )
    except (OSError, UnicodeError, ValidationError) as error:
        sys.stderr.write(
            json.dumps(
                {"status": "error", "message": str(error)},
                sort_keys=True,
            )
            + "\n"
        )
        return 2
    report = progress.to_dict()
    report["authoring_digest"] = sha256_file(arguments.authoring)
    if packet_path is not None:
        report["review_packet"] = str(packet_path)
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0


def _read_reviewer_label(path: Path) -> str:
    if path.stat().st_size > _MAXIMUM_REVIEWER_LABEL_BYTES:
        raise ValidationError("Reviewer label exceeds 4096 UTF-8 bytes")
    value = path.read_text(encoding="utf-8").strip()
    if not value or len(value.encode("utf-8")) > _MAXIMUM_REVIEWER_LABEL_BYTES:
        raise ValidationError("Reviewer label must contain at most 4096 UTF-8 bytes")
    return value


def _read_private_text(path: Path) -> str:
    if path.stat().st_size > _MAXIMUM_TRANSCRIPT_BYTES:
        raise ValidationError("Transcript review text exceeds 65536 UTF-8 bytes")
    value = path.read_text(encoding="utf-8")
    if len(value.encode("utf-8")) > _MAXIMUM_TRANSCRIPT_BYTES:
        raise ValidationError("Transcript review text exceeds 65536 UTF-8 bytes")
    return value


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
