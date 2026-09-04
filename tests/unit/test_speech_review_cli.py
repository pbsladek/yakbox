from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import click
import pytest
from click.testing import CliRunner

from yakbox._files import sha256_file
from yakbox.cli_speech_review import register_speech_review_commands
from yakbox.speech.analysis_disposition import (
    BoundReviewFile,
    HumanReviewCandidate,
    HumanReviewStatus,
    HumanReviewStore,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _cli() -> click.Group:
    @click.group()
    def root() -> None:
        pass

    def emit(
        value: dict[str, object],
        message: str,
        *,
        status: str = "ok",
        exit_code: int = 0,
    ) -> None:
        del message
        click.echo(json.dumps({"status": status, "data": value}, sort_keys=True))
        if exit_code:
            raise click.exceptions.Exit(exit_code)

    def fail(error: Exception) -> NoReturn:
        raise click.ClickException(str(error))

    register_speech_review_commands(root, emit=emit, fail=fail, expose=True)
    return root


def _workspace(tmp_path: Path) -> tuple[Path, HumanReviewCandidate]:
    manifest = tmp_path / "yakbox.toml"
    manifest.write_text("schema_version = 2\n", encoding="utf-8")
    records = (
        ("context_audio", "context.wav", b"audio", SHA_A),
        ("spoken_text_plan", "text-plan.json", b"plan", SHA_A),
        ("policy", "policy.json", b"policy", SHA_B),
        ("analysis_evidence", "evidence.json", b"evidence", SHA_C),
    )
    bound: list[BoundReviewFile] = []
    for kind, name, content, fingerprint in records:
        path = tmp_path / "evidence" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        bound.append(
            BoundReviewFile(
                kind,
                path.relative_to(tmp_path).as_posix(),
                sha256_file(path),
                fingerprint,
            )
        )
    candidate = HumanReviewCandidate(
        "review-1",
        "2026-08-14T00:00:00+00:00",
        bound[0].sha256,
        SHA_A,
        SHA_B,
        SHA_B,
        (SHA_C,),
        ("persistent_valid_dissent",),
        True,
        (),
        tuple(bound),
    )
    store = HumanReviewStore(
        tmp_path / ".yakbox" / "speech-analysis" / "reviews",
        evidence_root=tmp_path,
    )
    store.register(candidate)
    return manifest, candidate


def test_gated_review_commands_emit_clean_bounded_json(tmp_path: Path) -> None:
    manifest, candidate = _workspace(tmp_path)
    notes = tmp_path / "notes.txt"
    notes.write_text("Approved after contextual listening.", encoding="utf-8")
    runner = CliRunner()

    listed = runner.invoke(_cli(), ["speech", "reviews", "list", str(manifest)])
    shown = runner.invoke(
        _cli(), ["speech", "reviews", "show", str(manifest), candidate.review_id]
    )
    resolved = runner.invoke(
        _cli(),
        [
            "speech",
            "reviews",
            "resolve",
            str(manifest),
            candidate.review_id,
            "--decision",
            "accept",
            "--notes-file",
            str(notes),
            "--reviewer",
            "reviewer-one",
        ],
    )

    assert listed.exit_code == shown.exit_code == resolved.exit_code == 0
    listed_json = json.loads(listed.output)
    shown_json = json.loads(shown.output)
    resolved_json = json.loads(resolved.output)
    assert listed_json["data"]["reviews"][0]["state"] == "pending"
    assert shown_json["data"]["reviews"][0]["review_id"] == candidate.review_id
    assert resolved_json["data"]["decision"] == "accept"
    assert "reviewer-one" not in resolved.output
    assert "manuscript" not in listed.output.casefold()


def test_resolve_rechecks_evidence_after_show(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, candidate = _workspace(tmp_path)
    notes = tmp_path / "notes.txt"
    notes.write_text("", encoding="utf-8")
    original = HumanReviewStore.show

    def mutate_after_show(self: HumanReviewStore, review_id: str) -> HumanReviewStatus:
        status = original(self, review_id)
        (tmp_path / candidate.bound_files[-1].relative_path).write_bytes(b"changed")
        return status

    monkeypatch.setattr(HumanReviewStore, "show", mutate_after_show)
    result = CliRunner().invoke(
        _cli(),
        [
            "speech",
            "reviews",
            "resolve",
            str(manifest),
            candidate.review_id,
            "--decision",
            "accept",
            "--notes-file",
            str(notes),
            "--reviewer",
            "reviewer-one",
        ],
    )

    assert result.exit_code != 0
    assert "evidence is stale" in result.output
    assert not (
        tmp_path
        / ".yakbox/speech-analysis/reviews/dispositions/review-1.disposition.json"
    ).exists()


def test_resolve_rejects_notes_over_byte_limit_before_write(tmp_path: Path) -> None:
    manifest, candidate = _workspace(tmp_path)
    notes = tmp_path / "notes.txt"
    notes.write_text("é" * 2_049, encoding="utf-8")

    result = CliRunner().invoke(
        _cli(),
        [
            "speech",
            "reviews",
            "resolve",
            str(manifest),
            candidate.review_id,
            "--decision",
            "accept",
            "--notes-file",
            str(notes),
            "--reviewer",
            "reviewer-one",
        ],
    )

    assert result.exit_code != 0
    assert "4096 UTF-8 bytes" in result.output
