from __future__ import annotations

import importlib.util
import json
import tomllib
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from tests.schema_helpers import validate_contract

from yakbox.cli import main
from yakbox.cloud import Page
from yakbox.speech import AudioFormat, SpeechArtifact


def test_help_does_not_import_torch() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "build" in result.output
    assert "chatterbox" not in result.output.casefold()


def test_whisper_calibration_cli_emits_versioned_metrics() -> None:
    result = CliRunner().invoke(main, ["--json", "whisper", "calibrate"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["data"]["false_accepts"] == 0
    assert payload["data"]["false_rejects"] == 0
    validate_contract("cli-output", payload)
    validate_contract("whisper-calibration", payload["data"])


def test_whisper_and_short_review_commands_are_discoverable() -> None:
    runner = CliRunner()

    whisper = runner.invoke(main, ["whisper", "--help"])
    review = runner.invoke(main, ["short-review", "--help"])

    assert whisper.exit_code == 0
    assert {
        "inspect",
        "reinspect",
        "verify-manuscript",
        "inspect-joins",
        "models",
        "calibrate",
    }.issubset(set(whisper.output.split()))
    assert review.exit_code == 0
    assert {"list", "play", "approve", "reject"}.issubset(set(review.output.split()))


def test_localized_repair_cli_generates_approves_and_explains(
    book_workspace: Path,
) -> None:
    runner = CliRunner()
    manifest = book_workspace / "yakbox.toml"
    built = runner.invoke(
        main,
        ["--json", "build", str(manifest), "--through", "synthesize"],
    )
    assert built.exit_code == 0, built.output
    planned = runner.invoke(main, ["--json", "plan", str(manifest)])
    plan_data = json.loads(planned.output)["data"]
    synthesis = next(
        node for node in plan_data["nodes"] if node["stage"] == "synthesize"
    )
    chunk_id = synthesis["chunks"][0]["id"]

    generated = runner.invoke(
        main,
        [
            "--json",
            "repair",
            "generate",
            str(manifest),
            "--chunk-id",
            chunk_id,
            "--mode",
            "target-only",
            "--takes",
            "2",
            "--no-whisper",
        ],
    )
    assert generated.exit_code == 0, generated.output
    session = json.loads(generated.output)["data"]
    validate_contract("audiobook-repair-session", session)
    assert len(session["takes"]) == 2

    approved = runner.invoke(
        main,
        [
            "--json",
            "repair",
            "approve",
            session["repair_id"],
            str(manifest),
            "--take",
            "1",
            "--no-rebuild",
        ],
    )
    assert approved.exit_code == 0, approved.output
    explained = runner.invoke(
        main,
        [
            "--json",
            "artifacts",
            "cache",
            "why-miss",
            chunk_id,
            str(manifest),
        ],
    )
    assert explained.exit_code == 0, explained.output
    assert json.loads(explained.output)["data"]["status"] == "approved_repair"

    located = runner.invoke(
        main,
        [
            "--json",
            "repair",
            "locate",
            synthesis["chapter_id"],
            "0",
            str(manifest),
        ],
    )
    assert located.exit_code == 0, located.output
    validate_contract("audiobook-repair-location", json.loads(located.output)["data"])


def test_new_whisper_qa_commands_emit_observable_json_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "chapter.wav"
    audio.write_bytes(b"placeholder")
    manuscript = tmp_path / "chapter.md"
    manuscript.write_text("# One\n\nWren asked.\n", encoding="utf-8")
    join_spec = tmp_path / "joins.yaml"
    join_spec.write_text("joins:\n  - at_seconds: 1.0\n", encoding="utf-8")

    async def fake_inspection(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return SimpleNamespace(
            accepted=True,
            reason_codes=(),
            to_dict=lambda: {"accepted": True, "kind": "targeted"},
        )

    async def fake_manuscript(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return SimpleNamespace(
            accepted=True,
            matched_token_count=2,
            expected_token_count=2,
            mismatches=(),
            to_dict=lambda: {"accepted": True, "kind": "manuscript"},
        )

    async def fake_joins(*args: object, **kwargs: object) -> object:
        del args, kwargs
        join = SimpleNamespace(accepted=True)
        return SimpleNamespace(
            accepted=True,
            joins=(join,),
            to_dict=lambda: {"accepted": True, "kind": "joins"},
        )

    monkeypatch.setattr("yakbox.cli_whisper.inspect_with_whisper", fake_inspection)
    monkeypatch.setattr("yakbox.cli_whisper.verify_manuscript", fake_manuscript)
    monkeypatch.setattr("yakbox.cli_whisper.inspect_joins", fake_joins)
    runner = CliRunner()

    targeted = runner.invoke(
        main,
        [
            "--json",
            "whisper",
            "reinspect",
            str(audio),
            "--expected",
            "Wren asked.",
            "--start",
            "0.5",
            "--end",
            "1.5",
        ],
    )
    verified = runner.invoke(
        main,
        [
            "--json",
            "whisper",
            "verify-manuscript",
            str(audio),
            str(manuscript),
        ],
    )
    joins = runner.invoke(
        main,
        [
            "--json",
            "whisper",
            "inspect-joins",
            str(audio),
            "--spec",
            str(join_spec),
        ],
    )

    for result, kind in (
        (targeted, "targeted"),
        (verified, "manuscript"),
        (joins, "joins"),
    ):
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["data"]["kind"] == kind
        validate_contract("cli-output", payload)


def test_whisper_join_spec_rejects_json_configuration(tmp_path: Path) -> None:
    audio = tmp_path / "chapter.wav"
    audio.write_bytes(b"placeholder")
    join_spec = tmp_path / "joins.json"
    join_spec.write_text('{"joins": [{"at_seconds": 1.0}]}', encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "--json",
            "whisper",
            "inspect-joins",
            str(audio),
            "--spec",
            str(join_spec),
        ],
    )

    assert result.exit_code == 1
    assert "must use .yaml or .yml" in json.loads(result.output)["error"]["message"]


def test_json_usage_errors_use_stable_envelope_and_exit_two() -> None:
    result = CliRunner().invoke(
        main,
        ["--json", "build", "--from", "not-a-stage"],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 2
    assert payload["error"]["code"] == "invalid_argument"
    assert payload["command"] == "build"
    validate_contract("cli-output", payload)


def test_ctrl_c_uses_exit_130_in_human_and_json_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "script.txt"
    script.write_text("Tiny line.\n", encoding="utf-8")
    monkeypatch.setenv("RESEMBLE_API_KEY", "test")

    def interrupted(coroutine: Coroutine[object, object, object]) -> object:
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr("yakbox.cli.asyncio.run", interrupted)
    human = CliRunner().invoke(main, ["cloud", "batch", str(script)])
    assert human.exit_code == 130
    assert "Aborted!" in human.output

    machine = CliRunner().invoke(
        main,
        ["--json", "cloud", "batch", str(script)],
    )
    assert machine.exit_code == 130
    payload = json.loads(machine.output)
    assert payload["status"] == "aborted"
    assert payload["exit_code"] == 130
    validate_contract("cli-output", payload)


def test_init_validate_plan_and_doctor_json(tmp_path: Path) -> None:
    runner = CliRunner()
    workspace = tmp_path / "book"
    initialized = runner.invoke(main, ["init", str(workspace)])
    assert initialized.exit_code == 0, initialized.output

    validated = runner.invoke(main, ["validate", str(workspace / "yakbox.toml")])
    assert validated.exit_code == 0, validated.output

    planned = runner.invoke(main, ["--json", "plan", str(workspace / "yakbox.toml")])
    assert planned.exit_code == 0, planned.output
    payload = json.loads(planned.output)
    assert payload["schema_version"] == 1
    assert payload["data"]["nodes"]
    validate_contract("cli-output", payload)
    validate_contract("audiobook-plan", payload["data"])

    doctor = runner.invoke(main, ["--json", "doctor", str(workspace / "yakbox.toml")])
    assert doctor.exit_code == 0, doctor.output
    report = json.loads(doctor.output)
    assert report["data"]["healthy"] is True
    validate_contract("cli-output", report)
    validate_contract("doctor-report", report["data"])


def test_validate_and_plan_report_character_routes_and_attribution(
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.md"
    source.write_text(
        "# One\n\nNarration remains on the narrator profile.\n\n"
        "<!-- yakbox:speech:speaker name=wren -->\n\n"
        '"Mara, we need to leave before the signal returns," Wren said.\n\n'
        '"No."\n',
        encoding="utf-8",
    )
    manifest = tmp_path / "yakbox.toml"
    manifest.write_text(
        '"$schema" = "https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"\n'
        'schema_version = 1\nsources = ["book.md"]\n'
        '[book]\ntitle = "Routed"\n'
        '[voices.narrator]\ndisplay_name = "Narrator"\n'
        '[voices.wren]\ndisplay_name = "Wren (male)"\n'
        '[profiles.narrator]\nbackend = "fake"\nvoice = "narrator"\n'
        '[profiles.wren]\nbackend = "fake"\nvoice = "wren"\n'
        '[characters.narrator]\nprofile = "narrator"\n'
        '[characters.wren]\nprofile = "wren"\n'
        '[targets.default]\nprofile = "narrator"\n',
        encoding="utf-8",
    )
    runner = CliRunner()

    validated = runner.invoke(main, ["--json", "validate", str(manifest)])
    assert validated.exit_code == 0, validated.output
    validate_data = json.loads(validated.output)["data"]
    assert validate_data["attribution_finding_count"] == 1
    assert validate_data["attribution_findings"][0]["code"] == "unrouted-dialogue"

    planned = runner.invoke(main, ["--json", "plan", str(manifest)])
    assert planned.exit_code == 0, planned.output
    plan_data = json.loads(planned.output)["data"]
    chunks = plan_data["nodes"][0]["chunks"]
    assert [chunk["speaker"] for chunk in chunks] == [
        "narrator",
        "wren",
        "narrator",
        "narrator",
    ]
    assert chunks[1]["profile"] == "wren"


def test_plan_strips_middle_attribution_tag_when_configured(tmp_path: Path) -> None:
    source = tmp_path / "book.md"
    source.write_text(
        "# One\n\n"
        "<!-- yakbox:speech:speaker name=wren -->\n\n"
        '"What could be doing that?" Wren asked. "Some kind of magic?"\n',
        encoding="utf-8",
    )
    manifest = tmp_path / "yakbox.toml"
    manifest.write_text(
        '"$schema" = "https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"\n'
        'schema_version = 1\nsources = ["book.md"]\n'
        '[book]\ntitle = "Stripped Tags"\n'
        '[voices.narrator]\ndisplay_name = "Narrator"\n'
        '[voices.wren]\ndisplay_name = "Wren"\n'
        '[profiles.narrator]\nbackend = "fake"\nvoice = "narrator"\n'
        '[profiles.wren]\nbackend = "fake"\nvoice = "wren"\n'
        '[characters.narrator]\nprofile = "narrator"\n'
        '[characters.wren]\nprofile = "wren"\n'
        "[dialogue]\nstrip_attribution_tags = true\n"
        '[targets.default]\nprofile = "narrator"\n',
        encoding="utf-8",
    )

    planned = CliRunner().invoke(main, ["--json", "plan", str(manifest)])

    assert planned.exit_code == 0, planned.output
    chunks = json.loads(planned.output)["data"]["nodes"][0]["chunks"]
    assert [chunk["speaker"] for chunk in chunks] == ["wren"]


def test_dialogue_route_suggestions_require_review_before_use(tmp_path: Path) -> None:
    source = tmp_path / "book.md"
    source.write_text(
        '# One\n\n"Ready?" Wren asked. "Now," Wren snapped.\n\n"No."\n',
        encoding="utf-8",
    )
    manifest = tmp_path / "yakbox.toml"
    manifest.write_text(
        '"$schema" = "https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"\n'
        'schema_version = 1\nsources = ["book.md"]\n'
        '[book]\ntitle = "Route Review"\n'
        '[voices.narrator]\ndisplay_name = "Narrator"\n'
        '[voices.wren]\ndisplay_name = "Wren"\n'
        '[profiles.narrator]\nbackend = "fake"\nvoice = "narrator"\n'
        '[profiles.wren]\nbackend = "fake"\nvoice = "wren"\n'
        '[characters.narrator]\nprofile = "narrator"\n'
        '[characters.wren]\nprofile = "wren"\n'
        '[targets.default]\nprofile = "narrator"\n',
        encoding="utf-8",
    )
    runner = CliRunner()

    suggested = runner.invoke(
        main,
        ["--json", "dialogue", "routes", "suggest", str(manifest)],
    )

    assert suggested.exit_code == 0, suggested.output
    suggested_data = json.loads(suggested.output)["data"]
    assert suggested_data["suggested_routes"] == 1
    assert suggested_data["unresolved_dialogue_paragraphs"] == 1
    routes = tmp_path / "dialogue-routes.toml"
    route_data = tomllib.loads(routes.read_text(encoding="utf-8"))
    validate_contract("dialogue-routes", route_data)
    assert route_data["routes"][0]["status"] == "suggested"

    manifest.write_text(
        manifest.read_text(encoding="utf-8")
        + '\n[dialogue]\nroutes = "dialogue-routes.toml"\n'
        + "strip_attribution_tags = true\n"
        + 'expressive_tag_handling = "strip"\n',
        encoding="utf-8",
    )
    pending = runner.invoke(
        main,
        ["--json", "dialogue", "routes", "check", str(manifest)],
    )
    assert pending.exit_code == 1
    assert "require review" in json.loads(pending.output)["error"]["message"]

    routes.write_text(
        routes.read_text(encoding="utf-8").replace(
            'status = "suggested"',
            'status = "approved"',
        ),
        encoding="utf-8",
    )
    checked = runner.invoke(
        main,
        ["--json", "dialogue", "routes", "check", str(manifest)],
    )
    assert checked.exit_code == 0, checked.output
    assert json.loads(checked.output)["data"]["ready"] is True

    previewed = runner.invoke(
        main,
        ["--json", "dialogue", "preview", str(manifest)],
    )
    assert previewed.exit_code == 0, previewed.output
    preview = json.loads((tmp_path / "dialogue-preview.json").read_text())
    validate_contract("dialogue-transformation", preview)
    assert preview["paragraph_count"] == 1
    assert preview["stripped_tag_count"] == 2
    assert preview["paragraphs"][0]["spans"] == [
        {"speaker": "wren", "speaker_explicit": True, "spoken": "Ready? Now."}
    ]
    assert preview["paragraphs"][0]["stripped_tags"] == [
        {"kind": "pure", "text": "Wren asked."},
        {"kind": "expressive", "text": "Wren snapped."},
    ]
    refused = runner.invoke(
        main,
        ["--json", "dialogue", "preview", str(manifest)],
    )
    assert refused.exit_code == 1
    forced = runner.invoke(
        main,
        ["--json", "dialogue", "preview", str(manifest), "--force"],
    )
    assert forced.exit_code == 0, forced.output


def test_build_dry_run_and_preview_are_audiobook_first(tmp_path: Path) -> None:
    runner = CliRunner()
    workspace = tmp_path / "book"
    assert runner.invoke(main, ["init", str(workspace)]).exit_code == 0
    manifest = workspace / "yakbox.toml"

    planned = runner.invoke(
        main,
        ["--json", "build", str(manifest), "--dry-run"],
    )
    assert planned.exit_code == 0, planned.output
    build_payload = json.loads(planned.output)
    assert build_payload["exit_code"] == 0
    assert build_payload["data"]["status"] == "planned"
    assert build_payload["data"]["preflight"]["pending_nodes"] == 4
    assert not (workspace / ".yakbox").exists()
    validate_contract("cli-output", build_payload)

    draft = runner.invoke(
        main,
        ["--json", "build", str(manifest), "--mode", "draft", "--dry-run"],
    )
    assert draft.exit_code == 0, draft.output
    draft_payload = json.loads(draft.output)
    assert draft_payload["data"]["target"] == "draft"
    assert draft_payload["data"]["preflight"]["planned_nodes"] == 1

    preview = runner.invoke(
        main,
        ["--json", "preview", str(manifest), "--text", "A tiny preview."],
    )
    assert preview.exit_code == 0, preview.output
    preview_payload = json.loads(preview.output)
    artifact = preview_payload["data"]["artifact"]
    assert artifact["kind"] == "preview"
    assert "previews" in Path(artifact["path"]).parts
    validate_contract("cli-output", preview_payload)
    validate_contract("audiobook-artifact", artifact)


def test_cloud_batch_confirmation_happens_before_credentials_or_network(
    tmp_path: Path,
) -> None:
    script = tmp_path / "script.txt"
    script.write_text("Hello hosted world.\n", encoding="utf-8")
    output = tmp_path / "output"
    runner = CliRunner()

    refused = runner.invoke(
        main,
        [
            "cloud",
            "batch",
            str(script),
            "--out-dir",
            str(output),
            "--voice-uuid",
            "voice",
            "--confirm-above-characters",
            "0",
        ],
    )
    assert refused.exit_code == 1
    assert "--yes" in refused.output
    assert not output.exists()

    dry_run = runner.invoke(
        main,
        [
            "--json",
            "cloud",
            "batch",
            str(script),
            "--out-dir",
            str(output),
            "--voice-uuid",
            "voice",
            "--confirm-above-characters",
            "0",
            "--dry-run",
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    payload = json.loads(dry_run.output)
    assert payload["data"]["preflight"]["logical_items"] == 1
    assert payload["data"]["results"][0]["status"] == "not_run"
    assert not output.exists()
    validate_contract("cli-output", payload)
    validate_contract("batch-report", payload["data"])


def test_cloud_batch_json_reports_partial_failure_and_effective_exit_code(
    tmp_path: Path,
) -> None:
    script = tmp_path / "script.txt"
    script.write_text("x" * 3_001, encoding="utf-8")
    runner = CliRunner()
    common = [
        "--json",
        "cloud",
        "batch",
        str(script),
        "--voice-uuid",
        "voice",
        "--dry-run",
        "--no-report",
    ]

    failed = runner.invoke(main, common)
    assert failed.exit_code == 1, failed.output
    failed_payload = json.loads(failed.output)
    assert failed_payload["status"] == "partial_failure"
    assert failed_payload["exit_code"] == 1
    assert failed_payload["data"]["summary"]["failed"] == 1
    validate_contract("cli-output", failed_payload)

    ignored = runner.invoke(main, [*common, "--ignore-errors"])
    assert ignored.exit_code == 0, ignored.output
    ignored_payload = json.loads(ignored.output)
    assert ignored_payload["status"] == "partial_failure"
    assert ignored_payload["exit_code"] == 0
    assert ignored_payload["data"]["summary"]["failed"] == 1
    validate_contract("cli-output", ignored_payload)


def test_cloud_profile_reads_optional_keyring_without_exposing_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESEMBLE_API_KEY", raising=False)
    monkeypatch.setenv("YAKBOX_CONFIG", str(tmp_path / "missing.toml"))
    captured: list[str] = []
    keyring = SimpleNamespace(
        get_password=lambda service, profile: (
            "keyring-secret"
            if (service, profile) == ("yakbox/resemble", "studio")
            else None
        )
    )
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        "yakbox.cli.importlib.util.find_spec",
        lambda name: object() if name == "keyring" else real_find_spec(name),
    )
    monkeypatch.setattr(
        "yakbox.cli.importlib.import_module",
        lambda name: keyring if name == "keyring" else __import__(name),
    )

    async def list_projects(api_key: str, page: int, page_size: int) -> Page[object]:
        captured.append(api_key)
        assert (page, page_size) == (1, 10)
        return Page(items=(), page=1, page_count=1, total_results=0)

    monkeypatch.setattr("yakbox.cli._list_projects", list_projects)
    result = CliRunner().invoke(
        main,
        ["--json", "cloud", "--profile", "studio", "projects", "list"],
    )

    assert result.exit_code == 0, result.output
    assert captured == ["keyring-secret"]
    assert "keyring-secret" not in result.output


def test_generic_hosted_tts_uses_keyring_before_legacy_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[cloud]\napi_key = "legacy-secret"\n', encoding="utf-8")
    monkeypatch.delenv("RESEMBLE_API_KEY", raising=False)
    monkeypatch.setenv("YAKBOX_CONFIG", str(config))
    monkeypatch.setattr(
        "yakbox.cli._keyring_password",
        lambda profile: "keyring-secret" if profile == "default" else None,
    )
    captured: list[str | None] = []

    async def direct_tts(
        *_args: object, **kwargs: object
    ) -> tuple[SpeechArtifact, None]:
        captured.append(
            kwargs["api_key"] if isinstance(kwargs["api_key"], str) else None
        )
        return (
            SpeechArtifact(
                path=tmp_path / "speech.wav",
                backend="cloud",
                voice="narrator",
                output_format=AudioFormat.WAV,
                bytes_written=1,
                sha256="digest",
            ),
            None,
        )

    monkeypatch.setattr("yakbox.cli._direct_tts", direct_tts)
    result = CliRunner().invoke(
        main,
        ["--json", "tts", "hello", "--backend", "cloud", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert captured == ["keyring-secret"]
    assert "keyring-secret" not in result.output
    assert "legacy-secret" not in result.output


def test_local_batch_rejects_hosted_backends_before_execution(tmp_path: Path) -> None:
    script = tmp_path / "script.txt"
    script.write_text("A billable line.\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["--json", "batch", str(script), "--backend", "cloud"],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "usage_error"
    assert "local-only" in payload["error"]["message"]


def test_conflicting_output_modes_and_empty_verify_are_usage_errors() -> None:
    modes = CliRunner().invoke(main, ["--json", "--quiet", "--verbose", "models"])
    verify = CliRunner().invoke(main, ["--json", "verify"])

    assert modes.exit_code == 2
    assert json.loads(modes.output)["error"]["code"] == "usage_error"
    assert verify.exit_code == 2
    assert json.loads(verify.output)["error"]["code"] == "missing_parameter"


def test_purge_all_requires_an_explicit_scope() -> None:
    missing_scope = CliRunner().invoke(
        main,
        ["--json", "artifacts", "trash", "purge", "--yes"],
    )
    conflicting_scope = CliRunner().invoke(
        main,
        [
            "--json",
            "artifacts",
            "trash",
            "purge",
            "cleanup-1",
            "--all",
            "--yes",
        ],
    )

    assert missing_scope.exit_code == 2
    assert conflicting_scope.exit_code == 2
    assert "CLEANUP_ID or --all" in json.loads(missing_scope.output)["error"]["message"]


def test_inspect_uses_the_selected_target(monkeypatch: pytest.MonkeyPatch) -> None:
    selected: list[str] = []
    manifest = SimpleNamespace(
        target=lambda name: (
            selected.append(name) or SimpleNamespace(output_root=Path("unused"))
        )
    )
    monkeypatch.setattr("yakbox.cli.load_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        "yakbox.cli.inventory_artifacts",
        lambda _root: SimpleNamespace(records=()),
    )

    result = CliRunner().invoke(
        main,
        ["--json", "inspect", "book.toml", "--target", "release"],
    )

    assert result.exit_code == 0, result.output
    assert selected == ["release"]


def test_unexpected_json_failures_do_not_expose_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_name: str) -> object:
        raise RuntimeError("sensitive implementation detail")

    monkeypatch.setattr("yakbox.cli.importlib.util.find_spec", fail)
    result = CliRunner().invoke(main, ["--json", "models"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == {
        "code": "internal_error",
        "message": "An unexpected internal error occurred",
    }
    assert "sensitive" not in result.output


def test_report_failures_publish_the_effective_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unhealthy_doctor(*_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(healthy=False, to_dict=lambda: {"diagnostics": []})

    monkeypatch.setattr("yakbox.cli.run_doctor", unhealthy_doctor)
    doctor = CliRunner().invoke(main, ["--json", "doctor"])

    manifest = SimpleNamespace(
        root=tmp_path,
        sources=(),
        pronunciations=None,
        max_pause_ms=30_000,
        dialogue=SimpleNamespace(
            strip_attribution_tags=False,
            routes=None,
            expressive_tag_handling="context",
            retain_first_attribution_per_scene=False,
        ),
        target=lambda _name: SimpleNamespace(output_root=tmp_path),
    )
    monkeypatch.setattr("yakbox.cli.load_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        "yakbox.cli.check_release",
        lambda *_args, **_kwargs: SimpleNamespace(
            complete=False,
            issues=("missing release artifact",),
            to_dict=lambda: {"complete": False, "issues": ["missing release artifact"]},
        ),
    )
    release = CliRunner().invoke(main, ["--json", "release", "check"])

    monkeypatch.setattr(
        "yakbox.cli.audit_pronunciations",
        lambda *_args, **_kwargs: SimpleNamespace(
            rules=(object(),),
            unused_rules=1,
            shadowed_matches=0,
            to_dict=lambda **_kwargs: {"unused_rules": 1},
        ),
    )
    pronunciations = CliRunner().invoke(
        main,
        ["--json", "pronunciations", "audit", "--fail-unused"],
    )

    record = SimpleNamespace(path=tmp_path / "artifact.wav")
    monkeypatch.setattr(
        "yakbox.cli.inventory_artifacts",
        lambda _root: SimpleNamespace(records=(record,)),
    )
    monkeypatch.setattr("yakbox.cli.verify_artifact", lambda _record: (False, "bad"))
    artifacts = CliRunner().invoke(main, ["--json", "artifacts", "verify"])

    for result in (doctor, release, pronunciations, artifacts):
        assert result.exit_code == 1, result.output
        payload = json.loads(result.output)
        assert payload["status"] == "partial_failure"
        assert payload["exit_code"] == result.exit_code
        validate_contract("cli-output", payload)
