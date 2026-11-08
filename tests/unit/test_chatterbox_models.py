from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner
from jsonschema import ValidationError as SchemaError
from tests.schema_helpers import validate_contract

from yakbox import chatterbox_models
from yakbox.chatterbox_models import (
    RECORD_NAME,
    SnapshotVerifier,
    record_snapshot,
    verify_snapshot,
)
from yakbox.cli import main
from yakbox.errors import ModelIntegrityError, ValidationError
from yakbox.speech.fingerprints import speech_request_fingerprint
from yakbox.speech.models import ChatterboxSynthesisOptions, SpeechSynthesisRequest


def snapshot_files(root: Path, variant: str = "original") -> Path:
    root.mkdir(parents=True)
    names = {
        "original": (
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "conds.pt",
        ),
        "turbo": (
            "ve.safetensors",
            "t3_turbo_v1.safetensors",
            "s3gen_meanflow.safetensors",
            "tokenizer_config.json",
        ),
        "mlx-turbo": (
            "config.json",
            "weights.safetensors",
            "conds.safetensors",
            "tokenizer_config.json",
            "s3tokenizer/model.safetensors",
        ),
    }[variant]
    for name in names:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(
            b'{"model_type":"chatterbox_turbo"}'
            if name == "config.json"
            else b"{}"
            if name.endswith(".json")
            else b"model fixture"
        )
    return root


def test_mlx_snapshot_requires_main_weights_in_addition_to_conditioning(
    tmp_path: Path,
) -> None:
    root = snapshot_files(tmp_path / "model", "mlx-turbo")
    (root / "weights.safetensors").unlink()
    with pytest.raises(ModelIntegrityError, match="no safetensors weights"):
        record_snapshot(
            root,
            variant="mlx-turbo",
            repository="owner/model",
            revision="a" * 40,
            license_id="Apache-2.0",
        )


def test_worker_verification_reuses_digests_until_snapshot_files_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = snapshot_files(tmp_path / "model")
    result = record_snapshot(
        root, variant="original", repository="owner/model", revision="a" * 40
    )
    hashed: list[Path] = []
    original = chatterbox_models.sha256_file

    def digest(path: Path) -> str:
        hashed.append(path)
        return original(path)

    monkeypatch.setattr(chatterbox_models, "sha256_file", digest)
    verifier = SnapshotVerifier()
    verifier.verify(root, variant="original", fingerprint=str(result["fingerprint"]))
    count = len(hashed)
    assert count == 5
    verifier.verify(root, variant="original", fingerprint=str(result["fingerprint"]))
    assert len(hashed) == count
    (root / "ve.safetensors").write_bytes(b"other fixture")
    with pytest.raises(ModelIntegrityError):
        verifier.verify(
            root, variant="original", fingerprint=str(result["fingerprint"])
        )
    assert len(hashed) > count


@pytest.mark.parametrize(
    "field,value",
    [("yakbox_version", '""'), ("timestamp", '"2026-01-01T00:00:00"')],
)
def test_snapshot_verification_rejects_invalid_envelope(
    tmp_path: Path, field: str, value: str
) -> None:
    root = snapshot_files(tmp_path / "model")
    result = record_snapshot(
        root, variant="original", repository="owner/model", revision="a" * 40
    )
    record = root / RECORD_NAME
    record.write_text(
        record.read_text().replace(
            f'"{field}" = {json.dumps(result[field])}', f'"{field}" = {value}'
        )
    )
    with pytest.raises(ModelIntegrityError, match="envelope"):
        verify_snapshot(root, variant="original")


@pytest.mark.parametrize("variant", ["original", "turbo", "mlx-turbo"])
def test_snapshot_records_and_verifies_exact_local_files(
    tmp_path: Path, variant: str
) -> None:
    root = snapshot_files(tmp_path / "model", variant)
    (root / ".gitattributes").write_text("*.safetensors filter=lfs\n")
    (root / "t3.yaml").write_text("model: archived\n")
    (root / "NOTICE").write_text("Original notices\n")
    result = record_snapshot(
        root,
        variant=variant,
        repository="ResembleAI/chatterbox",
        revision="a" * 40,
        license_id="Apache-2.0" if variant == "mlx-turbo" else "MIT",
    )
    validate_contract("chatterbox-model", result)
    assert result == verify_snapshot(
        root, variant=variant, fingerprint=str(result["fingerprint"])
    )
    assert tomllib.loads((root / RECORD_NAME).read_text()) == result
    files = cast(list[dict[str, str]], result["files"])
    assert [item["path"] for item in files] == sorted(item["path"] for item in files)
    with pytest.raises(ModelIntegrityError, match="identity differs"):
        verify_snapshot(root, variant=variant, fingerprint="b" * 64)


@pytest.mark.parametrize(
    "change",
    [
        "bytes",
        "missing",
        "extra",
        "symlink",
        "remote_code",
        "provenance",
        "identity",
        "schema",
        "invalid_toml",
        "duplicate",
        "bad_digest",
    ],
)
def test_snapshot_rejects_changed_or_unsafe_files(tmp_path: Path, change: str) -> None:
    root = snapshot_files(tmp_path / "model")
    record = record_snapshot(
        root, variant="original", repository="ResembleAI/chatterbox", revision="a" * 40
    )
    path = root / "ve.safetensors"
    manifest = root / RECORD_NAME
    if change == "bytes":
        path.write_bytes(b"changed bytes")
    elif change == "missing":
        path.unlink()
    elif change == "extra":
        (root / "extra.txt").write_text("unexpected")
    elif change == "symlink":
        outside = tmp_path / "outside"
        outside.write_text("external data")
        path.unlink()
        path.symlink_to(outside)
    elif change == "remote_code":
        (root / "tokenizer.json").write_text(
            '{"nested": [{"auto_map": {"model": "unsafe"}}]}'
        )
    elif change in {"provenance", "identity", "schema"}:
        before, after = {
            "provenance": ("a" * 40, "invalid"),
            "identity": (str(record["fingerprint"]), "b" * 64),
            "schema": ('schema_version" = 1', 'schema_version" = 2'),
        }[change]
        manifest.write_text(manifest.read_text().replace(before, after))
    elif change == "invalid_toml":
        manifest.write_text("not valid toml")
    elif change == "duplicate":
        manifest.write_text(
            manifest.read_text()
            + '\n[[files]]\npath="ve.safetensors"\nsize=13\nsha256="'
            + "a" * 64
            + '"\n'
        )
    else:
        manifest.write_text(
            manifest.read_text().replace(
                str(cast(list[dict[str, object]], record["files"])[0]["sha256"]), "bad"
            )
        )
    with pytest.raises(ModelIntegrityError):
        verify_snapshot(root, variant="original")
    assert not list(root.glob("*.part"))


@pytest.mark.parametrize("variant", ["original", "turbo", "mlx-turbo"])
def test_cli_records_and_verifies_snapshots(tmp_path: Path, variant: str) -> None:
    root = snapshot_files(tmp_path / "model", variant)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--json",
            "models",
            "--record",
            str(root),
            "--variant",
            variant,
            "--repository",
            "ResembleAI/chatterbox",
            "--revision",
            "a" * 40,
            "--license",
            "Apache-2.0" if variant == "mlx-turbo" else "MIT",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    validate_contract("cli-output", data)
    validate_contract("chatterbox-model", data["data"])
    checked = runner.invoke(
        main, ["--json", "models", "--verify", str(root), "--variant", variant]
    )
    assert checked.exit_code == 0, checked.output
    assert json.loads(checked.output)["data"] == data["data"]
    (root / "extra.py").write_text("raise RuntimeError('must not run')")
    rejected = runner.invoke(
        main, ["--json", "models", "--verify", str(root), "--variant", variant]
    )
    assert rejected.exit_code == 1
    assert json.loads(rejected.output)["error"]["code"] == "model_integrity_error"


@pytest.mark.parametrize(
    "options",
    [
        {"variant": "unknown"},
        {"variant": "turbo", "cfg_weight": 0.4},
        {"variant": "mlx-turbo"},
        {"temperature": 0},
        {"temperature": float("inf")},
        {"top_p": 2},
        {"repetition_penalty": -1},
        {"cfg_weight": True},
        {"seed": 1.2},
        {"model_directory": {}},
        {"model_fingerprint": {}},
        {"model_directory": "model"},
        {"model_directory": "model", "model_fingerprint": "bad"},
    ],
)
def test_options_reject_invalid_values_before_loading_models(
    options: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ChatterboxSynthesisOptions.from_dict(options)


def test_options_round_trip_and_cache_identity_uses_model_content_not_location(
    tmp_path: Path,
) -> None:
    options = ChatterboxSynthesisOptions(
        variant="turbo",
        model_directory=tmp_path / "first",
        model_fingerprint="a" * 64,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        seed=42,
    )
    assert ChatterboxSynthesisOptions.from_dict(options.to_dict()) == options
    request = SpeechSynthesisRequest(
        "A short line.", "narrator", backend="chatterbox-local", chatterbox=options
    )
    identity = speech_request_fingerprint(request)
    assert identity == speech_request_fingerprint(
        replace(
            request, chatterbox=replace(options, model_directory=tmp_path / "second")
        )
    )
    for updated in (
        replace(options, model_fingerprint="b" * 64),
        replace(options, temperature=0.6),
        replace(options, variant="original"),
    ):
        assert identity != speech_request_fingerprint(
            replace(request, chatterbox=updated)
        )


@pytest.mark.parametrize(
    "field", ["revision", "fingerprint", "source_url", "files", "$schema"]
)
def test_snapshot_schema_rejects_invalid_identity(tmp_path: Path, field: str) -> None:
    root = snapshot_files(tmp_path / "model")
    value = record_snapshot(
        root, variant="original", repository="ResembleAI/chatterbox", revision="a" * 40
    )
    value[field] = [] if field == "files" else "invalid"
    with pytest.raises(SchemaError):
        validate_contract("chatterbox-model", value)


@pytest.mark.parametrize(
    ("repository", "revision"), [("invalid", "a" * 40), ("owner/model", "main")]
)
def test_snapshot_requires_immutable_provenance(
    tmp_path: Path, repository: str, revision: str
) -> None:
    root = snapshot_files(tmp_path / "model")
    with pytest.raises(ValidationError):
        record_snapshot(
            root, variant="original", repository=repository, revision=revision
        )
    assert not (root / RECORD_NAME).exists()


@pytest.mark.parametrize("license_id", [None, "unknown", "CC-BY-NC-4.0"])
def test_mlx_record_requires_an_explicit_supported_license(
    tmp_path: Path, license_id: str | None
) -> None:
    root = snapshot_files(tmp_path / "model", "mlx-turbo")
    with pytest.raises(ValidationError, match="reviewed model license"):
        record_snapshot(
            root,
            variant="mlx-turbo",
            repository="owner/model",
            revision="a" * 40,
            license_id=license_id,
        )
    assert not (root / RECORD_NAME).exists()
