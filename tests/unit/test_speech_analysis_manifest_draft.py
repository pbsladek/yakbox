from __future__ import annotations

import json
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator
from jsonschema import ValidationError as SchemaError

from yakbox.errors import ValidationError
from yakbox.speech.analysis_manifest import (
    default_draft_speech_analysis_config,
    parse_draft_manifest_speech_analysis,
)

ROOT = Path(__file__).parents[2]
SCHEMA_PATH = ROOT / "src" / "yakbox" / "schemas" / "audiobook-manifest-v2.schema.json"


def _manifest() -> dict[str, object]:
    defaults = default_draft_speech_analysis_config()
    return {
        "schema_version": 2,
        "book": {"language": "en"},
        "speech_analysis": defaults.to_manifest_value(),
    }


def test_default_draft_round_trips_and_validates_its_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    manifest = _manifest()

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)
    parsed = parse_draft_manifest_speech_analysis(manifest)

    assert parsed == default_draft_speech_analysis_config()
    assert parsed.policy.language == "en"
    assert parsed.policy.baseline_recognizers == ("whisper", "parakeet")
    assert parsed.policy.escalation_recognizer == "qwen"
    assert parsed.policy.forced_aligner == "qwen-forced"


def test_draft_defaults_can_change_policy_without_changing_code() -> None:
    manifest = _manifest()
    speech = cast(dict[str, object], manifest["speech_analysis"])
    assert isinstance(speech, dict)
    engines = cast(dict[str, object], speech["engines"])
    assert isinstance(engines, dict)
    whisper = cast(dict[str, object], engines["whisper"])
    assert isinstance(whisper, dict)
    original = parse_draft_manifest_speech_analysis(manifest)
    whisper["timeout_seconds"] = 90
    cache = cast(dict[str, object], speech["cache"])
    cache["directory"] = "build/analysis-cache"

    changed = parse_draft_manifest_speech_analysis(manifest)

    assert changed.policy.engines[0].timeout_seconds == 90
    assert changed.cache_directory == "build/analysis-cache"
    assert changed.cache.version == 2
    assert changed.short_utterances.require_independent_crop_verification is True
    assert changed.fingerprint != original.fingerprint


def test_draft_rejects_model_substitution_and_unknown_fields() -> None:
    manifest = _manifest()
    replaced = deepcopy(manifest)
    speech = cast(dict[str, object], replaced["speech_analysis"])
    assert isinstance(speech, dict)
    engines = cast(dict[str, object], speech["engines"])
    assert isinstance(engines, dict)
    qwen = cast(dict[str, object], engines["qwen"])
    assert isinstance(qwen, dict)
    qwen["model"] = "unreviewed/remote-code-model"

    with pytest.raises(ValidationError, match="reviewed registry"):
        parse_draft_manifest_speech_analysis(replaced)

    unknown = deepcopy(manifest)
    unknown_speech = cast(dict[str, object], unknown["speech_analysis"])
    assert isinstance(unknown_speech, dict)
    unknown_speech["worker_command"] = "arbitrary-command"
    with pytest.raises(ValidationError, match="Unknown draft"):
        parse_draft_manifest_speech_analysis(unknown)


def test_draft_schema_rejects_non_strict_truth_table() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    manifest = _manifest()
    speech = cast(dict[str, object], manifest["speech_analysis"])
    assert isinstance(speech, dict)
    consensus = cast(dict[str, object], speech["consensus"])
    assert isinstance(consensus, dict)
    consensus["reject_unexpected_speech"] = False

    with pytest.raises(SchemaError):
        Draft202012Validator(schema).validate(manifest)
    with pytest.raises(ValidationError, match="Strict policy"):
        parse_draft_manifest_speech_analysis(manifest)


def test_internal_cutover_manifest_fixture_stays_parseable_and_schema_valid() -> None:
    manifest = tomllib.loads(
        (ROOT / "tests/fixtures/speech-analysis-manifest-v2.toml").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    Draft202012Validator(schema).validate(manifest)
    parsed = parse_draft_manifest_speech_analysis(manifest)

    assert parsed.cache.version == 2
    assert parsed.chapters.verification
    assert parsed.policy.always_escalate_repairs
