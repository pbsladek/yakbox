from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from yakbox.audiobook.manifest import ChatterboxOptions, load_manifest
from yakbox.errors import ValidationError


def _write_manifest(tmp_path: Path, extra: str = "") -> Path:
    (tmp_path / "book.md").write_text("# One\n\nText.", encoding="utf-8")
    path = tmp_path / "yakbox.toml"
    path.write_text(
        '"$schema" = '
        '"https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"\n'
        "schema_version = 1\n"
        'sources = ["book.md"]\n'
        "[book]\n"
        'title = "Book"\n'
        "[profiles.default]\n"
        'backend = "fake"\n'
        'voice = "narrator"\n'
        "[targets.default]\n"
        'profile = "default"\n'
        f"{extra}",
        encoding="utf-8",
    )
    return path


def _write_character_manifest(tmp_path: Path, *, characters: str) -> Path:
    (tmp_path / "book.md").write_text("# One\n\nText.", encoding="utf-8")
    for name in ("narrator", "mara", "wren"):
        (tmp_path / f"{name}.wav").write_bytes(b"reference")
    path = tmp_path / "yakbox.toml"
    path.write_text(
        '"$schema" = '
        '"https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"\n'
        "schema_version = 1\n"
        'sources = ["book.md"]\n'
        "[book]\n"
        'title = "Book"\n'
        "[voices.narrator]\n"
        'reference_audio = "narrator.wav"\n'
        "[voices.mara]\n"
        'reference_audio = "mara.wav"\n'
        "[voices.wren]\n"
        'display_name = "Wren (male)"\n'
        'reference_audio = "wren.wav"\n'
        "[profiles.narrator]\n"
        'backend = "chatterbox-local"\n'
        'voice = "narrator"\n'
        'device = "cpu"\n'
        "[profiles.mara]\n"
        'backend = "chatterbox-local"\n'
        'voice = "mara"\n'
        'device = "cpu"\n'
        "[profiles.wren]\n"
        'backend = "chatterbox-local"\n'
        'voice = "wren"\n'
        'device = "cpu"\n'
        "[targets.default]\n"
        'profile = "narrator"\n'
        f"{characters}",
        encoding="utf-8",
    )
    return path


def test_manifest_parses_hosted_confirmation_and_storage_limits(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(
        _write_manifest(
            tmp_path,
            "confirm_above_characters = 1000\n"
            "confirm_above_requests = 10\n"
            "storage_budget_bytes = 123456\n"
            'max_estimated_spend = "5.00"\n'
            'currency = "USD"\n'
            'pricing_source = "resemble-2026-07"\n'
            'price_per_character = "0.001"\n'
            "[retention]\n"
            "keep_successful_runs = 2\n"
            "audition_days = 14\n"
            "preview_days = 3\n"
            "raw_until_release = true\n",
        )
    )

    target = manifest.target("default")
    assert target.confirm_above_characters == 1_000
    assert target.confirm_above_requests == 10
    assert target.storage_budget_bytes == 123_456
    assert target.max_estimated_spend == Decimal("5.00")
    assert manifest.retention.keep_successful_runs == 2
    assert manifest.retention.audition_days == 14
    assert manifest.retention.preview_days == 3
    assert manifest.retention.raw_until_release


def test_manifest_parses_localized_repair_defaults(tmp_path: Path) -> None:
    manifest = load_manifest(
        _write_manifest(
            tmp_path,
            "\n[repairs]\n"
            'mode = "paragraph"\n'
            "takes = 6\n"
            "minimum_passing_takes = 3\n"
            "whisper_qa = false\n"
            "rebuild_on_approval = false\n",
        )
    )

    assert manifest.repairs.mode == "paragraph"
    assert manifest.repairs.takes == 6
    assert manifest.repairs.minimum_passing_takes == 3
    assert not manifest.repairs.whisper_qa
    assert not manifest.repairs.rebuild_on_approval


def test_manifest_parses_persistent_runtime_policy(tmp_path: Path) -> None:
    manifest = load_manifest(
        _write_manifest(
            tmp_path,
            "\n[runtime]\n"
            "enabled = true\n"
            "idle_timeout_seconds = 120\n"
            "conditioning_cache_size = 4\n"
            "maximum_memory_bytes = 8589934592\n",
        )
    )

    assert manifest.runtime.enabled
    assert manifest.runtime.idle_timeout_seconds == 120
    assert manifest.runtime.conditioning_cache_size == 4
    assert manifest.runtime.maximum_memory_bytes == 8_589_934_592


def test_local_chatterbox_defaults_to_cpu(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    content = path.read_text(encoding="utf-8").replace(
        'backend = "fake"', 'backend = "chatterbox-local"'
    )
    path.write_text(content, encoding="utf-8")

    options = load_manifest(path).profile("default").options

    assert isinstance(options, ChatterboxOptions)
    assert options.device == "cpu"
    assert options.seed == 0
    assert not load_manifest(path).dialogue.strip_attribution_tags


def test_whisper_qa_policy_is_configurable_and_workspace_scoped(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(
        _write_manifest(
            tmp_path,
            "[whisper_qa]\n"
            "chapter_verification = true\n"
            "cache_enabled = true\n"
            'cache_directory = ".cache/whisper"\n'
            "join_coalesce_gap_ms = 250\n"
            "phoneme_alignment = true\n"
            "minimum_phoneme_confidence = 0.35\n"
            'manuscript_aliases = { mara = ["marah"] }\n',
        )
    )

    assert manifest.whisper_qa.chapter_verification
    assert manifest.whisper_qa.cache_directory == (tmp_path / ".cache/whisper")
    assert manifest.whisper_qa.join_coalesce_gap_ms == 250
    assert manifest.whisper_qa.phoneme_alignment
    assert manifest.whisper_qa.minimum_phoneme_confidence == 0.35
    assert manifest.whisper_qa.manuscript_alias_map == {"mara": ("marah",)}


def test_character_profiles_and_performance_overrides_are_resolved(
    tmp_path: Path,
) -> None:
    routes = tmp_path / "dialogue-routes.toml"
    routes.write_text(
        '"$schema" = "https://yakbox.dev/schemas/dialogue-routes-v1.schema.json"\n'
        "schema_version = 1\n"
        "[[routes]]\n"
        'source = "book.md"\nline = 3\nspeaker = "mara"\n'
        'status = "approved"\n',
        encoding="utf-8",
    )
    manifest = load_manifest(
        _write_character_manifest(
            tmp_path,
            characters=(
                "[characters.narrator]\n"
                'profile = "narrator"\n'
                "[characters.mara]\n"
                'profile = "mara"\n'
                'gender = "female"\n'
                "cfg_weight = 0.3\n"
                "exaggeration = 0.7\n"
                "seed = 17\n"
                "[characters.wren]\n"
                'display_name = "Wren"\n'
                'profile = "wren"\n'
                'gender = "male"\n'
                "[dialogue]\n"
                'attribution_assistance = "warn"\n'
                "short_utterance_words = 4\n"
                "strip_attribution_tags = true\n"
                "retain_first_attribution_per_scene = true\n"
                'routes = "dialogue-routes.toml"\n'
            ),
        )
    )

    mara = manifest.profile_for_speaker("mara", fallback_profile="narrator")
    assert isinstance(mara.options, ChatterboxOptions)
    assert mara.voice == "mara"
    assert mara.options.cfg_weight == 0.3
    assert mara.options.exaggeration == 0.7
    assert mara.options.seed == 17
    assert manifest.character("wren").display_name == "Wren"
    assert manifest.character("mara").gender == "female"
    assert manifest.character("wren").gender == "male"
    assert manifest.character("narrator").gender == "unspecified"
    assert manifest.dialogue.short_utterance_words == 4
    assert manifest.dialogue.strip_attribution_tags
    assert manifest.dialogue.retain_first_attribution_per_scene
    assert manifest.dialogue.routes == routes


@pytest.mark.parametrize(
    ("characters", "match"),
    [
        (
            '[characters.mara]\nprofile = "mara"\n',
            "define a narrator",
        ),
        (
            '[characters.narrator]\nprofile = "missing"\n',
            "unknown profile",
        ),
        (
            '[characters.narrator]\nprofile = "narrator"\n'
            '[characters.mara]\nprofile = "mara"\n'
            '[dialogue]\nattribution_assistance = "sometimes"\n',
            "must be off, warn, or error",
        ),
        (
            '[characters.narrator]\nprofile = "narrator"\n'
            '[characters.mara]\nprofile = "mara"\ngender = "robot"\n',
            "must be female, male, or unspecified",
        ),
        (
            '[characters.narrator]\nprofile = "narrator"\n'
            '[characters.mara]\nprofile = "mara"\n'
            '[dialogue]\nstrip_attribution_tags = "yes"\n',
            "dialogue.strip_attribution_tags must be boolean",
        ),
    ],
)
def test_manifest_rejects_invalid_character_configuration(
    tmp_path: Path,
    characters: str,
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        load_manifest(_write_character_manifest(tmp_path, characters=characters))


def test_manifest_rejects_character_profile_with_different_runtime(
    tmp_path: Path,
) -> None:
    path = _write_character_manifest(
        tmp_path,
        characters=(
            '[characters.narrator]\nprofile = "narrator"\n'
            '[characters.wren]\nprofile = "wren"\n'
        ),
    )
    content = path.read_text(encoding="utf-8").replace(
        '[profiles.wren]\nbackend = "chatterbox-local"\nvoice = "wren"\ndevice = "cpu"',
        '[profiles.wren]\nbackend = "chatterbox-local"\n'
        'voice = "wren"\ndevice = "cuda"',
    )
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValidationError, match="narrator runtime settings"):
        load_manifest(path)


def test_target_inheritance_supports_draft_proof_and_release_modes(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(
        _write_manifest(
            tmp_path,
            "media_concurrency = 3\n"
            "[targets.draft]\n"
            'extends = "default"\n'
            'output_root = "build/draft"\n'
            "mastering = false\n"
            'through_stage = "synthesize"\n'
            "[targets.release]\n"
            'extends = "default"\n'
            'output_root = "build/release"\n'
            "m4b = true\n",
        )
    )

    draft = manifest.target("draft")
    release = manifest.target("release")
    assert draft.profile == "default"
    assert draft.media_concurrency == 3
    assert draft.mastering is False
    assert draft.through_stage == "synthesize"
    assert release.m4b is True
    assert release.through_stage == "inspect"


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ('output_root = "."\n', "workspace root"),
        (
            'max_estimated_spend = "1.00"\n',
            "monetary budget requires",
        ),
        ("provider_concurrency = 101\n", "at most 100"),
        ("media_concurrency = 33\n", "at most 32"),
        ("storage_budget_bytes = -1\n", "non-negative"),
        ('mastering = "yes"\n', "mastering must be boolean"),
        ('mp3_bitrate = "fast"\n', "must look like"),
        ("chunk_chars = true\n", "positive integer"),
    ],
)
def test_manifest_rejects_unsafe_or_incomplete_target_settings(
    tmp_path: Path,
    extra: str,
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        load_manifest(_write_manifest(tmp_path, extra))


def test_manifest_rejects_source_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.md"
    outside.write_text("# Outside", encoding="utf-8")
    path = _write_manifest(tmp_path)
    content = path.read_text(encoding="utf-8").replace(
        'sources = ["book.md"]',
        'sources = ["../outside.md"]',
    )
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValidationError, match="escapes"):
        load_manifest(path)


def test_manifest_rejects_output_root_containing_source(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, 'output_root = "generated"\n')
    generated = tmp_path / "generated"
    generated.mkdir()
    (tmp_path / "book.md").replace(generated / "book.md")
    content = path.read_text(encoding="utf-8").replace(
        'sources = ["book.md"]',
        'sources = ["generated/book.md"]',
    )
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValidationError, match="contains source"):
        load_manifest(path)


def test_manifest_rejects_wrong_profile_and_book_value_types(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    content = path.read_text(encoding="utf-8").replace(
        'title = "Book"',
        'title = "Book"\nlanguage = 42',
    )
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValidationError, match=r"book\.language"):
        load_manifest(path)

    path = _write_manifest(tmp_path)
    content = path.read_text(encoding="utf-8").replace(
        'backend = "fake"',
        'backend = "fake"\nexecutor = "local-process"',
    )
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValidationError, match="incompatible"):
        load_manifest(path)


def test_manifest_parses_opt_in_short_utterance_policy(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    with path.open("a", encoding="utf-8") as manifest_file:
        manifest_file.write(
            "\n[short_utterances]\n"
            'strategy = "context_extract"\n'
            "maximum_words = 4\n"
            "candidate_count = 6\n"
            "prefer_natural_context = false\n"
            'carrier_positions = ["middle", "initial"]\n'
            'alignment_backend = "mlx-whisper"\n'
            'alignment_model = "mlx-community/whisper-medium-mlx"\n'
            'alignment_revision = "0123456789abcdef"\n'
            'alignment_aliases = { liora = ["leora"] }\n'
            "prompted_timing = false\n"
            "decode_consensus = false\n"
            "prompt_sensitivity = false\n"
            "maximum_consensus_timing_delta_ms = 125\n"
            "hallucination_silence_threshold = 1.25\n"
            "automatic_join_inspection = false\n"
            "join_inspection_window_seconds = 2.25\n"
            "alignment_timeout_seconds = 90.0\n"
            "minimum_alignment_confidence = 0.7\n"
            "minimum_extracted_confidence = 0.25\n"
            "minimum_one_word_confidence = 0.72\n"
            "minimum_short_phrase_confidence = 0.61\n"
            "minimum_segment_average_log_probability = -0.8\n"
            "maximum_segment_compression_ratio = 2.2\n"
            "maximum_segment_no_speech_probability = 0.4\n"
            "maximum_segment_temperature = 0.1\n"
            "candidate_confidence_tolerance = 0.04\n"
            "maximum_extra_speech_ms = 45\n"
            "maximum_internal_token_gap_ms = 240\n"
            "maximum_token_duration_ms = 800\n"
            "acoustic_refinement = false\n"
            "acoustic_threshold_dbfs = -52.5\n"
            "speech_island_gap_ms = 275\n"
            "minimum_edge_silence_ms = 12\n"
            "maximum_edge_silence_ms = 95\n"
            "maximum_clipped_sample_ratio = 0.003\n"
            "maximum_boundary_jump_ratio = 0.25\n"
            "maximum_vad_disagreement_ms = 300\n"
            "maximum_stationary_voiced_ms = 900\n"
            "minimum_pause_ms = 160\n"
            "pre_roll_ms = 25\n"
            "post_roll_ms = 35\n"
            "fade_ms = 6\n"
            'failure = "review"\n'
            "require_review_for_one_word = false\n"
            "keep_candidates = true\n"
        )

    policy = load_manifest(path).short_utterances

    assert policy.strategy == "context_extract"
    assert policy.maximum_words == 4
    assert policy.candidate_count == 6
    assert policy.prefer_natural_context is False
    assert policy.carrier_positions == ("middle", "initial")
    assert policy.alignment_model.endswith("whisper-medium-mlx")
    assert policy.alignment_alias_map == {"liora": ("leora",)}
    assert policy.prompted_timing is False
    assert policy.decode_consensus is False
    assert policy.prompt_sensitivity is False
    assert policy.maximum_consensus_timing_delta_ms == 125
    assert policy.hallucination_silence_threshold == 1.25
    assert policy.automatic_join_inspection is False
    assert policy.join_inspection_window_seconds == 2.25
    assert policy.alignment_timeout_seconds == 90.0
    assert policy.minimum_alignment_confidence == 0.7
    assert policy.minimum_extracted_confidence == 0.25
    assert policy.minimum_one_word_confidence == 0.72
    assert policy.minimum_short_phrase_confidence == 0.61
    assert policy.minimum_segment_average_log_probability == -0.8
    assert policy.maximum_segment_compression_ratio == 2.2
    assert policy.maximum_segment_no_speech_probability == 0.4
    assert policy.maximum_segment_temperature == 0.1
    assert policy.candidate_confidence_tolerance == 0.04
    assert policy.maximum_extra_speech_ms == 45
    assert policy.maximum_internal_token_gap_ms == 240
    assert policy.maximum_token_duration_ms == 800
    assert policy.acoustic_refinement is False
    assert policy.acoustic_threshold_dbfs == -52.5
    assert policy.speech_island_gap_ms == 275
    assert policy.minimum_edge_silence_ms == 12
    assert policy.maximum_edge_silence_ms == 95
    assert policy.maximum_clipped_sample_ratio == 0.003
    assert policy.maximum_boundary_jump_ratio == 0.25
    assert policy.maximum_vad_disagreement_ms == 300
    assert policy.maximum_stationary_voiced_ms == 900
    assert policy.failure == "review"
    assert policy.keep_candidates


@pytest.mark.parametrize(
    ("setting", "match"),
    [
        ('strategy = "guess"', "must be one of"),
        ("candidate_count = 0", "positive integer"),
        ('carrier_positions = ["direct"]', "must be one of"),
        ("minimum_alignment_confidence = 1.1", "between 0 and 1"),
        ("candidate_confidence_tolerance = 1.1", "between 0 and 1"),
        ("alignment_timeout_seconds = 0", "must be positive"),
        ("maximum_segment_compression_ratio = 0", "must be positive"),
        ("maximum_boundary_jump_ratio = 1.1", "between 0 and 1"),
        ("minimum_edge_silence_ms = 121", "must not exceed"),
        ("acoustic_threshold_dbfs = -121", "between -120 and 0"),
        ('alignment_backend = "hosted"', "must be mlx-whisper"),
    ],
)
def test_manifest_rejects_invalid_short_utterance_policy(
    tmp_path: Path, setting: str, match: str
) -> None:
    path = _write_manifest(tmp_path)
    with path.open("a", encoding="utf-8") as manifest_file:
        manifest_file.write(f"\n[short_utterances]\n{setting}\n")

    with pytest.raises(ValidationError, match=match):
        load_manifest(path)
