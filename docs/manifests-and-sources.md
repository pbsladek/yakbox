# Manifests, sources, and pronunciations

`yakbox.toml` is the only required project file. Paths are relative to its
directory, and generated output must not contain source files.

```toml
"$schema" = "https://yakbox.dev/schemas/audiobook-manifest-v1.schema.json"
schema_version = 1
sources = ["source/book.md"]
pronunciations = "pronunciations.toml"

[book]
title = "Example Book"
author = "Example Author"
narrator = "Example Narrator"
language = "en"
subtitle = "An Example"
publisher = "Example Press"
genre = "Science Fiction"
series = "Example Cycle"
series_position = 1
isbn = "9780000000000"
publication_date = "2026-07-31"
cover = "assets/cover.jpg"

[voices.narrator]
display_name = "Narrator"
rights_basis = "not_applicable"

[profiles.local]
backend = "chatterbox-local"
voice = "narrator"
device = "cpu"
cfg_weight = 0.5
exaggeration = 0.5
seed = 42
max_processes = 1
threads_per_process = 1
worker_timeout_seconds = 3600

[targets.default]
profile = "local"
output_root = "build/yakbox"
chunk_chars = 500
mastering = true
wav_sample_rate = 44100
mp3_bitrate = "192k"
m4b = false
m4b_bitrate = "192k"
provider_concurrency = 5
media_concurrency = 2

[targets.release]
extends = "default"
output_root = "build/yakbox-release"
m4b = true
quality_min_lufs = -23.0
quality_max_lufs = -16.0
quality_max_true_peak_dbfs = -1.0
quality_max_leading_silence_seconds = 2.0
quality_max_trailing_silence_seconds = 2.0

[retention]
keep_successful_runs = 3
audition_days = 30
preview_days = 7
raw_until_release = true
```

Logical voices describe identity, consent/rights, and optional reference
audio. Profiles describe backend render settings. Targets describe execution,
formats, limits, and output policy. Keeping those concerns separate allows the
same narrator identity to be auditioned on more than one backend.

Local Chatterbox defaults to `cpu`. Set `device = "auto"` only when Yakbox
should select CUDA, then MPS, then CPU from the available PyTorch runtimes.
Chatterbox requests are capped at 500 characters even when a target declares a
larger generic `chunk_chars` value. The base seed defaults to `0`; each request
derives a stable chunk-specific seed from it.

## Character voices and performance

Map the narrator and each speaking character to a declared profile. Character
names are lowercase identifiers because the same names appear in source
directives and plan output.

```toml
[characters.narrator]
display_name = "Narrator"
profile = "andy-minter"
gender = "male"

[characters.character-1]
display_name = "Character 1"
profile = "caro-davy"
gender = "female"
cfg_weight = 0.34
exaggeration = 0.44
seed = 42

[characters.character-2]
display_name = "Character 2"
profile = "nick-whitley"
gender = "male"
cfg_weight = 0.2
exaggeration = 0.44
seed = 42

[characters.character-3]
display_name = "Character 3"
profile = "ruth-golding"
gender = "female"

[characters.character-4]
display_name = "Character 4"
profile = "bill-boerst"
gender = "male"

[characters.character-5]
display_name = "Character 5"
profile = "john-greenman"
gender = "male"

[dialogue]
attribution_assistance = "warn"
short_utterance_words = 3
strip_attribution_tags = true
expressive_tag_handling = "context"
retain_first_attribution_per_scene = false
# routes = "dialogue-routes.toml"

[whisper_qa]
chapter_verification = true
cache_enabled = true
cache_directory = ".yakbox/cache/whisper"
join_coalesce_gap_ms = 100
manuscript_aliases = { mara = ["marah"] }
phoneme_alignment = true
phoneme_backend = "wav2vec2-ctc"
phoneme_model = "facebook/wav2vec2-lv-60-espeak-cv-ft"
phoneme_revision = "c43348bbaa5a77692c8e7bf3409d683474fdf2a4"
phoneme_language = "en-us"
minimum_phoneme_confidence = 0.2

[short_utterances]
strategy = "context_extract"
maximum_words = 3
candidate_count = 5
prefer_natural_context = true
carrier_positions = ["middle"]
alignment_backend = "mlx-whisper"
alignment_model = "mlx-community/whisper-large-v3-turbo"
alignment_revision = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
alignment_aliases = { liora = ["leora"] }
decode_consensus = true
prompt_sensitivity = true
maximum_consensus_timing_delta_ms = 180
hallucination_silence_threshold = 0.8
automatic_join_inspection = true
join_inspection_window_seconds = 1.5
minimum_alignment_confidence = 0.5
minimum_extracted_confidence = 0.2
minimum_one_word_confidence = 0.6
minimum_short_phrase_confidence = 0.5
maximum_segment_temperature = 0.2
candidate_confidence_tolerance = 0.05
maximum_extra_speech_ms = 60
acoustic_refinement = true
acoustic_threshold_dbfs = -48.0
speech_island_gap_ms = 300
minimum_edge_silence_ms = 10
maximum_edge_silence_ms = 120
minimum_pause_ms = 180
pre_roll_ms = 30
post_roll_ms = 40
fade_ms = 8
failure = "error"
require_review_for_one_word = true
keep_candidates = true
```

The bundled local example continues the generic mapping through
`character-24`: Karen Savage, Amanda Friday, Cori Samuel, Mil Nicholson, and
Lucy Burgoyne occupy female slots 6–10; Mark F. Smith, Bob Neufeld, Mark
Nelson, David Barnes, Simon Evers, Gregg Margarite, Tony Foster, Martin
Geeson, Phil Chenevert, and Peter Yearsley occupy male slots 11–20; Kirsten
Ferreri, Sibella Denton, and Laurie Anne Walden occupy female slots 21–23.
John Burlinson occupies male slot 24. These are editable defaults, not inferred
casting rules.

The settings under a character override that character's Chatterbox profile.
They don't change the profile used by anyone else. All routed profiles must use
the narrator's backend, executor, and Chatterbox device so one open worker can
switch voices safely between chunks.

`gender` is role metadata and accepts `female`, `male`, or `unspecified`. It
does not select or transform a voice. Choose the actual voice with `profile`;
this keeps role metadata separate from synthesis behavior.

Change `characters.narrator.profile` to select a different default narrator.
Keep `targets.default.profile` set to the same profile so the target remains
clear to readers and continues to work if the character map is removed.

Short-utterance context extraction is opt-in; omit its table or use
`strategy = "direct"` to retain ordinary synthesis. On Apple Silicon,
`context_extract` uses the optional local MLX Whisper adapter to verify and
time every marked narrator or character chunk at or below `maximum_words`.
Candidate count, carrier position, confidence, timing, crop padding, review,
and retention behavior are manifest settings. A custom alignment model must
provide an immutable `alignment_revision`. The default model revision is also
pinned explicitly above so project intent remains visible. Explicit
`alignment_aliases` handle reviewed ASR spelling variants for invented names;
they canonicalize exact single tokens and do not alter synthesis text or permit
extra words.

`whisper_qa.chapter_verification` adds a managed chapter-verification node and
blocks MP3 encoding and release publication when it fails. The
content-addressed cache is workspace-scoped and may be moved or disabled
without changing output semantics. Reviewed single-token spelling and
homophone variants belong in `whisper_qa.manuscript_aliases`; unlike
`short_utterances.alignment_aliases`, they affect only the final chapter
comparison and do not invalidate synthesized short-utterance candidates.
`phoneme_alignment` adds an independent
IPA/CTC forced-alignment gate to context-extracted short utterances; it requires
the `phoneme` extra, the pinned model, and the external eSpeak NG executable.

## Markdown normalization

Level-one and level-two headings start chapters. Speakable paragraphs, lists,
quotes, and link text are normalized through CommonMark. Code and non-speech
markup are not treated as narration.

Speech chunks prefer paragraph, sentence, clause, and word boundaries in that
order and never split a Unicode grapheme. Plans record the chosen boundary;
assembly applies stable boundary spacing and short PCM fades.

Use namespaced HTML comments when printed and spoken text differ:

```markdown
Visible setup.

<!-- yakbox:speech:exclude:start -->
This remains in the manuscript but is not spoken.
<!-- yakbox:speech:exclude:end -->

<!-- yakbox:speech:only:start -->
This narration-only bridge is spoken.
<!-- yakbox:speech:only:end -->

<!-- yakbox:speech:pause ms=750 -->
```

Route one paragraph by placing a speaker directive on its own line immediately
before that paragraph:

```markdown
The receiver woke with a pale blue pulse.

<!-- yakbox:speech:speaker name=character-2 -->

“Step away from the console,” the second technician said. “If it recognizes
us, we may never get another chance to leave.”

<!-- yakbox:speech:speaker name=character-1 -->

“We have waited years for this answer,” the first technician said. “I will not
let fear choose for us now.”

The signal sharpened. Every unmarked paragraph returns to the narrator.
```

A speaker directive applies to exactly the next spoken paragraph. Within a
routed paragraph, paired straight, curly, or guillemet double quotation marks
identify the character's spoken text. Yakbox routes the quoted spans to the
character and surrounding action back to the narrator. With
`strip_attribution_tags = true`, common speech tags are omitted—including
`Wren said`, `she replied`, and `asked Wren`—because the routed voice already
identifies the speaker. Action beats without a speech-attribution verb remain
on the narrator route. Set the field to `false` (the compatibility default) to
have the narrator read every tag.

Tag stripping also handles an attribution between two quoted spans. For
example, `“What could be doing that?” Wren asked. “Some kind of magic?”`
becomes one Wren turn: `What could be doing that? Some kind of magic?`. If a
quote ends in a comma solely because a tag follows it, Yakbox closes it with a
period when the tag is removed. An interrupted sentence such as `“If you
think,” Wren said, “that I am leaving...”` instead becomes `If you think that I
am leaving...`; removing the tag does not split the sentence. Unmarked narrator
dialogue is never rewritten.

Yakbox classifies recognized tags as pure (`said`, `asked`, `replied`) or
expressive (`snapped`, `whispered`, `with contempt`).
`expressive_tag_handling = "context"` is the default: expressive tags are not
spoken, but they are available as hidden carrier context when a short line is
synthesized and cropped. Use `"narrate"` to keep expressive tags on the
narrator route, or `"strip"` to discard that performance context. Pure stripped
tags are also available to short-utterance synthesis. Build plans contain only
their kind, position, and SHA-256 digest—not their text.

Set `retain_first_attribution_per_scene = true` to have the narrator speak the
first recognized tag for each routed character after a level-one or level-two
heading and after each Markdown thematic break (`---`, `***`, or `___`). Later
tags for that character are stripped normally. An untagged first turn does not
consume the retained introduction.

The paired quote delimiters identify the route but are not submitted to speech
synthesis; punctuation inside them is otherwise preserved. A routed paragraph
without paired double quotation marks remains one character turn. The directive
does not remain active and cannot be placed inline. `yakbox plan --json` records
the resolved speaker, profile, and performance settings for every speech chunk.

To retry one narrator paragraph with another declared profile while preserving
the logical narrator and dialogue parsing, add `profile=` to the one-shot
directive:

```markdown
<!-- yakbox:speech:speaker name=narrator profile=narrator-retry -->

“No cameras,” the clerk said.
```

The override profile must exist in `[profiles]`. It affects only that paragraph
and is included in source identity, planning, and artifact lineage.
For routed character dialogue, `narrator_profile=` retries only the surrounding
action or attribution while preserving the configured character voice:

```markdown
<!-- yakbox:speech:speaker name=liora narrator_profile=narrator-retry -->

"Quill," Liora said.
```

Speaker changes inside one source paragraph preserve their local punctuation:
commas use the configured clause pause, sentence-ending punctuation uses the
sentence pause, and only the final routed span uses the paragraph pause. This
avoids inserting a full paragraph break between a line and its attribution.
Artifact sidecars keep the primary narrator in `logical_voice` for backward
compatibility and list the complete routed cast in `logical_voices`.

### Reviewed dialogue route sidecars

For manuscripts without inline speaker directives, configure a reviewed route
file and generate conservative suggestions:

```toml
[dialogue]
routes = "dialogue-routes.toml"
strip_attribution_tags = true
expressive_tag_handling = "context"
retain_first_attribution_per_scene = false
```

```console
yakbox dialogue routes suggest yakbox.toml
```

The generated TOML records the source-relative path, paragraph start line,
speaker, reason, and `status = "suggested"`. Review each entry and change its
status to `approved` or `rejected`; suggested routes deliberately fail builds.
Then validate exact line matching and declared speakers:

```console
yakbox dialogue routes check yakbox.toml
```

Approved routes behave like one-shot inline speaker directives. Yakbox rejects
duplicate, stale, out-of-workspace, undeclared, and inline-conflicting entries
instead of guessing. Regenerate and review the sidecar after moving manuscript
paragraphs.

Before synthesis, write a local, explicit-text transformation report:

```console
yakbox dialogue preview yakbox.toml --output dialogue-preview.json
```

The versioned report shows every routed paragraph, final spoken spans, speakers,
and stripped tags. It intentionally contains manuscript text, stays inside the
workspace, and will not overwrite an existing file unless `--force` is passed.

Attribution assistance is advisory and never rewrites the manuscript. In
`warn` mode, `validate` and `plan` report quoted paragraphs that still use the
narrator, very short routed turns, and adjacent characters that share one
voice. Use `off` to suppress the findings or `error` to make any finding fail
validation. A line such as “No.” often sounds abrupt when rendered alone;
keeping it inside a fuller attributed turn usually gives the model enough
context for a natural read.

Directives are paired, validated with source locations, and bounded by
`source.max_pause_ms`. `yakbox validate` and `yakbox plan` perform no model
load or network operation.

## Pronunciations

Pronunciations are an explicit, reviewable TOML sidecar:

```toml
schema_version = 1

[[terms]]
written = "Weyland"
spoken = "Way land"
language = "en"
match = "whole_word"
case = "sensitive"
priority = 100
status = "approved"
enabled = true
notes = "Auditioned in the selected narrator voice."
```

Only enabled, approved entries are applied. Replacement is deterministic and
non-recursive. A source or pronunciation change invalidates only dependent
nodes. Explicit `yakbox audition --text/--text-file` and `yakbox preview`
samples apply the same manifest lexicon as chapter-derived samples and
production builds, so a voice comparison does not silently bypass a tested
pronunciation.

Audit the lexicon before a paid or long local render:

```console
yakbox pronunciations audit
yakbox pronunciations audit --fail-unused
```

The report identifies applied rules, unused approved rules, priority-shadowed
matches, and manuscript line locations.

Use UTF-8 `.md` or `.txt` sources. Keep manuscript files, the pronunciation
sidecar, and licensed reference audio under normal project backup/versioning;
they are never cleanup candidates.
# Localized repair defaults

The optional `[repairs]` table controls the normal editing loop without tying
the build engine to a particular book:

```toml
[repairs]
mode = "context"
takes = 4
minimum_passing_takes = 2
whisper_qa = true
rebuild_on_approval = true
```

See [Localized regeneration and repair](localized-repair.md) for the selection,
audition, approval, and rebuild commands.

## Persistent local runtime

Local model startup can be retained across separate CLI commands:

```toml
[runtime]
enabled = true
idle_timeout_seconds = 900
conditioning_cache_size = 8
# maximum_memory_bytes = 51539607552
```

With this opt-in policy, local speech and MLX Whisper requests use one
authenticated loopback process scoped to the manifest directory. It lazily
loads each model, keeps up to the configured number of voice-conditioning
states, stops after the idle timeout, and can stop after a request if the
resident-memory ceiling is exceeded.

```console
yakbox runtime start yakbox.toml
yakbox runtime status yakbox.toml
yakbox runtime stop yakbox.toml
```

The endpoint token is stored mode `0600` under `.yakbox/runtime/`; status output
never includes it. When `enabled = false` (the default), existing isolated
per-command workers remain unchanged.
