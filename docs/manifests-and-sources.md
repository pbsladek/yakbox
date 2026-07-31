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
device = "auto"
cfg_weight = 0.5
exaggeration = 0.5
seed = 42
max_processes = 1
threads_per_process = 1
worker_timeout_seconds = 3600

[targets.default]
profile = "local"
output_root = "build/yakbox"
chunk_chars = 2800
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

## Markdown normalization

Level-one and level-two headings start chapters. Speakable paragraphs, lists,
quotes, and link text are normalized through CommonMark. Code and non-speech
markup are not treated as narration.

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
nodes.

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
