# yakbox

Yakbox is a local-first audiobook build system. It turns Markdown or text
manuscripts into deterministic build plans, generated narration, mastered WAV
chapters, delivery MP3 chapters, inspection reports, and release evidence.

The same typed speech services power audiobook builds and direct `tts`, `vc`,
local batch, and Resemble.ai commands. Local Chatterbox remains synchronous and
GPU-aware; hosted requests use bounded asynchronous concurrency.

## Quick start

```console
uv tool install yakbox
yakbox init my-book
cd my-book
yakbox validate
yakbox plan
yakbox audition --profile default
yakbox build
yakbox release check --write-manifest
```

The generated starter uses the deterministic fake backend so the complete
workflow can be exercised without a model or provider account. Configure
`chatterbox-local` or `resemble` in `yakbox.toml` for real narration.

## Direct speech

```console
yakbox tts "A short test." --backend fake --out test.wav
yakbox tts "A short local test." --backend chatterbox-local --out local.wav
RESEMBLE_API_KEY=... yakbox cloud tts "A hosted test." \
  --voice-uuid VOICE_UUID --out hosted.wav
yakbox cloud batch lines.csv --voice-uuid VOICE_UUID --concurrency 5
```

Install local model support separately:

```console
uv tool install "yakbox[local]"
```

Use only voice material for which you have the necessary rights and consent.

## Generated files and cleanup

Yakbox keeps run journals under `.yakbox/runs/` and generated files under
`build/yakbox/`. Cleanup is a dry run by default:

```console
yakbox artifacts usage
yakbox artifacts clean
yakbox artifacts clean --apply
yakbox artifacts trash restore CLEANUP_ID
yakbox artifacts trash purge CLEANUP_ID --yes
```

The current plan's mastered WAV and delivery MP3 artifacts are protected;
obsolete chapter outputs become eligible only after retention permits it.
Unknown files, source manuscripts, active work, and immutable release
snapshots are never selected by normal cleanup.

## Diagnostics

```console
yakbox doctor
yakbox doctor yakbox.toml --backend chatterbox-local --deep
yakbox doctor yakbox.toml --backend resemble --network
```

Doctor is offline and read-only unless `--network` is supplied. Deep local
checks do not synthesize audio or load a model.

See [AUDIOBOOK_BUILD_SYSTEM_PLAN.md](docs/AUDIOBOOK_BUILD_SYSTEM_PLAN.md) for the
full architecture and phased specification. The user documentation starts at
[docs/README.md](docs/README.md), including manifests and sources, backend
selection, hosted budgets, cleanup/recovery, releases, operations, the Python
API, and migration.
