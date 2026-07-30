# Migration guide

## From one-off scripts

Move source truth into Markdown/text, put pronunciation changes in
`pronunciations.toml`, and put backend/output settings in `yakbox.toml`. Do not
copy generated audio beside source chapters.

1. Run `yakbox init new-book`.
2. Replace `source/book.md` or set `sources` to existing UTF-8 files.
3. Express printed-versus-spoken differences with speech directives.
4. Create a logical narrator and one or more backend profiles.
5. Use short `audition` matrices to select settings.
6. Run `validate`, `plan`, and `build --dry-run`.
7. Build, inspect, and write release evidence.
8. Replace custom deletion scripts with dry-run cleanup and quarantine.

Example script-heavy layout:

```text
old-project/
  scripts/generate.py
  scripts/concat.sh
  chapters/*.txt
  output/*.wav
```

becomes:

```text
book/
  yakbox.toml
  pronunciations.toml
  source/*.md
  build/yakbox/       generated
  .yakbox/            resumable tool state
```

Do not import the old project as a runtime dependency. Preserve old outputs as
external evidence until the yakbox release passes digest/media checks.

## Cloud rewrite compatibility

Existing direct cloud TTS/stream behavior remains file-oriented. Batch parsing
matches local `.txt`, `.csv`, and `.jsonl`. The canonical recording command is:

```console
yakbox cloud voices recordings create VOICE_UUID AUDIO_FILE \
  --name NAME --text TRANSCRIPT
```

The old `cloud voices recording` spelling remains a deprecated hidden alias.
Deprecated `--api-key` remains accepted but warns; migrate to environment or
keyring profiles. `is_public` is not a Resemble project-create field and is
never sent.

## Versioned contracts

Manifests and JSON outputs carry schema version 1 and a schema URI. Automation
should branch on `schema_version` and `status`, not parse human prose. Use
`yakbox --json ...` and treat exit 1 as runtime/partial failure, 2 as usage
error, and 130 as interruption.
