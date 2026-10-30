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

## Local models only

Hosted Resemble support has been removed after the provider discontinued its
commercial voice-generation service. This is an intentional breaking change
for the next minor release; the reviewed SDK surface is now
`tests/public-api-v2.yaml`. The earlier snapshot remains as historical evidence.

The `cloud` and `config auth` commands, the `credentials` extra,
`yakbox.cloud`, and `ResembleOptions` are removed. Build, preview, audition,
repair, speech-factory, and Doctor APIs no longer accept provider credentials;
build and direct TTS commands no longer expose hosted spending options.
Doctor's provider-only `--network` option is removed.

For an existing project:

1. Change each hosted profile to `backend = "chatterbox-local"`.
2. Remove `voice_uuid`, `project_uuid`, `use_hd`, and hosted-only profile fields.
3. Set the profile's logical `voice` and configure an authorized local reference
   recording and rights basis when cloning a voice.
4. Install the local dependencies using the versioned overrides in the
   [installation guide](installing-and-releasing.md).
5. Run `yakbox validate`, a short audition, and `yakbox build --dry-run` before
   rebuilding. Local and hosted audio have different fingerprints and cannot
   share synthesis-cache entries.

Legacy `[cloud]` configuration and provider environment variables are ignored.
Yakbox does not read or delete old keyring entries; remove saved Resemble tokens
using your operating system's credential manager if desired.

Existing JSON schema versions and historical usage fields remain readable.
New local build reports set `hosted_work` and `hosted_usage` to `null`.
Legacy target spending/concurrency fields and backend-neutral usage types remain accepted
for record compatibility but have no effect on local execution.

## Versioned contracts

Manifests and JSON outputs carry schema version 1 and a schema URI. Automation
should branch on `schema_version` and `status`, not parse human prose. Use
`yakbox --json ...` and treat exit 1 as runtime/partial failure, 2 as usage
error, and 130 as interruption.
