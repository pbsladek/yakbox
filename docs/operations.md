# Doctor, configuration, security, and troubleshooting

## Doctor

```console
yakbox doctor
yakbox doctor yakbox.toml --target default
yakbox doctor yakbox.toml --backend chatterbox-local --deep
yakbox doctor yakbox.toml --backend resemble --network
yakbox --json doctor
```

The default is offline and read-only. It checks Python/package configuration,
FFmpeg/FFprobe, workspace paths, source readability, free storage, locks, and
atomic writes. `--deep` inspects local runtime/device readiness without loading
a model. `--network` performs only safe provider discovery (`GET` voices); it
never synthesizes or mutates.

## Configuration

User configuration is optional TOML at
`~/.config/yakbox/config.toml`, or `YAKBOX_CONFIG`:

```toml
[defaults]
backend = "fake"

[cloud]
voice_uuid = "..."
project_uuid = "..."
concurrency = 5
```

Supported environment overrides include `YAKBOX_BACKEND`,
`YAKBOX_CLOUD_VOICE_UUID`, `YAKBOX_CLOUD_PROJECT_UUID`,
`YAKBOX_CLOUD_CONCURRENCY`, and `RESEMBLE_API_KEY`. Legacy
`RESEMBLE_VOICE_UUID`/`RESEMBLE_PROJECT_UUID` aliases remain accepted.

Missing or malformed manifests are application failures because manifest
loading performs domain validation and may report several related issues.
Direct file arguments such as audio, recording, batch, resume, and shard inputs
are validated by Click as usage errors before an operation begins.

## Security

- Do not place API keys in `yakbox.toml`, source control, batch rows, or shell
  arguments.
- Use environment secrets in automation and OS keyring profiles interactively.
- Worker children do not inherit the Resemble token.
- Provider bodies and request IDs are bounded and token-redacted in errors.
- Output and restore paths are contained beneath declared roots.
- Reference voice use requires an explicit rights basis. Keep licenses and
  consent evidence with project records.
- Hosted confirmation and budget caps are independent of concurrency.

## Failure recovery

Every build stage journals its start, completion, or failure. A failed or
cancelled build releases its target lock and is resumed by the next ordinary
`yakbox build`. Completed artifacts are reused only after their sidecar,
fingerprint, size, and digest match.

Use these read-only checks before repairing anything:

```console
yakbox status
yakbox artifacts verify
yakbox doctor yakbox.toml
```

Provider authentication failures are never retried. Provider timeouts are
bounded and leave no destination file. FFmpeg, FFprobe, local-worker, disk
preflight, and stage failures retain the run journal and do not leave owned
`.part` or chunk files. If valid audio was externally changed,
`yakbox artifacts verify` reports the mismatch; rebuild from source rather
than accepting unexplained bytes.

## Troubleshooting

`FFmpeg is required`: install FFmpeg and ensure both `ffmpeg` and `ffprobe` are
on `PATH`, then rerun Doctor.

`Local Chatterbox package is not installed`: install `yakbox[local]` in the
same environment using the versioned security override command in the
[installation guide](installing-and-releasing.md). Start with a one-sentence
direct test. Worker heartbeat logs are under
`.yakbox/runs/RUN_ID/logs/local-worker.log`.

`No Resemble API key found`: set `RESEMBLE_API_KEY` or install
`yakbox[credentials]`, run `config auth login --credential-profile NAME`, and
select the same profile before the cloud subcommand.

`may have been accepted by the provider`: a management mutation failed after
the request could have reached Resemble. Inspect the provider project/voice
before retrying.

`build already locked`: another target build/cleanup may be active. Do not
delete an active lock. Use Doctor and inspect the recorded run first.

`stage prerequisite ... missing`: a `--from` build requires digest-verified
outputs from earlier stages. Build from synthesis or remove the stage bound.

## Automation and shell completion

Use `--json` before the command. JSON is a versioned, single-document envelope
with a stable `command` discriminator, snake-case error codes, statuses, and
exit codes 0, 1, 2, and 130.

Click completion can be installed for the current shell:

```console
_YAKBOX_COMPLETE=bash_source yakbox > ~/.local/share/bash-completion/completions/yakbox
_YAKBOX_COMPLETE=zsh_source yakbox > ~/.zfunc/_yakbox
_YAKBOX_COMPLETE=fish_source yakbox > ~/.config/fish/completions/yakbox.fish
```

Create the destination directory first and restart the shell. Regenerate
completion after upgrading command surfaces.
