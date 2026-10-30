# Doctor, configuration, security, and troubleshooting

## Configuration formats

Yakbox user-authored configuration uses TOML or YAML only. Existing project,
voice, pronunciation, dialogue-route, QA-review, and global configuration files
remain TOML. Structured batch inputs and Whisper join specifications use YAML;
JSON configuration files are rejected.

JSON remains in generated reports, journals, CLI `--json` output, JSON Schema
documents, SBOMs, and upstream model files such as `config.json`. Those files
are machine protocols or evidence rather than user configuration. A `$schema`
URI ending in `.schema.json` identifies the standard JSON Schema document and
does not change the TOML or YAML format of the file that references it.

## Doctor

```console
yakbox doctor
yakbox doctor yakbox.toml --target default
yakbox doctor yakbox.toml --backend chatterbox-local --deep
yakbox --json doctor
yakbox doctor --strict
```

The default is offline and read-only. It checks Python/package configuration,
FFmpeg/FFprobe, workspace paths, source readability, free storage, locks, and
atomic writes. `--deep` inspects local runtime/device readiness without loading
a model. Doctor does not contact hosted speech services.

`--strict` also returns exit code 1 when a diagnostic has `warn` status. Skipped
checks remain allowed. Without `--strict`, warnings retain exit code 0 when
there are no failures. The JSON envelope's status and exit code reflect this
policy; the nested Doctor report's `healthy` field continues to describe the
absence of failed checks.

## Initialize a workspace

Set the book metadata when creating a new project:

```console
yakbox init my-book --title "My Book" --author "Author Name"
```

The defaults remain `My Audiobook` and `Author Name`. Titles and authors must
contain non-whitespace text. Quotes, backslashes, and Unicode are safely
encoded in `yakbox.toml`. Initialization refuses an existing manifest.

Choose a backend when creating the project:

```console
yakbox init local-book --backend chatterbox-local
```

`--backend` accepts `fake` (the default) or `chatterbox-local`.
The local starter uses CPU execution with one model process and one thread.
Install the local dependencies separately using the
[installation guide](installing-and-releasing.md). For voice cloning, configure
your authorized reference audio and its rights basis in the manifest.
Creating a project does not load models or download anything.

## Listen to previews and auditions

Add `--play` to open generated audio in the desktop's default audio player:

```console
yakbox preview --text "A short listening check." --play
yakbox audition --profile default --text "Compare this voice." --play
yakbox audition --profile narrator --profile guest --text "Compare voices." --play
```

The last command requires profiles named `narrator` and `guest`. Audition
samples open in render order; whether they queue or use separate windows
depends on the player. Playback uses `open` on macOS and `xdg-open` on Linux,
so it requires a configured desktop audio application. Player-launch attempts
have a 15-second timeout. A launch failure returns an error and leaves the
generated audio available at the path in the message.

Without `--play`, these commands only generate and report their artifacts.
`--json` can be combined with `--play`: player diagnostics stay out of the
single JSON response. Launching a player does not approve a listening review
or establish narration quality.

## Watch a build

In another terminal, run:

```console
yakbox status --watch
yakbox status --target draft --watch
```

Each command refreshes every two seconds until Ctrl+C, which exits with code
130. It follows the most recently active run for the selected target and
shows completed chapters and stages, reused stages, failures, and active
chapter/stage pairs. It can display an in-progress build before `run.json`
exists and wait for the first build in a new project. An unfinished run with
no target lock is shown as `stopped`.

Watch mode is read-only. It ignores an incomplete final journal append until
the next refresh and never repairs or truncates the journal. Complete corrupt
records stop the command with an error. Interactive terminals update in
place; redirected output appends a plain snapshot at each refresh.

`--watch` cannot be combined with `--json` or `--quiet`. Use ordinary
`yakbox --json status` for the existing single-document status response.

## Configuration

User configuration is optional TOML at
`~/.config/yakbox/config.toml`, or `YAKBOX_CONFIG`:

```toml
[defaults]
backend = "fake"
```

Supported environment overrides are `YAKBOX_CONFIG` and `YAKBOX_BACKEND`.
Legacy `[cloud]` settings and provider environment variables are ignored.
See the [migration guide](migration.md#local-models-only) for retired commands.

Missing or malformed manifests are application failures because manifest
loading performs domain validation and may report several related issues.
Direct file arguments such as audio, batch, and shard inputs are validated by Click as usage errors before an operation begins.

## Security

- Worker environments omit credentials and unrelated process settings.
- Output and restore paths are contained beneath declared roots.
- Reference voice use requires an explicit rights basis. Keep licenses and
  consent evidence with project records.

## Failure recovery

Every build stage journals its start, completion, or failure. A failed or
cancelled build releases its target lock and is resumed by the next ordinary
`yakbox build`. Completed artifacts are reused only after their sidecar,
planned path, identity, dependencies, fingerprint, size, and digest match.
Sidecar paths must stay inside the target's managed output root.

Cancellation stops active FFmpeg/FFprobe processes and waits for background
writers to finish before releasing the target lock. A filesystem copy already
in progress must finish before cancellation returns; another build cannot
start writing the same target during that interval. Mixed-format chunk
conversion uses the same media concurrency limit as mastering and encoding,
and skips chunks already in the required PCM format.

Audio inspection and canonical analysis use the first audio stream (`a:0`)
consistently, including files with multiple audio tracks. Quality measurements
are cached independently of policy limits for an unchanged file within the
current process. Changing a policy re-evaluates the stored measurements;
changing the file invalidates them. Edge-silence limits below 250 ms are
measured without the former 250 ms detection floor. Media diagnostics are
drained continuously with a bounded in-memory tail; required quality metrics
are extracted as they arrive.

Existing mastered, delivery, and inspection artifacts need to rebuild under
the updated media algorithms.

Use these read-only checks before repairing anything:

```console
yakbox status
yakbox artifacts verify
yakbox doctor yakbox.toml
```

FFmpeg, FFprobe, local-worker, disk preflight, and stage failures retain the run journal and do not leave owned
`.part` or chunk files. If valid audio was externally changed,
`yakbox artifacts verify` reports the mismatch; rebuild from source rather
than accepting unexplained bytes.

Local worker timeouts cover stalled request writes as well as waits for
per-item progress. Analysis batch deadlines cover the worker handshake,
request write, and response reads.
Rejected analysis-worker handshakes and response frames discard the process,
so the next request starts with a fresh handshake.

Persistent-runtime idle shutdown counts time since the latest authenticated
request or completed operation and keeps an active operation alive.

## Troubleshooting

`FFmpeg is required`: install FFmpeg and ensure both `ffmpeg` and `ffprobe` are
on `PATH`, then rerun Doctor.

`Local Chatterbox package is not installed`: install `yakbox[local]` in the
same environment using the versioned security override command in the
[installation guide](installing-and-releasing.md). Start with a one-sentence
direct test. Worker heartbeat logs are under
`.yakbox/runs/RUN_ID/logs/local-worker.log`.

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
yakbox completion --shell bash > ~/.yakbox-complete.bash
yakbox completion --shell zsh > ~/.yakbox-complete.zsh
yakbox completion --shell fish > ~/.config/fish/completions/yakbox.fish
```

For Bash, source `~/.yakbox-complete.bash` in `~/.bashrc`; for Zsh, source
`~/.yakbox-complete.zsh` in `~/.zshrc`. Bash completion requires Bash 4.4 or
newer. Create the Fish destination directory first and restart the shell.
Regenerate completion after upgrading command surfaces. The command prints
the raw script without terminal formatting; `--json completion --shell zsh`
instead returns the script and shell in the standard CLI envelope. The
existing `_YAKBOX_COMPLETE=SHELL_source yakbox` interface remains available.
