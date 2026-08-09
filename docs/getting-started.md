# Getting started

## Install

Yakbox supports Python 3.14 only. Install the default CLI without
PyTorch:

```console
uv tool install yakbox
```

Install local Chatterbox only on a machine intended to run the model:

```console
uv tool install "yakbox[local]" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.1.0/constraints/chatterbox-security-overrides.txt
```

On Apple Silicon, add the local MLX Whisper aligner when using verified
context extraction for short utterances:

```console
uv tool install "yakbox[local,alignment]" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.1.0/constraints/chatterbox-security-overrides.txt
```

The aligner is optional and runs locally. Install the pinned Whisper model
explicitly with `yakbox whisper models install`; audiobook builds never
download it. The model uses about 1.61 GB on disk.

For optional phoneme-level forced alignment, install
`yakbox[local,alignment,phoneme]`, install eSpeak NG with the platform package
manager, and run `yakbox whisper phoneme-models install`. Builds remain
offline; model downloads occur only through the explicit install command.

FFmpeg and FFprobe are required for mastering, encoding, inspection, and M4B
assembly. Run `yakbox doctor` after installation.

## Build a tiny book

```console
yakbox init my-book
cd my-book
yakbox validate
yakbox plan
yakbox audition --profile default --text "A short voice check."
yakbox build --dry-run
yakbox build
yakbox status
yakbox release check --write-manifest
```

The starter manifest uses the fake backend, so this workflow is fast and
offline. The fake backend produces test fixtures, not listening-quality audio.
Switch the profile to `chatterbox-local` or `resemble` when the source and plan
are correct.

The normal build DAG is:

```text
normalized source
  -> raw synthesis
  -> mastered WAV chapter
  -> optional manuscript-verification release gate
  -> delivery MP3 chapter
  -> technical inspection
  -> optional M4B and immutable release evidence
```

Mastered WAV chapters are retained as the production masters. MP3 chapters are
the default delivery set. M4B assembly is optional.

## Everyday loop

Use `yakbox audition` for short, isolated profile comparisons. Use
`yakbox preview` for a bounded sample that does not mutate production state.
Then run `yakbox build`; unchanged nodes are digest-verified and reused.
The local Chatterbox worker stays alive for the build so the model is loaded
once, while hosted profiles use one pooled client and bounded concurrency.

Interrupted builds retain append-only journals and resume by default:

```console
yakbox build
yakbox build --no-resume
```

`--no-resume` starts a new run but still permits safe artifact reuse. Expert
stage selectors cannot make an incomplete graph releasable:

```console
yakbox build --through synthesize
yakbox build --from master
yakbox build --stage inspect
yakbox build --changed
yakbox build --changed --since RELEASE_ID
yakbox build --failed
yakbox build --missing
```

`--chapter 2-4,7` accepts one selector containing ranges and comma-separated
titles or chapter IDs. `--chapters` remains a compatibility alias; it is not a
repeatable option.
Targets can inherit from another target and define their normal stage range.
Starter projects include `--mode draft`, `--mode proof`, and `--mode release`
targets. Interactive builds show live node progress; `--json`, `--quiet`,
non-TTY output, and `--no-progress` remain clean for automation.
`--quiet` and `--verbose` are mutually exclusive.

All commands accept `--help`. Put global machine-output options before the
command, for example `yakbox --json plan`.
