# Getting started

## Install

Yakbox supports Python 3.12 through 3.14. Install the default CLI without
PyTorch:

```console
uv tool install yakbox
```

Install local Chatterbox only on a machine intended to run the model:

```console
uv tool install "yakbox[local]"
```

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
```

All commands accept `--help`. Put global machine-output options before the
command, for example `yakbox --json plan`.
