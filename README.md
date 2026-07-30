# yakbox

Yakbox is a local-first audiobook build system. It turns Markdown manuscripts
into generated narration, mastered WAV chapters, delivery MP3 chapters, and
verified releases.

It supports local Chatterbox and hosted Resemble.ai synthesis. The same speech
backends also power direct text-to-speech commands.

## Install

Yakbox requires Python 3.12 or newer. Install the CLI with
[uv](https://docs.astral.sh/uv/):

```console
uv tool install yakbox
yakbox doctor
```

For local Chatterbox synthesis:

```console
uv tool install "yakbox[local]" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.1.0/constraints/chatterbox-security-overrides.txt
```

FFmpeg and FFprobe are required to master, encode, inspect, and assemble audio.
You can also install a wheel downloaded from the
[GitHub Releases](https://github.com/pbsladek/yakbox/releases) page:

```console
uv tool install ./yakbox-*.whl
```

## Build an audiobook

```console
yakbox init my-book
cd my-book
yakbox validate
yakbox audition --profile default --text "A short voice check."
yakbox build
yakbox release check --write-manifest
```

New projects use the fast offline fake backend. Change the profile in
`yakbox.toml` to use local Chatterbox or Resemble.ai for real narration.

## Examples

The [examples](examples/README.md) cover a tiny offline book, local Chatterbox,
Resemble.ai, multiple voice editions, pronunciations, selective rebuilding,
and M4B assembly. Start with the tiny book:

```console
cp -R examples/tiny-book my-tiny-book
cd my-tiny-book
yakbox validate
yakbox plan
yakbox build
```

## Direct text to speech

```console
yakbox tts "A short local test." --backend chatterbox-local --out local.wav
RESEMBLE_API_KEY=... yakbox cloud tts "A hosted test." \
  --voice-uuid VOICE_UUID --out hosted.wav
```

Use only voices and reference audio for which you have the necessary rights
and consent.

## Documentation

See the [documentation](docs/README.md) for manifests, backends, hosted
budgets, cleanup, releases, troubleshooting, and the Python API.
