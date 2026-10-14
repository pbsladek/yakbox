# yakbox

Yakbox is a local-first audiobook build system. It turns Markdown manuscripts
into generated narration, mastered WAV chapters, delivery MP3 chapters, and
verified releases.

It supports local Chatterbox and hosted Resemble.ai synthesis. The same speech
backends also power direct text-to-speech commands.

## Install

Yakbox supports macOS and Linux with Python 3.14. Install the CLI from
[PyPI](https://pypi.org/project/yakbox/) with
[uv](https://docs.astral.sh/uv/):

```console
uv tool install --python 3.14 yakbox
yakbox --version
yakbox doctor
```

To upgrade, run `uv tool upgrade yakbox`.

For local Chatterbox synthesis, install the `local` extra with the matching
security overrides:

<!-- x-release-please-start-version -->
```console
uv tool install --python 3.14 "yakbox[local]==0.4.0" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.4.0/constraints/chatterbox-security-overrides.txt
```
<!-- x-release-please-end -->

FFmpeg and FFprobe are required to master, encode, inspect, and assemble audio.
Alternatively, install a wheel downloaded from the
[GitHub Releases](https://github.com/pbsladek/yakbox/releases) page:

```console
uv tool install --python 3.14 ./yakbox-*.whl
```

GitHub Releases include `SHA256SUMS`. See the
[installation and release guide](docs/installing-and-releasing.md) for checksum
verification and the automated GitHub and PyPI release process.

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
For a new local project, use `yakbox init my-book --backend chatterbox-local`.
Add `--play` to `preview` or `audition` to listen, and use `yakbox status --watch`
in another terminal to follow a build. See the [operations guide](docs/operations.md).

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

The local Chatterbox example includes four public-domain LibriVox prompts with
verified provenance and checksums. See the
[third-party notices](THIRD_PARTY_NOTICES.md) and
[licensing policy](docs/licensing.md).

For one- to three-word dialogue, Yakbox can generate longer carrier sentences,
use local Whisper to crop and verify the target, and stop for a hash-bound
listening review. See [Whisper inspection and short-audio
QA](docs/whisper-and-short-audio.md).

## Documentation

See the [documentation](docs/README.md) for manifests, backends, hosted
budgets, cleanup, releases, troubleshooting, and the Python API.
