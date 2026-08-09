# Backends and direct speech

## Backend selection

```console
yakbox backends list
yakbox backends capabilities fake
yakbox backends capabilities chatterbox-local
yakbox backends capabilities resemble
```

`fake` is deterministic test audio. `chatterbox-local` is the on-device
PyTorch backend. `resemble` is hosted Resemble.ai. A remotely hosted
Chatterbox backend is intentionally unavailable until a real service contract
is selected and verified.

## Local Chatterbox

Local direct commands remain synchronous. Audiobook builds isolate local model
execution in a worker process, use one model sequentially, and never share a
PyTorch model through threads or `asyncio.to_thread()`.

CPU is the default device. `auto` remains available as an opt-in and resolves
to CUDA, MPS, or CPU before the upstream model is loaded. Long direct inputs,
previews, and audiobook chapters use the same semantic 500-character maximum.
Reference-voice conditionals are prepared once per reference and exaggeration
setting instead of being recomputed for each chunk.

```console
yakbox tts "A short test." \
  --backend chatterbox-local --voice narrator --out short.wav
yakbox vc input.wav \
  --backend chatterbox-local --voice narrator --out converted.wav
yakbox batch lines.csv \
  --backend chatterbox-local --voice narrator --out-dir local-output
yakbox verify short.wav
yakbox models
```

Keep auditions short. Profile controls include `device`, `cfg_weight`,
`exaggeration`, `seed`, worker timeout, and one-process resource declarations.
Reference audio must have a documented rights basis.
Yakbox adds no additional watermark; upstream Chatterbox embeds its PerTh
watermark in generated local speech.

The local Chatterbox example includes the provenance-tracked Caro Davy, Nick
Whitley, Andy Minter, and Ruth Golding LibriVox prompts. Its default target uses
Caro Davy. Change only the target's `profile` value to select another voice;
the logical voice identity, prompt path, and tuning remain together in the
named profile.

Chunk audio is assembled at the backend's native PCM format. Explicit pauses
inherit that format, while sentence, clause, and paragraph boundaries receive
stable spacing and short fades. A deterministic per-chunk seed is derived from
the profile seed and chunk identity so cached and regenerated builds agree.

The current upstream Chatterbox package is maintained but pins several stale
ML/UI dependencies. Yakbox applies reviewed, versioned uv overrides to fixed
releases and audits the complete local stack as a release blocker. Use the
local installation command in the
[installation guide](installing-and-releasing.md); omitting its override file
is intentionally unsupported. The default yakbox install does not contain the
machine-learning stack.

Resemble Perth 1.0.1 still uses the removed `pkg_resources.resource_filename`
API to locate its bundled watermark model. Yakbox supplies that one resource
lookup through a temporary `importlib`-based compatibility bridge while Perth
imports, allowing the secure Setuptools release to remain installed. The
bridge does not replace or disable the upstream watermark implementation.
Torchaudio 2.11 writes audio through TorchCodec, so the local extra includes
the matching BSD-licensed TorchCodec runtime instead of failing after model
generation.

## Verified short utterances

Chatterbox can add sounds or words to isolated one-, two-, and three-word
requests. The opt-in `context_extract` strategy synthesizes deterministic
longer carriers, locates the target with local MLX Whisper word timestamps,
crops at guarded PCM boundaries, and transcribes the extracted result again.
Candidates with a missing, substituted, repeated, prefixed, or suffixed word
are rejected. An independent energy detector also guards against extra speech
that Whisper did not tokenize.

This path currently requires Apple Silicon, the `local` and `alignment`
extras, and the explicitly installed, pinned 1.61 GB Whisper Large-v3-Turbo model. One-word results require a
hash-bound listening review by default. Candidate audio and privacy-safe reports
are retained when `keep_candidates = true`; manuscript and carrier text are
represented by hashes in reports. See the
[short-utterance design and test plan](short-utterance-synthesis.md).
Use [Whisper inspection and short-audio QA](whisper-and-short-audio.md) for
model management, take inspection, focused generation, and listening review.

## Direct text sources

Text-taking commands accept one of a positional value, a file, or stdin:

```console
yakbox tts "One line." --backend fake --out one.wav
yakbox tts --text-file passage.txt --backend fake --out passage.wav
printf 'Piped text.' | yakbox tts --text-file - --backend fake --out pipe.wav
```

The direct commands and audiobook nodes share validation, backend resolution,
artifact models, hosted-budget checks, and errors. Direct commands are useful
for testing and automation, but the audiobook lifecycle is the primary
interface for chapter-scale work.

`yakbox batch` is local-only. It rejects `cloud` and `resemble` backend aliases
so hosted work cannot bypass confirmation, budget, journal, and resume rules.
Use `yakbox cloud batch` for provider batches.

## Audition matrices

```console
yakbox audition --profile local \
  --text "A brief comparison." \
  --matrix cfg_weight=0.4,0.6 \
  --matrix exaggeration=0.3,0.5
```

The Cartesian product is deterministic. Outputs are profile-named ordinary
audio files plus a report containing resolved values and a persistable
manifest snippet. Listening and choosing does not create hidden approval
state.
