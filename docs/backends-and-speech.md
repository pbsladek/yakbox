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

The current upstream Chatterbox package is maintained but deliberately pins
its ML/UI dependency stack. Some pinned versions currently have published
advisories and cannot be safely overridden without breaking the supported
model contract. The default yakbox install does not contain that stack. Read
the [operations guide](operations.md), use only trusted model/input files, and
treat the full local extra audit as a release blocker rather than ignoring it.

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
