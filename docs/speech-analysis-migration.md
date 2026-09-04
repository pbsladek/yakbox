# Speech-analysis migration preview

Yakbox still accepts manifest schema version 1, and the `yakbox whisper`
commands remain canonical. The version-2 manifest and `yakbox speech` command
names are internal cutover fixtures. They let the ensemble implementation
exercise real projects without changing current builds or asking users to
migrate early.

## What the preview does

`preview_manifest_migration()` reads a version-1 project and returns a
deterministic in-memory preview. It does not write a manifest, pronunciation
file, cache entry, repair record, or build artifact. The preview:

- maps `whisper_qa` into cache, join, chapter, and optional phoneme policies;
- maps short-utterance candidate limits and the legacy Whisper timeout into the
  shared speech-analysis policy;
- selects the reviewed Whisper, Parakeet, Qwen3-ASR, and Qwen3-ForcedAligner
  registry records;
- moves analysis caching into version 2 and labels version-1 analysis reports
  as historical evidence;
- preserves approved repair audio and approval fingerprints while requiring
  new analysis before that audio can support a version-2 release; and
- reports every setting that cannot transfer exactly.

The internal planner validates the draft version-2 envelope and then delegates
to the existing version-1 planner. The internal build adapter permits fake
profiles only. Normal CLI builds continue to load version 1 directly.

## Pronunciation migration

The old `spoken` field served two different purposes. The preview splits it
into:

- `synthesis_hint`: text sent to the synthesis backend;
- `expected_lexical`: normalized words the listener should hear; and
- `phonemes`: optional pronunciation evidence.

The expected words come from `written`, not from the synthesis hint. For
example, `written = "Asterion"` and `spoken = "As tear ee on"` produce a
synthesis hint of `As tear ee on` while retaining `asterion` as lexical truth.
The preview requires review whenever the two forms tokenize differently. It
never teaches the recognizers to accept a broken-up hint as the intended word.

## When review is still required

The migration itself is automatic except where the old setting has no honest
equivalent. Review remains necessary for:

- pronunciation hints that change lexical tokenization;
- legacy Whisper confidence thresholds, because Whisper, Parakeet, and Qwen
  scores do not share one scale; and
- custom single-engine model or decode controls replaced by reviewed ensemble
  policy and calibration.

Audio qualification does not require routine manual approval. A candidate must
pass independent recognition, consensus, authorized forced alignment, signal
checks, crop re-recognition, and join checks. Human listening remains a
fallback for an unresolved or explicitly review-only case, never a substitute
for missing machine evidence.

## Evidence states

A localized rebuild first creates a **repair candidate**. It may be ready for
listening after changed-chunk, affected-join, mapped post-master-window, and
technical checks, but it is not a release.

A **release verified** artifact is tied to the exact mastered WAV and every
exact delivery container. Yakbox recognizes the complete new master with
Whisper and Parakeet, escalates disagreements and high-risk spans to Qwen, and
uses forced alignment only after lexical correctness is established. It then
decodes each delivery stream and applies the same baseline to the bytes users
receive. A metadata-only container rebuild can reuse lexical evidence only
when its decoded canonical PCM is byte-identical.

The internal report schemas carry these states explicitly. A repair candidate
cannot satisfy a release-verified schema or release check.
Managed candidates live under `.yakbox/repair-candidates/`; release evidence
lives under `release/verified/`. Neither path is reused for the other state.

## Planned command names

The tested cutover map is:

| Current command | Version-2 command |
| --- | --- |
| `yakbox whisper inspect` | `yakbox speech inspect` |
| `yakbox whisper reinspect` | `yakbox speech reinspect` |
| `yakbox whisper verify-manuscript` | `yakbox speech verify-manuscript` |
| `yakbox whisper inspect-joins` | `yakbox speech inspect-joins` |
| `yakbox whisper inspect-phonemes` | `yakbox speech inspect-phonemes` |
| `yakbox whisper calibrate` | `yakbox speech calibrate` |
| `yakbox whisper qualify-voices` | `yakbox speech qualify-voices` |
| `yakbox whisper models ...` | `yakbox models ...` |
| none | `yakbox runtimes ...` |

These names are documentation for the eventual cutover, not current CLI
commands.
