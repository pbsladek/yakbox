# Multi-model speech analysis plan

Status: Draft for review
Last updated: 2026-08-13

Decision state: Architecture and automation-first defaults proposed. The initial
release is English-only. No routine build outcome requires listening review.

Upstream snapshot: package, model, and API observations were checked on
2026-08-13. Phase 0 must repeat that review against the exact revisions selected
for implementation.

## Purpose

Yakbox currently uses Whisper Large-v3-Turbo for transcript verification, word
timing, short-utterance extraction, join inspection, and localized repair. That
has worked well enough to expose the next set of problems:

- one recognizer can confidently misread a name or very short utterance;
- Whisper's inferred word times are not always precise enough for a clean crop;
- expected-text prompting can improve timing but cannot remain independent
  evidence that the expected words were spoken;
- Whisper-specific configuration and report fields have leaked into otherwise
  backend-neutral code; and
- full-chapter analysis is too expensive to repeat after a small repair.

This plan adds Parakeet TDT 0.6B v3 and Qwen3-ASR with Qwen3-ForcedAligner while
retaining Whisper Large-v3-Turbo. Each model gets a narrow role. The goal is
better defect detection and more accurate repair boundaries without turning
every build into three full chapter transcriptions.

This is a design plan, not a description of shipped behavior.

## Goals

The finished system should:

- catch transcript defects that one recognizer misses;
- make crop and splice boundaries more accurate than Whisper timing alone;
- keep correctness evidence independent from manuscript-guided timing;
- turn a detected defect into a stable, localized repair request;
- make repair verification scale with changed audio instead of chapter length;
- remain reproducible, offline after installation, and explainable; and
- keep model-specific dependencies outside Yakbox's default installation.

Quality takes priority over speed. Performance work removes repeated or
provably irrelevant computation; it does not relax a release gate.

### Delivery sequence at a glance

| Phase | Primary outcome | Gate before advancing |
| --- | --- | --- |
| 0 | Dependencies, models, licenses, and corpus qualified | Python 3.14, BF16, provenance, offline, quality, and memory gates pass |
| 1 | Backend-neutral internal contracts and draft schemas | Fake-backed type, schema, cache-identity, and import-boundary tests pass |
| 2 | Local model registry, adapters, workers, and audio preparation | Real adapter contracts and isolated runtime checks pass |
| 3 | Deterministic consensus running in shadow mode | Held-out safety truth table and calibration are approved |
| 4 | Read-only schema-v2 migration prepared privately | Migrated fixtures plan correctly; public behavior is unchanged |
| 5 | Short utterance and localized/multi-repair workflows integrated | Carrier, crop, splice, candidate, and affected-scope gates pass |
| 6 | Join, chapter, mastering, and delivery verification integrated | Exact release bytes produce complete, mapped evidence |
| 7 | Layered evidence caches and durable release snapshots integrated | Invalidation, atomicity, cleanup, and repeat-build gates pass |
| 8 | Supervised scheduling and model lifecycle integrated | Cancellation, crash, memory, batching, and accelerator tests pass |
| 9 | Full qualification followed by one public cutover | Quality, performance, packaging, migration, docs, and API gates pass |

No phase makes the new public contract canonical before Phase 9. A failed gate
stops the rollout without weakening the configured policy.

## Non-goals

This work does not add:

- general-purpose transcription or meeting diarization;
- live microphone or a user-facing real-time transcription service;
- Qwen or mlx-audio text-to-speech;
- cloud speech services;
- model training or fine-tuning;
- automatic pronunciation correction based only on ASR guesses; or
- a claim that automated analysis replaces listening review.

The first supported runtime remains Apple Silicon macOS with Python 3.14. The
domain contracts should not prevent later CPU or CUDA workers, but this plan
does not require them.

## Terminology

There is no upstream model named "Turbo ForcedAligner." In this document, the
requested model set means:

- Whisper Large-v3-Turbo through `mlx-whisper`;
- Parakeet TDT 0.6B v3 through `parakeet-mlx`;
- Qwen3-ASR through `mlx-audio`; and
- Qwen3-ForcedAligner 0.6B through `mlx-audio`.

An **independent recognizer** receives audio and a language, but not the
expected spoken text it is being asked to verify. A **forced aligner** receives
both audio and expected text and returns timing. Forced-alignment output is
timing evidence, never correctness evidence.

Here, independent means separate unprompted evidence paths, not statistically
independent errors. Training data, tokenizers, conversions, or acoustic failure
modes may overlap. Do not derive ensemble safety from a binomial voting formula;
measure the complete policy on paired held-out audio.

## Upstream capabilities and constraints

### Whisper Large-v3-Turbo

Whisper remains useful because Yakbox already has a tested MLX adapter,
calibrated clip classes, segment diagnostics, multi-pass decode evidence,
content-addressed caching, and a persistent runtime. It is also the current
baseline against which new behavior must be measured.

Yakbox currently installs `mlx-whisper>=0.4.3,<0.5` on Apple Silicon macOS and
uses `mlx-community/whisper-large-v3-turbo` at a pinned revision.

### Parakeet TDT 0.6B v3

Parakeet is a 600-million-parameter FastConformer-TDT recognizer. NVIDIA
documents automatic punctuation and capitalization, word- and segment-level
timestamps, and support for 25 European languages. The model supports long
audio, but its model card cautions that incomplete sentences and isolated words
can be less accurate because they lack context.

Use offline transcription as the qualification baseline. `parakeet-mlx` also
offers streaming, but its context, attention, and encoder-depth choices can
change computation and output. Streaming is a separately fingerprinted future
optimization and cannot replace offline evidence until corpus tests prove
decision equivalence.

That makes Parakeet a good fit for:

- fast chapter and paragraph transcription;
- the first independent pass over full carrier sentences;
- native word timing used as a cross-check; and
- narrowing the windows that need a more expensive third opinion.

It must not be the sole judge of an isolated word such as `No.` or `Alone?`.

Sources:

- [NVIDIA Parakeet TDT 0.6B v3 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [parakeet-mlx repository](https://github.com/senstella/parakeet-mlx)

The `parakeet-mlx` code is Apache-2.0. The NVIDIA model weights are CC BY 4.0
and require attribution.

### Qwen3-ASR

Qwen3-ASR is available in 0.6B and 1.7B variants. The 1.7B model is the initial
quality-oriented candidate for Yakbox. Because the reference Mac has 64 GB of
unified memory, the qualification baseline is
`mlx-community/Qwen3-ASR-1.7B-bf16`. The 8-bit conversion is a performance
candidate, not the assumed default. It replaces BF16 only if the defect corpus
shows no meaningful loss in false-accept rate, names, numbers, or short speech.

The current mlx-audio Qwen documentation demonstrates 8-bit model identifiers
more prominently than BF16. Treat BF16 loading through the reviewed mlx-audio
revision as an unproven Phase-0 requirement, not as evidence that the generic
loader will necessarily accept that conversion.

Qwen3-ASR is a good fit for:

- a third independent vote when Whisper and Parakeet disagree;
- mandatory verification of names, numbers, abbreviations, and codes;
- one-word and short-dialogue verification; and
- verification of newly regenerated regions.

It should not receive the expected spoken text during these checks.

Sources:

- [Qwen3-ASR repository](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR 1.7B model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [MLX Qwen3-ASR 1.7B BF16 model](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16)
- [MLX Qwen3-ASR model collection](https://huggingface.co/collections/mlx-community/qwen3-asr)

The `mlx-audio` implementation is MIT. The upstream and converted model weights
are Apache-2.0.

### Qwen3-ForcedAligner

Qwen3-ForcedAligner 0.6B aligns supplied text to audio and returns word- or
character-level timestamps. Upstream documents a maximum input duration of five
minutes and support for 11 languages. The MLX implementation supports batch
alignment. As with Qwen3-ASR, BF16 is the qualification baseline and 8-bit is a
candidate optimization.

It is a good fit for:

- locating a verified short target inside a carrier sentence;
- locating verified anchor words around an original repair region;
- measuring word boundaries around a splice or join; and
- bracketing a confirmed chapter mismatch from surrounding accepted speech.

It must not decide whether the supplied text was actually spoken. A forced
aligner is told which text to find, so using it as a correctness vote would
create circular evidence.

Sources:

- [Qwen3-ForcedAligner model card](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- [MLX Qwen3-ForcedAligner BF16 model](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-bf16)
- [mlx-audio Qwen3 ASR and alignment documentation](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/qwen3_asr/README.md)

The upstream and converted model weights are Apache-2.0.

## Design principles

1. Correctness and timing are separate decisions.
2. Expected text never reaches an independent recognition pass.
3. A forced aligner never votes on transcript correctness.
4. Confidence values from different models are never averaged.
5. Each engine has its own calibrated quality gates.
6. Analysis is content addressed and scoped to changed audio.
7. The full model set is optional and imported lazily.
8. Models are installed explicitly at immutable revisions; builds do not
   download them.
9. Story names, pronunciations, and aliases stay in book configuration.
10. The core implementation contains no Anima Cara characters or phrases.
11. Reports retain enough evidence to explain every acceptance and rejection.
12. Performance optimizations may skip redundant computation, but not required
    evidence.
13. Recognizers agreeing with each other is not enough; the accepted transcript
    must match the resolved expected-spoken span.
14. A valid, high-confidence dissent is uncertainty, not noise to average away.
15. Missing required models, unsupported languages, timeouts, and malformed
    results fail closed. There is no silent downgrade of a strict policy.
16. Model conversion provenance is part of model identity, not incidental
    download metadata.
17. Decode seeds, retry behavior, execution identity, and calibration identity
    are explicit. A retry cannot quietly use a different decision procedure.
18. Timestamps, wall time, memory samples, cache-access time, and presentation
    order never affect semantic evidence fingerprints or acceptance.

## Model authority

| Engine | Primary responsibility | Not authoritative for |
| --- | --- | --- |
| Whisper Turbo | Independent transcription and Whisper-specific diagnostics | Forced timing after it has seen expected text |
| Parakeet | Fast contextual transcription and native word timing | Isolated one-word correctness |
| Qwen3-ASR | Independent third vote and high-risk-span verification | Timing supplied by expected text |
| Qwen3-ForcedAligner | Precise boundaries for already-verified text | Transcript correctness |

Multiple decodes from one model are stability evidence for that engine, not
independent votes. Whisper's current authority and sampled-consensus passes can
decide whether the Whisper vote is valid, but together they still count as one
recognizer.

The existing Whisper expected-prompt timing pass should leave the acceptance
path once Qwen forced alignment is qualified. If retained for diagnostics, it
uses `DiagnosticRecognitionRequest`, is labeled `non_voting`, and cannot supply
crop boundaries preferred over the dedicated forced aligner.

### Relationship to phoneme alignment

Yakbox's optional wav2vec2/eSpeak phoneme aligner remains a separate
pronunciation and acoustic check. Qwen ForcedAligner supplies word or character
boundaries; it does not replace phoneme-level evidence. Both consume expected
text, so neither counts as an independent transcript vote.

For a configured pronunciation term, the intended order is:

1. independent recognizers establish the lexical transcript;
2. Qwen ForcedAligner locates the verified word span;
3. the optional phoneme aligner evaluates pronunciation inside that span; and
4. signal and boundary checks decide whether a crop or repair is safe.

The phoneme runtime, model, eSpeak dependency, calibration, and license
exception remain independently configurable and fingerprinted.

## Target decision flow

Ordinary prose begins with two independent passes:

```text
Whisper independent transcript  --+
                                  +--> token consensus --> verified text
Parakeet independent transcript --+                         |
                                                            v
                                                Qwen forced alignment
                                                            |
                                                            v
                                                  precise boundaries
```

Qwen3-ASR runs when:

- Whisper and Parakeet disagree;
- either initial decode fails its engine-specific quality gate;
- the expected span contains a pronunciation entry, name, number,
  abbreviation, or code;
- the clip contains three or fewer words;
- the audio is a new repair candidate; or
- policy explicitly requires all three recognizers.

### Consensus rules

- A recognizer votes for acceptance only when its normalized output matches the
  complete expected-spoken span. Two recognizers producing the same wrong
  text is not a pass.
- Whisper and Parakeet must both match every expected token before ordinary
  prose may pass without a Qwen3-ASR call.
- If they disagree, Qwen3-ASR analyzes the smallest safe merged window covering
  the disagreement. Yakbox also rechecks the initial recognizers on that same
  bounded window so the votes describe comparable audio context.
- Two recognizers matching the expected span may resolve a disagreement only
  when the dissenting result fails its own engine-quality gate or maps to a
  pre-approved normalization equivalence.
- A valid dissent that survives the expanded-context recheck produces a
  `persistent_valid_dissent` rejection in the automation-first strict policy.
  A generated take is discarded and regenerated within its fixed budget; a
  chapter or release fails with an exact repair selector. It is not discarded
  merely because the other two recognizers agree.
- Three distinct results reject the span.
- Any unresolved insertion, deletion, or substitution rejects the span.
- A decode that fails its own engine-quality policy does not receive a vote.
- One-word clips require both Whisper and Qwen3-ASR to match the expected word.
  Parakeet may be recorded as diagnostic evidence but does not satisfy the
  required vote.
- High-risk spans run Qwen3-ASR even when Whisper and Parakeet agree.
- Unexpected speech reported by any valid bounded recheck produces
  a rejection in strict policy.
- PCM clipping, corruption, detached speech, and boundary or VAD disagreement
  beyond their calibrated clip-class limits remain independent hard failures.

Use the following strict-policy truth table after bounded recheck. `M` means a
valid result matching the complete expected-spoken span, `D` means a valid
dissent, `I` means engine-invalid and therefore non-voting, and `-` means Qwen
was not required.

| Whisper | Parakeet | Qwen | Outcome | Reason |
| --- | --- | --- | --- | --- |
| M | M | - | accepted | ordinary baseline agreement |
| M | M | M | accepted | required high-risk evidence agrees |
| M | M | D | rejected | persistent valid dissent after recheck |
| M | M | I | rejected | required high-risk evidence is missing |
| M | D | M | rejected | two matches cannot erase valid dissent |
| M | I | M | accepted | two valid matches and no valid dissent |
| M | I | D | rejected | insufficient matching evidence |
| D | D | M | rejected | only one recognizer matches expected-spoken span |
| I | I | M | rejected | insufficient independent evidence |

Other permutations are symmetric between the baseline recognizers. Three
distinct valid outputs, an unresolved edit, or any hard signal failure rejects.
For one-word clips, Whisper and Qwen must both be `M`; other combinations do not
pass automatically.

The consensus layer compares normalized lexical tokens to the resolved
expected-spoken plan. Raw engine tokens needed for reuse remain in the managed local
evidence cache. Reports contain hashes, token indices, reason codes, and only
the bounded previews permitted by Yakbox's existing privacy rules. Engine
confidence values are not combined into a synthetic score.

Aliases are comparison rules, not pronunciation evidence. Each alias is
directional, reason-coded, and limited to a configured lexical term. Applying
an alias can reconcile spelling, punctuation, or a reviewed spoken-number form;
it cannot turn a phonetic mismatch into a pass. When pronunciation matters,
the span still requires the configured phoneme check. Missing or failed required
phoneme evidence rejects or regenerates the take; optional human review remains
an explicit escape hatch. This matters especially for names that an ASR model
may spell correctly despite a wrong vowel or stress pattern.

The automation-first strict policy has two terminal outcomes: `accepted` and
`rejected`. A soft rejection may be marked `review_eligible`, but Yakbox does not
pause or create a mandatory review step. It regenerates when policy permits and
otherwise fails with actionable selectors. A release cannot silently treat a
rejection as accepted; it needs new passing evidence or an explicit permitted
human disposition.

A human disposition is bound to the exact audio digest, spoken-text-plan and
expected-span hashes, policy fingerprint, and evidence fingerprints. It records
reviewer, timestamp, and notes. It may resolve only a `review_eligible` soft
rejection; it cannot convert clipping, corruption, unsafe boundaries, missing
required evidence, or another hard rejection into a verified release. Changing
any bound input makes the disposition stale.

If human disposition is approved, expose it through explicit review commands,
not a generic `--force` flag:

```console
yakbox speech reviews list MANIFEST
yakbox speech reviews show MANIFEST REVIEW_ID
yakbox speech reviews resolve MANIFEST REVIEW_ID --decision accept --notes-file NOTES
```

`show` identifies the managed contextual audio and evidence without printing
complete manuscript text. `resolve` revalidates every bound digest immediately
before an atomic write, refuses hard failures, and records a stable reviewer
identifier supplied by user configuration. JSON output follows the normal CLI
contract and never includes Rich formatting. Reviewer notes are bounded UTF-8
user input; Yakbox never auto-populates them with manuscript or transcript text.

### Contextual target projection

Never verify a suspicious word by searching for that word anywhere in an ASR
transcript. A bounded recheck contains the target plus deterministic left and
right speech context whenever available. Its plan identifies the owned target
tokens and the context anchors separately.

Align the complete recognized window to the complete expected-spoken window,
then project the result onto the owned target. The engine may vote on that target
only when its location is unique and the surrounding sequence alignment is
unambiguous. Unexpected speech before or after the target remains an insertion;
hearing `No` inside `naaah ... No` therefore does not satisfy the target.

If an edge token or weak context prevents a unique projection, expand to the
next deterministic window once and rerun every participating recognizer on the
same frames. If the target remains ambiguous, reject it. The automation-first
strict policy marks this as review eligible without creating a mandatory pause.
Do not use substring matching, the forced aligner, or the manuscript position
to manufacture a recognition vote.

### Defect localization rules

Direct forced alignment is valid only for text whose presence was independently
verified. When consensus reports a deletion, substitution, or unexpected
insertion, do not force the missing expected token onto the defective audio and
accept the resulting timestamp as its location.

Instead, choose the nearest independently accepted lexical anchors before and
after the defect, force-align those anchors in a bounded contextual window, and
bracket the repair interval between their safe boundaries. Cross-check native
ASR timing, VAD, waveform evidence, and the assembly map. For an insertion, a
recognized unexpected sequence may be aligned for diagnostic timing, but it
does not become accepted manuscript text.

If two safe anchors cannot be established, expand once to the next stable
assembly or paragraph boundaries. A remaining one-sided or ambiguous location
requires broader replacement or fails closed; the user may explicitly review it
but it never receives a falsely precise automatic crop. Reports distinguish
direct verified-target timing from anchor-bracketed defect timing.

### High-risk span selection

Do not put a general named-entity recognizer in the first version. It would add
another model and another uncertain decision to a safety gate. Build the
high-risk set deterministically from:

- entries in the reviewed pronunciation lexicon;
- an optional book-level list of verification terms and phrases;
- tokens containing digits or normalized spoken-number sequences;
- abbreviations and codes matched by documented lexical rules;
- utterances at or below the configured short-word threshold; and
- every changed repair span.

Capitalization may add a diagnostic candidate, but it is not authoritative in
a system that normalizes Markdown and sentence starts. The selected high-risk
span and the rule that selected it appear in the plan fingerprint and report.

### Spoken-text authority and source mapping

Verify the resolved words intended for the listener, not raw Markdown and not
whatever spelling happened to be sent to one TTS backend. Yakbox's current
`SpeechSegment.text` is the closest authority because source parsing has already
removed non-spoken structure, routed dialogue, stripped configured attribution
tags and quote delimiters, and applied pronunciation substitutions. Version 2
must preserve those stages without collapsing their distinct meanings.

Create an immutable `SpokenTextPlan` before synthesis. For every segment it
records, by hash or managed internal value as privacy rules require:

- source file digest and exact source line and character spans;
- display text from the manuscript;
- synthesis text sent to the selected backend;
- expected lexical tokens representing the words the listener should hear;
- optional expected phonemes for reviewed pronunciation terms;
- speaker, profile, language, and boundary identity; and
- a lossless transform map covering Markdown removal, dialogue and attribution
  handling, pronunciation rules, number expansion, and backend-specific text
  preparation.

Every transform has a stable kind, input and output spans, configuration rule
identity, and deterministic version. Deleted punctuation or attribution text
maps to an empty spoken span rather than disappearing from provenance. This lets
an ASR defect map back through a pronunciation substitution or stripped dialogue
delimiter to the original source location.

Pronunciation configuration must distinguish four concepts that the current
`written` and `spoken` pair can blur:

- the written manuscript form;
- an optional synthesis hint or respelling;
- the intended lexical form used for independent recognition; and
- optional phoneme evidence used for pronunciation QA.

A synthesis hint such as a broken-up phonetic spelling is never automatically
an accepted transcript alias. For example, a backend hint intended to produce
the name `Asterion` still has the expected lexical token `asterion`; recognition
of four unintended words must not pass merely because those words appeared in
the TTS input. Version-2 migration treats the old `spoken` field as a synthesis
hint and requires review when it changes lexical tokenization.

Consensus compares recognizer output with `expected_lexical_tokens`. Forced
alignment receives the corresponding reviewed aligner text. Neither operation
re-tokenizes raw source Markdown. The `SpokenTextPlan` digest is part of
synthesis, analysis-window, consensus, repair, and release fingerprints.

### Lexical normalization contract

Implement normalization as a versioned sequence of typed stages rather than a
collection of regular expressions called in different orders. Unicode form,
case handling, punctuation, contractions, hyphenation, digits, abbreviations,
and language-specific tokenization each produce a trace from input indices to
output lexical-unit indices.

Represent reviewed equivalences as bounded directional token sequences. Reject
cycles, empty accepted forms, duplicate or conflicting canonical forms,
unbounded sequence lengths, and ambiguities that cannot be resolved from the
expected side. Compile them into a bounded matcher rather than enumerating the
Cartesian product of every alias choice. The normalization version, language
rules, and equivalence-set digest are policy inputs.

Sequence alignment uses one documented cost model and deterministic tie-break
order, emitting source, expected-spoken, and recognized token indices for every
edit. A rule may classify an edit as an approved lexical equivalence; it may not
erase the underlying evidence or satisfy a separate pronunciation requirement.

## Fit with the current Yakbox architecture

Yakbox already has several pieces this design should preserve:

- backend-neutral timing contracts in
  [`src/yakbox/speech/alignment.py`](../src/yakbox/speech/alignment.py);
- a lazy MLX Whisper adapter in
  [`src/yakbox/local_alignment.py`](../src/yakbox/local_alignment.py);
- content-addressed analysis caching in
  [`src/yakbox/whisper_cache.py`](../src/yakbox/whisper_cache.py);
- full-chapter verification with bounded mismatch rechecks in
  [`src/yakbox/whisper_qa.py`](../src/yakbox/whisper_qa.py);
- changed-chunk and affected-join verification in
  [`src/yakbox/audiobook/build.py`](../src/yakbox/audiobook/build.py);
- content-addressed generation, extraction, DSP, and verification stages in
  [`src/yakbox/audiobook/repair.py`](../src/yakbox/audiobook/repair.py) and
  [`src/yakbox/audiobook/repair_cache.py`](../src/yakbox/audiobook/repair_cache.py);
  and
- a persistent local runtime in
  [`src/yakbox/local_runtime.py`](../src/yakbox/local_runtime.py).

The main structural problems are:

- `WhisperQaPolicy` owns settings that are no longer Whisper-specific;
- `ShortUtterancePolicy` selects the aligner used by unrelated chapter, join,
  and repair workflows;
- `AlignmentResult` combines generic timing with Whisper-only decode fields;
- independent transcription and expected-text timing share one cache contract;
- analysis cache keys include expected text even when recognition is
  unprompted; and
- the runtime exposes a single Whisper-shaped `align` operation.

## Target runtime topology

Do not keep extending one process that owns Chatterbox, Torch/MPS, Whisper, and
every MLX analysis model. That makes dependency resolution, memory reclamation,
crash recovery, and model upgrades unnecessarily coupled.

The target is a small Yakbox supervisor with capability-specific workers:

```text
Yakbox core and build planner
  |
  +-- TTS worker
  |     chatterbox-tts, Torch, voice-conditioning cache
  |
  +-- Whisper worker
  |     mlx-whisper, Whisper model
  |
  +-- Parakeet worker
  |     parakeet-mlx, Parakeet model
  |
  +-- Qwen analysis worker
        mlx-audio STT code, Qwen3-ASR, Qwen3-ForcedAligner
```

Each dependency family may run in a built-in, pinned `uv` environment using
Python 3.14. The audiobook manifest selects a registered engine name; it cannot
provide package requirements, indexes, executable paths, environment variables,
or arbitrary commands. This follows the isolated-runtime direction already
described in `docs/AUDIOBOOK_BUILD_SYSTEM_PLAN.md`.

Workers use a versioned local protocol and receive only validated local paths,
bounded options, hashes, language identifiers, and the expected text required
by a forced-alignment request. Independent-recognition requests cannot carry
expected text. The supervisor owns timeout, cancellation, restart, memory, and
idle-lifetime policy. A worker never writes a final Yakbox report or artifact
record directly, and protocol diagnostics never echo manuscript or transcript
payloads.

Keep the current authenticated workspace runtime as the client-facing control
plane, but make analysis workers its supervised child processes. A worker has
no listening socket. Use a framed, size-bounded message protocol over inherited
stdin and stdout, reserve stderr for bounded diagnostics, and launch the worker
without a shell. This leaves one authenticated loopback boundary while reducing
the network and lifecycle surface of every model process.

Spawn workers with an allowlisted environment and managed working and temporary
directories. Do not inherit cloud credentials, provider tokens, arbitrary
Python paths, package-index settings, proxy credentials, or debug hooks. Supply
only required locale, device, offline-model, and runtime variables whose values
are owned and validated by Yakbox.

This process split is the default architecture even if all packages happen to
resolve in one environment. A measured shared-MLX worker may be added later as
an optimization, but it must produce byte-equivalent normalized evidence and
must not become a requirement for correctness.

The direct in-process adapters remain useful for adapter tests and embedded
development. Production audiobook builds go through the worker protocol so a
model crash or dependency conflict cannot corrupt the build coordinator or TTS
session.

### Failure and degradation policy

- A missing required worker or model fails during planning or preflight.
- An unsupported book language fails policy validation before audio work.
- An out-of-memory result reduces batch size or evicts an idle worker; it does
  not switch precision, model, or consensus policy automatically.
- A crashed worker may restart once for an idempotent analysis request. A
  second failure stops the stage with preserved diagnostics.
- A timeout or cancellation never commits partial evidence as a cache hit.
- A malformed engine result fails adapter validation and cannot vote.
- A strict analysis or release-promotion path never falls back to fewer
  recognizers.
- A diagnostic or explicitly configured reduced policy records that weaker
  policy in every fingerprint and cannot create release artifacts.

### Shared audio preparation

Decode and resample each source digest once into a canonical analysis WAV before
calling recognizers. The provisional format is mono 16 kHz PCM, selected because
all target models consume 16 kHz speech. The exact sample format, resampler,
FFmpeg version, channel-mix rule, dither rule, and preprocessing implementation
form one fingerprint.

All engines analyzing a consensus span receive the same canonical frames. This
avoids each upstream loader choosing different resampling and window rounding,
and it avoids decoding the same chapter three times. Phase 0 must compare this
path with each backend's native loader before it becomes authoritative. If one
backend requires materially different preparation, record that as an
engine-specific preprocessing identity rather than hiding the difference.

Hash and materialize each planned canonical window once in the managed analysis
cache, atomically, with its exact PCM digest and frame count. All recognizers
reference that same window identity. Do not ask three upstream loaders to decode
the chapter or create three nominally equivalent temporary crops. A backend may
consume a memory view or the managed WAV, but both paths must validate the same
PCM digest and produce equivalent adapter evidence.

### Shared analysis-window plan

Create one immutable `AnalysisWindowPlan` for each canonical audio digest. Each
entry records an integer-frame span, expected-spoken-plan reference, reason
for the boundary, overlap ownership, and maximum permitted engine context. The
baseline recognizers receive the same windows. Qwen disagreement rechecks use
the same merged source frames for all three recognizers, rather than comparing
results produced from different context.

Prefer chapter assembly boundaries and verified pauses that do not split a
lexical token. When a model limit forces another split, use deterministic
fixed-frame windows with calibrated overlap. Assign every overlap token to one
window using a stable timestamp-and-token rule so stitching cannot duplicate or
drop speech. The window plan, stitching version, and canonical-audio digest are
fingerprinted and reported.

Do not delegate authoritative splitting or overlap stitching to an upstream
package's undocumented defaults. An adapter may sub-window internally to meet a
hard model limit, but it must return the mapping to the Yakbox source frames and
pass equivalence tests. Forced-alignment requests are always bounded below the
Qwen five-minute limit and normally cover only one carrier, join, or repair
region.

The canonical analysis file also carries a `CanonicalAudioIdentity`: source
audio digest and format, normalized-audio digest and format, exact frame counts,
preprocessing fingerprint, and the tested mapping between both coordinate
spaces. Analysis boundaries are converted back to source frames with explicit
outward-rounding rules and resampler-delay compensation before cropping. A
16 kHz analysis frame is never used directly as a 44.1 or 48 kHz splice frame.

### Capability resolution

At planning time, resolve the book's canonical language identifier against a
versioned capability matrix for every required engine. The matrix records
recognition, word-timing, character-timing, batching, and maximum-duration
support separately; a package being importable does not imply that a model can
perform every operation for every language.

Normalize book languages to a documented BCP 47 subset, then map that value to
each backend's accepted language name or code inside the adapter. The mapping,
lexical tokenizer, Unicode normalization, number and abbreviation rules, and
calibration table are language-specific fingerprints. Do not pass arbitrary
manifest strings through to an upstream model.

Upstream model coverage is not Yakbox release coverage. A language becomes
release-qualified only after the complete ensemble, text transforms, timing,
and defect corpus pass for that language. The first release qualifies English
only, without hard-coding English behavior into the contracts. Other model-
supported languages remain unavailable for strict or release use until they
have equivalent evidence.

For the first release, strict and release manifests require
`book.language = "en"`. Adapters map it to each engine's reviewed English value.
The qualification corpus still covers the American, British, Australian, and
other English accents represented by supported voices. Accent and phoneme-locale
configuration may refine pronunciation checks, but ASR language detection is
diagnostic and cannot silently change the resolved policy language.

Each authoritative analysis window has one resolved language. A future segment-
level language override can split windows at stable speech-segment boundaries,
but an unmarked code-switch inside a window is not silently treated as ordinary
English evidence. It requires an explicitly qualified multilingual policy or
review.

If the requested strict pipeline cannot be constructed, preflight fails with
the missing capability and supported alternatives. A different configured
aligner or an explicitly weaker diagnostic policy may be valid for another
book, but Yakbox never silently substitutes it or labels its evidence as the
original policy.

## Target contracts

Replace the overloaded alignment contract with distinct recognition, forced
alignment, consensus, and verification models.

Use sample-indexed windows at contract and cache boundaries:

```python
@dataclass(frozen=True, slots=True)
class AudioSpan:
    audio_digest: str
    start_frame: int
    end_frame: int
    sample_rate: int
```

Seconds are derived for human reports. Integer frames avoid floating-point
rounding changing a cache key or moving a splice by one sample. The digest
prevents a valid span from being applied to a different audio coordinate space.

```python
class SpeechRecognizer(Protocol):
    @property
    def fingerprint(self) -> str: ...

    async def recognize(
        self,
        audio: Path,
        *,
        language: str,
        span: AudioSpan | None = None,
    ) -> RecognitionResult: ...


class ForcedAligner(Protocol):
    @property
    def fingerprint(self) -> str: ...

    async def force_align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
        span: AudioSpan | None = None,
    ) -> ForcedAlignmentResult: ...
```

`ForcedAligner` is the low-level backend contract and cannot establish text
authority. Authoritative crop and repair services accept a `VerifiedTextSpan`
containing the consensus-result fingerprint and accepted lexical indices, then
construct the low-level request. Anchor-bracket requests carry two verified
anchor spans. Diagnostic alignment of observed unexpected text is explicitly
typed `non_authoritative` and cannot flow into an automatic crop.

`RecognitionRequest` has no expected-text or prompt field. A separate
`DiagnosticRecognitionRequest` may support explicitly biased experiments, but
its result is marked `non_voting` in the type and schema.

Every result carries a `ModelArtifactIdentity` containing:

- backend package name and version;
- adapter and worker protocol versions;
- converted model repository and immutable revision;
- converted model directory fingerprint;
- upstream model repository and immutable revision;
- conversion source, tool, version, recipe, and precision policy;
- precision or quantization settings; and
- decode configuration fingerprint.

This distinguishes an upstream model update from a new MLX conversion of the
same upstream weights.

Each result also carries a redacted `ExecutionIdentity`: worker-artifact and
lock digests, Python version, operating-system family and version, architecture,
MLX and Metal runtime versions when observable, device class, determinism mode,
and declared decode seeds. It never contains a machine serial number, username,
or cache path. Calibration tables declare the execution classes they qualify;
release preflight rejects an unqualified class instead of assuming that an M5
threshold applies unchanged to another backend or device.

Use deterministic greedy decoding where it meets quality gates. When an
engine-quality policy intentionally uses several sampled decodes, pin and
record the complete ordered seed set. Those passes remain one engine vote. A
worker restart repeats the identical request; changing a seed, beam width,
temperature, precision, or preprocessing creates new evidence and cannot be
presented as a transparent retry.

Every serialized public document uses Yakbox's common contract envelope: an
absolute `$schema` URI, schema version, Yakbox version, and UTC timestamp where
applicable. Reports refer to managed evidence by digest and never expose model
cache paths, credentials, or complete manuscript text.

Define a canonical semantic serialization for every evidence fingerprint.
Exclude generated-at timestamps, operation IDs, wall time, memory measurements,
cache-hit state, local paths, and human display ordering. Keep those fields in
the report envelope for operations and diagnosis, but changing them cannot
invalidate or alter a speech decision.

### RecognitionResult

The generic recognition result contains:

- normalized and raw transcript hashes;
- recognized lexical tokens and optional word timings;
- detected or requested language;
- engine, package, model, and revision fingerprints;
- engine-specific diagnostics in a typed evidence variant;
- the analyzed sample span; and
- issues that invalidate the engine's vote.

Word scores include an explicit score kind and calibration identifier. A score
from one engine is never treated as equivalent to a score from another.

### Adapter result validation

Treat every upstream result as untrusted structured data. Before it can be
cached or voted, require bounded token and segment counts, valid Unicode, a
known language mapping, finite timestamps, monotonic ordering, and boundaries
inside the requested frames. Reject NaN, infinity, negative duration, malformed
batch shape, output too large for the analyzed duration, and—when an engine
reports them—unknown finish state, truncated generation, or repetition
termination. Do not silently clamp a model's timestamp into range or truncate
its text to make a result valid.

Convert upstream floating-point seconds to analysis frames with one documented
rounding policy and retain the original bounded numeric values in engine
evidence. Engine-specific finish reasons, token counts, and timing shape live in
typed variants; they never escape as arbitrary objects or determine acceptance
without a calibrated gate.

### ForcedAlignmentResult

The forced-alignment result contains:

- aligner-text and expected-lexical-span hashes;
- aligned words or characters with boundaries;
- coverage and monotonicity checks;
- verified-target, verified-anchor, or non-authoritative diagnostic purpose;
- analyzed sample span;
- model fingerprint; and
- timing diagnostics and reason codes.

It does not contain an `accepted_transcript` field.

### ConsensusResult

The consensus result contains:

- recognition-result fingerprints;
- expected lexical-unit sequence hash;
- per-token votes;
- normalized aliases applied during comparison;
- disagreement spans;
- high-risk spans that required Qwen3-ASR;
- accepted and rejected spans, including review-eligible soft rejections; and
- stable reason codes.

### SpeechVerification

The final verification document combines consensus, forced timing when needed,
signal evidence, policy identity, and an acceptance decision. This becomes the
common report used by chapter, repair, short-utterance, and join workflows.

## Proposed manifest configuration

Schema version 2 should centralize model and decision settings under
`speech_analysis`.

```toml
schema_version = 2

[speech_analysis]
preset = "strict"
cache_enabled = true
cache_directory = ".yakbox/cache/speech-analysis"
baseline_recognizers = ["whisper", "parakeet"]
escalation_recognizer = "qwen"
forced_aligner = "qwen-forced"
always_use_escalation_for = [
    "one_word",
    "verification_term",
    "pronunciation_entry",
    "number",
    "repair",
]

[speech_analysis.engines.whisper]
backend = "mlx-whisper"
model = "mlx-community/whisper-large-v3-turbo"
revision = "PINNED_COMMIT"
timeout_seconds = 180

[speech_analysis.engines.parakeet]
backend = "parakeet-mlx"
model = "mlx-community/parakeet-tdt-0.6b-v3"
revision = "PINNED_COMMIT"
decoding = "greedy"
chunk_seconds = 120
overlap_seconds = 15

[speech_analysis.engines.qwen]
backend = "mlx-audio-qwen3-asr"
model = "mlx-community/Qwen3-ASR-1.7B-bf16"
revision = "PINNED_COMMIT"

[speech_analysis.engines.qwen-forced]
backend = "mlx-audio-qwen3-forced"
model = "mlx-community/Qwen3-ForcedAligner-0.6B-bf16"
revision = "PINNED_COMMIT"
maximum_window_seconds = 300

[speech_analysis.consensus]
ordinary_acceptance = "all_baseline_match"
one_word_required_recognizers = ["whisper", "qwen"]
reject_unresolved_disagreement = true
reject_unexpected_speech = true
valid_dissent = "retry_then_reject"
missing_required_engine = "error"

[short_utterances]
strategy = "context_extract"
maximum_words = 3
candidate_count = 5
maximum_rounds = 3
automatic_join_inspection = true

[repairs]
verification = "strict"
approval_output = "repair_candidate"
candidate_verification_scope = "affected"

[repairs.candidates]
per_round = 5
maximum_rounds = 3

[runtime.analysis]
isolation = "dependency_family"
maximum_resident_workers = 2
idle_timeout_seconds = 900
# maximum_memory_bytes is optional and selected from measured platform data.
```

The exact revisions are selected and recorded during dependency qualification.
The default preset should make the safe choice. Advanced fields exist for
calibration and experimentation, not because every book should need to tune
model internals.

`strict` is the default analysis policy for builds and repairs. Release
promotion is a separate operation scope that adds complete-master evidence
requirements; it is not a preset that can accidentally be selected for an
ordinary audition. A `fast` or diagnostic preset cannot create a release
artifact. Presets expand into canonical policy values before fingerprinting so
changing a preset definition invalidates dependent evidence.

## Phase 0: Qualify dependencies and models

This phase changes no public Yakbox contract.

### Work

1. Resolve `parakeet-mlx` and `mlx-audio[stt]` under Python 3.14 in a clean
   `uv` environment.
2. Audit the full dependency graph, licenses, wheel hashes, and known
   advisories.
3. Verify package source and wheel contents against the tagged upstream source.
4. Pin reviewed package ranges and refresh `uv.lock` with `uv`.
5. Pin immutable revisions for all selected MLX model repositories.
6. Verify model licenses, source URLs, conversion provenance, file allowlists,
   and checksums under `docs/licensing.md`.
7. Confirm that builds perform no network access after explicit model
   installation.
8. Reject any path that requires `trust_remote_code` or unsafe weight loading.
9. Run import, model-load, transcription, forced-alignment, and unload smoke
   tests on the M5 Mac.
10. Prove that the pinned mlx-audio revision loads the selected BF16 Qwen ASR
    and forced-aligner conversions from verified local paths. If it cannot, stop
    qualification rather than substituting 8-bit.
11. Measure cold load, warm inference, peak resident memory, Metal memory, and
    model-switch time.
12. Use Qwen3-ASR 1.7B BF16 and Qwen3-ForcedAligner BF16 as the quality
    baselines. Compare their 8-bit conversions as candidate optimizations.
13. Exercise at least 150 consecutive calls—or twice the largest qualified
    chapter workflow, whichever is greater—per long-lived worker. Mix short and
    long windows, batches, invalid requests, cancellation, and unload cycles,
    then verify output, memory, and latency stability. A short smoke test is not
    enough evidence for a persistent model process.
14. Verify timeout, cancellation, worker termination, restart, and model unload
    behavior under memory pressure.
15. Record whether each upstream repository has CI, tests, a security policy,
    signed releases, and PyPI Trusted Publishing. Missing controls do not
    automatically reject a package, but they raise the review and pinning bar.
16. Compare one combined analysis environment with dependency-family-isolated
    environments. Choose from resolved dependency evidence and runtime
    behavior, not installation convenience.
17. Compare Parakeet offline and streaming output on the frozen corpus. Keep
    streaming non-authoritative unless decisions and owned-token projections
    are equivalent under every qualified window class.
18. Encode the defect corpus with each supported delivery preset and determine
    whether a signal-and-timing equivalence gate can safely replace full
    delivery-stream recognition. Treat full recognition as the baseline.

### Evaluation corpus

Create a licensed, versioned corpus containing:

- isolated one-word dialogue;
- one-to-three-word questions;
- full carrier sentences;
- proper names and unusual spellings;
- numbers, room identifiers, abbreviations, and codes;
- clean and intentionally corrupted joins;
- clipped beginnings and endings;
- extra syllables such as `naaah ... No`;
- long pauses and unnatural internal pauses;
- chapter excerpts; and
- MP3 or other supported delivery encodes with delay, boundary, clipping, and
  low-bitrate stress cases; and
- real repaired regions represented by source-controlled metadata and approved
  reference audio.

Record transcript ground truth and manually reviewed word boundaries. Do not
commit third-party audio without the source, rights basis, and checksum required
by `docs/licensing.md`.

### Measurements

- word and character error rate;
- exact token accuracy;
- name and number accuracy;
- insertion, deletion, and substitution counts;
- hallucination rate on silence and boundary noise;
- word-boundary mean absolute error and P95 error;
- crop contamination and clipped-word rate;
- false acceptance and false rejection;
- cold and warm real-time factor;
- batch throughput;
- peak memory; and
- model-load count per workflow.

### Evaluation discipline

Split the corpus into calibration and held-out sets by source passage and voice,
not by individual clip. Carrier and extracted versions of the same utterance
must stay in one partition. Include multiple voices, accents, speaking rates,
genders, dialogue lengths, and acoustic conditions so thresholds do not fit one
Anima Cara chapter or one narrator. Partition and qualify the corpus separately
for every claimed release language; success on English does not qualify another
language from the model card.

Set the strict-policy truth table and success thresholds before evaluating the
held-out set. Do not tune on held-out failures and then report the same set as
independent evidence. Add newly discovered production defects to a future
regression set after recording the original failure.

Pre-register non-inferiority margins, the required one-sided confidence bound
for false acceptance, and the minimum sample count in each risk class. Derive
corpus size from those bounds; zero observed failures in a small set is not
enough. Compare candidate and baseline policies on paired clips, and calculate
uncertainty with source-passage and voice clusters so several crops from one
recording are not treated as independent observations.

Boundary ground truth needs human review. For a representative subset, use two
reviewers or one reviewer on two separate passes and record disagreement. A
boundary metric finer than reviewer agreement is not meaningful.

Randomize and blind listening comparisons where the reviewer is judging repair
or join quality rather than labeling objective transcript content. Preserve the
review order and anonymized candidate identities with the evaluation result so
a later model or DSP change can use the same protocol.

Report results by clip class and risk class, not only as one aggregate WER. The
primary safety result is false acceptance of known bad audio. False rejection,
latency, and memory are secondary constraints. Preserve raw per-case outcomes
so a later model upgrade can be compared against the same frozen corpus.

Measure repair selection as a workflow, not only one candidate at a time.
Repeatedly generating until something passes increases the probability of a
false acceptance. Evaluate the configured candidate count, round limit, early-
stopping rule, and ranking rule on grouped candidate sets, and require the
workflow-level false-accept bound to pass. Candidate budgets and selection rules
are policy inputs and fingerprints, not invisible retry behavior.

### Exit criteria

- Python 3.14 resolves and runs all selected packages.
- All selected model artifacts are pinned, licensed, and verified.
- Runtime inference works offline.
- No new path increases false acceptance on the known-defect corpus.
- The Qwen precision is selected from measured quality and stability results.
- Peak memory remains within the configured runtime limit.
- Persistent workers survive the endurance and restart tests without output
  drift, leaked memory, or unrecoverable Metal state.
- The selected worker topology has a reproducible environment fingerprint.

## Phase 1: Define internal contracts and draft schema version 2

Build the backend-neutral domain and worker protocol before changing the public
manifest. Keep the version-2 schema as an internal draft until real adapters and
shadow consensus prove that the fields describe actual evidence.

### Work

- Add `SpokenTextPlan`, `AudioSpan`, `CanonicalAudioIdentity`,
  `MasteringAudioIdentity`, `DeliveryAudioIdentity`, `AnalysisWindowPlan`,
  `ModelArtifactIdentity`, `VerifiedTextSpan`, capability-matrix, recognition,
  forced-alignment, consensus, and verification domain types.
- Add separate `SpeechRecognizer` and `ForcedAligner` protocols.
- Define the versioned worker request, response, cancellation, status, and error
  contracts.
- Add fake recognizers, fake forced aligners, and worker protocol fixtures.
- Draft `SpeechAnalysisPolicy` and the version-2 manifest/schema shape without
  making it canonical yet.
- Draft generic structured-report schemas and reason-code registries.
- Define cache identities for recognition, forced alignment, consensus, and
  final verification.
- Add import boundaries so core domain and planning code cannot import an MLX
  package or concrete adapter.
- Extend Import Linter contracts so speech domain and consensus modules cannot
  import worker adapters, worker code cannot import audiobook or CLI modules,
  and the supervisor depends on protocol contracts rather than backend classes.

### Exit criteria

- Generic contracts represent every required piece of evidence without an
  untyped engine dictionary at a public boundary.
- The worker protocol rejects expected text in independent recognition
  requests.
- The consensus service cannot accept `ForcedAlignmentResult` as a vote.
- Draft schemas validate all fake-backed reports.
- The default fake-backend suite remains deterministic and offline.
- The new package boundaries pass `lint-imports` without ignored imports or
  file-wide lint exemptions.

## Phase 2: Add model management and adapters

### Model registry

Generalize the current Whisper model manager into a backend-neutral service.
The target cutover command surface is:

```console
yakbox models list
yakbox models install whisper
yakbox models install parakeet
yakbox models install qwen
yakbox models install qwen-forced
yakbox models status
yakbox models verify
yakbox models path qwen
yakbox models licenses
```

Before Phase 9, exercise this service through tests and the explicit
qualification harness while the existing Whisper model command remains the
canonical public entry point. Do not expose a partially migrated generic CLI
before the manifest and report contracts are ready.

Extend planning and deep doctor checks to validate the complete resolved
pipeline—worker artifact and lock, Python 3.14, model integrity, language
capabilities, calibration availability, disk space, and measured memory policy—
before Chatterbox generates anything. Baseline doctor remains lightweight,
offline, and free of heavy optional imports; explicit deep checks may start a
worker but never download a package or model.

Each model record contains:

- logical engine name;
- backend package and version;
- converted model repository and immutable revision;
- upstream model repository and immutable revision;
- conversion source, tool, version, recipe, precision policy, and verification
  state;
- tensor precision or quantization;
- local directory and directory fingerprint;
- model license and source URL;
- expected file allowlist;
- total size; and
- integrity and readiness state.

Builds never install models. Installation is an explicit lifecycle operation.

### Adapters

Implement four lazy adapters:

- `MlxWhisperRecognizer`;
- `ParakeetMlxRecognizer`;
- `MlxAudioQwenRecognizer`; and
- `MlxAudioQwenForcedAligner`.

Adapters accept only workspace-controlled local paths. They do not accept URLs,
base64 input, arbitrary model identifiers at runtime, or server endpoints even
if an upstream package supports them.

At worker launch, the supervisor supplies resolved, allowlisted workspace,
analysis-cache, and model roots. Requests use relative paths plus expected file
digests. The worker independently rejects symlinks, non-regular files, root
escapes, and digest mismatches after opening the input. Parent-side validation
alone is insufficient because a path can change before the child reads it.

The `mlx-audio` adapter imports only the STT/Qwen implementation. Yakbox does
not expose or initialize mlx-audio's TTS, microphone, server, or
speech-to-speech features.

Implement the canonical-audio preparation service and deterministic window
planner beside the adapters. Adapters consume its immutable output rather than
decoding arbitrary source formats independently. Preserve the source-to-analysis
frame map needed by repair and join services.

Wrap each dependency family in the Phase 1 worker protocol. The worker
definition, not the audiobook manifest, owns its Python version, package
constraints, executable module, allowed model families, and environment
fingerprint. The Qwen worker may keep ASR and forced-alignment models in one
dependency environment, but their model lifetimes remain independent.

Add an explicitly opted-in harness under `tests/live/` for model installation
verification, real adapter contracts, corpus runs, and shadow comparisons. It
uses only reviewed built-in engine definitions and local model records; it does
not accept arbitrary dependencies or commands from a book manifest.
The final local audiobook gate also runs through
[`tests/live/test_local_chatterbox_e2e.py`](../tests/live/test_local_chatterbox_e2e.py)
and preserves its technical report and completed listening-review artifact.

### Exit criteria

- All adapters satisfy the same fake-backed contract suite.
- Missing packages and models produce typed, actionable errors.
- Importing Yakbox does not import MLX or any model package.
- Model revisions and adapter settings contribute to stable fingerprints.
- Direct test adapters and worker-backed adapters emit equivalent normalized
  results.
- Killing one analysis worker does not terminate the coordinator, TTS worker,
  or another analysis worker.
- A worker cannot load a model outside its built-in allowed family.

## Phase 3: Implement consensus, calibration, and shadow evaluation

### Work

1. Apply the versioned, language-specific lexical normalization pipeline and
   bounded directional equivalences.
2. Align recognizer token sequences against expected lexical tokens.
3. Produce per-token votes and merged disagreement windows.
4. Apply engine-specific decode-quality gates before counting a vote.
5. Trigger Qwen3-ASR only under the configured policy.
6. Record why the third recognizer did or did not run.
7. Calibrate thresholds independently for each engine and clip class.
8. Version and fingerprint every calibration table.
9. Run the ensemble beside the current Whisper decision without changing build
   outcomes. Record where the proposed policy agrees, catches an additional
   known defect, or would add a false rejection.
10. Review every valid dissent and every proposed automatic acceptance in the
    qualification corpus before enabling the policy as a gate.

Retain the current semantic clip classes:

- one word;
- short phrase;
- sentence;
- join;
- chapter; and
- repaired region.

Do not carry Whisper's thresholds into Parakeet or Qwen by name or numerical
analogy. Each threshold must come from corpus measurements.

### Exit criteria

- Consensus decisions are deterministic for identical evidence.
- Result ordering does not depend on model completion order.
- A forced-alignment result cannot satisfy a recognition vote in the type
  system or service API.
- Known mismatches produce stable, actionable reason codes.
- The ensemble does not increase false acceptance over current Whisper QA.
- Shadow reports explain every difference from the current Whisper decision.
- Reviewers approve the final strict-policy truth table and threshold set.

## Phase 4: Prepare migration without cutting over

Implementation status (2026-08-14): complete as an internal boundary. The
read-only migration service, draft schema, cutover fixture, command map, cache
transition, pronunciation split, and real fake-project planning path are in
place. Manifest version 1 and the current CLI remain canonical. See
[Speech-analysis migration preview](speech-analysis-migration.md).

Do not make schema version 2 public merely because isolated adapters and shadow
consensus work. Short repairs, joins, chapters, delivery verification, caching,
and worker scheduling must exercise the same contracts first.

### Work

- Keep manifest schema version 1 and the current CLI canonical.
- Maintain one internal `SpeechAnalysisPolicy`; translate the current
  `WhisperQaPolicy` into it at the existing boundary only while qualification is
  underway. Do not create two analysis engines.
- Finalize the draft schema-version-2 `speech_analysis` shape using evidence
  emitted by the shadow implementation.
- Draft generic `speech-*` report schemas with new absolute schema URIs starting
  at contract schema version 1. If an existing URI is retained while its
  meaning changes, plan an increment to contract schema version 2.
- Implement and test a migration service in preview-only form. It maps
  `whisper_qa`, model settings in `ShortUtterancePolicy`, and join, chapter, and
  cache settings into the new policy.
- Replace ambiguous pronunciation `spoken` values in the draft with distinct
  synthesis-hint, expected-lexical, and optional phoneme fields. Preview flags
  any old rule whose replacement changes tokenization for explicit review.
- Define the new Python exports, CLI command map, cache-version transition,
  examples, and documentation changes as tested cutover fixtures without yet
  exposing them as the canonical public surface.
- Run migrated real project fixtures through the internal version-2 planner and
  fake backend, while continuing to run normal builds through version 1.

Existing approved repair audio is not deleted. Migration preview preserves its
source and approval records, but marks old analysis evidence stale when the new
policy fingerprint requires evidence it never contained. Version-1 reports
remain historical artifacts and are never treated as version-2 release
evidence.

### Exit criteria

- The draft manifest represents the complete proposed ensemble without fields
  known to be unusable by the implemented adapters.
- Preview is deterministic, read-only, and identifies every lossy or ambiguous
  transformation for review.
- Migrated fixtures plan successfully through the internal version-2 path.
- Schema fixtures validate every shadow report and reject invalid envelopes.
- No public command, schema default, or manifest behavior has changed yet.

### Command direction

| Version 1 | Version 2 |
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

At cutover, the canonical inspection commands run the configured policy. An
explicit `--engine` option may run one registered recognizer for diagnosis, but
its single-engine output is not a release verification.

## Phase 5: Integrate short utterances and localized repair

Implementation status (2026-08-14): complete behind the internal ensemble
boundary. Carrier and extracted audio require independent consensus and
authorized forced timing; sentence repairs use accepted anchors; and multiple
approved frame replacements are reconstructed once from immutable base audio.
The current public version-1 workflow remains unchanged until the later
cutover.

Qwen forced alignment should provide its largest immediate benefit here.

### Carrier extraction

For each candidate:

1. Synthesize the complete carrier sentence.
2. Run Parakeet and Whisper without expected-text prompts.
3. Run Qwen3-ASR when the consensus policy requires it.
4. Establish lexical consensus for the complete carrier.
5. Give the verified carrier text to Qwen3-ForcedAligner.
6. Locate the exact target span.
7. Run optional phoneme QA inside the forced target span when configured.
8. Compare forced boundaries with independent ASR timing, VAD regions, and
   waveform edges.
9. Reject the candidate when boundary disagreement exceeds a calibrated
   tolerance.
10. Crop with conservative pre-roll and post-roll.
11. Apply the existing acoustic refinement and edge fades.
12. Verify the extracted audio independently.
13. Verify both final joins.

For a one-word extracted clip, Whisper and Qwen3-ASR must both recognize the
word. Parakeet remains diagnostic because its model card warns about isolated
incomplete input.

Every generated take keeps an immutable candidate identity and terminal reason.
Only candidates that independently pass every required gate enter ranking.
Ranking may use calibrated naturalness, boundary, and signal evidence; it cannot
average incomparable ASR confidence scores or rescue a rejected take. The
maximum candidate and round counts are fixed before generation and included in
the repair-session fingerprint.

Do not average boundary timestamps from different engines. Treat them as an
agreement envelope around a forced-alignment proposal. The target occurrence
must map uniquely to its designated expected-token indices even when the same
surface word appears elsewhere. Alignment must cover the complete target in
monotonic order, and a safe cut must exist between the target and adjacent
speech after calibrated pre-roll and post-roll. If the independent timing, VAD,
phoneme timing when enabled, and waveform evidence leave no safe intersection,
reject the candidate instead of choosing a compromise timestamp. Re-recognize
the final cropped PCM; recognizing only the carrier is insufficient.

### Sentence and clause repair

Use forced alignment twice:

- align independently accepted anchors around the original defect to bracket
  the replacement interval; and
- align the newly generated carrier with its verified text to locate the new
  replacement audio.

Then retain the existing level matching, high-frequency matching, adaptive
crossfade, reconstructed-chunk verification, and two-boundary join inspection.

### Multi-repair candidate assembly

Allow a repair session to accumulate several independently approved replacement
takes against one exact base raw-assembly digest. Before rebuilding, resolve all
selectors to chunk and frame intervals, reject overlaps or ambiguous ordering,
and revalidate that the base and every selected take still match their approval
evidence. Reconstruct each affected chunk once from immutable base slices and
the canonically ordered replacements; do not apply edits sequentially to frame
coordinates already shifted by an earlier edit.

Build one raw chapter candidate, run changed-chunk and unioned affected-join
checks, master once, and analyze the union of outward-mapped post-master windows.
Coalesce overlapping QA windows without losing the per-repair provenance that
explains them. One failing repair or interaction rejects the batch candidate but
does not erase the individually approved takes, so the user can remove or
replace the failing member without regenerating the others.

The batch identity contains repair IDs in canonical timeline order, the base
digest, individual take and approval fingerprints, resolved non-overlapping
intervals, splice policy, and union-window plan. Reordering the same input set
does not change the identity. Adding, removing, or changing a repair invalidates
only batch assembly and downstream evidence, not unchanged take generation and
qualification.

Approval rebuilds a named repair candidate and runs affected-scope analysis. It
does not overwrite or label a prior verified release as current. A separate
release verification step analyzes the exact new mastered chapter and promotes
it only after every full release gate passes. This keeps the listening loop fast
without lowering the standard for distributable audio.

### Exit criteria

- Every crop is backed by independent lexical consensus and forced timing.
- No forced-alignment-only candidate can be approved.
- Known `naaah ... No`, clipped ending, and extra-attribution defects reject.
- Repeating an unchanged repair reuses generation, recognition, forced
  alignment, extraction, DSP, reconstruction, and join evidence.
- The repair report identifies the exact stage responsible for a cache miss.

## Phase 6: Integrate joins, chapters, and releases

Implementation status (2026-08-14): complete behind the internal ensemble
boundary. Join, hierarchical chapter, mastering-map, decoded-delivery, stable
repair-selector, invalidation, and digest-bound release evidence contracts are
implemented. Repair-candidate and release-verified states remain distinct.

### Join inspection

Preserve PCM join measurements as independent hard evidence. For each affected
join:

1. inspect sample jump, local peak change, and surrounding silence;
2. transcribe a contextual window with Whisper and Parakeet;
3. invoke Qwen3-ASR on disagreement or high-risk text;
4. establish lexical consensus across the join; and
5. use Qwen3-ForcedAligner to measure the last word before and first word after
   the join when transcript correctness is already established.

This can distinguish an unnatural pause from a clipped word more precisely than
PCM or inferred Whisper timing alone.

### Chapter verification

Use a hierarchical pass:

1. Parakeet scans the mastered chapter in deterministic bounded windows.
2. Whisper verifies the same expected-spoken plan.
3. Yakbox merges both token streams and finds disagreements.
4. Qwen3-ASR analyzes disagreement windows, high-risk expected-spoken spans,
   newly generated regions, and affected joins.
5. Qwen3-ForcedAligner gives accepted words direct locations and brackets
   defective regions from verified surrounding anchors.
6. Yakbox maps each defect to the assembly manifest's chunk ID, source file,
   source lines, speaker, profile, and adjacent join indices.

The report becomes an actionable repair map rather than only a pass/fail
transcript.

### Evidence lifetime across mastering

Yakbox currently applies FFmpeg `loudnorm` to the complete chapter. A localized
raw repair may therefore change samples outside the repaired span even when
duration and words are unchanged. Do not claim that old mastered-chapter
evidence remains content-identical after such a rebuild.

Create a `MasteringAudioIdentity` containing raw-assembly and mastered digests,
formats and frame counts, FFmpeg and filter-graph fingerprints, and a verified
mapping between their frame coordinates. Measure filter and resampler delay on
controlled marker fixtures and cross-check real chapter boundaries. Do not
assume that `loudnorm` or resampling is time preserving because nominal duration
looks unchanged.

Map a repaired raw span outward through that identity before targeted
post-master analysis. Include the calibrated uncertainty margin and adjacent
join context. If the mapping is absent, non-monotonic, or outside its qualified
tolerance, targeted post-master evidence cannot create a repair candidate; run
the broader verification path instead. The delivery timing map extends the same
chain from mastered frames to decoded-delivery frames.

Use two explicit artifact states:

- **repair candidate**: the rebuilt chapter has passed changed-raw-chunk,
  affected-join, post-master repaired-window, and whole-file technical checks;
  it is ready for listening but not a release; and
- **release verified**: the exact mastered WAV digest has passed complete
  chapter recognition, and every encoded delivery audio stream has passed its
  delivery gates.

Raw synthesis evidence for byte-identical unchanged chunks remains reusable.
Full mastered-chapter recognition becomes stale whenever the mastered WAV
digest changes. A future mastering implementation may prove narrower
invalidation only if it records a verified time-preserving transform and tests
that claim; this plan does not assume it.

### Release verification

A release report aggregates reusable raw-chunk evidence, then analyzes the
exact mastered WAV being released. A complete Parakeet and Whisper chapter pass
remains the strict release baseline whenever that WAV digest is new. Qwen3-ASR
is mandatory for all disagreements and configured high-risk spans, but it does
not need to retranscribe every ordinary span when both baseline recognizers
match the manuscript and no valid dissent exists.

Then encode the delivery artifact, bind its file digest and encoder identity to
the release, decode its audio stream back to canonical analysis PCM, and inspect
duration, channel layout, leading and trailing audio, codec delay, clipping, and
chapter boundaries. Until Phase 0 qualifies a versioned codec-equivalence gate,
run the complete Parakeet and Whisper release pass on that decoded delivery
stream as well. This verifies the bytes users receive, not merely their source.

Keep container and speech evidence separate. A `DeliveryAudioIdentity` records
the container digest, selected stream, encoder and decoder fingerprints,
decoded canonical-PCM digest, timing map, and metadata fingerprint. A metadata-
only rebuild may reuse lexical evidence when the decoded PCM is byte-identical,
but it still reruns container, metadata, chapter, and stream-selection checks.

A future codec-equivalence gate may reuse mastered-WAV lexical evidence only if
held-out testing proves that its signal and timing checks catch every encoded
speech or boundary defect in the corpus. The gate is bound to the exact encoder,
codec settings, audio-stream layout, decoder, and calibration fingerprints. A
different bitrate, FFmpeg build, decoder, or audio-stream layout invalidates
that proof. If equivalence is absent or fails, full delivery recognition remains
mandatory.

### Exit criteria

- A reported mismatch maps directly to a stable repair selector.
- A byte-identical mastered chapter reuses all recognition and consensus
  evidence.
- A byte-identical delivery artifact reuses all delivery evidence. A new
  container may reuse lexical evidence only when its decoded PCM is
  byte-identical or its qualified codec-equivalence gate passes.
- One repaired chunk invalidates that raw chunk, affected joins, targeted
  post-master windows, and full-master evidence tied to the prior WAV digest.
- Strict release verification cannot pass with an unresolved disagreement.
- Repair-candidate output cannot be mistaken for release-verified output in a
  schema, path, CLI message, or release check.

## Phase 7: Redesign analysis caching

The current cache includes expected text in an otherwise independent Whisper
request. Separate caches by evidence type.

### Recognition cache key

- cache format version;
- exact normalized PCM span hash;
- audio preprocessing fingerprint;
- language;
- recognizer fingerprint;
- execution-identity fingerprint;
- decode settings; and
- sample-indexed span boundaries.

It does not include expected text.

### Forced-alignment cache key

- cache format version;
- exact normalized PCM span hash;
- audio preprocessing fingerprint;
- aligner-text and expected-lexical-span hashes;
- language;
- forced-aligner fingerprint;
- execution-identity fingerprint;
- alignment settings; and
- sample-indexed span boundaries.

### Consensus cache key

- recognition-result fingerprints;
- expected lexical-unit sequence hash;
- alias and normalization policy fingerprint;
- consensus policy fingerprint; and
- calibration fingerprint.

### Final-verification cache key

- consensus-result fingerprint;
- forced-alignment fingerprints used by the decision;
- signal and boundary evidence fingerprints;
- spoken-text-plan and assembly-map fingerprints;
- mastered or delivery audio identity, as applicable;
- verification policy and calibration fingerprints; and
- human-disposition fingerprint when one resolves review-eligible evidence.

This layer may change when source mapping or review state changes even if the
underlying independent recognition remains reusable.

### Repair-stage cache

Continue storing separate content-addressed stages for:

- synthesis;
- recognition per engine;
- consensus;
- forced alignment;
- extraction;
- DSP splice;
- reconstructed verification; and
- join verification;
- post-master verification; and
- decoded-delivery verification.

Reports store text hashes where full manuscript text is not required. Cache
integrity checks continue to use safe workspace paths and atomic writes.

The managed local analysis cache may store recognized tokens needed for exact
reuse. Logs, journals, release metadata, runtime status, and exception messages
must not store full manuscript or transcript text. Human-facing defect reports
use hashes plus short bounded previews only where a reviewer needs them.

Cache entries are acceleration data, not the sole durable proof of a release.
Promotion atomically snapshots a compact evidence bundle under the release root:
all audio, text-plan, model, execution, calibration, policy, decision, and human-
disposition fingerprints; per-span outcomes and reason codes; and hashes needed
to verify the evidence graph without copying complete text. Detailed cache
entries referenced by an active repair session or configured audit-retention
period are pinned. Cleanup cannot delete a pin or leave a release record pointing
to a required artifact that was never durably committed.

Cache successful and deterministic rejected evidence, because a mismatch is
still useful analysis. Do not cache timeouts, cancellations, worker crashes,
malformed results, or partial batches as complete evidence. Each batch item
commits independently only after its result validates.

### Performance outcome

Producing a one-line repair candidate should require:

- no synthesis outside the selected repair;
- no recognition of unchanged speech chunks;
- forced alignment only for changed carriers and splice windows;
- join QA only for adjacent joins;
- one chapter mastering pass;
- at most one clearly labeled preview encoding pass when requested; and
- no full-chapter Qwen3-ASR pass.

Promoting that candidate to a release still requires complete baseline
recognition of the exact new mastered WAV. This cost is paid once per release
candidate, not once per generated take or intermediate repair audition.

### Exit criteria

- Repeating an identical analysis or repair candidate performs no model
  inference and explains every cache hit.
- Changing text mapping, policy, calibration, model, execution, audio,
  mastering, delivery, or human disposition invalidates only the dependent
  evidence layers defined above.
- Corrupt, truncated, path-escaping, or schema-invalid entries are quarantined
  or ignored without becoming votes.
- Cache writes are atomic, concurrent identical requests coalesce safely, and a
  cancelled producer cannot leave an apparent hit.
- Cache status and miss diagnostics remain bounded and contain no complete
  manuscript or transcript text.
- Release promotion remains auditable after ordinary cache cleanup, and cleanup
  preserves active-session and retention pins.

## Phase 8: Implement scheduling and worker lifecycle

Replace the Whisper-shaped `align` request with supervisor-to-worker operations:

- `recognize`;
- `recognize_many`;
- `force_align`;
- `force_align_many`;
- `status`;
- `unload`; and
- `shutdown`.

Every request carries a protocol version, operation ID, deadline, engine
fingerprint, workspace-relative audio path, and sample span. Every response
carries the operation ID, normalized evidence, duration and memory metrics, and
a typed terminal status. Cancellation and timeout are protocol operations, not
signals inferred from a disconnected socket.

Before accepting work, a child sends one handshake containing its protocol,
worker-artifact, environment-lock, adapter, Python, execution, and capability
fingerprints. The supervisor compares them with the planned identities. Any
mismatch terminates the stale worker and fails preflight or starts the exact
planned runtime; protocols do not silently negotiate down to an older evidence
contract.

Assume an upstream inference call may not be cooperatively cancellable. On
deadline or user cancellation, the supervisor first sends a soft cancellation,
stops scheduling new batch items, and waits a bounded grace period. It then
terminates the worker process if the active call does not return, marks only the
uncommitted items cancelled, and starts a clean worker for later requests. A
hard-killed operation never contributes a cache entry or vote.

Batch responses are item-framed and carry the batch ID, stable item index,
request fingerprint, and terminal status. The supervisor may atomically commit
each validated completed item before another item fails, but reconstructs user-
visible ordering from the input indices. Duplicate completions are idempotent;
an index with conflicting evidence is a protocol violation, not last-write-wins.
Deadlines use the supervisor's monotonic clock and are translated to remaining
durations rather than comparing wall clocks across processes.

### Model-major scheduling

Avoid changing resident models for every candidate. Process repairs in small
rounds:

1. generate a small round of Chatterbox candidates;
2. run backend-independent format and hard signal checks, removing only
   candidates that already have a terminal hard failure;
3. run Parakeet over every remaining candidate;
4. run Whisper over that same remaining set—Parakeet alone cannot filter it;
5. compute baseline evidence and run Qwen3-ASR over every disputed, invalid-
   baseline, and high-risk candidate required by policy;
6. batch Qwen forced alignment for carriers with accepted lexical evidence;
7. extract, independently verify, inspect, and rank passing candidates; and
8. generate another round only for unresolved repairs within the fixed budget.

This keeps early stopping without repeatedly loading four models.

Model-major execution may reorder work but cannot alter the truth table. A
candidate leaves the required recognition path early only after an engine-
independent hard failure or a complete terminal policy decision. Preserve the
reason and unrun stages explicitly; absence of a model result is never inferred
to be agreement.

Calls within one MLX worker remain serialized unless upstream batch APIs are
proven deterministic. Different dependency-family workers do not run Metal
inference concurrently by default because concurrent command buffers can raise
peak memory and destabilize latency. FFmpeg work, PCM inspection, hashing, and
report preparation may run concurrently within existing bounded semaphores.

Use one supervisor-owned accelerator lease across Chatterbox Torch/MPS and all
MLX workers, not merely one lock per dependency family. A process may be alive
without holding the lease, but model load, inference, and framework cache cleanup
must be scheduled as measured accelerator operations. This prevents a TTS task
and an analysis task from independently deciding that unified memory is
available. CPU-only preparation remains parallel and bounded.

The existing performance report should add, per engine and evidence stage:

- cold loads, warm reuses, unloads, evictions, and restarts;
- requested and analyzed audio seconds;
- wall time and real-time factor;
- batch size and survivor count;
- cache hits, misses, and miss reasons;
- peak resident and Metal memory when measurable; and
- time spent preparing audio, running inference, normalizing tokens, reaching
  consensus, forced-aligning, and writing evidence.

These metrics contain no transcript or manuscript text.

### Resident-model policy

Add a supervisor-level worker policy and a per-worker model policy with:

- maximum resident workers and models;
- maximum total resident memory;
- per-worker and per-model last-use time;
- explicit unload support;
- graceful worker shutdown and forced-termination deadlines;
- Metal cache cleanup after unload; and
- model-load, restart, and eviction metrics.

Treat an upstream `unload` as advisory until memory measurements confirm it.
After the cleanup grace period, compare resident and Metal memory with the
calibrated post-unload watermark. If memory does not return, recycle the worker
process before granting another model load. Do not report a model as evicted
merely because Yakbox dropped its Python reference.

Do not assume that 64 GB of unified memory means all models should remain loaded
alongside Chatterbox. Choose defaults from measured peak memory and model-switch
cost.

### Exit criteria

- Each model loads at most once in a successful no-fault processing round;
  recovery reloads are explicit failure metrics.
- Batched repairs retain input order in reports.
- Runtime eviction cannot change fingerprints or acceptance results.
- A runtime crash leaves durable caches and repair sessions recoverable.
- Direct test execution and worker execution produce equivalent structured
  evidence.
- Repeated-call endurance tests prove stable output and bounded memory.
- Restarting a failed worker retries only idempotent, uncommitted analysis
  operations.

## Phase 9: Test, calibrate, and make the ensemble default

Implementation status (2026-08-14): the automated contracts, migration,
preflight, quality and performance evaluators, cutover gate, managed runtimes,
real-model smoke suite, runtime-specific endurance gate, licensed source-window
inventory, checksum-verified archive expansion to 75 independent passages, and
restartable three-engine transcript-authoring stage are implemented. Safe
per-case transcript approval, a voice-disjoint 18/57 calibration and held-out
partition, and two-pass boundary-truth contracts are also implemented. Their
real qualification artifacts remain gated on transcript review. The public
cutover remains intentionally inactive until the approved transcript and
boundary truth, frozen risk corpus, calibration, reference-M5 workflow
benchmark, clean-release artifacts, and required reviews all pass. See
[Speech-analysis qualification record](speech-analysis-qualification.md).

### Automated tests

- unit tests with fake recognizers and forced aligners;
- adapter contract tests;
- adapter rejection tests for NaN or out-of-range timing, malformed batch shape,
  invalid Unicode, excessive output, truncation, and repetition termination;
- token normalization and sequence-consensus property tests;
- equivalence validation and complexity tests for cycles, ambiguous rules,
  multi-token forms, deterministic edit ties, and adversarial alias counts;
- insertion, deletion, substitution, alias, and tie tests;
- contextual-projection tests for repeated words, missing anchors, clipped edge
  tokens, and unexpected prefixes such as an extra vocalization before `No`;
- tests proving an alias cannot satisfy a required pronunciation check;
- transform-map tests covering deleted quote delimiters, stripped attribution,
  pronunciation respelling, number expansion, and exact source mapping;
- migration tests that reject ambiguous legacy pronunciation replacements;
- migration preview/write tests for determinism, atomicity, backup or clean-
  destination enforcement, and preservation of historical report readability;
- tests proving forced alignment cannot vote on correctness;
- defect-localization tests proving missing or substituted expected tokens
  cannot provide their own forced boundary and that ambiguous one-sided anchors
  expand or require review;
- tests rejecting expected text in independent worker requests;
- tests proving independent recognition-cache keys ignore expected text;
- tests for sample-indexed window and cache-key stability;
- single-flight cache tests for concurrent identical requests, waiter
  cancellation, producer failure, atomic commit, and corrupt-entry quarantine;
- cleanup tests proving release evidence, active repair sessions, and configured
  retention pins remain auditable after unpinned cache entries are removed;
- resampling round-trip tests proving analysis spans map outward to safe source
  frames without drift at the start or end of long chapters;
- marker and real-boundary tests for raw-assembly to mastered timing through the
  exact resampler and `loudnorm` filter graph;
- tests proving every consensus engine receives identical canonical frames and
  that overlap stitching neither duplicates nor drops a token;
- language-capability and maximum-duration preflight tests;
- preflight tests proving missing workers, models, calibration, disk, or memory
  capability fails before synthesis, plus lightweight doctor import tests;
- worker protocol tests for malformed, oversized, late, duplicate, and
  out-of-order messages;
- framed-transport tests for truncated messages, unexpected stdout, bounded
  stderr, and a child process that never closes its pipes;
- handshake tests for stale protocol, lock, worker, adapter, Python, and
  capability fingerprints without downgrade negotiation;
- reproducible worker-artifact tests plus installed-wheel execution under each
  frozen dependency-family runtime;
- worker crash, timeout, cancellation, restart, and partial-batch tests;
- scheduler tests proving TTS and analysis cannot hold the accelerator lease
  concurrently while bounded CPU preparation still progresses;
- scheduler tests proving one recognizer cannot filter work required by another
  recognizer and that every skipped stage has a terminal policy reason;
- tests for soft-cancel grace, forced termination during blocked inference,
  duplicate completion, conflicting completion, and monotonic deadline expiry;
- model allowlist, revision, digest, and path-escape tests;
- tests that model, execution, seed, and calibration changes invalidate the
  correct evidence layers without invalidating unrelated synthesis;
- tests proving timestamps, operation IDs, latency, memory, cache-hit state, and
  report ordering do not change semantic fingerprints or decisions;
- socket-blocked tests proving inference is offline after installation;
- worker-environment tests proving credentials, proxies, package indexes,
  Python paths, and unrelated parent variables are not inherited;
- schema tests for every new report;
- tests that human dispositions become stale when any bound digest or policy
  identity changes and cannot override hard failures;
- review-command tests for bounded notes, atomic resolution, JSON cleanliness,
  and a digest change between `show` and `resolve`;
- tests that repair-candidate artifacts cannot satisfy release selectors;
- delivery tests that decode the exact output artifact, detect codec delay and
  boundary changes, and invalidate evidence when codec settings change;
- CLI JSON contract tests;
- deterministic offline integration tests with controlled WAV fixtures;
- opt-in Apple Silicon real-model tests;
- repeated-call determinism and memory-endurance tests;
- unload-watermark tests that recycle a worker when MLX or MPS memory is not
  actually released;
- performance tests for cold start, warm inference, batching, cache reuse, and
  one-line repair;
- grouped-candidate tests proving early stopping and ranking cannot admit a
  rejected take, plus performance and false-accept tests at the configured
  candidate and round limits;
- multi-repair tests for stale bases, overlapping selectors, canonical ordering,
  coalesced QA windows, one-pass mastering, interaction failures, and reuse of
  unaffected approved takes; and
- listening-review workflows for short utterances and joins.

### Quality release criteria

- Zero observed false accepts in the held-out known-defect corpus, with corpus
  size and a pre-registered one-sided uncertainty bound reported rather than
  implying universal perfection. The bound, not merely the observed count, must
  meet the approved threshold in every safety-critical risk class.
- Clean-audio false rejection does not regress beyond a threshold approved
  before held-out evaluation.
- The complete configured candidate-generation and ranking workflow meets the
  same approved false-accept bound; per-candidate accuracy is not sufficient.
- Name and number accuracy is no worse than the current Whisper baseline.
- Forced-boundary median and P95 error improve over current Whisper timing, and
  clipped-target or extra-speech crops do not increase.
- The known one-word, clipped-ending, extra-syllable, and join-artifact cases
  reject automatically.
- Every automatic acceptance is explainable from stored model and signal
  evidence.

### Performance release criteria

Benchmark a frozen set of unchanged builds, one-line repair candidates,
multi-repair rounds, mastered release promotions, and delivery verifications on
the reference M5. Record cold-process/cold-cache, warm-process/cold-evidence,
and fully repeated-cache-hit cases separately. For expensive chapter workflows,
use at least five measured runs after one unmeasured warm-up and report median
and range. Use at least 20 runs before reporting a latency P95 for bounded model
or cache operations. Always report audio seconds, model loads, and peak memory.
Compare only workflows with identical quality policy and terminal artifact
state.

- Repeating an unchanged repair-candidate build performs no model inference.
- Producing a localized repair candidate spends less than 25% of the previous
  full-chapter verification time on speech analysis.
- Applying several approved non-overlapping repairs performs one raw assembly,
  one mastering pass, and only unioned affected-scope candidate analysis before
  release promotion.
- Each required engine loads no more than once per successful no-fault
  processing round; recovery reloads are reported separately.
- Qwen3-ASR runs only for policy-required windows in the default strict preset.
- Full offline operation succeeds after explicit model installation.
- Full release verification remains a separately measured chapter-level cost.
- Delivery verification time is reported separately from mastered-WAV
  verification, including whether full recognition or a qualified equivalence
  gate was used.

### Public cutover after all gates pass

- Make manifest schema version 2 canonical.
- Rename `WhisperQaPolicy` to `SpeechAnalysisPolicy` at the public boundary and
  remove the temporary version-1 translation.
- Replace the `whisper_qa` table with `speech_analysis`, remove generic analysis
  settings from `ShortUtterancePolicy`, and move join, chapter, and cache policy
  under the new table.
- Activate the reviewed pronunciation schema and require explicit resolution of
  migration warnings before writing.
- Replace Whisper-named schemas and public Python exports with the generic
  contracts, then update the public API snapshot.
- Rename `yakbox whisper` to `yakbox speech`, move model lifecycle commands to
  `yakbox models`, and expose `yakbox runtimes`.
- Expose `yakbox migrate manifest --check` and explicit `--write`. Write mode is
  atomic and either preserves a backup or requires a clean destination.
- Bump the analysis-cache version and ignore version-1 entries.
- Update examples, user documentation, schemas, CLI contract tests, and package
  consumer tests in the same change.

Do not maintain two public policy implementations after cutover. Version-1
manifests fail with a focused migration message. Old cache data may remain until
normal cleanup removes it, but version-1 evidence is never interpreted under
version-2 rules. Version-1 reports remain readable as historical artifacts
through their packaged schemas; they are not valid inputs to a version-2 release
decision.

After these gates pass, schema version 2 and the strict ensemble become the
default. The old Whisper-specific configuration and command surface are removed
rather than maintained indefinitely.

## Dependency and environment layout

Keep model packages out of Yakbox's default environment. Package built-in
worker definitions and locks, for example:

```text
runtimes/
  whisper/
    runtime.toml
    pyproject.toml
    uv.lock
  parakeet/
    runtime.toml
    pyproject.toml
    uv.lock
  qwen/
    runtime.toml
    pyproject.toml
    uv.lock
```

All runtime projects use Python 3.14. `runtime.toml` names a Yakbox-owned worker
module and allowed model family; it does not contain a shell command. The lock
and its digest become part of the worker fingerprint.

The installer invokes a reviewed `uv` version with a fixed argument vector and
`shell=False`. It validates the Python 3.14 interpreter, built-in project path,
lock digest, package indexes, and destination before mutation. Neither a book
manifest nor an environment variable may replace the runtime project, indexes,
constraints, Python executable, or worker entry point. Missing or incompatible
`uv` produces an actionable preflight error; a build never installs it or
modifies a runtime implicitly.

Build the protocol validator, path guard, and lazy backend adapters into a small
Python zip application shipped as a package resource in the Yakbox wheel.
`yakbox runtimes install` copies that exact hash-verified artifact into the
managed runtime and applies the dependency-family lock. `runtime.toml` selects
one built-in allowed adapter; it cannot select an arbitrary module or entry
point. The worker artifact contains no model package and cannot import the
audiobook planner.

The artifact version, digest, protocol schema, and selected adapter version are
part of every worker fingerprint. Release tests install the built Yakbox wheel,
extract the packaged worker artifact, and run it under every runtime lock. An
explicit development command rebuilds the artifact; production never injects
editable source or the supervisor's `PYTHONPATH` into a managed environment.

Build the zip application reproducibly from reviewed source: sorted entries,
fixed timestamps, no host paths, no generated bytecode, and a manifest of every
included file. Two builds from the same source tree and toolchain must have the
same digest. The normal package check verifies that the wheel contains exactly
the declared worker artifact and can execute it outside the repository.

This avoids copying ad hoc scripts, recursively installing Yakbox into its own
optional runtime, or coordinating a second published distribution. Supervisor
and worker exchange schema-validated values, not pickled Python objects, so
deployment does not require their in-memory classes to be identical.

At public cutover, install workers explicitly:

```console
yakbox runtimes install whisper parakeet qwen
yakbox runtimes verify
```

For contributors and embedded testing, a `speech-analysis` optional extra may
install all three dependency families into the current environment if Phase 0
proves that the combined graph resolves. It is a convenience path, not the
production isolation boundary. Do not create a self-referential extra such as
`yakbox[speech-analysis]` inside Yakbox's own package metadata.

The provisional package constraints are:

```text
mlx-whisper >=0.4.3,<0.5
parakeet-mlx >=0.5.2,<0.6
mlx-audio[stt] >=0.4.5,<0.5
```

Apple Silicon platform markers apply to all three. Phase 0 selects exact locks
and verifies Python 3.14 before any range becomes a supported contract.

## Security and licensing requirements

- Pin package versions through `uv.lock` with artifact hashes.
- Pin every model by immutable revision and verify its local file fingerprint.
- Record converted-model and upstream-model revisions separately.
- Reject a converted model with unknown provenance. Require a published,
  reproducible conversion recipe or rebuild it from the pinned upstream model
  and compare tensor metadata plus reference outputs.
- Allow model installation only through explicit lifecycle commands.
- Reject runtime model downloads.
- Reject `trust_remote_code`.
- Prefer Safetensors and reject unexpected executable or pickle-based model
  files.
- Reject model-cache symlinks or resolved paths that escape the verified model
  snapshot.
- Accept only workspace-controlled local audio paths in adapters.
- Revalidate worker input type, root containment, and digest in the worker to
  prevent a parent-validation/path-open race.
- Do not expose mlx-audio's server, microphone, URL, base64, TTS, or STS paths.
- Spawn model workers with a minimal allowlisted environment that excludes
  provider, cloud, package-index, proxy, and Python-path credentials or hooks.
- Generate and audit an SBOM for every built-in worker environment, not only the
  small Yakbox core environment.
- Include worker-lock and model-directory fingerprints in doctor and release
  provenance without logging user cache paths unnecessarily.
- Audit package and model licenses separately.
- Record Apache-2.0 obligations for Qwen models.
- Record CC BY 4.0 attribution for Parakeet model weights.
- Keep model source URLs, rights basis, revisions, and checksums in the existing
  licensing documentation.

Neither `parakeet-mlx` nor `mlx-audio` currently uses PyPI Trusted Publishing.
Locked wheel hashes reduce the risk for reviewed versions, but every upgrade
still requires source, provenance, dependency, and behavior review.

## Main risks and mitigations

### Python 3.14 compatibility

Both packages declare broad Python compatibility but do not clearly document a
Python 3.14 test matrix. Phase 0 is a hard gate. If either package cannot meet
the repository's sole supported runtime, keep its adapter unshipped until
upstream or a narrowly maintained compatibility patch resolves the issue.

### Fast-moving mlx-audio API

`mlx-audio` supports many model families and changes quickly. Keep all imports
behind one narrow adapter, pin a reviewed minor range, and test the adapter
against real model output. Never allow upstream result objects to escape into
Yakbox's public contracts.

### Incomparable confidence values

Do not average or directly compare scores across engines. Calibrate each engine
against the same corpus, store the score kind, and convert it only into a valid
or invalid vote under that engine's policy. Some adapters may not expose a
meaningful scalar confidence at all. Do not invent one; validate those results
with the engine's available signals, output invariants, repeated-call stability,
and corpus-calibrated error behavior.

### Forced-alignment circularity

Keep `SpeechRecognizer` and `ForcedAligner` as separate protocols and result
types. The consensus service accepts only recognition results. This makes it
difficult to accidentally treat supplied-text alignment as transcript proof.

### Quantization quality

BF16 Qwen MLX conversions are the qualification baseline. Benchmark 8-bit as a
possible optimization. Prefer quality over memory savings when the corpus shows
a meaningful difference.

### Converted-model provenance

An immutable community-model revision proves which files were downloaded, not
how they were derived. Require a reproducible recipe tied to the upstream model
and conversion code. When the published provenance is incomplete, rebuild the
conversion under Yakbox's qualification process and compare tensor names,
shapes, dtypes, tokenizer/config files, and fixed reference outputs. Do not make
an opaque conversion release-authoritative because it happens to load.

### Memory pressure and model thrashing

Use model-major scheduling, small candidate rounds, a resident-model LRU, and
measured memory limits. Cache every completed stage so eviction affects latency,
not correctness or resumability.

### Language coverage differences

Parakeet and Qwen ForcedAligner do not support the same languages as Whisper or
Qwen3-ASR. Manifest validation must construct a valid pipeline for the book's
language. Unsupported engines cause an explicit configuration error or a
documented policy choice; there is no silent fallback that weakens evidence.

## Automation and review policy

The default workflow does not pause for routine listening review:

1. When the recognizers disagree, Yakbox retries them on the same expanded
   context. Persistent disagreement discards and regenerates a generated take
   within its fixed budget. If regeneration is unavailable or exhausted, the
   build fails with an exact repair selector. A two-model majority does not
   silently accept uncertain audio.
2. A local repair that passes its affected-scope checks becomes a clearly
   labeled repair candidate. Multiple candidates can be accumulated without
   repeating full-chapter work. Complete mastered-chapter and delivery checks
   run once when the selected candidate is promoted to a release.
3. A one-word repair does not require listening after all strict automated gates
   pass. Carrier consensus, forced boundaries, isolated-word recognition,
   signal checks, and both joins must pass. Otherwise Yakbox regenerates within
   the fixed budget and then fails closed.
4. Manual resolution remains an explicit escape hatch for a review-eligible
   model disagreement. Yakbox never opens it automatically, and it cannot
   override clipping, corruption, unsafe boundaries, missing evidence, or any
   other hard failure.

This removes per-build listening as a normal gate. It does not remove the
one-time human work needed to label the qualification corpus and approve model
thresholds; automating those judgments with the same models under test would
make the evidence circular. Requalification is required only when a dependency,
model, policy, language, or calibration change invalidates that evidence.

## Architecture decisions taken

The following are not held open because they follow from the requested freedom
to break compatibility or from current correctness constraints:

- schema version 2 is a deliberate hard cutover with an explicit migration
  command, not a second permanent implementation;
- production TTS, Whisper, Parakeet, and Qwen execution is isolated by
  dependency family behind supervised workers; and
- every changed mastered WAV receives complete Whisper and Parakeet release
  passes. Targeted analysis is sufficient only for a repair candidate while
  chapter-wide `loudnorm` can change the complete mastered output; and
- every new delivery audio stream receives complete baseline recognition until
  its exact codec pipeline passes the plan's held-out equivalence gate; and
- the first release qualifies English only. Other languages require their own
  corpus, normalization, pronunciation, calibration, and held-out gates.

## Decisions resolved by measurement

Phase 0 and shadow evaluation decide these without product preference:

1. Whether Qwen3-ASR and ForcedAligner 8-bit are non-inferior to BF16.
2. Whether Parakeet greedy or beam decoding is better for audiobook prose.
3. The maximum safe analysis span for each backend on the M5.
4. The safe worker and model residency limits.
5. Batch and candidate-round sizes.
6. Engine-specific thresholds for each clip class.
7. The allowed timing disagreement between Qwen ForcedAligner, independent ASR
   timing, phoneme timing, and VAD boundaries.
8. Whether shared canonical audio preparation is non-inferior to each native
   loader.
9. Exact dependency ranges, worker locks, and model revisions.
10. Whether any delivery codec and preset qualifies for lexical-evidence reuse
    instead of full decoded-delivery recognition.

## Definition of done

The integration is complete when:

- the manifest, CLI, Python API, reports, and caches are backend neutral;
- Whisper, Parakeet, and Qwen3-ASR provide independent evidence under a single
  deterministic consensus policy;
- Qwen3-ForcedAligner supplies timing only after transcript verification;
- short-word extraction and localized repair use verified forced boundaries;
- chapter defects map directly to stable repair locations;
- byte-identical raw-chunk evidence survives localized repair through
  content-addressed provenance;
- repeat repairs avoid generation and analysis work already proven valid;
- repair candidates and release-verified artifacts are impossible to confuse;
- every changed mastered WAV receives the release evidence required by its
  policy;
- every released delivery audio stream is verified from its exact encoded bytes
  by full recognition or a qualified codec-equivalence gate;
- the full workflow runs offline with pinned local models;
- licensing and attribution are complete; and
- real-model quality and performance gates pass on the M5 reference system.
