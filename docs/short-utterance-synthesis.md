# Short-utterance synthesis design

Status: implemented as an experimental, opt-in strategy  
Scope: local Chatterbox audiobook builds  
Last updated: 2026-08-02

## Problem

Chatterbox is unreliable when Yakbox asks it to synthesize an isolated phrase
of only a few words. The failure is not limited to flat delivery. Listening
tests have found words that were never requested:

- `Wren asked.` can sound like `Or Wren asked.`;
- `No,` can sound like a drawn-out `naaah` followed by `No`; and
- `Liora added.` can sound like `Liora added it.`

These failures affect narration tags as well as character dialogue, so a
dialogue-only rule would miss them. Removing quote delimiters fixes malformed
Chatterbox input, but it does not solve the model's general instability on
one-, two-, and three-word requests.

The leading hypothesis is that Chatterbox needs enough surrounding language to
establish a stable beginning, cadence, and ending. Yakbox can give it that
context, locate the intended phrase in the longer waveform, extract only that
phrase, and merge the result into the audiobook. The hard part is proving that
the extracted audio contains every requested word and no extra speech.

## Goals

The short-utterance pipeline must:

1. detect risky narration and dialogue chunks before synthesis;
2. generate the requested phrase inside longer, voice-appropriate context;
3. find the target phrase with word- or token-level timing evidence;
4. reject leading, trailing, repeated, or substituted speech;
5. crop without removing initial or final consonants;
6. join the crop without clicks, doubled silence, or an unnatural gap;
7. try a bounded number of deterministic candidates;
8. fail clearly when no candidate is safe enough to use;
9. preserve reproducible builds, cache correctness, and resumability; and
10. make the strategy and thresholds easy to change in `yakbox.toml`.

The release requirement is exactness, not perfect first-generation behavior.
Chatterbox may produce bad candidates. Yakbox must never select a candidate
known to contain extra or missing speech.

## Non-goals

This work will not:

- retrain or patch Chatterbox itself;
- promise that automated metrics can judge acting or naturalness;
- rewrite the manuscript to avoid short lines;
- make context words audible in the final book;
- send manuscript audio or text to a hosted alignment service by default; or
- silently fall back to a candidate that failed validation.

## Implemented approach

Short utterances become a small candidate-selection pipeline inside the
synthesis stage:

```text
planned speech chunk
        |
        v
short-utterance risk check ------ no ------> normal synthesis
        |
       yes
        v
build natural and synthetic carriers
        |
        v
generate deterministic candidates
        |
        v
align expected carrier + detect speech regions
        |
        v
crop target + zero-crossing refinement + edge fades
        |
        v
validate extracted transcript and acoustic guards
        |
        +------ no accepted candidate ------> fail/review/explicit override
        |
        v
rank accepted candidates --> cache selection --> normal chapter assembly
```

This belongs in the audiobook synthesis orchestration, not inside the
Chatterbox adapter. The adapter should continue to turn one request into one
waveform. Planning identifies the risk, a backend-neutral short-utterance
service manages candidates, an optional local aligner supplies timing evidence,
and the existing WAV assembly layer joins the selected crop.

## 1. Detect risky chunks

The current `dialogue.short_utterance_words` setting produces an advisory
finding. The new policy must cover every routed speaker, including the
narrator.

Initial eligibility should use normalized spoken text after dialogue quote
removal and pronunciation replacement:

- one to three lexical words: always risky;
- four words: measured as a control group before deciding the default;
- punctuation-only or empty text: invalid, not a synthesis candidate;
- explicit pauses: never eligible; and
- acronyms, initials, numbers, and hyphenated words: counted by the same
  Unicode-aware tokenizer used by tests and reports.

Word count is only the first signal. Later versions may add character count,
punctuation shape, prior failure history for the voice, or an estimated duration
model. The first implementation should stay explainable: a plan must be able to
say exactly why a chunk was classified as short.

The plan records a `short_utterance` marker and policy fingerprint without
recording additional manuscript text.

## 2. Construct carriers

A carrier is hidden text synthesized in the same voice as the target. It gives
Chatterbox enough language before and after the phrase to stabilize its output.
Only the target survives cropping.

Yakbox should try carriers in this order:

1. **Natural same-speaker context.** Reuse nearby manuscript text assigned to
   the same speaker when it makes a coherent carrier. This is most likely to
   preserve the intended emotion.
2. **Attribution context.** For a short narrator tag such as `Wren asked.`, use
   nearby narrator prose before and after it.
3. **Neutral synthetic carrier.** Put the target between short neutral anchor
   sentences when natural context is unavailable.

The target should normally appear in the middle of a carrier. A middle target
tests whether Chatterbox's leading and trailing hallucinations are primarily
request-edge failures. Initial and final positions remain experimental
variants because punctuation and position may change delivery.

Example carriers:

```text
Target: Wren asked.
Carrier: The exchange continued. Wren asked. No one answered immediately.

Target: No.
Natural carrier: No. Not after I got here.
Middle carrier: She understood the question. No. Then she looked away.

Target: You first in?
Natural carrier: You first in? Anyone touch him?
Middle carrier: He stopped at the tape. You first in? Then he waited.
```

Synthetic carriers are production configuration, not arbitrary prose assembled
inside a test. Each template needs an ID and version. Templates must contain
the target placeholder exactly once, avoid names or gendered pronouns, avoid
emotion words that could distort delivery, and end each anchor with clear
sentence punctuation.

For comma-ended source fragments such as `No,` followed by a narrator tag, a
carrier may render the target as `No.` to create an alignable boundary. The
planner must record this punctuation adaptation. The spoken lexical target
remains unchanged.

## 3. Generate deterministic candidates

Each risky chunk gets a bounded set of candidates. A useful first matrix is:

- one direct request retained as a baseline;
- two natural-context candidates with distinct deterministic seeds when natural
  context exists;
- two middle-carrier candidates with distinct deterministic seeds; and
- one alternate-position candidate during live testing, not by default.

Candidate seeds derive from the chapter ID, chunk index, normalized target
hash, profile fingerprint, carrier template ID, carrier position, attempt
number, and policy version. A resumed build must request the same candidates in
the same order.

The default attempt ceiling should be small. Short-utterance recovery cannot
turn a chapter with many short tags into unbounded model work. Preflight must
report the maximum additional local generations or hosted requests before the
build starts.

## 4. Align the target

Energy-based silence detection is useful but insufficient for production. It
works when a carrier creates a long pause after `No.` or `Third.`, but it cannot
prove that the audio before the pause says only the requested word. It also
cannot safely isolate a phrase in continuous speech.

Yakbox needs a lazy, optional alignment interface:

```python
class SpeechAligner(Protocol):
    async def align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
    ) -> AlignmentResult: ...
```

`AlignmentResult` contains normalized recognized tokens, token start and end
times, confidence, segment-level Whisper evidence, parser issues, detected
speech regions, and the backend/model fingerprint. The selector then locates one exact,
contiguous occurrence of the target in the carrier.

The first implementation uses `mlx-whisper` on Apple Silicon. It runs Whisper
Large-v3-Turbo locally through MLX and requests word timestamps. An unprompted
pass remains the transcript authority. A prompted pass may refine timing only
after the unprompted tokens exactly match the target. The default Hugging Face model revision is
pinned and included in the aligner, policy, plan, and cache fingerprints. A
custom model must declare its own immutable revision. Model installation is an
explicit `yakbox whisper models install` operation; builds never download it.

This adapter is intentionally platform-specific. Normal builds and the direct
strategy do not import MLX. The backend-neutral alignment contract and fake
tests continue to run on Linux, macOS, and Windows; `context_extract` currently
requires an Apple Silicon Mac with the `alignment` extra installed.

Voice activity detection can refine speech boundaries. An implementation such
as [Silero VAD](https://github.com/snakers4/silero-vad) is a candidate, but the
same optional-dependency and model-rights review applies. A simple PCM energy
detector remains useful as an independent check and offline test fixture.

Alignment must fail when:

- the target tokens do not appear exactly once and contiguously;
- the carrier transcript drops, substitutes, or repeats a target word;
- the target timing is missing or confidence is below policy;
- unexpected speech occurs inside the proposed pre-roll or post-roll; or
- the target touches an audio edge without enough evidence for a safe crop.

## 5. Validate hallucinations before cropping

The three reported failures need explicit guards:

- `Or Wren asked.` must fail because a recognized token precedes the target;
- `Liora added it.` must fail because a recognized token follows the target;
- `naaah ... No` must fail when VAD or alignment finds speech before the aligned
  `No`, even if an ASR system normalizes both sounds to one word.

No single recognizer is reliable enough to prove a one-word clip by itself.
Validation therefore combines lexical and acoustic evidence:

### Hard gates

- expected target tokens match exactly after case and punctuation normalization;
- extracted audio transcribes to only those tokens;
- no speech region longer than the configured guard threshold exists before or
  after the aligned target;
- the crop contains finite, readable PCM with the expected channel and sample
  format;
- duration falls inside broad per-word-count safety bounds;
- peak, clipping, and edge-silence checks pass; and
- speaker similarity stays above a conservative threshold once a vetted local
  embedding implementation exists.

### Ranking signals

- alignment confidence;
- distance from the median accepted duration for the same target and voice;
- amount of safe pre-roll and post-roll;
- distance from the expected voice embedding;
- boundary energy and zero-crossing quality; and
- join-preview continuity with the preceding and following selected chunks.

Hard gates decide whether a candidate is usable. Ranking chooses among usable
candidates. A high score must never override a lexical or extra-speech failure.

Non-lexical artifacts remain the hardest case. A prolonged vowel may still be
recognized as the intended word. Duration outlier detection, extra VAD regions,
phoneme-aware alignment, and human review all need evaluation against `naaah ...
No`. Until those checks are proven, one-word candidates remain review-required
for release builds.

## 6. Crop safely

The initial crop comes from aligned target boundaries, then passes through an
acoustic refinement step:

1. retain configurable pre-roll and post-roll, initially 20–50 ms;
2. refuse padding that overlaps another speech region;
3. move the cut to a nearby low-energy zero crossing within a small bounded
   search window;
4. apply a short equal-power or linear fade, initially 5–10 ms;
5. preserve the source sample rate and channel count; and
6. inspect the crop again after writing it atomically.

Pre-roll protects initial consonants such as the `Wr` in `Wren`. Post-roll
protects final consonants such as the `d` in `asked` and `added`. The cropper
must not maximize tightness; it must maximize evidence that the complete phrase
survived without another word.

## 7. Select, cache, and fall back

Selection outcomes are:

- `accepted_automatic`: every hard gate passes and review is not required;
- `accepted_reviewed`: a reviewer approved the exact candidate and report;
- `rejected`: the candidate failed at least one hard gate; or
- `unresolved`: no candidate passed and the build cannot continue under the
  configured policy.

Release builds should default to `failure = "error"`. Draft builds may use
`failure = "review"`, which produces a listening package and stops before a
release can be marked complete. Silent direct-synthesis fallback would recreate
the original defect and is not acceptable.

An explicit audio override is the final escape hatch. It must bind a stable
chunk ID to a managed WAV, checksum, rights basis, expected speaker, and source
fingerprint. Changing the source text invalidates the override.

The candidate cache key includes:

- target and carrier hashes;
- logical voice, reference-audio checksum, and profile settings;
- Chatterbox runtime fingerprint and candidate seed;
- aligner and VAD model fingerprints;
- template, alignment, crop, validation, and policy versions; and
- expected output format.

Rejected candidate audio does not need to become a durable release artifact.
Keep it only when a configured local QA package needs it. Durable reports must
follow Yakbox's privacy rule: store hashes, token counts, timing, scores, and
reason codes rather than full manuscript text or unrestricted transcripts.

## 8. Assemble and review the join

The selected crop enters the existing `WavJoinPart` pipeline. Context extraction
must not invent its own chapter concatenator.

The joiner should:

- apply the semantic pause selected by the planner;
- avoid counting carrier silence as part of that pause;
- avoid applying two overlapping edge fades;
- record the crop and final join timestamps; and
- run the existing PCM discontinuity and silence-window checks.

Every short-utterance QA package needs three files per candidate: direct,
full-context, and extracted. It also needs a merged passage containing the
selected crop. Reviewers must be able to tell whether a defect came from
generation, alignment, cropping, or assembly.

## Configuration

The feature is additive and opt-in. Existing manifests keep direct synthesis.
All operational choices are in `yakbox.toml`:

```toml
[short_utterances]
strategy = "context_extract"
maximum_words = 3
candidate_count = 5
prefer_natural_context = true
carrier_positions = ["middle"]
alignment_backend = "mlx-whisper"
alignment_model = "mlx-community/whisper-large-v3-turbo"
alignment_revision = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
prompted_timing = true
decode_consensus = true
prompt_sensitivity = true
maximum_consensus_timing_delta_ms = 180
hallucination_silence_threshold = 0.8
automatic_join_inspection = true
join_inspection_window_seconds = 1.5
alignment_timeout_seconds = 180.0
alignment_aliases = { liora = ["leora"] }
minimum_alignment_confidence = 0.5
minimum_extracted_confidence = 0.2
minimum_one_word_confidence = 0.6
minimum_short_phrase_confidence = 0.5
minimum_segment_average_log_probability = -1.0
maximum_segment_compression_ratio = 2.4
maximum_segment_no_speech_probability = 0.6
maximum_segment_temperature = 0.2
candidate_confidence_tolerance = 0.05
maximum_extra_speech_ms = 60
maximum_internal_token_gap_ms = 350
maximum_token_duration_ms = 1200
acoustic_refinement = true
acoustic_threshold_dbfs = -48.0
speech_island_gap_ms = 300
minimum_edge_silence_ms = 10
maximum_edge_silence_ms = 120
maximum_clipped_sample_ratio = 0.005
maximum_boundary_jump_ratio = 0.35
maximum_vad_disagreement_ms = 500
maximum_stationary_voiced_ms = 1200
minimum_pause_ms = 180
pre_roll_ms = 30
post_roll_ms = 40
fade_ms = 8
failure = "error"
require_review_for_one_word = true
keep_candidates = true
```

After lexical validation, Yakbox groups independent energy detections into
speech islands. A detached prefix or suffix, less than
`minimum_edge_silence_ms`, or more than `maximum_edge_silence_ms` at either
edge triggers a crop around the dominant island. The refined file is
transcribed and acoustically inspected again; it is rejected if any word,
detached island, or unsafe edge remains. Set `acoustic_refinement = false` to
make those findings hard failures without attempting the second crop.

Candidate ranking first keeps only accepted takes within
`candidate_confidence_tolerance` of the best exact-transcript confidence. In
that safe band, `prefer_natural_context = true` prefers the natural-context
take, followed by lower edge-silence and internal-pause penalties and then the
stable confidence and candidate-index tiebreakers. Set the tolerance to `0` for
strict confidence-first selection.

The first shipped version should retain `strategy = "direct"` for existing
manifests and make `context_extract` opt-in. New-project defaults may change
only after the live acceptance gates pass. Draft, proof, and release targets
may override candidate count and failure policy, but they must not weaken
lexical exactness.

When a one-word result passes the automated hard gates, Yakbox writes
`listening-review.toml` beside the candidate report and stops. Listen to the
selected candidate, change only `status = "pending"` to `status = "pass"`, and
rerun the same build. Approval is bound to the report hash and selected
candidate, so regeneration or a policy change invalidates it. Failed hard gates
cannot be overridden by this review file.

## Suggested types and module boundaries

Keep the workflow split into small typed pieces:

- `speech.short_utterances`: policy, risk classification, carrier construction,
  candidate metadata, and selection;
- `speech.alignment`: backend-neutral alignment contracts and validation;
- `audio.crop`: PCM boundary refinement, zero crossing, padding, and fades;
- `local_alignment`: lazy optional local aligner adapter;
- `audiobook.planner`: risk markers and policy fingerprints; and
- `audiobook.build`: candidate orchestration, caching, failure handling, and
  artifact records.

Likely domain types include `ShortUtterancePolicy`, `CarrierRecipe`,
`SynthesisCandidate`, `AlignmentToken`, `AlignmentResult`, `CropEvidence`,
`CandidateDecision`, and `ShortUtteranceSelection`.

The audio layer must not import audiobook or Chatterbox code. Alignment models
must remain optional and lazy so the normal package import and fake-backend test
suite do not load them.

## Testing plan

### Phase 0: freeze the failure corpus

Create a small original-text fixture with permissive rights and these classes:

| Class | Required examples | Purpose |
| --- | --- | --- |
| One word | `No.`, `Third.`, `Wait.`, `Yes.` | Detect vowel artifacts, repeats, and edge instability |
| Two words | `Wren asked.`, `Liora added.`, `She said.`, `He waited.` | Detect leading and trailing hallucinated words |
| Three words | `You first in?`, `Anyone touch him?`, `Not this time.` | Exercise short questions and statements |
| Four words | `Did anyone touch him?`, `You were here first.` | Establish the default eligibility boundary |

Test straight and typographic source quotes separately from synthesized text.
Include period, comma, and question-mark endings. Use the twenty-five approved local
reference voices so the result does not overfit one speaker.

The three user-reported failures are mandatory regression observations:

- leading `Or` before `Wren asked`;
- extra vocalization before `No`; and
- trailing `it` after `Liora added`.

### Phase 1: offline unit tests

These tests run in the default suite without Chatterbox or an alignment model.

#### Risk classification

- count Unicode words, contractions, hyphens, initials, acronyms, and numbers;
- classify narrator and character chunks identically;
- exclude pauses and empty text;
- verify three- and four-word policy boundaries; and
- prove quote removal and pronunciation replacement happen before counting.

#### Carrier construction

- select only context routed to the same speaker;
- preserve the target exactly once;
- never expose carrier text as final narration;
- choose templates and positions deterministically;
- reject a template containing no target or multiple targets; and
- preserve question versus statement intent.

#### Alignment and hallucination validation

Use fake alignment tokens to prove that:

- exact `Wren asked` passes;
- `Or Wren asked` fails with `unexpected_prefix`;
- `Liora added it` fails with `unexpected_suffix`;
- a missing, substituted, or repeated word fails;
- multiple target occurrences are ambiguous and fail;
- low confidence fails;
- speech in a guard window fails even when lexical tokens match; and
- punctuation and case normalization do not hide lexical differences.

#### Crop behavior

Build synthetic PCM fixtures with known speech envelopes and assert:

- pre-roll and post-roll are retained;
- the crop never crosses another speech region;
- zero-crossing search remains within its bound;
- fades change only the configured edges;
- initial and final consonant fixtures are not truncated;
- invalid PCM and missing safe boundaries fail; and
- Windows, macOS, and Linux write identical PCM for identical input.

#### Selection, cache, and policy

- hard-gate failures cannot win through ranking;
- deterministic ties resolve stably;
- attempt ceilings are honored;
- changing any carrier, model, crop, or policy fingerprint invalidates cache;
- unchanged accepted candidates reuse cache;
- interrupted candidate batches resume safely;
- `failure = "error"` stops the build;
- `failure = "review"` cannot produce an approved release; and
- explicit overrides require matching checksums and source fingerprints.

### Phase 2: fake-backend integration tests

Add a fake synthesizer and fake aligner that can return exact speech, leading
speech, trailing speech, ambiguous timing, and no safe boundary on demand.

Exercise the observable build contract:

1. plan marks the risky chunk and estimated attempts;
2. build creates candidates in deterministic order;
3. rejected candidates carry bounded reason codes;
4. one accepted crop replaces only its target chunk;
5. surrounding chunks remain byte-identical;
6. the merged chapter preserves source order and semantic pauses;
7. artifacts and schemas describe the selection without full text;
8. cache reuse avoids synthesis and alignment on the second build; and
9. cancellation or worker failure leaves resumable state and no partial durable
   output.

Add CLI tests for validation errors, dry-run estimates, JSON output, review
requirements, and explicit overrides. Add schema compatibility tests for every
new machine-readable field.

### Phase 3: alignment benchmark

Before selecting a dependency, run the frozen corpus through each candidate
aligner. Measure:

- exact transcript rate on direct and carrier audio;
- target start/end error against manually marked boundaries;
- detection rate for added leading and trailing words;
- behavior on `naaah ... No` and other non-lexical vocalizations;
- confidence calibration for one-word clips;
- CPU time, memory, model size, and supported platforms;
- offline operation after installation; and
- code, model, and transitive dependency licenses.

The selected adapter must support Python 3.14 and the repository's Linux,
macOS, and Windows CI policy, or it must be isolated behind a documented local
runtime boundary with equivalent contract tests on every platform.

Do not select an aligner from long-form transcription accuracy alone. The
benchmark is specifically about one- to four-word timing and hallucination
detection.

### Phase 4: live Chatterbox experiment

Run the approved voices across:

- every frozen phrase;
- direct, natural-context, middle-carrier, initial-carrier, and final-carrier
  strategies;
- at least five deterministic seeds per strategy; and
- the current approved `cfg_weight` and `exaggeration` settings before trying a
  parameter grid.

Record generation success separately from selection success. This shows whether
context improves Chatterbox and whether the validator can identify the good
result. Do not tune generation parameters and carrier strategy in the same
first experiment; that would hide which change mattered.

For every attempt, preserve locally:

- direct audio;
- full carrier audio;
- extracted target audio;
- a join preview with its neighboring chunks;
- hashes and runtime fingerprints;
- alignment and crop evidence;
- rejection reason codes; and
- a structured human review bound to the exact report hash.

### Phase 5: listening review

Listen first without reading, then with the expected text. Score each selected
candidate for:

- exact words with no prefix, suffix, repetition, or vocalization;
- intact first and last consonants;
- natural pronunciation and duration;
- appropriate question or statement contour;
- consistent speaker identity;
- clean crop boundaries; and
- natural entry into and exit from the surrounding passage.

Any extra or missing speech is a blocking defect, regardless of the numerical
score. A reviewer must explicitly record `pass` or `fail` for the three known
failure shapes: prefix speech, suffix speech, and non-lexical vocalization.

### Phase 6: full-chapter validation

Enable the strategy for one Chapter 1 build after the corpus passes. Review:

- every one- to four-word source chunk;
- the ten seconds before and after each replacement;
- character-to-narrator transitions;
- repeated tags such as `Wren asked` and `Liora said`;
- chapter duration changes;
- cache reuse on an identical rebuild; and
- selective invalidation after changing one profile or one short line.

The full chapter must still pass existing loudness, peak, silence, cadence,
join-step, artifact, and release checks.

## Acceptance gates

The feature is ready for opt-in production use only when:

1. every automatically selected core-corpus clip has an exact normalized target
   transcript;
2. no selected clip has detected prefix or suffix speech above the configured
   guard threshold;
3. all three reported failure shapes are rejected in controlled tests;
4. every crop passes PCM, duration, peak, clipping, and edge checks;
5. every required merged-passage observation receives a human `pass`;
6. no selected one-word clip has a blocking human defect;
7. repeated runs select the same candidate and produce the same PCM;
8. a cached rebuild performs no model or aligner work;
9. the configured attempt ceiling bounds worst-case work; and
10. the full Chapter 1 release review is approved with no short-utterance
    blocking defects.

Track first-candidate success and retry counts as quality metrics, but do not
weaken exactness to improve them. If the system cannot produce a verified clip,
the correct result is an unresolved build, not incorrect narration.

## Implementation sequence

1. Add typed policy, risk classification, and carrier recipes behind
   `strategy = "direct"`.
2. Add fake aligner contracts, hallucination gates, and PCM crop utilities with
   complete offline tests.
3. Add the optional MLX Whisper adapter, pin its model revision, and document
   its MIT-licensed software lineage.
4. Extend the live QA corpus and structured listening review.
5. Integrate candidate orchestration and cache fingerprints into audiobook
   builds as opt-in behavior.
6. Add the review-required workflow. A managed explicit-audio override remains
   future work and is not silently emulated by direct synthesis.
7. Run the full Chapter 1 experiment and tune only thresholds supported by its
   evidence.
8. Consider changing new-project defaults after the acceptance gates pass.

## Open questions

- Does a middle carrier outperform a target at the beginning for all voices?
- Can the aligner reliably distinguish `No` from a preceding `naaah`?
- Should comma-ended fragments always use sentence-final carrier punctuation?
- Is natural same-speaker context consistently better than neutral templates?
- What is the smallest safe pre-roll for initial consonants in each voice?
- Can speaker-embedding checks be licensed and run locally on every platform?
- How many candidates are needed before retries stop producing useful gains?
- Should reviewed accepted crops be reusable across identical text occurrences,
  or remain bound to one source location and performance context?

Threshold changes are checked against the frozen corpus with
`yakbox whisper calibrate`. See [Whisper inspection and short-audio
QA](whisper-and-short-audio.md) for the operator workflow.

## Initial implementation evidence

The 2026-08-02 Apple Silicon experiment used Python 3.14, Chatterbox 0.1.7,
`mlx-whisper` 0.4.3, and the pinned 1.61 GB Large-v3-Turbo conversion. Whisper
Small was rejected as the default after it failed to preserve the spelling of
`Wren` in the known regression take; Large-v3-Turbo recognized the contextual
name correctly and exposed an extra leading token in the direct take.

The six-phrase cut-and-merge corpus completed for `You first in?`, `Wren
asked.`, `Third.`, `Liora added.`, `Anyone touch him?`, and `No.`. The run also
provided two useful calibration results: invented-name and written-number ASR
variants need explicit project aliases, and isolated-crop confidence is lower
than in-carrier confidence. The implementation therefore keeps separate
confidence gates without weakening exact-token or extra-speech checks.

This is engineering evidence, not listening approval. The generated `Third.`
and `No.` crops remain one-word review items, and the no-pause continuous cuts
must be checked for consonant loss, clicks, and join quality before production
defaults are changed.
