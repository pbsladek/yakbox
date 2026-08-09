# Whisper inspection and short-audio QA

Yakbox uses local MLX Whisper as an independent check on risky one- to
three-word Chatterbox output. Whisper does not decide what the script should
say. The first, unprompted transcription remains authoritative. A second pass
may use the expected text to improve word boundaries, but only after the first
pass already matches every expected token. The prompted pass cannot turn a
failed transcript into a pass.

This feature currently supports Apple Silicon macOS. It is optional and never
downloads a model during an audiobook build.

## Install and verify the runtime

Install the optional package, then explicitly install the pinned model:

```console
uv sync --extra alignment
yakbox whisper models install
yakbox whisper models status
yakbox whisper models verify
yakbox doctor --whisper
```

`yakbox whisper models path` prints the resolved local snapshot. All four
model commands accept `--model` and `--revision`. A remote model must have an
immutable revision. `status`, `verify`, `path`, `doctor`, `build`, and
`short-test` are local-only; only `models install` may use the network.

Whisper results are cached by the audio digest, aligner fingerprint, language,
time range, and a hash of the expected text. The default cache is
`.yakbox/cache/whisper`; use `--no-cache` when measuring a cold run. Cached
documents omit the expected manuscript and full transcript.

Run `yakbox doctor --whisper --deep` when you want to load the model and test
the word-timestamp call. The normal doctor check reports the platform, Python
package, pinned revision, cache path, model size and fingerprint, and free
storage without loading model weights.

## Inspect an existing take

```console
yakbox whisper inspect take.wav --expected "You first in?"
yakbox --json whisper inspect take.wav --expected "You first in?"
```

The report includes the recognized text, word times and probabilities, token
diff, segment log probability, compression ratio, no-speech probability,
temperature, language, speech islands, edge speech, and parser errors. It also
shows an independently sampled best-of decode, its agreement score, timing drift, and
the result of an expected-prompt sensitivity pass. Invalid numbers, missing
words, out-of-bounds times, and nonmonotonic times are retained as rejection
reasons. They are never silently dropped.

The unprompted decode remains authoritative. If the prompted pass fixes a word,
the report marks it `prompt_rescued`, but the original mismatch still fails.
If greedy and sampled decoding disagree, the report fails with
`decode_consensus_mismatch`. A timing difference above the configured limit
fails with `decode_timing_instability`.

Use a targeted reinspection when a full paragraph report points to one suspect
range:

```console
yakbox whisper reinspect paragraph.wav \
  --start 4.20 --end 6.10 --expected "You first in?" \
  --out build/whisper/you-first-in.json
```

This uses Whisper's clip-timestamp path. Times in the report remain relative to
the original audio, so they can be compared directly with an editor timeline.

## Verify a complete chapter

```console
yakbox whisper verify-manuscript \
  build/mastered/0001-room-609.wav source/01-room-609.md \
  --out build/reports/0001-room-609.manuscript.json
```

Yakbox normalizes the Markdown with the same source parser used by audiobook
builds, then aligns the full transcription against every expected lexical
token. The report gives token counts, token accuracy, bounded mismatch previews,
and audio times for substitutions, additions, and deletions. It stores a hash
of the normalized manuscript instead of copying the complete chapter into the
report. When one source file contains several chapters, select one with
`--chapter CHAPTER_ID`. Pass the build's pronunciation file with
`--pronunciations` when applicable.

To make this a build and release gate, enable it in the repository manifest:

```toml
[whisper_qa]
chapter_verification = true
```

Yakbox then inserts `verify_manuscript` between `master` and `encode_mp3` for
every chapter. A mismatch stops the graph before delivery encoding. Release
checks independently require a current, digest-valid, passing verification
artifact.

## Inspect every audio join

Create a JSON join spec from the same timeline or assembly data used to make the
chapter:

```json
{
  "joins": [
    {
      "at_seconds": 12.48,
      "boundary": "dialogue",
      "expected_before": "Wren asked.",
      "expected_after": "Anyone touch him?"
    },
    {
      "at_seconds": 18.02,
      "boundary": "sentence"
    }
  ]
}
```

Then inspect all joins in one run:

```console
yakbox whisper inspect-joins build/mastered/0001-room-609.wav \
  --spec build/0001-room-609.joins.json \
  --out build/reports/0001-room-609.joins.json
```

Each join gets a targeted decode-consensus pass plus a PCM check at the exact
sample boundary. Yakbox reports transcript mismatches, repeated words or
phrases, words crossing a splice, clicks, local peak changes, and the silence on
both sides. Supplying `expected_before` and `expected_after` enables exact local
text verification; without them, the command still checks decoding stability,
repetition, and the waveform.

Context-free windows that overlap, or are separated by at most 100 ms by
default, are decoded once and projected back onto each exact join. Joins with
expected context remain separate so prompt-sensitivity evidence stays local.
The report records the physical join count and the smaller Whisper window
count. Change the merge distance with `--coalesce-gap-ms` or
`whisper_qa.join_coalesce_gap_ms`.

## Inspect final consonants with forced phoneme alignment

Install the optional runtime and pinned Apache-2.0 acoustic model. eSpeak NG is
an external phonemizer and must also be installed on the host:

```console
uv sync --extra phoneme
brew install espeak-ng
yakbox whisper phoneme-models install
yakbox whisper phoneme-models status
```

Then inspect a suspect take:

```console
yakbox whisper inspect-phonemes third.wav \
  --expected "Third." --out build/whisper/third.phonemes.json
```

The report contains a forced start, end, and path confidence for every expected
IPA phoneme. The hard gate scores the final acoustic boundary: a lone final
consonant must pass itself, while a natural final consonant cluster is scored as
a cluster. This makes a weak or clipped final `/d/` visible without rejecting a
normally unreleased `/t/` in a cluster such as `/skt/`. The implementation uses
a pinned Wav2Vec2 phoneme CTC model and Yakbox's deterministic Viterbi path; it
does not use torchaudio's removed forced-alignment API.

Vowel-final words such as “No” retain complete phoneme evidence but are not
rejected on vowel-model confidence. Their endings are instead protected by the
exact Whisper transcript, duration, speech-island, stationary-voicing, pitch,
and waveform gates.

Phoneme evidence may clear only a lone `low_confidence` Whisper reason when the
unprompted transcript is already exact, decode consensus is stable, every other
gate passes, the final boundary passes, and the median full phoneme path is at
least 0.60. It cannot rescue a substituted, missing, added, or unstable word.
For a one-word consonant-final clip, an exact stable transcript may instead use
a minimum 0.50 Whisper confidence plus a minimum 0.80 final-boundary confidence;
this prevents an accent-sensitive vowel from vetoing a clean word ending.
Boundary reconciliation includes up to 70 ms of tolerance for the acoustic
model's frame quantization, with a hard 130 ms word-tail ceiling. This is added
only after the phoneme and transcript gates pass.

The cropper applies a stricter 25 ms tolerance when an energy region begins just
after an ASR word boundary, but protects it only when the complete region ends
within 130 ms of that boundary. Longer adjacent speech remains a hard failure.

A candidate that is too short for a valid CTC phoneme path is rejected with
`phoneme_alignment_failed`; it does not abort the phrase. Yakbox continues with
the next deterministic carrier. Missing runtimes and invalid installation state
still fail the build globally.

`candidate_count` is a hard maximum, not a count of unique carrier layouts.
After every safe direct, synthetic, and natural layout has been tried once,
Yakbox cycles the same bounded layouts with new deterministic seeds. Generation
still stops as soon as two candidates pass, so only difficult phrases pay for
the additional retries.

Internal word gaps retain the configured 350 ms limit unless the source has an
explicit internal sentence boundary, such as “Static? So what?” That deliberate
rhetorical boundary permits up to 900 ms; ordinary phrases such as “You first
in?” do not receive the wider allowance.

Enable the independent phoneme gate for context-extracted short utterances:

```toml
[whisper_qa]
phoneme_alignment = true
minimum_phoneme_confidence = 0.2
phoneme_language = "en-us"
```

A candidate must pass Whisper, waveform and acoustic checks, plus the final
phoneme-boundary gate. Evidence for every phoneme is written into the
short-utterance QA report and participates in the synthesis fingerprint.

The exact-text gate is necessary but not sufficient. Yakbox also rejects low
confidence, suspicious Whisper segments, unexpected speech before or after the
target, excessive pauses between target words, detached acoustic islands,
clipping, boundary clicks, excessive token duration, prolonged stationary
voicing, pitch variation, high-frequency energy, and strong
disagreement between fixed and adaptive energy detectors.

## Generate a focused short-utterance package

```console
yakbox short-test yakbox.toml \
  --profile nick-whitley \
  --text "You first in?" \
  --previous-context "The door buckled behind them." \
  --next-context "Liora checked the corridor."
```

Each run gets its own directory below `build/short-tests`. It contains the
direct take, full carrier takes, initial crops, any refined crops, the selected
WAV, `report.json`, and `short-test.json`. The test manifest records the exact
text used by each recipe. QA reports keep manuscript text private by storing
text hashes instead.

`candidate_count` is a maximum generation budget. Yakbox evaluates recipes in
deterministic order and stops after two candidates pass every hard gate, then
ranks those passing candidates. If fewer than two pass, it exhausts the matrix
so a difficult phrase retains every available recovery attempt.

Use the review commands after listening:

```console
yakbox short-review list build/short-tests
yakbox short-review play build/short-tests/RUN/report.json
yakbox short-review approve build/short-tests/RUN/report.json \
  --notes "Clean consonants and natural ending"
yakbox short-review reject build/short-tests/RUN/report.json \
  --notes "Extra syllable before No"
```

An approval is bound to the complete QA report, selected candidate number, and
selected audio checksum. Regenerating either the report or audio invalidates
the decision. One-word audiobook output still stops for listening review unless
`require_review_for_one_word = false` is set explicitly.

## Calibrate threshold changes

```console
yakbox whisper calibrate
yakbox --json whisper calibrate --out build/whisper-calibration.json
```

The checked-in v1 corpus freezes evidence for the failures found during the
chapter-one tests: prefixed “Wren asked,” suffixed “Liora added,” elongated
“No,” unnatural pauses in “You first in,” low-confidence “Third,” and known
good controls. The report tracks false accepts, false rejects, token accuracy,
crop-boundary error, consonant-sensitive case accuracy, listening scores, and a
fingerprint that binds the corpus to the active policy thresholds. The command
also records its evaluation runtime and peak Python memory; live model timing
remains part of the explicit `doctor --whisper --deep` smoke test.

CI requires zero false accepts and zero false rejects on this frozen corpus.
That protects threshold changes without requiring the Apple-only model on
Linux and Windows runners.

## Configuration

The defaults live in `yakbox.toml` and can be changed per repository:

```toml
[whisper_qa]
chapter_verification = false
cache_enabled = true
cache_directory = ".yakbox/cache/whisper"
join_coalesce_gap_ms = 100
phoneme_alignment = false
phoneme_backend = "wav2vec2-ctc"
phoneme_model = "facebook/wav2vec2-lv-60-espeak-cv-ft"
phoneme_revision = "c43348bbaa5a77692c8e7bf3409d683474fdf2a4"
phoneme_language = "en-us"
phoneme_timeout_seconds = 180.0
minimum_phoneme_confidence = 0.2

[short_utterances]
strategy = "context_extract"
alignment_backend = "mlx-whisper"
alignment_model = "mlx-community/whisper-large-v3-turbo"
alignment_revision = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
prompted_timing = true
alignment_timeout_seconds = 180.0
decode_consensus = true
prompt_sensitivity = true
maximum_consensus_timing_delta_ms = 180
hallucination_silence_threshold = 0.8
automatic_join_inspection = true
join_inspection_window_seconds = 1.5

minimum_alignment_confidence = 0.5
minimum_extracted_confidence = 0.2
minimum_one_word_confidence = 0.6
minimum_short_phrase_confidence = 0.5
minimum_segment_average_log_probability = -1.0
maximum_segment_compression_ratio = 2.4
maximum_segment_no_speech_probability = 0.6
maximum_segment_temperature = 0.2
maximum_extra_speech_ms = 60
maximum_internal_token_gap_ms = 350
maximum_token_duration_ms = 1200

maximum_clipped_sample_ratio = 0.005
maximum_boundary_jump_ratio = 0.35
maximum_vad_disagreement_ms = 500
maximum_stationary_voiced_ms = 1200
```

Treat these as calibrated safety settings. Run `yakbox whisper calibrate` and
listen to a representative short-test set before changing them.

When context extraction opens Whisper during a build,
`automatic_join_inspection = true` also checks every non-explicit chunk splice
immediately after WAV assembly. Its versioned report is written under
`OUTPUT_ROOT/qa/joins/`. A failed join stops the build. Explicit pause chunks
are still checked for clicks and unstable speech, but their deliberate silence
is not treated as an excessive internal pause.

PCM click evidence is the hard join gate. Window-level Whisper concerns such as
low confidence, timing instability, a word spanning the declared boundary, or
source-authored repeated tokens are retained as `diagnostic_reason_codes` in
the join report. They do not fail the splice because short-utterance QA and the
whole-manuscript transcript gate own lexical correctness.

Whole-manuscript verification expands hyphenated spoken compounds and applies
the manifest's scoped `alignment_aliases` before comparing tokens. Exact
canonical transcript equality remains the hard release gate. Decode consensus,
minimum-confidence, token-duration, and narrative-pause findings remain visible
as `diagnostic_reason_codes`; they do not replace or weaken the exact lexical
comparison.

General inspection also selects confidence thresholds by clip type. One-word
clips use the strictest word-confidence, no-speech, duration, and fallback
temperature limits. Short phrases, sentences, join windows, and chapters each
have separate defaults. `whisper inspect --clip-type TYPE` can override the
automatic one-word, short-phrase, or sentence classification for a deliberate
test. The report always records the chosen profile.
