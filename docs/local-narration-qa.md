# Local narration QA

The short live canary proves only that Chatterbox can load and write a WAV. The
local narration E2E suite separately exercises the installed `yakbox` CLI,
semantic chunk planning, all bundled reference voices, a complete audiobook
build, technical inspection, cache reuse, and a human listening review.

It is opt-in because it loads the real model and generates several minutes of
audio. Install the runtime and run it with:

```console
uv sync --frozen --all-groups --extra local
YAKBOX_RUN_LOCAL_E2E=1 \
YAKBOX_LIVE_LOCAL_DEVICE=auto \
YAKBOX_LOCAL_E2E_OUTPUT="$PWD/build/local-chatterbox-e2e" \
uv run pytest --force-enable-socket -m live -s \
  tests/live/test_local_chatterbox_e2e.py
```

Each invocation creates a new `run-*` directory; it never overwrites an earlier
review. Pytest reports the package path in its warning summary. That path contains:

- two profile-named audition WAVs for every voice listed in `qa.toml`: one
  sustained narration paragraph and one escalating Mara–Wren argument;
- the raw joined chapter, mastered WAV, and delivery MP3;
- the CLI plan, command results, chunk boundary evidence, audio measurements,
  hashes, duration, loudness, peak, edge silence, speaking rate, and exact join
  timestamps in `qa/report.json`; and
- `qa/listening-review.md`, with a repeatable one-to-five listening guide; and
- `qa/listening-review.toml`, a structured review bound to the exact
  `qa/report.json` hash.

## What the corpus tests

The original-fiction fixture under `tests/fixtures/local-chatterbox-e2e/`
contains reflective narration, suspense, dialogue, a final emotional turn, a
fictional pronunciation rule, and an explicit pause. Its first and final long
paragraphs force sentence-level chunks; each Markdown paragraph creates a
paragraph boundary. The plan must therefore contain sentence, paragraph, and
explicit-pause boundaries, with every chunk at or below the configured limit.

The narration audition holds suspense and an emotional turn inside one
uninterrupted synthesis request. The separate dialogue audition uses clearly
attributed Mara–Wren turns and escalating accusations. The joined chapter
routes narration to Andy Minter, Mara to Caro Davy, and Wren—a man—to Nick
Whitley. Reviewers can compare the single-voice auditions with character-routed
delivery across real paragraph boundaries and chunk joins.

The join policy inserts 100 milliseconds after a sentence chunk and 250
milliseconds after a paragraph chunk. The fixture also requests a 450
millisecond explicit pause. Five-millisecond edge fades reduce clicks where
separately generated chunks meet. The plan and technical report establish what
was assembled; the listening review establishes whether those joins actually
sound natural.

The automated cadence gate currently accepts 110–190 words per minute. The
minimum applies to chunks of at least eight words because fixed pauses make a
lower-bound WPM misleading for brief dialogue. The corpus keeps short answers
inside fuller attributed turns rather than synthesizing a bare “No.” The
maximum still applies to every speech chunk. The gate also
requires −30 to −10 LUFS, no true peak above 0.1 dBFS, and no more than one
second of leading or trailing silence. These are deliberately broad defect
bounds, not a claim that every technically conforming read is good narration.
Cadence is checked both for the complete files and for every synthesized speech
chunk, so a reasonable chapter average cannot conceal one rushed or dragging
segment. At every planned silence, the raw PCM step into and out of silence must
remain below −50 dBFS; this catches click-producing discontinuities even when
the silence duration itself is correct.
The generated join table gives the start/end time of every synthesized chunk,
the inserted silence, and the source line so audible seams can be reviewed
directly. The E2E gate also reads the raw PCM at every planned pause window and
requires every sample in that window to be silence.

## Narration review

Listen once without reading the source, then again while following it. Review
each voice for:

- audiobook narration fit rather than assistant or advertising delivery;
- first-listen clarity of names, dialogue speakers, and event order;
- sentence, paragraph, dialogue, and explicit-pause rhythm;
- suspense, reflection, dialogue, and the final emotional turn;
- natural, unsplit pronunciation of “Asterion”; and
- timbre, loudness, pace, and character continuity across chunks.

For the dialogue passage, also score whether Mara and Wren remain identifiable
without reading, whether the conflict escalates instead of beginning at full
intensity, and whether the attributed turns form one continuous confrontation.
For the joined chapter, confirm that each routed voice enters at the intended
source paragraph and returns cleanly to narration afterward.

Technical checks deliberately do not assign a narration-quality score. A file
can have valid loudness, peaks, duration, and encoding while still sounding
flat, rushed, overacted, or visibly stitched. Quality approval requires a named
reviewer to complete the generated scorecard and record the preferred profile,
approved settings, and any blocking defects.

Record the final decision in `qa/listening-review.toml`. Leave `status` as
`pending` while listening. For approval, change it to `approved`, supply the
reviewer and timezone-aware `reviewed_at` timestamp, choose a configured
`preferred_profile`, describe the approved settings, score every voice and
chapter dimension at or above the configured passing score, and mark every
required join observation `pass`. Keep `blocking_defects` empty only when the
run is approved.

Validate the completed review without loading Chatterbox or regenerating audio:

```console
YAKBOX_NARRATION_REVIEW="$PWD/build/local-chatterbox-e2e/run-123/qa/listening-review.toml" \
uv run pytest -m live tests/live/test_narration_review.py
```

Replace `run-123` with one specific run directory. Validation fails if the
report hash changed, a configured voice or dimension is missing, a score is
below the threshold, a required join failed, or reviewer metadata is incomplete.

## Configuration

`qa.toml` is the QA contract. It controls the voice list, required chunk
boundaries, audition passage list, automated duration and peak bounds, score
scale, passing score, narration/dialogue audition dimensions, and joined-chapter
dimensions. `yakbox.toml` independently controls the narrator and character
profiles, per-character Chatterbox settings, and build target. This keeps the
routed cast and review policy easy to change without editing the test
implementation.
