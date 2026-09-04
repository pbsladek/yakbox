# Localized regeneration and repair

When one line sounds wrong, you shouldn't have to send the whole chapter back
through the speech model. Yakbox can regenerate that line, keep several takes
for listening, and rebuild the chapter from the approved take plus cached audio.

## The editing loop

Start by finding the passage. You can select it by stable chunk ID, source line,
or a unique text fragment:

```console
yakbox repair plan yakbox.toml --chapter 0002 --line 81
yakbox repair plan yakbox.toml --chunk-id 25da6c92c04a26a059d2d55b
yakbox repair plan yakbox.toml --text "Micah Levi"
```

The plan reports the source location, routed speaker and profile, number of
chunks that will be synthesized, and the joins that need another inspection.
It doesn't load Chatterbox or write audio.

Generate audition takes next:

```console
yakbox repair generate yakbox.toml \
  --chunk-id 25da6c92c04a26a059d2d55b \
  --takes 8 \
  --minimum-passing 2
```

Yakbox keeps the local speech worker open for the entire session, so the model
and current voice conditioning aren't reloaded for every take. Each take is
written under `.yakbox/repair-sessions/<repair-id>/`. A one-chunk repair points
directly to each WAV. Repairs that cover several chunks also include an M3U
playlist.

`--takes` is a maximum budget. Generation stops as soon as
`--minimum-passing` takes have passed every configured gate. Failed takes stay
in the session as diagnostic evidence; they never count toward the stopping
condition.

After listening, approve one take:

```console
yakbox repair approve 20260810T044921Z-f6f3aff7 yakbox.toml --take 2
```

Approval is explicit. Yakbox verifies the candidate digest and checks that the
source text, chunk ID, and profile still match the session. It then rebuilds
the affected chapter. Unchanged speech comes from the synthesis cache; only the
approved replacement changes. Mastering, manuscript verification, MP3 encoding,
and final inspection still run over the chapter because loudness and encoded
audio are chapter-level outputs.

For several problems in one chapter, use a transactional batch. Staging never
changes approved repair state. Finalization revalidates every source and audio
digest, commits the complete set in one manifest write, and rebuilds the chapter
once:

```console
yakbox repair batch begin yakbox.toml
yakbox repair batch approve BATCH_ID REPAIR_ID_1 yakbox.toml --take 2
yakbox repair batch approve BATCH_ID REPAIR_ID_2 yakbox.toml --take 1
yakbox repair batch finalize BATCH_ID yakbox.toml
```

For several known locations, generate every audition under one warm model
lifetime and create the approval batch up front:

```yaml
# repairs.yaml
repairs:
  - line: 81
    chapter: 0001-room-609
    mode: sentence
  - text: "Anyone touch him?"
    chapter: 0001-room-609
    mode: context
```

```console
yakbox repair batch generate repairs.yaml yakbox.toml
yakbox repair batch approve BATCH_ID REPAIR_ID yakbox.toml --take 2
yakbox repair batch finalize BATCH_ID yakbox.toml
```

The generation result includes one `review.m3u` playlist. Requests are grouped
by profile while reports retain input order. The shared service and aligner
lifetime avoids repeated model startup without weakening per-take QA.

A batch is intentionally limited to one target and chapter so finalization can
never trigger several hidden full-chapter pipelines. `--no-rebuild` on finalize
commits the approvals without running the one chapter build.

## Context and scope

The repair mode controls how much speech is regenerated:

- `target-only` synthesizes the selected chunk directly.
- `context` speaks the target with neighboring same-voice prose, uses Whisper
  to find it, and keeps only the aligned target. This is the default because it
  helps short dialogue retain natural rhythm.
- `sentence` replaces only the sentence containing a unique `--text` selector.
- `clause` replaces only the clause containing a unique `--text` selector.
  Both modes align the old chunk, speak the selected span with hidden adjacent
  prose, extract it with Whisper, match its level and high-frequency balance to
  the original, use quiet-boundary adaptive crossfades, and verify the complete
  reconstructed chunk plus both new splice boundaries. A versioned `-splice.json`
  report records the crop and every bounded DSP adjustment.
- `neighbors` replaces the target and its immediate speech neighbors.
- `paragraph` replaces every planned chunk from the same source paragraph.
- `scene` replaces speech up to the nearest explicit pause boundaries.

Context speech is used only to shape delivery. It isn't added to the chapter.
Whisper transcript, timing, waveform-edge, clipping, detached-speech, and
stationary-voicing checks must pass before a take can be approved. Use
`--no-whisper` only when you deliberately want listening review to be the gate;
signal checks still run.

## Configurable defaults

Put editing defaults in `yakbox.toml`:

```toml
[repairs]
mode = "context"
takes = 8
minimum_passing_takes = 2
whisper_qa = true
rebuild_on_approval = true
```

Command-line options override these values for one session. Repair decisions
remain target-specific and book-specific under `.yakbox/repairs/`; the repair
engine itself contains no character names, manuscript phrases, or book rules.

Local repairs are content addressed twice: raw synthesis requests and fully
verified candidates. A repeated request therefore skips generation, alignment,
crop/splice work, and reconstructed/join QA only when every input fingerprint
matches a candidate that previously passed those gates. Inspect the durable
decision with:

```console
yakbox repair cache why-miss REPAIR_ID yakbox.toml --take 1
```

Each event reports `content_match`, `metadata_missing`, `audio_missing`,
`metadata_invalid`, or `integrity_mismatch`; manuscript text is not stored.

## Finding a problem by timestamp

Every raw chapter build writes an exact frame timeline under
`.yakbox/assemblies/<target>/<chapter>.json`. Map a heard timestamp back to its
source with:

```console
yakbox repair locate 0002-new-partner 197.42 yakbox.toml
```

The result contains the stable chunk ID, source lines, speaker/profile route,
and the two adjacent join indices. After a repair, automatic join QA runs only
for joins whose audio, ordering, or boundary changed.

Changed speech chunks are also verified independently before mastering. Their
Whisper evidence is keyed to each chunk's audio bytes and expected-text hash,
so later repairs keep evidence for unchanged chunks. The final mastered chapter
still receives a complete manuscript-verification pass before delivery.

## Performance evidence

Every terminal build writes `.yakbox/runs/<run-id>/performance.json`. It records
wall time, node and stage durations, node reuse, pending/reused synthesis chunks,
generated candidates and generation/extraction/evaluation reuse, Whisper cache
entries, and estimated model loads. Individual Whisper verification reports
also include cache hit/miss counts. This makes a slow synthesis run, local
repair, or chapter verification visible without exposing manuscript text.

## Understanding reuse

Build preflight now reports synthesis work separately from chapter-level work:

```text
synthesize 1/77 chunk(s), inspect 2 affected join(s)
```

For a detailed cache decision, run:

```console
yakbox artifacts cache why-miss CHUNK_ID yakbox.toml
yakbox artifacts cache inspect FINGERPRINT yakbox.toml
```

The report uses hashes instead of manuscript text. It distinguishes missing or
invalid entries from changes to the profile, backend runtime, reference audio,
sample rate, Chatterbox controls, seed, and context/QA policy.

Current assembly manifests pin the synthesis chunks they reference. Normal
cache cleanup won't delete those chunks, even when a byte or age limit would
otherwise select them.

## What makes insertion cheap

Speech seeds and cache keys use a content-based chunk ID rather than the chunk's
position in the chapter. Inserting a paragraph near the beginning can change
assembly order without assigning new seeds to every unchanged passage after it.
If a paragraph itself is split into several model-sized chunks, edits within
that paragraph may still change the identities of its later pieces. Other
paragraphs remain reusable.
