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
  --takes 4
```

Yakbox keeps the local speech worker open for the entire session, so the model
and current voice conditioning aren't reloaded for every take. Each take is
written under `.yakbox/repair-sessions/<repair-id>/`. A one-chunk repair points
directly to each WAV. Repairs that cover several chunks also include an M3U
playlist.

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

Use `--no-rebuild` if you want to approve several repairs before rebuilding:

```console
yakbox repair approve REPAIR_ID yakbox.toml --take 2 --no-rebuild
yakbox build yakbox.toml --chapter 0002
```

## Context and scope

The repair mode controls how much speech is regenerated:

- `target-only` synthesizes the selected chunk directly.
- `context` speaks the target with neighboring same-voice prose, uses Whisper
  to find it, and keeps only the aligned target. This is the default because it
  helps short dialogue retain natural rhythm.
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
takes = 4
whisper_qa = true
rebuild_on_approval = true
```

Command-line options override these values for one session. Repair decisions
remain target-specific and book-specific under `.yakbox/repairs/`; the repair
engine itself contains no character names, manuscript phrases, or book rules.

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
