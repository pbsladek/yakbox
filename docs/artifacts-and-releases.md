# Artifacts, cleanup, and releases

Yakbox keeps state and output separate:

```text
.yakbox/runs/       append-only journals, summaries, worker logs
.yakbox/cache/      verified reusable synthesis chunks
.yakbox/locks/      target locks
.yakbox/trash/      reversible cleanup quarantine
build/yakbox/raw/
build/yakbox/mastered/
build/yakbox/reports/
build/yakbox/release/
build/yakbox/auditions/
build/yakbox/previews/
```

Every managed artifact has a typed sidecar with identity, stage fingerprint,
digest, size, dependencies, run, and provenance. Unknown files are reported
but never adopted or removed automatically.

```console
yakbox artifacts list
yakbox artifacts usage
yakbox artifacts verify
yakbox artifacts cache list
yakbox status
yakbox explain --chapter 0001
```

`artifacts verify --repair-metadata` is explicit recovery: it first validates
the current audio/JSON payload, then accepts those bytes and refreshes only
size/digest metadata. Do not use it to conceal unexplained corruption; retain
the old journal and investigate first.

## Cleanup and recovery

Cleanup prints a revalidated plan by default:

```console
yakbox artifacts clean
yakbox artifacts clean --kind preview --older-than 7
yakbox artifacts clean --apply
yakbox artifacts trash list
yakbox artifacts trash restore CLEANUP_ID
yakbox artifacts trash restore CLEANUP_ID --path previews/RUN/sample.wav
yakbox artifacts trash purge CLEANUP_ID --yes
```

Apply moves eligible artifacts and sidecars into quarantine atomically under a
target lock. Restore verifies digests and refuses collisions. Purge is the only
irreversible step.

Normal cleanup protects:

- sources, pronunciations, config, and unknown files;
- active or interrupted runs;
- the current plan's mastered WAVs, delivery MP3s, inspection reports, and all
  immutable release evidence;
- artifacts owned by configured recent successful runs;
- raw synthesis until a release exists when `raw_until_release = true`.

Audition and preview expiration are independently configurable.
Master/delivery files left by chapters removed from the current plan become
eligible after their run-retention window; current chapter outputs do not.

The synthesis cache has an independent, preview-first policy:

```console
yakbox artifacts cache clean --max-age-days 30
yakbox artifacts cache clean --max-bytes 2147483648 --apply
```

Invalid entries are always candidates. A byte limit keeps the newest valid
chunks. Applying cache cleanup removes only cache files; built chapter audio
and source files remain untouched.

## Release

The default release retains mastered WAV chapters and matching ordered MP3
chapters. M4B is optional:

```toml
[targets.default]
mastering = true
wav_sample_rate = 44100
mp3_bitrate = "192k"
m4b = true
m4b_bitrate = "192k"
```

```console
yakbox inspect
yakbox assemble
yakbox release check
yakbox release check --write-manifest
yakbox release diff RELEASE_ID_1 RELEASE_ID_2
```

Release check re-inspects media and verifies graph completeness, ordering,
metadata, paths, sizes, and digests. It refuses missing or incomplete stage
output. Inspection reports include integrated LUFS, true peak, loudness range,
and leading/trailing silence. Optional target thresholds turn those metrics
into release gates.

Writing release evidence copies the verified WAV and MP3 sets into a new,
release-ID-scoped immutable directory. Its manifest records book and chapter
metadata, cover digest, artifact digests, node fingerprints, logical-voice
provenance, and backend/FFmpeg versions. MP3 and M4B outputs receive available
book metadata and cover art; M4B chapters use manuscript titles. A later source
rebuild replaces only mutable working artifacts, so older release snapshots
remain byte-for-byte verifiable and are never silently replaced.

For distributed hosted work, `yakbox shards export` creates stable shard
manifests and `yakbox shards verify` proves complete, non-overlapping
recombination before release.
