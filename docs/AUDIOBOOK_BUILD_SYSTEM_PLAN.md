# yakbox: production audiobook build system

Status: working design; Phases 1–7 implemented, later phases pending

Last reviewed: 2026-07-29

Target release: `yakbox` 1.0

## 1. Executive summary

Yakbox is an audiobook build system. Its primary job is to turn long-form
source material into planned, reproducible, inspectable, and release-ready
audiobook artifacts. Source normalization, spoken-text overlays, pronunciation
policy, voice/profile auditions, chunking, synthesis, mastering, technical
inspection, chapter assembly, manifests, resumability, generated-file lifecycle
management, and release evidence form the center of the product.

Text-to-speech and voice/speech transformation are the build system's
foundational services. Yakbox also exposes those services directly for one-off
synthesis, voice conversion, testing, and automation. The audiobook build
system must consume the exact same typed application services and backend
adapters as the direct interfaces: no duplicate provider calls, model wrappers,
request models, retry logic, output writers, or CLI-to-CLI orchestration.
Yakbox builds audiobooks by eating its own dog food.

Local Chatterbox models, remotely hosted Chatterbox models, and Resemble.ai are
interchangeable synthesis capabilities beneath that build system where their
semantics align. Backend-specific controls remain typed and visible; local GPU
lifecycle, remote-worker execution, and provider HTTP concurrency are not
forced through one unsuitable runtime model.

One implementation workstream rewrites yakbox's existing Resemble.ai
integration around a reusable, fully typed async Python client. Keep Click
command callbacks synchronous and bridge them to the async application layer
with `asyncio.run()`. Add bounded concurrent synthesis, deterministic results,
retries, atomic file writes, progress reporting, machine-readable output, and
failure isolation so audiobook builds can use hosted synthesis reliably.

The local PyTorch commands remain synchronous and are not internally rewritten
by this async workstream. They are still fully in scope for yakbox as a
product: packaging, command discovery, compatibility, end-to-end usage,
documentation, CI, and interaction with shared configuration must treat them
as core functionality. Existing or planned hosted Chatterbox and
Resemble integrations are foundational build capabilities; their provider
contracts must be inventoried and specified rather than guessed.

Speech-to-text, transcription, and transcription-assisted QA are explicitly
deferred from 1.0 so the implementation can focus on excellent text-to-speech
and audiobook production. Do not scaffold an STT backend, public transcribe
command, transcription service, or QA stage until a later specification is
explicitly requested.

Every backend implementation must be useful through direct CLI/Python
interfaces and through audiobook build nodes. Transport, provider mapping,
model lifecycle, orchestration, filesystem policy, artifact state, and
terminal presentation remain separate so direct use and builds cannot drift.

The default distribution contains the audiobook planner, artifact graph,
source pipeline, direct speech interfaces, and lightweight hosted backends.
GPU/local dependencies are isolated in a `local` extra and imported lazily, so
hosted-only users, CI planning jobs, and `yakbox --help` never need to import or
install PyTorch.

The canonical Python namespace is `yakbox`; an existing `yakbox_cli` namespace
is retained only as a documented compatibility shim.

The project will use a `src/` layout, PEP 621 metadata, Astral `uv` for Python,
environments, locking, building, and publishing, `uv_build` as the pure-Python
build backend, Ruff for linting and formatting, Astral `ty` as the blocking
type checker, pytest with native async tests, RESPX for HTTP mocking, and
GitHub Actions for CI and trusted PyPI publication.

## 2. Scope and constraints

### 2.1 Product scope

Yakbox's core product surface includes:

- audiobook workspace initialization, validation, planning, auditioning,
  building, resuming, inspecting, assembling, and release checking;
- source-to-speech normalization, pronunciation policy, artifact lineage,
  mastering, technical QA, and deterministic incremental rebuilds;
- generated audio/intermediate inventory, storage accounting, retention,
  cleanup planning, quarantine, restore, and explicit purge;
- direct text-to-speech and voice/speech transformation interfaces backed by
  the same services used in audiobook builds;
- local Chatterbox synthesis, voice conversion, batch, verification, and
  model-management capabilities;
- remotely hosted Chatterbox inference where an existing or separately
  specified service contract is available;
- Resemble.ai synthesis, streaming, voices, recordings, and projects;
- shared text/audio input conventions, output safety, config and credential
  behavior, human and JSON presentation, documentation, and release quality.

Optional installation extras and different sync/async implementations are
deployment boundaries. User-facing documentation and top-level help lead with
the audiobook build lifecycle, then explain how to choose backends and use the
same speech services directly.

Yakbox is local-CLI-first. The default product assumes one user, a terminal,
ordinary files, and source-controlled manifests—not a daemon, web application,
account system, approval workflow, remote coordinator, or database. Commands
should complete useful work directly, print the next practical action, and
avoid introducing persistent status or ceremony unless it materially improves
build correctness, recovery, or safety. Advanced team/distributed services may
integrate later without changing this local core.

### 2.2 Target 1.0 implementation workstreams

- Build the canonical audiobook manifest, normalized speech document, artifact
  DAG, planner, audition/build modes, mastering/inspection/assembly stages,
  resumability, sharding, and release checks specified in section 3.
- Retain mastered WAV chapters and derive delivery MP3 chapters as the default
  release outputs, with combined M4B available only when configured.
- Build typed artifact inventory, storage/status diagnostics, reference-aware
  cleanup planning, quarantine/restore, and explicit purge services.
- Define shared typed TTS and speech-transformation services with backend
  capability contracts. Both audiobook nodes and direct commands use them.
- Port `cloud tts`, `cloud stream`, `cloud voices`, and `cloud projects` from
  `requests` to one `httpx.AsyncClient` per command or client context.
- Add `cloud batch` with bounded concurrency, retry/backoff, progress,
  per-row errors, systemic-failure short-circuiting, deterministic output
  naming, a durable journal, resumability, an audit report, and correct exit
  status.
- Add conservative hosted-synthesis character, request, and estimated-spend
  guardrails shared by direct hosted operations and audiobook builds.
- Add a typed, non-synthesizing `yakbox doctor` diagnostic for installations,
  backends, credentials, tools, devices, storage, and workspace safety.
- Publish typed, documented speech-service, audiobook-build, and
  provider-specific Python APIs.
- Preserve all existing cloud command flags and observable behavior unless a
  change is explicitly called out as a bug fix in this specification.
- Add professional packaging, developer tooling, CI, release automation,
  security checks, documentation, and measurable quality gates.
- Split optional local/GPU dependencies from the default lightweight
  audiobook/hosted installation without changing local generation semantics.
- Extend existing configuration for audiobook defaults, backend profiles,
  resources, and credentials without rewriting unrelated user settings.
- Adapt existing local and hosted synthesis services into audiobook build
  nodes without rewriting local model-generation internals.

### 2.3 Workstream non-goals

- No algorithm changes to local model generation merely to fit the audiobook
  coordinator. Backend adapters, worker boundaries, packaging, lazy imports,
  compatibility shims, integration tests, and documentation remain in scope.
- No async or threaded sharing of PyTorch models.
- No speaker playback.
- No undocumented adaptive rate-limit algorithm.
- No attempt to replace Resemble.ai's official SDK for every API endpoint.
- No WebSocket streaming, webhook server, or distributed job queue in 1.0.
- No multi-user review server, mandatory chapter-approval state, or embedded
  audio player in 1.0.
- No backend abstraction so generic that Chatterbox- or Resemble-specific
  capabilities become awkward. Share speech operations and artifact contracts,
  not a lowest-common-denominator settings dictionary.
- No speech-to-text backend, transcription command, transcript-alignment QA, or
  STT dependency in 1.0. Preserve an architectural extension point in prose,
  not unused production abstractions or placeholder commands.

### 2.4 Repository-state rule

Before implementation, inventory the existing repository and record:

- current command help and exit behavior;
- current batch input conventions;
- current config keys and precedence;
- every import of `requests`;
- current package/distribution names and entry points;
- current local and hosted-Chatterbox commands, model/service boundaries, and
  install requirements;
- tests that encode compatibility.

If the repository is still empty, use the fresh package layout and canonical
contracts in this document. If existing code appears later, compatibility
tests take precedence over guessed behavior. Do not claim `requests` can be
removed until `rg` and the dependency graph prove nothing else imports it.

### 2.5 Parked post-1.0 phases

The design reserves two optional phases after the local CLI reaches its release
gate:

1. a localhost-served browser UI;
2. a separately hardened service/Kubernetes deployment.

They are planned in phases 9 and 10 so the core application boundaries do not
block them, but they are inactive. Their presence in this document is not
authorization to scaffold code, add dependencies, build frontend assets,
publish containers, or change the 1.0 release criteria. Each phase starts only
after the user explicitly asks to start that named phase. Completing phase 8
does not automatically activate phase 9, and completing phase 9 does not
automatically activate phase 10.

## 3. Product goals and service-level targets

### 3.1 Functional goals

1. Build long-form source into complete, technically inspected,
   provenance-bearing audiobook artifacts.
2. Make every build stage planned, resumable, selectively rebuildable, and
   safe to run locally or in hosted automation.
3. Let one audiobook target choose local Chatterbox, remotely hosted
   Chatterbox, Resemble, or another verified backend without changing the
   source pipeline or artifact contract.
4. Use the same TTS and transformation services for audiobook nodes and direct
   CLI/Python calls.
5. Support fast, bounded auditions across voices/profiles before expensive
   production rendering.
6. Preserve direct synthesis, streaming, voice conversion,
   batch, verification, model, and provider-management usability.
7. Synthesize independent hosted rows concurrently with a user-controlled
   limit and isolate row-local failures.
8. Make retries, cancellation, progress, output commits, and systemic failures
   predictable, observable, and testable.
9. Provide stable typed Python APIs and machine-readable schemas for speech
    services, audiobook planning/execution, and artifacts.
10. Produce outputs safe for scripts, CI, audiobook-scale runs, and releases.
11. Manage generated audio and intermediate files safely with typed inventory,
    storage accounting, preview-first cleanup, quarantine, and restore.
12. Bound hosted-provider exposure with preflight estimates and atomic runtime
    limits that concurrency, retries, and resume cannot bypass.
13. Diagnose installation, configuration, backend, tool, device, and workspace
    readiness without generating audio.
14. Leave narrow, capability-based seams for verified additional synthesis
    backends
    without flattening backend-specific controls.

### 3.2 Measurable targets

- A clean fake audiobook target with at least 20 chapters must pass normalize,
  plan, audition, render, master, inspect, assemble, and release-check through
  the same speech services exercised by direct-command contract tests.
- Architecture tests must fail if audiobook modules import provider transports,
  local model packages, Click commands, or backend-private request types
  directly; all speech work crosses the typed application-service boundary.
- Ten mocked 200 ms syntheses at concurrency 10 must overlap, with observed
  maximum in-flight requests greater than one and no greater than ten.
- A batch of any supported size must create only `O(concurrency)` active
  synthesis tasks, not one task per row.
- The input parser and work queue use `O(concurrency)` active-row memory.
  Durable journal size and optional collected Python results are `O(rows)`;
  the CLI does not retain full input text or all audio in memory.
- No incomplete final audio file may remain after an error or cancellation.
- Cleanup must never select source, unknown, active-run, or release-protected
  files, and a quarantined artifact must restore with the same digest.
- Batch results and reports must be in input order, regardless of completion
  order.
- A process killed after a journaled success can resume without repeating that
  synthesis when request and output hashes still match.
- A systemic failure stops new scheduling promptly and records all untouched
  rows as `not_run`.
- Hosted usage counters must never exceed configured submitted-character or
  request-attempt limits under concurrency, retries, cancellation, or resume.
- `yakbox doctor` performs no synthesis and makes no network call unless its
  explicit network checks are enabled.
- All HTTP clients and response streams must close on success, error, and
  cancellation.
- Replanning an unchanged audiobook on another supported operating system must
  produce the same normalized segment, shard-assignment, and content
  fingerprints; absolute workspace paths and platform-specific tool locations
  are excluded from portable identities.
- A fake 20-chapter audiobook must prove selective rebuild: changing one
  pronunciation or source chapter invalidates only the dependent render and
  downstream mastering/assembly artifacts.
- Release checking must fail closed when any expected chapter, shard manifest,
  technical inspection, provenance field, or digest is missing or incompatible.
- CLI cold-start time is not a primary optimization target; avoid obvious
  heavy imports in the command-registration path.

### 3.3 Reference use case: reproducible audiobook production

The `anima-cara` repository was reviewed on 2026-07-29 as an evidence-bearing
consumer case study. It is not a runtime dependency and its story-specific
names or defaults must not be copied into yakbox. Its scripts demonstrate that
the useful product boundary is larger than a one-request TTS command.

Its current scripts, Make variables, directory names, Markdown comments, YAML
shape, and CI wrappers are observations, not compatibility contracts. Yakbox
defines the canonical audiobook workspace model and file formats below; the
consumer repository can migrate to those contracts later. Preserve the
workflow need, not incidental implementation details.

Observed workflows include:

- converting one Markdown manuscript into naturally ordered chapter sources;
- keeping printed text and spoken text different through inline regions, plus
  optional explicit pause markers;
- applying a reviewed sidecar pronunciation lexicon without changing the
  manuscript;
- selecting one chapter, several chapters, a direct passage, or a text file;
- using named voice/render profiles with model-specific settings;
- reusing a human voice identity across engines while keeping each engine's
  controls distinct;
- using reference audio for voice conditioning, including preparation and
  rights/provenance concerns;
- auditioning short previews before committing to full chapters;
- rendering the same passage across several profiles and parameter grids for
  comparison;
- chunking long prose at paragraphs, sentences, clauses, and finally safe word
  boundaries;
- inserting stable inter-chunk and explicit pauses;
- writing local model output incrementally and clearing model intermediates
  between chunks to prevent chapter-sized memory growth;
- selecting CPU, CUDA, or MPS, controlling CPU threads, and using deterministic
  seeds where the model permits;
- measuring characters, chunk counts, render time, generated duration, and
  realtime factor;
- keeping auditions, previews, production chapters, processed audio, logs, and
  release artifacts in separate namespaces;
- normalizing loudness and true peak, optionally using two-pass analysis, and
  exporting WAV, MP3, and ACX-oriented outputs;
- inspecting duration, codec, sample rate, channels, bitrate, loudness, true
  peak, and loudness range;
- validating that every expected chapter exists exactly once and combining
  chapters in manuscript order with audiobook metadata;
- sharding long renders for hosted CI time limits, staging per-shard manifests,
  recombining verified artifacts, and writing checksums;
- using isolated Astral `uv` runtimes when model packages require different
  Python or dependency versions;
- preserving engine-specific provenance such as generated-audio watermarks and
  the rights status of reference voice clips.

This evidence defines yakbox's reproducible audiobook build pipeline; low-level
`tts`, `vc`, and provider-management commands expose the same underlying speech
services directly.

### 3.4 Required audiobook-build capabilities

The product design must support the following without forcing all engines into
identical parameters.

#### Source preparation

- Accept plain text, Markdown, an ordered directory, or an explicit manifest of
  source files.
- Provide configurable Markdown chapter splitting with stable natural order
  and deterministic slugs.
- Parse Markdown through a CommonMark token stream. By default speak headings,
  paragraphs, block quotes, list content, and link labels; omit code
  blocks/inline code, raw HTML, images, and link destinations unless a manifest
  explicitly changes that policy. Preserve line/source maps for diagnostics.
- Define namespaced yakbox directives:
  `yakbox:speech:exclude:start/end`, `yakbox:speech:only:start/end`, and
  `yakbox:speech:pause ms=N`. They are HTML comments so ordinary Markdown
  renderers ignore the controls. Paired regions may be inline or block-level;
  reject unbalanced, overlapping, nested, malformed, negative, or excessive
  directives with exact source locations before rendering. Pause values are
  integer milliseconds from 0 through a manifest-configurable ceiling whose
  safe default is 30,000.

Canonical examples:

```markdown
<!-- yakbox:speech:exclude:start -->אמת<!-- yakbox:speech:exclude:end --><!-- yakbox:speech:only:start -->Emet<!-- yakbox:speech:only:end -->

<!-- yakbox:speech:pause ms=750 -->
```
- Produce a normalized intermediate speech document containing ordered text
  segments, pauses, source locations, stable segment IDs, and hashes.
- Keep source preparation backend-neutral. Engine token/character limits are
  applied only after the normalized document exists.
- Define a schema-versioned TOML pronunciation lexicon using repeated
  `[[terms]]` records with written form, spoken form, language, match mode,
  case policy, optional integer priority, status, notes, and an explicit
  enabled flag. TOML keeps audiobook, profile, and lexicon configuration on
  the standard-library `tomllib` boundary. Show a dry-run diff of every
  applied replacement and reject
  ambiguous overlapping rules unless precedence is explicit.

```toml
schema_version = 1

[[terms]]
written = "Wai Tong"
spoken = "Way Tong"
language = "en"
match = "whole_word"
case = "sensitive"
priority = 100
status = "approved"
enabled = true
notes = "Reviewed in the selected narration voice."
```

Only enabled, approved terms apply by default. Replacement is a single,
non-recursive pass after Markdown/directive normalization and before chunking;
a generated spoken form is never fed back through the lexicon. Sort by explicit
higher priority and then longest written form so a shorter term cannot silently
steal part of an approved phrase.

#### Voices, profiles, and execution targets

- Separate a logical voice identity from an engine profile. A voice such as a
  particular licensed reader may map to different Chatterbox, hosted, or other
  engine settings without pretending those settings are equivalent.
- Profile precedence is command override, selected build target, audiobook
  profile, user profile, then backend default. The resolved non-secret profile
  is captured in the run manifest.
- Profiles are schema-versioned TOML, support composition only through an
  explicit `extends` field, reject unknown keys, and expose engine capability
  validation before model load or network use.
- Represent backend, executor, and hardware separately. For example,
  Chatterbox is a backend, `local-process` or a remote worker is an executor,
  and `cpu`/`cuda`/`mps` is a hardware choice. “Hosted” must not ambiguously
  mean either a provider API or a model package running in CI.
- Reference-voice assets carry an ID, digest, local path or managed remote
  reference, preparation recipe, source/rights note, and redistribution
  policy. Reports include the digest and ID, never an unapproved embedded clip.
- Voice-cloning targets require an affirmative rights/consent basis field.
  Yakbox cannot adjudicate that claim, but `release-check` refuses an unknown
  or explicitly restricted basis and records the user's declaration for
  provenance.
- Backend-specific settings remain in typed namespaces. Chatterbox
  `cfg_weight`, `exaggeration`, and sampling controls are not renamed to a
  generic speed or quality field; provider capabilities may offer separate
  high-level presets that resolve to documented settings.

#### Audition and production modes

- `audition` renders a selected passage/chapter across one or more profiles,
  supports deterministic parameter matrices, and defaults to a bounded preview
  rather than a complete book.
- Write profile-named audio files plus one typed `audition.json` containing
  resolved settings, duration, render time, and artifact paths. Print the same
  compact comparison to the terminal. Yakbox does not require a review queue,
  approval state, embedded player, or selection database.
- The user listens with their normal local audio player, may rerun the
  audition with different profiles/settings, then supplies the chosen
  `--profile` to a partial or full build. The CLI prints the exact manifest
  field to persist when the choice should become reproducible; it does not
  silently edit `yakbox.toml`.
- `production` renders every selected source with resumability, durable
  manifests, per-item failure isolation, and optional post-processing.
- `release` requires a clean, complete production graph. It rejects preview,
  limit, unsafe skip, keep-going, or partial-input options; verifies every
  expected artifact and digest; and writes an immutable release manifest.
- Preview outputs, unique timestamped auditions, stable production outputs, and
  release assets use separate directories so an experiment cannot overwrite a
  releasable file.

#### Rendering and scheduling

- Explicit audiobook building may chunk long text; this does not change the
  fail-fast length contract of the low-level `cloud batch` command.
- Chunking is deterministic, Unicode-safe, preserves source locations and
  explicit pauses, records the algorithm/version and hashes, and prefers
  paragraph, sentence, clause, then word boundaries. Hard wrapping is a
  last-resort policy that is visible in the plan/report.
- Each raw render records a content fingerprint over normalized source,
  actually applied pronunciations, logical voice, resolved backend profile,
  backend version, and chunker version. Each downstream stage fingerprints its
  input artifact plus only that stage's settings, such as mastering or
  assembly metadata. Resume and caching use these stage fingerprints rather
  than filename existence, so a mastering change does not re-run synthesis.
- A local model instance renders sequentially unless the backend explicitly
  documents safe parallelism. Multi-profile local concurrency uses isolated
  processes/model instances with a resource budget; hosted API work uses the
  async bounded scheduler.
- The audiobook coordinator invokes local model workers through an internal,
  versioned worker protocol rather than shelling out to yakbox's human CLI.
  Workers receive a non-secret plan fragment by file/stdin, emit structured
  progress separately from logs, and atomically return an artifact manifest.
  API tokens and licensed prompt audio are never placed in process arguments.
- Resource policy includes maximum processes, threads per process, device,
  estimated model memory, and provider concurrency. The planner rejects
  oversubscription it can prove and warns about unverified combinations.
- Local renderers write chunks incrementally and expose backend cleanup hooks.
  Cleanup is an implementation capability, not a public promise that cache
  eviction always improves performance.
- Long-running workers emit periodic heartbeat events and write per-job logs.
  A silent model invocation must still be distinguishable from a dead worker.

#### Mastering, inspection, and assembly

- Treat mastering as an optional, separately fingerprinted stage after raw
  synthesis. Preserve raw audio unless an explicit retention policy removes it.
- Provide an FFmpeg/FFprobe-backed adapter for format conversion, sample rate,
  bitrate, channel layout, head/tail silence, integrated loudness, true peak,
  loudness range, and optional measured two-pass normalization.
- Support named mastering presets, including an audiobook/ACX-oriented preset,
  while also exposing resolved technical values. Do not claim publisher
  compliance solely because a preset was selected; validate the output.
- Audio inspection produces human and versioned JSON results. Empty, truncated,
  malformed, wrong-format, or out-of-policy audio fails before assembly.
- The default audiobook release profile retains one mastered PCM WAV per
  chapter as the lossless source of record and derives one delivery MP3 per
  chapter from those WAV masters. Never create release MP3s by transcoding an
  already lossy provider MP3.
- WAV mastering sample rate/bit depth and MP3 codec/bitrate/channel/ID3 values
  are explicit resolved preset fields recorded in artifact manifests. Chapter
  MP3s include deterministic track order plus configured book, chapter,
  author/narrator, cover, and copyright metadata where supported.
- Master WAVs and default chapter MP3s are release-protected artifacts. Cleanup
  may remove superseded raw/intermediate renders according to policy but cannot
  quarantine the current masters or delivery files until a newer verified
  release supersedes them.
- A combined chapter-marked M4B may be enabled as an additional assembly
  target, but it is not required or generated by default. It is derived from
  the mastered WAV lineage and never replaces the per-chapter WAV/MP3 outputs.
- Assembly consumes an ordered artifact manifest, rejects missing, duplicate,
  stale, or mixed-profile chapters, and writes the combined file atomically
  with title/artist and chapter metadata where the container supports it.

#### Build state and incrementality

- Keep audiobook state entirely file-based. Each run directory contains an
  immutable resolved plan, an append-only NDJSON event journal, and
  schema-versioned JSON run/shard/artifact manifests. Generated audio remains
  in the declared artifact tree.
- The resolved plan records the complete DAG and canonical fingerprints.
  Resume reconstructs node state by replaying the run journal and validating
  artifact manifests, file sizes, and digests. Filename existence alone is
  never completion evidence.
- Plan and run reports automatically include a typed change summary against
  the most recent compatible successful run: added/removed/changed nodes,
  fingerprint reasons, resolved profile/backend/tool changes, and available
  duration/loudness/QA deltas. `explain` renders this evidence; there is no
  separate review workflow.
- One coordinator-owned writer serializes journal records from local workers
  and hosted tasks, flushes and `fsync`s terminal transitions before reporting
  them complete, and never allows workers to append directly.
- Commit manifests and target/run pointer files with write-to-temporary,
  `fsync`, and atomic replace. Append-only journals are never rewritten during
  a run. A derived summary/index may accelerate inspection but is disposable
  and must be reproducible from the plan, journal, and artifact manifests.
- Acquire an explicit target lock before mutating a build. A second
  coordinator fails clearly unless a separately designed read-only inspection
  mode is used. Detect filesystems without the required lock and atomic-rename
  behavior and fail safely.
- Recovery accepts only complete, schema-valid journal records. A torn final
  line caused by process death is copied to a diagnostic file, then the
  journal is truncated to its last complete record under the target lock
  before appending a recovery event. Corruption before the final record fails
  safely instead of guessing.
- Store portable identities, relative artifact paths, hashes, metrics, and
  safe failure state—not API tokens, full source text, reference audio, or
  provider response dumps. Unknown future schema versions fail with upgrade or
  recovery guidance; file formats evolve through versioned readers rather
  than in-place format mutation.
- Standalone `cloud batch` uses the same journal durability conventions but a
  separate, smaller schema suited to a linear row queue. Audiobook journals
  record DAG node and artifact transitions.

#### Artifact inventory, retention, and cleanup

- Treat generated WAV, MP3, mastered audio, chunk intermediates, auditions,
  previews, reports, worker logs, journals, and caches as managed artifact
  classes. Inventory records their owning run/target, stage, format, byte size,
  content digest, creation time, references, and retention status.
- Determine media type from validated artifact metadata and inspection, not
  only a filename extension. A renamed or malformed `.wav`/`.mp3` must not be
  silently classified as a valid artifact.
- Provide typed inventory, usage, verification, cleanup-plan, quarantine,
  restore, and purge application services. Human and JSON CLI output are
  adapters over those same services.
- Cleanup is manifest-driven and reference-aware. Never remove source files,
  `yakbox.toml`, pronunciation files, reference voices, artifacts used by an
  active/incomplete run, or files referenced by an immutable release manifest.
  Unknown/untracked files are reported but excluded unless a future explicit
  adoption workflow makes them managed.
- Cleanup defaults to a non-mutating plan. The plan lists every candidate,
  reason, size, digest, dependent references, bytes moved to quarantine, and
  bytes reclaimable only after purge. Applying it revalidates those facts
  under the target lock so a stale plan cannot remove a newly referenced
  artifact.
- Normal cleanup moves files atomically into
  `.yakbox/trash/CLEANUP_ID/` and writes a schema-versioned manifest sufficient
  to restore them. When a configured artifact root is on another filesystem,
  use a managed quarantine directory on that artifact's filesystem and record
  it in the central cleanup manifest; do not silently degrade an atomic move
  into copy-then-delete. Permanent purge is a distinct explicit operation with
  confirmation or an automation-safe `--yes`; immutable release artifacts are
  excluded unless individually named with an additional release override.
  Quarantine remains part of disk usage and does not count as reclaimed space
  until purged.
- Cancellation/error cleanup removes safe `.part` files immediately when
  ownership is certain. Startup recovery inventories abandoned temporaries and
  offers them as cleanup candidates rather than deleting ambiguous files.
- Manifest retention policy may keep the newest N successful runs, auditions
  for a duration, raw renders until a verified release, and selected mastered
  formats. Automatic post-build cleanup is opt-in and may quarantine only; it
  never permanently purges.
- Planning estimates required and reclaimable storage from known inputs and
  historical duration/bitrate data. Warn before synthesis when free space is
  clearly insufficient and support an optional workspace storage budget.
- `status` reports incomplete/failed runs and storage use. `explain` shows why
  a node is reusable, stale, blocked, or scheduled by comparing its canonical
  fingerprint and dependencies. These diagnostics use the same inventory and
  plan models as cleanup.

#### Sharding, artifacts, and provenance

- An audiobook plan can partition work into stable shards by segment/chapter and
  logical voice without hardcoded chapter numbers.
- Every shard has a schema-versioned manifest containing audiobook/run IDs,
  source and profile fingerprints, assigned items, produced artifacts,
  checksums, metrics, and terminal status.
- The combine stage accepts only a complete, non-overlapping set of compatible
  shard manifests and verifies file hashes before copying or concatenating.
- Build artifacts include raw/processed lineage, tool/backend versions,
  watermark disclosure where applicable, reference-voice provenance IDs, and
  checksums. Secrets and full licensed reference clips are excluded.
- Backends may run in isolated `uv` environments with pinned Python and package
  requirements. Environment resolution is planned and cached separately from
  synthesis; a generated environment lock/fingerprint is included in
  reproducibility metadata.
- An audiobook manifest cannot inject arbitrary shell commands, package indexes,
  wheel URLs, or dependency specifications into those environments. Built-in
  backend definitions own executable/module names and allowed dependency
  constraints. Any future custom-worker escape hatch requires an explicit
  trust boundary and is not enabled merely by loading an audiobook manifest.

### 3.5 Audiobook manifest and command direction

Add a schema-versioned `yakbox.toml` audiobook manifest for reproducible
multi-stage builds. It references sources, logical voices, backend-specific
profiles, build targets, mastering presets, assembly metadata, output roots,
and retention/provenance policy. Paths are relative to the manifest by default.
Environment variables and secrets are referenced by name, never serialized
into the manifest or resolved plan.

Canonical audiobook workspace layout:

```text
audiobook-root/
├── yakbox.toml
├── pronunciations.toml           optional canonical lexicon
├── source/                       workspace-owned input; name configurable
├── .yakbox/
│   ├── locks/                    one target lock per active coordinator
│   ├── runs/
│   │   └── RUN_ID/
│   │       ├── plan.json         immutable resolved DAG
│   │       ├── journal.ndjson    append-only node/artifact events
│   │       ├── run.json          atomically replaced derived summary
│   │       └── workers/          per-worker structured logs
│   ├── trash/
│   │   └── CLEANUP_ID/           restorable files + cleanup manifest
│   └── cache/                    disposable derived data
└── build/yakbox/                 declared generated artifacts
    ├── auditions/
    ├── previews/
    ├── raw/
    ├── mastered/
    ├── reports/
    └── release/
```

Only `yakbox.toml` is required and no source directory name is hardcoded.
`.yakbox/` is tool state and should normally be ignored by version control;
deleting it may discard resumability/cache but never source. `build/yakbox/`
contains reproducible user artifacts and is independently configurable.
Release outputs are staged under a new run/version directory and never replace
an older release implicitly. Yakbox does not write generated chapters back
beside source Markdown.

With resume enabled, yakbox scans run metadata for the newest incomplete run
whose target and resolved-plan fingerprint match. If none matches, it starts a
new run. Valid artifact manifests from older runs may still satisfy unchanged
DAG nodes when their fingerprints and output digests match. `--no-resume`
always starts a new run but does not disable safe content-addressed artifact
reuse; an explicit rebuild/clean command would be a separate destructive
operation.

Canonical command direction:

```text
yakbox init [DIRECTORY]
yakbox validate [MANIFEST]
yakbox plan [MANIFEST] [--target NAME] [--json]
yakbox audition [MANIFEST] --target NAME
  [--chapter SELECTOR | --text TEXT | --text-file PATH|-]
  [--profiles NAME...]
  [--matrix KEY=VALUES...]
yakbox build [MANIFEST] --target NAME
  [--chapter SELECTOR] [--profile NAME]
  [--resume | --no-resume] [--dry-run]
  [--from STAGE] [--through STAGE]
  [--max-submitted-characters INT]
  [--max-provider-requests INT]
  [--max-estimated-spend DECIMAL] [--yes]
yakbox inspect [MANIFEST] [PATH...]
yakbox assemble [MANIFEST] --target NAME
yakbox release check [MANIFEST] --target NAME
yakbox status [MANIFEST] [--target NAME] [--json]
yakbox explain [MANIFEST] --target NAME
  [--chapter SELECTOR | --artifact ID]
yakbox doctor [MANIFEST]
  [--target NAME] [--backend NAME] [--network] [--deep] [--json]

yakbox artifacts list [MANIFEST] [--target NAME] [--kind KIND] [--json]
yakbox artifacts usage [MANIFEST] [--target NAME] [--json]
yakbox artifacts verify [MANIFEST] [--target NAME] [--repair-metadata]
yakbox artifacts clean [MANIFEST]
  [--target NAME] [--kind KIND] [--older-than DAYS] [--keep-runs N]
  [--apply]
yakbox artifacts trash list [MANIFEST]
yakbox artifacts trash restore CLEANUP_ID [--path RELATIVE_PATH]
yakbox artifacts trash purge [CLEANUP_ID] --yes

yakbox tts [TEXT] --backend NAME [--profile NAME]
  [--max-submitted-characters INT]
  [--max-provider-requests INT]
  [--max-estimated-spend DECIMAL] [--yes] ...
yakbox vc INPUT_AUDIO --backend NAME [--profile NAME] ...
yakbox backends list
yakbox backends capabilities [NAME]
```

The first two groups are yakbox's primary audiobook lifecycle and managed
artifact surface. `build` executes the complete target DAG by default; stage
selectors are expert/debug controls and cannot weaken release completeness.
`artifacts clean` prints a plan by default; `--apply` quarantines its validated
candidates but does not permanently purge them. The direct speech commands
remain first-class for one-off calls and automation.

Both groups call the same typed synthesis, transformation, profile-resolution,
output, progress, and artifact services. Audiobook
commands additionally compose those primitives with worker isolation,
mastering, inspection, assembly, and release reporting. They never shell out
to direct yakbox commands, and direct commands never maintain parallel backend
implementations. `plan` performs no model load, network request, or output
mutation. `audition`, `production`, and `release` have the distinct safety
contracts above.

Backend selection precedence for direct operations is explicit option, selected
profile, user config, then the documented default. Existing `yakbox tts`/`vc`
local behavior remains the compatibility default until a deliberate migration
changes it. Existing `yakbox cloud tts` and `cloud stream` map to the same
speech services with the Resemble backend preselected; provider-management
commands remain under their provider group.

The initial manifest schema should describe Chatterbox local/remote execution
and Resemble-hosted synthesis cleanly. Other engines may implement the same
capabilities later, but the schema uses backend-owned settings rather than
baking Pocket TTS or Kokoro parameters into yakbox's universal core.

Do not add an `anima-cara` importer, Makefile parser, legacy directory mode, or
special YAML dialect to yakbox. Provide a worked migration document showing
how a script-heavy audiobook repository can replace its split/render/master/
assemble wrappers with canonical yakbox manifests and commands; changes needed
in that consumer repository belong there.

## 4. Corrected provider API contract

These details were checked against the official Resemble documentation on
2026-07-28. Provider behavior remains an external dependency; contract tests
must use representative fixtures and documentation must be rechecked before a
release.

```text
MGMT_BASE  = https://app.resemble.ai/api/v2
SYNTH_BASE = https://f.cluster.resemble.ai
```

Every request uses:

```text
Authorization: Bearer <api_token>
```

Never log or include the token in an exception, report, debug dump, URL, or
test snapshot.

### 4.1 Direct synthesis

```text
POST {SYNTH_BASE}/synthesize
```

Request:

```text
voice_uuid: str                       required
data: str                             required; text or SSML; max 3000 chars
project_uuid: str                     optional
title: str                            optional
precision: MULAW|PCM_16|PCM_24|PCM_32 optional; default PCM_32
output_format: wav|mp3                optional; default wav
sample_rate: int                      optional; provider-supported value
use_hd: bool                          optional; default false
apply_custom_pronunciations: bool     optional; default false
```

Successful response includes `audio_content` as base64 plus duration, synthesis
duration, format, sample rate, title, timestamps, and issues.

### 4.2 HTTP streaming synthesis

```text
POST {SYNTH_BASE}/stream
```

The response body is chunked WAV audio. The current documented `data` limit is
**2,000 characters**, not 3,000. Therefore `cloud stream` is not a valid
fallback for an over-limit direct-synthesis row. Long input guidance must point
to explicit text chunking (local `batch --chunk-chars` or the planned
`yakbox build` workflow), not to `cloud stream`.

Streaming accepts `voice_uuid`, `data`, optional `project_uuid`, `precision`,
`sample_rate`, `use_hd`, and `apply_custom_pronunciations`. It does not expose
the direct endpoint's `mp3` output contract; the CLI output is WAV.

### 4.3 Voices and recordings

```text
GET  {MGMT_BASE}/voices?page=&page_size=
POST {MGMT_BASE}/voices/{voice_uuid}/recordings
```

The recording request is multipart with a required file, name, and text.
`emotion`, `is_active`, and `fill` are optional in the current platform guide.
Validate the documented name/text size limits before a request, while leaving
audio-duration validation to the provider unless a reliable local metadata
reader already exists.

### 4.4 Projects

```text
GET  {MGMT_BASE}/projects?page=&page_size=
POST {MGMT_BASE}/projects
```

The current create schema is:

```text
name: str               required
description: str        optional
is_collaborative: bool  optional
is_archived: bool       optional
```

The earlier draft's `is_public` field is not in the current official contract
and must not be sent.

### 4.5 Clips

```text
GET {MGMT_BASE}/projects/{project_uuid}/clips?page=&page_size=
GET {MGMT_BASE}/projects/{project_uuid}/clips/{clip_uuid}
```

Clips are internal provider records when a project is supplied. Do not poll
clips during direct synthesis unless the endpoint response contract requires
it. Expose clip methods later without coupling them to the 1.0 batch engine.

### 4.6 Defensive status handling

- Success is any documented `2xx`, not only `200`, unless an endpoint requires
  a specific response body.
- `408`, `425`, `429`, `500`, `502`, `503`, and `504` are retryable.
- Other `4xx` responses fail immediately.
- Connect, read, write, pool, and protocol transport failures are retryable
  subject to the policy in section 9.
- A `2xx` JSON response that cannot be decoded or lacks required fields is a
  provider protocol error, not a successful result.
- Error bodies shown to users are sanitized and truncated to 2 KiB. Include a
  provider request ID header when present.

## 5. Supported Python and dependency policy

### 5.1 Current-stable component policy

Start each implementation phase and release candidate from the latest stable,
mutually compatible releases available from the components' official
registries. This applies to Python, Astral tooling, direct and development
Python dependencies, Node.js, the frontend package manager, TypeScript, Vite,
React, Material UI, CodeMirror 6 packages, browser-test tooling, container base
images, Helm, and CI actions. Do not begin new work from stale tutorial pins,
deprecated packages, archived mirrors, or an older major merely because it is
more familiar.

“Latest” is a selection policy, not a floating runtime constraint:

- exclude alpha, beta, release-candidate, nightly, and unreleased versions
  unless this specification explicitly approves one, such as the documented
  temporary use of Astral `ty` while it remains beta;
- select versions together and verify their Python/Node/platform/backend
  compatibility rather than independently demanding a version combination
  that cannot work;
- document any compatibility exception, its owner, reason, affected feature,
  and removal condition. Local Chatterbox/PyTorch/CUDA compatibility is a
  legitimate reason to hold that optional stack below an otherwise-current
  release; it is not a reason to hold back the lightweight core or UI;
- declare tested compatibility ranges for direct dependencies and commit exact
  transitive resolutions in lockfiles. Production, CI, release, and packaged
  frontend builds use frozen locks and never resolve “latest” at startup;
- use Astral `uv` to add, resolve, lock, synchronize, audit, build, and publish
  Python dependencies. Do not maintain a parallel `pip` requirements workflow;
- when Phase 9 is activated, record the exact Node.js active-LTS line and
  package-manager version, commit the frontend lockfile, and install only with
  its frozen-lock mode. Browser assets are bundled into the wheel, never loaded
  from a CDN;
- run automated monthly update proposals and an explicit update review before
  every release candidate. Merge an update only after typing, unit,
  integration, browser, packaging, performance, and applicable local-model
  compatibility gates pass;
- retain the minimum-direct-resolution CI job so declaring current versions
  does not conceal incorrect lower bounds. Raise or remove a lower bound when
  the project no longer tests it.

At the start of every phase, record the selected versions and official release
sources in its architecture decision or implementation PR. Exact versions
shown elsewhere in this design are the reviewed baseline as of this document's
date, not instructions to downgrade a newer implementation environment.

Every new direct dependency must also pass a maintenance admission review:

- identify the canonical source repository, official registry package,
  license, responsible upstream, release notes, security-reporting path, and
  reason yakbox needs it;
- confirm the canonical project is active and the selected release supports
  yakbox's runtimes. An archived mirror is acceptable only when upstream
  documents its live replacement, as CodeMirror does; an archived canonical
  repository with no maintained successor is not;
- inspect release and issue activity in context. Do not reject a small,
  finished library solely for a quiet release cadence, but require credible
  security ownership and a practical replacement or vendoring/removal plan;
- prefer a well-maintained standard-library or already-selected dependency
  capability over adding an overlapping package. Every direct dependency has
  an owner inside yakbox and a concise usage/rationale entry in the dependency
  inventory;
- prefer signed/provenance-linked registry releases when upstream provides
  them. Do not use mutable Git branches, unreviewed forks, arbitrary wheel
  URLs, runtime dependency downloads, or CDN-hosted production assets;
- review the complete resolved graph, not only direct packages. A vulnerable,
  yanked, deprecated, quarantined, malicious, or abandoned transitive
  dependency can block an update or release just as a direct dependency can.

Maintenance is continuous. Configure Dependabot for the `uv`, npm, GitHub
Actions, and container ecosystems used by activated phases. Scheduled CI
checks for compatible stable updates, runs `uv audit` against the frozen
Python graph, audits the frozen frontend graph when Phase 9 exists, verifies
available registry signatures/provenance, and reviews dependency changes in
pull requests. Security fixes bypass the normal monthly cadence.

If upstream marks a dependency deprecated, stops providing a credible security
contact, archives the canonical project without a maintained successor, or
leaves a relevant vulnerability without an acceptable mitigation, open a
tracked replacement/removal decision immediately. A temporary hold requires a
documented risk acceptance with an owner, expiry date, affected surfaces, and
compensating controls; it may not become a silent permanent pin. A release is
blocked by unresolved known high-severity vulnerabilities and by expired risk
acceptances.

### 5.2 Python versions

Python 3.14 is the current stable feature release as of this review. Use it for
local development and the release build, but publish with:

```toml
requires-python = ">=3.13"
```

Support CPython 3.13 and 3.14 in CI. This provides modern language and
`asyncio` features without needlessly excluding maintained installations.
Do not target the Python 3.15 prerelease in required CI; it may run as an
allowed-to-fail scheduled job once dependencies support it.

Commit:

```text
.python-version  # 3.14
uv.lock
```

### 5.3 Runtime dependencies

Use a deliberately small runtime set:

```toml
dependencies = [
    "click>=8.4,<9",
    "httpx>=0.28.1,<1",
    "markdown-it-py>=4.2,<5",
    "rich>=15,<16",
]

[project.optional-dependencies]
credentials = [
    "keyring>=25.7,<26",
]
local = [
    "chatterbox-tts>=0.1.7,<0.2",
    # Secure floors and the reviewed Torch/TorchAudio pair are declared
    # directly; see the versioned uv override file for upstream pin conflicts.
    "torch==2.13.0",
    "torchaudio==2.11.0",
]
```

- Click owns parsing, command composition, parameter validation, and exit
  behavior.
- HTTPX owns async HTTP, connection pooling, multipart, and response streaming.
- markdown-it-py provides a token/AST boundary for Markdown source extraction;
  do not implement book-scale Markdown semantics as a chain of regular
  expressions.
- Rich owns progress, tables, color, and terminal-safe rendering. Do not add
  `rich-click`; keep presentation under yakbox's control.
- Use standard-library dataclasses and enums. Do not add Pydantic merely for
  internal response mapping. If Phase 9 is activated, Pydantic is permitted
  only at the optional FastAPI HTTP boundary; domain/application models remain
  the typed dataclasses and value objects used by the CLI.
- Use a small, hand-written retry policy because provider-specific retry and
  stream semantics are central domain behavior. Do not add Tenacity initially.
- Keyring support is optional because headless Linux environments may not have
  a usable secret-service backend.
- FFmpeg and FFprobe are optional external executables for mastering,
  inspection, and assembly. Their absence must not break synthesis or help;
  affected audiobook stages fail at validation with installation guidance.
- The default install contains no PyTorch/chatterbox dependency. Install local
  support with the versioned uv security override command documented in the
  installation guide; direct dependencies imported by local code must be
  declared explicitly rather than assumed transitively.
- This split keeps audiobook installations that use only hosted backends
  small; it is not a product priority boundary. The local extra, supported
  model combinations, and local build/direct examples are documented and
  release-tested as prominently as hosted workflows.

Local commands register without importing their heavy dependencies. Invoking a
local command without the extra raises one concise installation error. The
audiobook-core/hosted test and release matrix must remain green even if the
local ML stack does not yet support the newest Python. A separate CPU-only
local-extra job tests
only the Python/platform combinations actually supported by that dependency
stack and documents the compatibility table.

Runtime lower bounds are compatibility claims, not a substitute for the lock
file. `uv.lock` pins the tested development/application environment. Dependency
updates should be automated monthly and merged only after the full matrix.

### 5.4 Development dependency groups

Define PEP 735 groups:

```toml
[dependency-groups]
test = [
    "hypothesis",
    "jsonschema",
    "pytest>=9",
    "pytest-asyncio>=1.4",
    "pytest-cov>=7",
    "pytest-socket>=0.8",
    "respx",
]
quality = [
    "ruff>=0.16",
    "ty>=0.0.64,<0.1",
]
dev = [
    { include-group = "test" },
    { include-group = "quality" },
    "pre-commit",
]
```

Exact tested versions live in `uv.lock`. Avoid duplicating every exact tool
version in prose. Use Astral `uv audit` for the frozen Python dependency graph
and `uv export --format cyclonedx1.5` for its CycloneDX SBOM instead of adding
separate Python packages that duplicate current uv capabilities.

## 6. Packaging and repository layout

Use the distribution name `yakbox` if it is available and consistent with the
existing project metadata. The canonical import namespace is also `yakbox`.
If phase 0 finds an established `yakbox_cli` import namespace, ship a thin,
warning-emitting re-export shim for the 1.x series; otherwise do not create it.

```text
.
├── .github/
│   ├── dependabot.yml
│   └── workflows/
│       ├── ci.yml
│       ├── live-canary.yml
│       ├── release.yml
│       └── scheduled.yml
├── docs/
│   ├── README.md
│   ├── AUDIOBOOK_BUILD_SYSTEM_PLAN.md
│   ├── artifacts-and-releases.md
│   ├── backends-and-speech.md
│   ├── cloud-and-budgets.md
│   ├── getting-started.md
│   ├── installing-and-releasing.md
│   ├── manifests-and-sources.md
│   ├── migration.md
│   ├── operations.md
│   └── python-api.md
├── examples/
│   ├── README.md
│   ├── local-chatterbox/
│   ├── m4b-release/
│   ├── multiple-voices/
│   ├── pronunciation-heavy/
│   ├── resemble/
│   ├── selective-rebuild/
│   └── tiny-book/
├── src/
│   ├── yakbox/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── cli.py
│   │   ├── config.py
│   │   ├── diagnostics/
│   │   │   ├── __init__.py
│   │   │   ├── checks.py
│   │   │   └── models.py
│   │   ├── local.py
│   │   ├── release_preflight.py
│   │   ├── textutils.py
│   │   ├── py.typed
│   │   ├── audio/
│   │   │   ├── __init__.py
│   │   │   ├── assemble.py
│   │   │   ├── inspect.py
│   │   │   └── master.py
│   │   ├── audiobook/
│   │   │   ├── __init__.py
│   │   │   ├── artifacts.py
│   │   │   ├── build.py
│   │   │   ├── cleanup.py
│   │   │   ├── manifest.py
│   │   │   ├── planner.py
│   │   │   ├── profiles.py
│   │   │   ├── sources.py
│   │   │   └── journal.py
│   │   ├── speech/
│   │   │   ├── __init__.py
│   │   │   ├── capabilities.py
│   │   │   ├── models.py
│   │   │   ├── services.py
│   │   │   └── workers.py
│   │   ├── schemas/
│   │   │   ├── cli-output-v1.schema.json
│   │   │   ├── batch-journal-v1.schema.json
│   │   │   ├── batch-report-v1.schema.json
│   │   │   ├── audiobook-manifest-v1.schema.json
│   │   │   ├── audiobook-plan-v1.schema.json
│   │   │   ├── audiobook-journal-v1.schema.json
│   │   │   ├── audiobook-artifact-v1.schema.json
│   │   │   ├── audiobook-cleanup-v1.schema.json
│   │   │   ├── audiobook-audition-v1.schema.json
│   │   │   ├── audiobook-run-v1.schema.json
│   │   │   ├── doctor-report-v1.schema.json
│   │   │   └── audio-inspection-v1.schema.json
│   │   └── cloud/
│   │       ├── __init__.py
│   │       ├── batch.py
│   │       ├── client.py
│   │       ├── errors.py
│   │       ├── journal.py
│   │       ├── models.py
│   │       ├── output.py
│   │       ├── rate_limit.py
│   │       └── retry.py
│   └── yakbox_cli/              compatibility shim only when required
│       └── __init__.py
├── tests/
│   ├── contract/
│   ├── integration/
│   ├── unit/
│   ├── conftest.py
│   └── fixtures/
├── .gitignore
├── .python-version
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock
```

Use PEP 621 metadata and Astral's native pure-Python build backend. The
bounded build requirement follows Astral's compatibility recommendation:

```toml
[build-system]
requires = ["uv_build>=0.12.0,<0.13"]
build-backend = "uv_build"

[tool.uv]
required-version = ">=0.12.0,<0.13"

[tool.uv.build-backend]
module-name = "yakbox"

[project.scripts]
yakbox = "yakbox.cli:main"
```

The wheel must include `py.typed`. Keep the version in one authoritative
PEP 621 location and use SemVer. Expose `__version__` through
`importlib.metadata.version("yakbox")`; never import the build backend at
runtime.

If the compatibility shim is required, configure `uv_build` to package both
root modules explicitly and make every shim import emit `DeprecationWarning`
with a link to the migration guide. The shim contains no implementation.

Only names listed in `yakbox.audiobook.__all__`, `yakbox.speech.__all__`,
`yakbox.diagnostics.__all__`, or `yakbox.cloud.__all__` are public. SemVer
applies to those names, documented CLI flags, exit codes, and versioned JSON
schemas.
Deprecations remain functional for at least two minor releases and 90 days,
whichever is longer, before removal in a major release.

Build and validate releases with:

```text
uv lock --check
uv build --no-sources
```

Install the built wheel and the sdist-built wheel into clean temporary
environments and smoke-test `yakbox --version`, `yakbox --help`, and a public
API import. `uv_build` must reject invalid project metadata or structure during
the build. Test what will be published, not only the editable source tree.

## 7. Architecture and dependency direction

### 7.1 Layers

```text
                    Click CLI / public Python APIs
                       │                 │
                       ▼                 ▼
             audiobook lifecycle     direct speech
             planner + build DAG      interfaces
                       │                 │
                       └───────┬─────────┘
                               ▼
              typed TTS / transformation services
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
       Chatterbox   Chatterbox  Resemble
       local engine remote API  adapter

Audiobook-only services beside the speech boundary:
source normalization → artifacts/state → mastering → inspection → assembly
```

Filesystem output is an application service dependency, not transport logic.
Click types and calls must not appear in `client.py`, `models.py`, `retry.py`,
or `batch.py`. The public Python API must work without constructing a Click
context or a Rich console. Audiobook modules cannot import provider transports
or local model packages; they depend only on typed speech services. The
Resemble client still owns exactly one injected `httpx.AsyncClient`; the
broader diagram does not weaken its lifecycle rules.

### 7.2 Module responsibilities

- `models.py`: immutable request/result dataclasses, enums, page types, and
  validation that is independent of Click.
- `errors.py`: public exception hierarchy and safe error formatting.
- `retry.py`: retry classification, delay calculation, `Retry-After` parsing,
  and an injectable async sleeper/random source.
- `client.py`: endpoint mapping, HTTP lifetime, serialization, decoding, and
  typed provider errors.
- `output.py`: safe path resolution, collision policy, atomic byte/stream
  writes, and filename normalization.
- `batch.py`: bounded worker orchestration, progress events, ordered reports,
  and per-row failure isolation.
- `speech/`: backend-neutral operation models, capability declarations,
  synthesis/transformation services, executor selection, and internal worker
  protocol.
- `audiobook/`: manifest/profile schemas, normalized speech documents,
  artifact fingerprints, file-backed run journals, DAG planning, sharding,
  resumability, inventory/cleanup, and cross-backend audiobook orchestration.
- `diagnostics/`: typed, read-only installation, configuration, backend,
  credential-presence, tool, device, storage, and workspace checks.
- `audio/`: optional FFmpeg/FFprobe mastering, inspection, and ordered assembly
  adapters; it does not own synthesis or audiobook policy.
- CLI module: config resolution, sync-to-async bridge, human/JSON rendering,
  and exit codes only.

### 7.3 Extension rule

Do not introduce a generic `CloudProvider` or universal settings dictionary.
Define two narrow application capabilities:

```python
class TextToSpeechService(Protocol):
    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact: ...


class SpeechTransformationService(Protocol):
    async def transform_to_file(
        self,
        request: SpeechTransformationRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact: ...
```

Application requests contain operation-wide semantics and a typed backend
profile reference. Backend-specific controls live in validated backend-owned
option dataclasses, never `dict[str, Any]`. Adapters translate these operations
to Resemble requests, local Chatterbox engines, or remote Chatterbox APIs.

The hosted batch engine depends on `TextToSpeechService`, not
`ResembleClient`. Direct commands construct the same services used by
audiobook nodes. Provider management operations such as listing Resemble
voices remain provider-specific and do not need a fictional cross-provider
equivalent.

### 7.4 Audiobook dogfood and execution boundaries

The audiobook build system is the primary consumer of the speech capability
boundary:

- local Chatterbox execution remains synchronous and model-lifecycle-aware;
- hosted Chatterbox may use an async transport adapter while preserving its
  own request, model, and deployment semantics;
- Resemble uses the typed async client specified here.

Dogfood invariants:

- audiobook synthesis calls `TextToSpeechService`; it never imports
  `ResembleClient`, Chatterbox packages, or provider payload models;
- audiobook voice transformation calls `SpeechTransformationService`; direct
  `yakbox vc` calls the same service;
- direct and audiobook paths share validation, resolved profiles, retry/error
  translation, progress events, atomic output, artifact metadata, and secret
  redaction;
- audiobook code never invokes yakbox's CLI as a subprocess or parses its
  human/JSON presentation;
- contract tests run the same fake backend once through a direct command and
  once through an audiobook node, then compare the resolved operation and
  artifact semantics.

A local engine implementation may be invoked in-process for an established
direct command and through an isolated worker for a build. Both execution modes
wrap the same typed local backend adapter and generation implementation; only
lifecycle/isolation differs. Never run GPU calls in `asyncio.to_thread()` or
share a model instance across concurrent work merely to resemble a hosted
client.

Top-level command discovery leads with `init`, `validate`, `plan`, `audition`,
`build`, `inspect`, `assemble`, and `release`. Direct `tts`, `vc`, backend
inspection, and provider management remain prominent secondary interfaces, not
separate products. Exact legacy aliases are informed by phase 0.

### 7.5 Optional UI and service adapters

If explicitly activated, the UI and deployable service remain outer adapters:

```text
local browser ──loopback HTTP/SSE──┐
                                   ├── typed application services
Click CLI / Python API ────────────┤   audiobook + speech + artifacts
                                   │   diagnostics + hosted budgets
Kubernetes API/worker adapters ────┘
```

Neither adapter may shell out to the CLI, duplicate backend logic, parse human
output, or own a second artifact/build model. Browser code never receives
provider secrets or arbitrary filesystem access. Long-running work continues
to use the existing planner, coordinator, journals, locks, progress events,
cancellation, and recovery contracts.

The local UI remains file-backed and single-coordinator; it requires no
database. A first Kubernetes deployment may also run one API/coordinator
replica against a persistent volume without a database. True multi-coordinator,
multi-user scale-out requires a separate coordination/storage decision and may
justify an external database or queue inside the optional service deployment,
but it must never become a dependency of the local CLI.

## 8. Public Python API

The primary supported APIs are exported from `yakbox.audiobook`,
`yakbox.speech`, and `yakbox.diagnostics`, each with an explicit `__all__`.
Provider-specific Resemble management/client APIs remain available from
`yakbox.cloud`. Everything else is internal and may change between minor
releases. A required legacy `yakbox_cli.cloud` module only re-exports the
compatible cloud objects with deprecation warnings.

`yakbox.speech` exports `SpeechSynthesisRequest`,
`SpeechTransformationRequest`, `SpeechArtifact`, `BackendCapabilities`,
`HostedUsageBudget`, `HostedUsageSnapshot`, and the two service protocols in
section 7.3. Factories resolve typed backend
profiles and return owned async context managers so direct callers and
audiobook builders receive identical lifecycle, validation, budget, and error
behavior. Provider transport requests and raw model objects are never part of
this public surface.

Hosted guardrails use typed values rather than loose option dictionaries:

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import NewType

CurrencyCode = NewType("CurrencyCode", str)
PricingSourceId = NewType("PricingSourceId", str)


@dataclass(frozen=True, slots=True)
class HostedUsageBudget:
    max_submitted_characters: int | None = None
    max_provider_requests: int | None = None
    max_estimated_spend: Decimal | None = None
    currency: CurrencyCode | None = None
    pricing_source: PricingSourceId | None = None
    confirm_above_characters: int | None = None
    confirm_above_requests: int | None = None


@dataclass(frozen=True, slots=True)
class HostedUsageSnapshot:
    logical_items: int
    provider_attempts: int
    submitted_characters: int
    estimated_spend: Decimal | None
    currency: CurrencyCode | None
    ambiguous_attempts: int
```

Counts and thresholds are non-negative. A monetary cap requires currency and a
recognized pricing source. Provider-specific estimators translate usage into
this common budget evidence while retaining their source/version; the budget
model does not assume that all providers bill per character.

### 8.1 Resemble client types

```python
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class AudioFormat(StrEnum):
    WAV = "wav"
    MP3 = "mp3"


class Precision(StrEnum):
    MULAW = "MULAW"
    PCM_16 = "PCM_16"
    PCM_24 = "PCM_24"
    PCM_32 = "PCM_32"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_backoff: float = 8.0
    max_retry_after: float = 60.0


@dataclass(frozen=True, slots=True)
class ClientOptions:
    management_base_url: str = "https://app.resemble.ai/api/v2"
    synthesis_base_url: str = "https://f.cluster.resemble.ai"
    connect_timeout: float = 10.0
    read_timeout: float = 120.0
    write_timeout: float = 30.0
    pool_timeout: float = 10.0
    max_connections: int = 20
    max_keepalive_connections: int = 20
    max_json_response_bytes: int = 64 * 1024 * 1024
    retry: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    text: str
    voice_uuid: str
    project_uuid: str | None = None
    title: str | None = None
    precision: Precision = Precision.PCM_32
    output_format: AudioFormat = AudioFormat.WAV
    sample_rate: int | None = None
    use_hd: bool = False
    apply_custom_pronunciations: bool = False


@dataclass(frozen=True, slots=True)
class StreamRequest:
    text: str
    voice_uuid: str
    project_uuid: str | None = None
    precision: Precision = Precision.PCM_32
    sample_rate: int | None = None
    use_hd: bool = False
    apply_custom_pronunciations: bool = False


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    audio: bytes
    duration_seconds: float | None
    synthesis_seconds: float | None
    output_format: AudioFormat
    sample_rate: int | None
    title: str | None
    issues: tuple[str, ...]
    request_id: str | None
    attempts: int


@dataclass(frozen=True, slots=True)
class FileSynthesisResult:
    path: Path
    bytes_written: int
    duration_seconds: float | None
    issues: tuple[str, ...]
    request_id: str | None
    attempts: int
```

Also define typed `Voice`, `Project`, `Recording`, and generic `Page[T]`
models. `Page[T]` contains `items: tuple[T, ...]`, `page`, `page_count`, and
`total_results`; pagination metadata may be `None` only when the provider
omits it. `Voice`, `Project`, and `Recording` expose the documented identifiers,
names, statuses, and timestamps while preserving no unvalidated provider
dictionary as public state.

Preserve unknown response fields internally only when required for forward
compatibility; do not expose raw untyped dictionaries as the primary API.

### 8.2 Client lifecycle and methods

The public signatures are:

```python
from pathlib import Path
from typing import Self

import httpx


class ResembleClient:
    def __init__(
        self,
        api_key: str,
        *,
        options: ClientOptions | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, *exc_info: object) -> None: ...
    async def aclose(self) -> None: ...

    async def synthesize(
        self,
        request: SynthesisRequest,
    ) -> SynthesisResult: ...

    async def synthesize_to_file(
        self,
        request: SynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> FileSynthesisResult: ...

    async def stream_to_file(
        self,
        request: StreamRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> FileSynthesisResult: ...

    def stream(
        self,
        request: StreamRequest,
    ) -> AbstractAsyncContextManager[AsyncIterator[bytes]]: ...

    async def list_voices(
        self,
        *,
        page: int = 1,
        page_size: int = 10,
    ) -> Page[Voice]: ...

    async def create_recording(
        self,
        voice_uuid: str,
        audio_path: Path,
        *,
        name: str,
        text: str,
        emotion: str | None = None,
        is_active: bool = True,
        fill: bool = False,
    ) -> Recording: ...

    async def list_projects(
        self,
        *,
        page: int = 1,
        page_size: int = 10,
    ) -> Page[Project]: ...

    async def create_project(
        self,
        name: str,
        *,
        description: str | None = None,
        is_collaborative: bool = False,
        is_archived: bool = False,
    ) -> Project: ...
```

Validate page values, paths, and documented request limits before sending a
request. `aclose()` is idempotent. Calls made before entering the context,
after closing it, or while it is closing raise a clear client-state error.

```python
import asyncio
import os
from pathlib import Path

from yakbox.cloud import (
    AudioFormat,
    ClientOptions,
    ResembleClient,
    StreamRequest,
    SynthesisRequest,
)


async def main() -> None:
    request = SynthesisRequest(
        text="Hello from yakbox.",
        voice_uuid="VOICE_UUID",
        output_format=AudioFormat.WAV,
    )
    stream_request = StreamRequest(
        text="Hello from yakbox.",
        voice_uuid="VOICE_UUID",
    )

    async with ResembleClient(
        os.environ["RESEMBLE_API_KEY"],
        options=ClientOptions(),
    ) as client:
        await client.synthesize(request)
        await client.synthesize_to_file(request, Path("line.wav"))
        await client.stream_to_file(stream_request, Path("stream.wav"))
        async with client.stream(stream_request) as chunks:
            async for chunk in chunks:
                print(f"received {len(chunk)} bytes")
        await client.list_voices(page=1, page_size=20)
        await client.list_projects(page=1, page_size=20)


if __name__ == "__main__":
    asyncio.run(main())
```

Advanced callers and tests may inject an already-open
`httpx.AsyncClient`. The yakbox client must not close an injected client. It
owns and closes only clients it constructs.

The client is async-only, event-loop-affine, and not advertised as thread-safe.
There is no sync Python wrapper in 1.0 because a wrapper would fail inside an
already-running event loop. The synchronous Click boundary is an application
adapter, not part of the library API.

`stream()` retries status and connection failures only before yielding its
first body chunk. Once bytes have reached the caller, a transport failure is
raised because an invisible restart would duplicate audio. The higher-level
`stream_to_file()` may retry from byte zero because it owns and discards the
temporary file before another attempt. Exiting the raw stream context early
closes the provider response immediately.

### 8.3 Batch API

```python
from datetime import datetime


@dataclass(frozen=True, slots=True)
class BatchRow:
    index: int
    text: str
    row_id: str | None = None
    voice_uuid: str | None = None
    title: str | None = None
    output_name: str | None = None


@dataclass(frozen=True, slots=True)
class BatchOptions:
    output_dir: Path
    default_voice_uuid: str | None
    project_uuid: str | None = None
    concurrency: int = 5
    output_format: AudioFormat = AudioFormat.WAV
    precision: Precision = Precision.PCM_32
    sample_rate: int | None = None
    use_hd: bool = False
    apply_custom_pronunciations: bool = False
    overwrite: bool = False
    dry_run: bool = False
    resume_from: Path | None = None
    journal_path: Path | None = None
    collect_results: bool = True


class BatchItemStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    SKIPPED = "skipped"
    NOT_RUN = "not_run"


class BatchRunStatus(StrEnum):
    OK = "ok"
    PARTIAL_FAILURE = "partial_failure"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class BatchItemResult:
    index: int
    row_id: str | None
    status: BatchItemStatus
    path: Path | None
    bytes_written: int
    output_format: AudioFormat
    duration_seconds: float | None
    elapsed_seconds: float
    attempts: int
    request_id: str | None
    issues: tuple[str, ...]
    error_code: str | None
    error_message: str | None
    text_sha256: str
    request_sha256: str
    output_sha256: str | None


@dataclass(frozen=True, slots=True)
class BatchProgress:
    completed: int
    total: int | None
    result: BatchItemResult


@dataclass(frozen=True, slots=True)
class BatchReport:
    status: BatchRunStatus
    started_at: datetime
    finished_at: datetime
    journal_path: Path
    results: tuple[BatchItemResult, ...] | None
    aborted_reason: str | None

    @property
    def ok_count(self) -> int: ...

    @property
    def failed_count(self) -> int: ...

    @property
    def skipped_count(self) -> int: ...

    @property
    def not_run_count(self) -> int: ...


async def synthesize_batch(
    client: TextToSpeechService,
    rows: Iterable[BatchRow],
    options: BatchOptions,
    *,
    on_progress: Callable[[BatchProgress], None] | None = None,
) -> BatchReport: ...
```

When collected, `BatchReport.results` is input-ordered. The CLI sets
`collect_results=False` and renders from progress events plus the durable
journal, avoiding `O(rows)` result memory during synthesis. The progress
callback receives immutable events in completion order and is called in the
event-loop thread; it may not block or raise. Callback exceptions are
orchestration failures, not row failures. Cancellation is never converted into
an error result.

### 8.4 Audiobook build API

Export the stable build surface from `yakbox.audiobook`. At minimum it includes
immutable `AudiobookManifest`, `BuildPlan`, `BuildTarget`,
`NormalizedSpeechDocument`, `ArtifactRecord`, `AuditionReport`,
`BuildChangeSummary`, `BuildProgress`, and `BuildReport` types plus:

```python
def load_manifest(path: Path) -> AudiobookManifest: ...


def plan_audiobook(
    manifest: AudiobookManifest,
    *,
    target: str,
) -> BuildPlan: ...


async def build_audiobook(
    plan: BuildPlan,
    *,
    resume: bool = True,
    on_progress: Callable[[BuildProgress], None] | None = None,
) -> BuildReport: ...
```

Planning is deterministic and side-effect-free. Execution is async at the
coordinator boundary so it can supervise provider requests and local worker
processes without sharing PyTorch models or blocking the event loop. This does
not make the low-level local Python model API async. Public types contain
resolved non-secret settings and artifact identities, not live model objects,
Click/Rich state, open HTTP clients, or subprocess handles.

Mastering, inspection, and assembly application services remain separately
testable and injectable behind the audiobook coordinator. Add them to public
`__all__` only after their parameter and external-tool contracts are stable.

Artifact lifecycle is also a typed public service:

```python
from typing import NewType


@dataclass(frozen=True, slots=True)
class ArtifactInventory: ...


CleanupId = NewType("CleanupId", str)


@dataclass(frozen=True, slots=True)
class CleanupPolicy: ...


@dataclass(frozen=True, slots=True)
class CleanupPlan: ...


@dataclass(frozen=True, slots=True)
class CleanupReport: ...


def inventory_artifacts(
    manifest: AudiobookManifest,
    *,
    target: str | None = None,
) -> ArtifactInventory: ...


def plan_cleanup(
    inventory: ArtifactInventory,
    policy: CleanupPolicy,
) -> CleanupPlan: ...


def apply_cleanup(plan: CleanupPlan) -> CleanupReport: ...


def restore_cleanup(
    manifest: AudiobookManifest,
    cleanup_id: CleanupId,
    *,
    relative_path: Path | None = None,
) -> CleanupReport: ...


def purge_cleanup(
    manifest: AudiobookManifest,
    cleanup_id: CleanupId | None,
) -> CleanupReport: ...
```

`CleanupPlan` contains immutable candidate identities, expected digests,
reference checks, quarantined/reclaimable-byte estimates, and the inventory
fingerprint used to create it. Applying, restoring, or purging reacquires the
appropriate lock and revalidates paths beneath managed roots. Public callers
cannot bypass release protection through an untyped boolean or arbitrary path
list.

### 8.5 Diagnostics API

`yakbox.diagnostics` exposes immutable `DoctorOptions`, `DoctorCheck`, and
`DoctorReport` models plus:

```python
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class DoctorCheckStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class DoctorOptions:
    manifest_path: Path | None = None
    target: str | None = None
    backend: str | None = None
    network: bool = False
    deep: bool = False


async def diagnose(options: DoctorOptions) -> DoctorReport: ...
```

Default checks are read-only and offline: Python/yakbox versions, installed
extras, import boundaries, manifest/config validity, credential presence
without reading secret values into reports, backend registration, model-file
presence, FFmpeg/FFprobe availability, paths/permissions, free space, artifact
roots, file-lock support, and atomic-replace behavior in a temporary probe
inside the workspace.

`network=True` permits only documented non-mutating authentication/connectivity
requests and never synthesis, uploads, project creation, or other potentially
billable mutations. If a backend has no safe diagnostic endpoint, report the
network check as `skipped`. `deep=True` may import an installed local backend
and inspect device/runtime capability, but still must not load a model or
generate audio. Each check has a stable ID, typed status/severity, safe
evidence, remediation text, elapsed time, and whether it was skipped by policy.
The CLI exits 1 when a required check fails, 0 for pass/warnings, and uses the
normal JSON envelope when requested.

### 8.6 Public exceptions

```text
YakboxError
├── SpeechServiceError
│   ├── SpeechValidationError
│   ├── BackendUnavailableError
│   ├── BackendCapabilityError
│   ├── HostedUsageLimitError
│   └── SpeechExecutionError
├── YakboxCloudError
│   ├── ConfigurationError
│   ├── RequestValidationError
│   ├── AuthenticationError
│   ├── ProviderAPIError
│   ├── ProviderProtocolError
│   ├── RetryExhaustedError
│   ├── OutputFileError
│   ├── BatchJournalError
│   └── ResumeMismatchError
├── AudiobookValidationError
├── AudiobookBuildError
├── ArtifactCleanupError
│   ├── CleanupPlanStaleError
│   ├── ProtectedArtifactError
│   └── CleanupRestoreConflictError
├── DiagnosticError
├── WorkerProtocolError
└── AudioInspectionError
```

The existing cloud subtree remains source-compatible:

```text
YakboxCloudError
├── ConfigurationError
├── RequestValidationError
├── AuthenticationError
├── ProviderAPIError
├── ProviderProtocolError
├── RetryExhaustedError
├── OutputFileError
├── BatchJournalError
└── ResumeMismatchError
```

Speech services translate backend/provider errors into the generic speech
subtree while retaining a safe typed cause and backend identifier. Provider
exceptions expose structured fields such as `status_code`, `request_id`,
`attempts`, and safe `detail`. Audiobook exceptions expose
book/target/stage/artifact IDs and a safe cause category. They never expose API
keys, licensed reference audio, or full sensitive response bodies. Batch
converts row-local exceptions to `error` results and uses systemic exceptions
to abort the run and mark untouched rows `not_run`; direct calls raise the same
speech errors that audiobook nodes record.

## 9. Async HTTP, retries, and cancellation

### 9.1 Shared client

Construct at most one HTTPX client for each `ResembleClient` context:

```python
timeout = httpx.Timeout(
    connect=options.connect_timeout,
    read=options.read_timeout,
    write=options.write_timeout,
    pool=options.pool_timeout,
)
limits = httpx.Limits(
    max_connections=options.max_connections,
    max_keepalive_connections=options.max_keepalive_connections,
)
```

Set `Authorization`, a stable `User-Agent` containing the yakbox version, and
JSON accept headers. Base URLs are configurable through `ClientOptions` for
testing and enterprise endpoints, but the CLI must not accept arbitrary base
URLs by default.

### 9.2 Retry contract

Use a generic internal operation helper rather than a decorator that assumes
every operation returns a buffered response:

```python
async def retry_operation(
    operation: Callable[[int], Awaitable[T]],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[Exception], bool],
    rate_limit_gate: RateLimitGate | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T: ...
```

Semantics:

1. `max_attempts` includes the first call and must be at least one.
2. Before attempt `n > 1`, sleep
   `min(max_backoff, base_delay * 2 ** (n - 2)) + jitter`, where jitter is
   uniformly distributed from 0 through 0.25 seconds.
3. Before each network attempt, await the shared rate-limit gate when present.
4. A valid `Retry-After` delta or HTTP-date on `429`/`503` takes precedence,
   capped by `max_retry_after`.
5. HTTP operations translate retryable status responses into a private typed
   exception after capturing safe metadata and closing the response. Streaming
   operations also clean their temporary output before raising it.
6. `is_retryable` classifies only `Exception` instances. Control-flow
   exceptions are never passed to it.
7. On exhaustion, raise `RetryExhaustedError` chained from the last transport
   exception or containing the last response error.
8. Re-raise `asyncio.CancelledError`, `KeyboardInterrupt`, and
   `SystemExit` immediately.
9. Make delay calculation, clock, randomness, and sleeper injectable so tests
   never perform real retry sleeps.

Retry classification is operation-aware:

- paginated `GET` requests retry all listed transient statuses and transport
  failures;
- direct/stream synthesis retries them as required for reliable batch work;
- project creation and recording upload retry rate-limit responses and
  failures known to occur before a request is sent, but do not automatically
  retry an ambiguous read timeout or `5xx` after a mutating request may have
  been accepted.

Retries of synthesis POSTs can still be ambiguous if a timeout occurs after
the provider accepted the request. A synthesis with `project_uuid` may also
create duplicate provider clip records. The provider does not document an
idempotency key for these endpoints. Follow the requested synthesis retry
behavior but document the small risk of duplicate provider work or billing;
do not send invented idempotency headers. For an ambiguous management mutation,
raise a specific error telling the caller to verify server state before
retrying manually.

### 9.3 Shared server-directed cooldown

All batch workers share a monotonic `RateLimitGate`. When any worker receives a
`429` with a valid `Retry-After`, it atomically extends the gate's deadline.
Every worker awaits that deadline before its next network attempt. A later,
shorter value never shortens an existing cooldown.

This gate reacts only to the documented server signal; it does not infer an
account rate or dynamically tune concurrency. A `429` without `Retry-After`
uses normal per-operation exponential backoff and does not invent a global
window. Cancellation interrupts gate waits normally.

### 9.4 Streaming retries

Stream to a same-directory temporary file. If a retryable failure occurs
before or during body consumption:

1. close the response;
2. remove the temporary file;
3. retry from byte zero under the normal policy;
4. atomically replace the destination only after a complete response.

Never append a retried stream. A non-retryable error or cancellation removes
the temporary file. Existing destinations follow the explicit collision
policy.

The public raw `stream()` follows the stricter contract in section 8.2 and
never restarts after yielding bytes.

### 9.5 Bounded batch concurrency

Do not create one task per row. Stream normalized rows from an `Iterable` into
an `asyncio.Queue` with `maxsize = max(1, 2 * concurrency)`. Create exactly
`concurrency` workers. `asyncio.TaskGroup` supervises the producer and
workers because each worker captures per-item exceptions; unexpected
orchestration failures still cancel the group.

Worker count is the concurrency limit. A worker sleeping for retry backoff
continues to occupy its slot, intentionally reducing pressure after throttling.
Validate `1 <= concurrency <= 100`. Set HTTP pool connections to at least the
effective concurrency for batch commands.

`asyncio.as_completed()` is acceptable for small fixed task sets but is not
the primary design because it creates `O(rows)` tasks. Progress is emitted by
workers through a queue and rendered by the CLI.

### 9.6 Row-local versus systemic failures

Continue after row-local validation errors, ordinary request-specific `4xx`
responses, exhausted transient retries, and output collisions for one
destination. Stop accepting new work after:

- `401` or `403`;
- failure to create/write/sync the output directory or journal;
- a global project UUID rejected in a way that applies to every row;
- a provider/account error explicitly classified as global;
- an internal invariant or worker-supervision failure.

On a systemic failure, cancel unnecessary in-flight requests, drain the input
iterator without synthesis when needed to journal each remaining row as
`not_run`, finalize an `aborted` report, and return exit 1. If the journal
itself is unavailable, raise `BatchJournalError` after best-effort cleanup.
`--ignore-errors` applies only to row-local `error` results and never converts
an aborted run to success.

Treat `ENOSPC`, a read-only output filesystem, loss of the output directory,
and shared directory permission failures as systemic. An invalid row-specific
name, an existing destination under the selected collision policy, or a
failure demonstrably isolated to one destination remains row-local.

### 9.7 Memory and response-size contract

The batch engine is bounded by rows and tasks, not by a false constant-byte
claim:

- the producer queue contains at most `2 * concurrency` normalized rows;
- exactly `concurrency` workers may hold request/response state;
- direct synthesis returns JSON containing base64 audio, so each active worker
  may temporarily buffer up to `max_json_response_bytes` plus decoded audio;
- `synthesize()` intentionally returns one decoded audio object to its caller;
- batch `synthesize_to_file()` releases each decoded audio object immediately
  after its atomic file commit;
- raw HTTP streaming and `stream_to_file()` do not buffer the complete WAV;
- journal state and the bounded progress channel remain incremental;
- final JSON report materialization and `collect_results=True` are explicit
  `O(rows)` metadata choices.

Reject a direct JSON response whose declared or observed body exceeds
`ClientOptions.max_json_response_bytes` with `ProviderProtocolError`, closing
the response before retry classification. The default is 64 MiB and the limit
must be positive. Therefore peak batch memory is
`O(concurrency * max_json_response_bytes + queue capacity)`, rather than
`O(rows)`; tests measure a representative large run and enforce a documented
regression budget.

### 9.8 Hosted usage and spending guardrails

Use one typed `HostedUsageBudget` per command/build run and one concurrency-safe
reservation gate shared by all hosted workers. Limits may come from the
selected audiobook target/profile or explicit CLI options; CLI overrides
manifest/profile, then user config. At minimum support:

- `max_submitted_characters`: characters placed into outgoing synthesis
  attempts, conservatively counted again for every retry because an ambiguous
  timeout may have reached and billed the provider;
- `max_provider_requests`: every outgoing synthesis HTTP attempt, including
  retries, but excluding cache hits and non-network validation failures;
- `max_estimated_spend`: an optional `Decimal` amount with an explicit currency
  and pricing-source/version.

The planner reports new logical items/characters after cache reuse, plus a
worst-case attempt range under the retry policy. It does not claim an exact
bill. Estimated-spend enforcement is available only when the provider exposes
an official quote or the user selects a dated, versioned pricing table; if no
reliable pricing input exists, requesting a monetary limit fails before any
network call rather than inventing a rate.

Immediately before each HTTP attempt, atomically reserve its request,
characters, and available estimate. A worker may send only after reservation.
This prevents concurrent workers from racing past a cap. Persist reservations
and confirmed/ambiguous outcomes in the run journal; resume reconstructs prior
usage before scheduling, so restarting cannot reset the budget. Release unused
reservations only when the transport proves the request was never sent.

Crossing a hard cap is a controlled systemic stop: start no new hosted work,
record remaining items as `not_run`, preserve successful artifacts, and explain
which limit was reached. `--ignore-errors` cannot mask it. An interactive
confirmation threshold may require approval for a large predicted run;
non-interactive execution fails unless the manifest already authorizes the
budget or `--yes` is explicit. Reports separate logical work, physical
attempts, submitted characters, known provider usage, and estimates so none is
misrepresented as an invoice.

## 10. Batch input, output, and report contracts

### 10.1 Input normalization

Reuse the existing local batch parser if present, but immediately normalize its
output into `BatchRow`. If no parser exists, define:

- `.txt`: one non-empty line per row; line number is the source index.
- `.csv`: header required; `text` required; `id`, `voice_uuid`, `title`, and
  `output` optional.
- `.jsonl`: one JSON object per non-empty line with the same keys.
- UTF-8 input, with an optional leading BOM accepted.
- Blank lines are ignored in text files and rejected as malformed records in
  structured formats when a record is otherwise present.
- `SCRIPT_FILE=-` reads stdin once into a private temporary spool so the full
  input can be hashed and retried/resumed consistently without retaining it in
  memory.

Parsing errors that make the whole file unreadable are command errors.
Record-local schema/validation errors become failed rows so other records can
continue.

### 10.2 Row validation

Before any network scheduling:

- require non-empty text and a row/default voice UUID;
- reject direct synthesis text over 3,000 Unicode code points;
- reject unsupported format/precision/sample-rate values;
- validate optional output names without trusting them as paths.

An over-limit error says to split the text into rows or use local
`yakbox batch --chunk-chars`; it must not recommend `cloud stream`, whose
documented limit is lower.

### 10.3 Output names and collision safety

Default filename:

```text
{index:06d}-{slug-or-row-id}.{format}
```

Rules:

- normalize a user filename to a basename; reject absolute paths, `..`, path
  separators, NULs, and reserved platform names;
- keep every output under the resolved output directory;
- make names deterministic and resolve duplicate normalized names by appending
  `-2`, `-3`, and so on in input order;
- flush and `fsync` a complete same-directory temporary file before commit;
- with `--overwrite`, commit using `os.replace`;
- without overwrite, atomically hard-link the complete temporary file to the
  destination and then unlink the temporary name, so an external race cannot
  overwrite an existing file; if the platform/filesystem cannot provide this
  no-replace commit, fail safely rather than silently overwrite;
- default to treating an existing final path as a row error;
- `--overwrite` permits replacement;
- resume may skip an existing output only after the journal's request hash,
  recorded byte count, and output SHA-256 all match.

After the atomic commit, `fsync` the parent directory where the platform
supports it before appending the row's success record. This ordering makes a
journaled success evidence of a durable output, not merely a completed HTTP
request. Isolate platform-specific directory-sync behavior behind the output
service and test its call order with fakes.

### 10.4 Durable journal and resume

Before the first network request, exclusively create
`<out-dir>/batch-journal.ndjson` or the explicit `--journal` path. Refuse to
replace an existing journal unless it is supplied through `--resume`. The
first record contains the journal schema version, run ID, input SHA-256,
yakbox version, non-secret effective synthesis options, and start time.

Append one schema-valid record after every terminal row result in completion
order. Flush and `fsync` each record before reporting that row as complete.
Journal records never contain full input text or credentials. A cancellation
or systemic failure appends an interruption/abort record when possible.
Serialize journal writes through one writer task and move blocking
flush/`fsync` work to `asyncio.to_thread()`; never block the event loop or let
multiple workers write the file directly. Each worker awaits a per-record
acknowledgement from the writer before emitting its terminal progress event.

`--resume JOURNAL_OR_REPORT`:

1. validates the schema version and rejects incompatible future schemas;
2. verifies the current input digest and effective options match;
3. reconstructs completed row state by input index and request SHA-256;
4. verifies every successful output's path, byte count, and output SHA-256;
5. emits `skipped` for proven-complete rows and schedules missing, failed,
   changed, or corrupt rows normally;
6. appends to the same journal without rewriting prior records.

Any mismatch that could combine different jobs raises `ResumeMismatchError`
before a network call. Do not infer completion from a filename, file existence,
or text hash alone.

`--dry-run` parses and validates the entire input, resolves effective options
and deterministic output paths, detects collisions/resume mismatches, and
prints a human or JSON plan. It performs no provider calls, creates no audio,
and does not create or append the run journal.

### 10.5 Batch result and final report

Each result contains:

```text
index, row_id, status, path, bytes_written, output_format,
duration_seconds, elapsed_seconds, attempts, request_id,
issues, error_code, error_message, text_sha256, request_sha256,
output_sha256
```

Do not include full input text in the report by default. Write an atomic
`<out-dir>/batch-report.json` after every batch. `--report PATH` overrides the
path; `--no-report` disables only this final materialized report, never the
durable journal. The report includes schema version, yakbox version, run ID,
start/end UTC timestamps, non-secret effective options, input-ordered results,
abort detail, journal path, and summary counts. Cost is omitted until the
provider supplies a documented cost field.

The worker pipeline and journal writer use bounded memory. Materializing the
input-ordered JSON report requires `O(rows)` result metadata during finalization
and is documented as such. `--no-report` keeps the entire CLI run bounded;
Python callers that set `collect_results=True` also explicitly opt into
`O(rows)` result memory.

## 11. CLI contract

### 11.1 General conventions

- Click command callbacks remain regular `def` functions.
- Each wrapper resolves and validates config before calling `asyncio.run()`.
- Shared options may be accepted at `yakbox cloud` and inherited by
  subcommands without removing existing per-command spellings in the first
  compatibility release.
- `--json` emits one document conforming to the bundled CLI output schema.
  Successes, provider/configuration failures, and Click usage errors all use
  the same versioned envelope. Diagnostics and progress go to stderr.
- Human mode uses Rich tables/progress only on an interactive terminal.
- Disable progress when stderr is not a TTY, when `--json` is used, or with
  `--no-progress`.
- Honor `NO_COLOR`; also provide `--no-color`.
- `-q/--quiet` suppresses non-error human output. `-v` enables safe diagnostic
  detail but never secrets or full input text.
- All commands support `--help` without requiring configuration or importing
  GPU dependencies.
- Text-taking commands accept exactly one source: the optional `TEXT`
  argument, `--text-file PATH`, or stdin via `--text-file -`. Multiline text
  and SSML never require unsafe shell quoting.

### 11.2 Config and authentication precedence

For non-secret values, use:

```text
explicit CLI option
> environment variable
> existing yakbox config
> documented built-in default
```

Cloud keys:

```text
RESEMBLE_API_KEY
YAKBOX_CLOUD_VOICE_UUID
YAKBOX_CLOUD_PROJECT_UUID
YAKBOX_CLOUD_CONCURRENCY
YAKBOX_HOSTED_MAX_SUBMITTED_CHARACTERS
YAKBOX_HOSTED_MAX_PROVIDER_REQUESTS
YAKBOX_HOSTED_MAX_ESTIMATED_SPEND
```

Add a persistent `cloud.concurrency` setting with default 5 if the existing
config schema can evolve compatibly. Preserve and migrate existing flat keys
without rewriting unrelated config.

Hosted limits may be stored in user config or audiobook profiles, but they
have no permissive built-in numeric default. Organizations/users choose their
own caps. Monetary configuration includes currency plus a supported
provider/pricing-table identifier; a bare decimal without that context is
invalid.

API-key precedence is:

```text
deprecated explicit --api-key
> RESEMBLE_API_KEY
> selected profile in the optional OS keyring
> legacy config token, read-only with a migration warning
```

Never write a new token to JSON config. `--api-key` remains accepted only as a
compatibility option because command arguments may appear in process listings
and shell history; hide it from canonical examples and emit a deprecation
warning. With the `credentials` extra installed, provide:

```text
yakbox config auth login [--profile NAME]
yakbox config auth logout [--profile NAME]
yakbox config auth status [--profile NAME]
```

`login` prompts without echo and stores service `yakbox/resemble` under the
profile name. It fails clearly when no secure keyring backend exists and never
falls back to plaintext. Environment-only/headless use requires no keyring
extra.

### 11.3 Canonical command surface

Existing flags discovered in phase 0 remain aliases or preserved options.
For a fresh build, the canonical surface is:

```text
yakbox cloud [--profile NAME] [--json] [--no-color] [-q|-v]

yakbox cloud tts [TEXT]
  --text-file PATH|-
  --voice-uuid UUID
  --project-uuid UUID
  --out PATH
  --format wav|mp3
  --precision MULAW|PCM_16|PCM_24|PCM_32
  --sample-rate INT
  --title TEXT
  --hd
  --custom-pronunciations
  --overwrite

yakbox cloud stream [TEXT]
  --text-file PATH|-
  --voice-uuid UUID
  --project-uuid UUID
  --out PATH
  --precision MULAW|PCM_16|PCM_24|PCM_32
  --sample-rate INT
  --hd
  --custom-pronunciations
  --overwrite

yakbox cloud batch SCRIPT_FILE
  --voice-uuid UUID
  --project-uuid UUID
  --out-dir PATH                 default: cloud_batch_output/
  --concurrency INT              config or default: 5; range: 1..100
  --max-submitted-characters INT
  --max-provider-requests INT
  --max-estimated-spend DECIMAL
  --yes
  --format wav|mp3               default: wav
  --precision MULAW|PCM_16|PCM_24|PCM_32
  --sample-rate INT
  --hd
  --custom-pronunciations
  --overwrite
  --journal PATH
  --resume JOURNAL_OR_REPORT
  --dry-run
  --report PATH
  --no-report
  --ignore-errors
  --no-progress

yakbox cloud voices list
  --page INT
  --page-size INT

yakbox cloud voices recordings create VOICE_UUID AUDIO_FILE
  --name TEXT
  --text TEXT
  --emotion TEXT
  --active / --inactive
  --fill / --no-fill

yakbox cloud projects list
  --page INT
  --page-size INT

yakbox cloud projects create NAME
  --description TEXT
  --collaborative / --not-collaborative
  --archived / --not-archived
```

Preserve `cloud voices recording` and established API-key spellings as
deprecated aliases during the compatibility window. Do not use ambiguous
provider-inaccurate antonyms such as `--private` for `is_collaborative`.

The top-level audiobook lifecycle in section 3.5 is the primary surface above
these low-level commands. Both compose the same application services and typed
backends; audiobook commands do not shell out to yakbox's own CLI or scrape
human output.

### 11.4 Output and exit codes

```text
0    complete success, or row-local partial failure with --ignore-errors
1    runtime/config/provider failure or batch partial failure
2    Click usage/parameter error
130  interrupted by Ctrl-C
```

Batch human summary:

```text
8 ok, 1 failed, 3 not run — aborted: authentication failed
row 4 [intro-4]: text exceeds the 3000-character synthesis limit
journal: cloud_batch_output/batch-journal.ndjson
```

JSON output includes a stable `$schema`/`schema_version`, `status` (`ok`,
`partial_failure`, `aborted`, or `error`), exit code, counts, journal/report
paths, safe error object, and ordered result records when materialized.
`--ignore-errors` changes only the process exit status for `partial_failure`;
it never masks `aborted`, configuration, usage, or internal errors.

On Ctrl-C, cancel outstanding work, close response/client contexts, remove
temporary files, append an interruption journal record, write a partial report
when possible, and exit 130.

### 11.5 Machine-readable schema policy

Ship Draft 2020-12 JSON Schemas in `yakbox/schemas` for:

- CLI envelopes;
- batch journal headers/results/terminal records;
- final batch reports;
- audiobook manifests and resolved build plans;
- audition reports and resolved profile comparisons;
- audiobook run/shard/artifact manifests;
- artifact inventories, cleanup plans, quarantine manifests, and cleanup
  reports;
- doctor reports and individual diagnostic evidence;
- audio inspection and release-check reports.

Every emitted runtime document contains an absolute `$schema` identifier,
integer `schema_version`, yakbox version, and UTC timestamp. A user-authored
audiobook manifest contains `$schema`/`schema_version` but does not need a
generated timestamp or tool version until it is resolved into a plan. Additive
optional fields may extend a schema version; removing, renaming, changing
meaning/type, or making an optional field required needs a new schema version.
Keep readers for the current and immediately previous journal/report versions
so upgrades can resume recent work. Validate every emitted fixture against the
packaged schema in tests and document examples generated from those fixtures.

### 11.6 Doctor command safety

`yakbox doctor` performs the typed checks in section 8.5. Its default invocation
is offline, read-only, does not import GPU packages merely to render help, and
does not create persistent workspace files. Temporary filesystem capability
probes are uniquely named, confined to the managed state directory, and
removed on success and failure.

`--network` is required for safe hosted connectivity/authentication checks;
`--deep` is required for heavier local runtime/device imports. Human output
groups checks by workspace, tool, and backend and prints actionable
remediation. JSON contains no tokens, complete environment dump, source text,
or licensed audio paths. A skipped check is distinct from pass, and a required
check that cannot run is a failure unless the target does not use that
capability.

## 12. Testing strategy

### 12.1 Test layers

- Unit: audiobook normalization/planning/artifacts, speech services,
  validation, retry math, safe path rules, response mapping, and batch
  orchestration with fakes.
- HTTP contract: RESPX-backed endpoint requests and representative response
  fixtures. No real provider calls.
- CLI: Click `CliRunner` tests for flags, output streams, summaries, and exit
  codes.
- Packaging: build/install/import/entry-point smoke tests from artifacts.
- Optional live integration: manually triggered or scheduled against a
  dedicated low-cost account; never required for pull requests.

Use `pytest-asyncio` in strict mode so async fixtures/tests are explicit.
Default test runs use `pytest-socket --disable-socket`; individual optional
live tests are the only marked exceptions. RESPX routes must assert that all
expected requests were called and unexpected requests fail.

### 12.2 Required retry tests

- immediate success;
- success after each retryable status;
- `Retry-After` delta and HTTP-date;
- invalid, negative, and excessive `Retry-After`;
- one worker's `429 Retry-After` delays every worker through the shared gate;
- later shorter cooldowns cannot shorten the active deadline;
- a `429` without the header does not create an invented global delay;
- jitter/backoff cap with injected deterministic randomness;
- retryable transport failures;
- non-retryable `4xx` fail immediately;
- response closed before retry;
- final exception retains safe status/request ID/attempt count;
- cancellation is never retried or swallowed.

### 12.3 Required client tests

- exactly one owned `AsyncClient` constructed for multiple operations;
- injected client is reused and not closed;
- owned client closes after success and mid-operation failure;
- auth and user-agent headers;
- correct management versus synthesis base URL;
- request serialization omits `None` and uses provider field names;
- base64 decode and required-field validation;
- streaming chunks write atomically and retry from byte zero;
- raw streaming retries before the first yielded chunk but never after bytes
  have been yielded;
- early raw-stream context exit closes the response;
- multipart file handles close on success/error;
- sanitized/truncated provider error details;
- all public request/result models satisfy strict typing.

### 12.4 Required batch tests

- mixed successes, validation failures, API failures, and exhausted retries;
- `ok`, `error`, `skipped`, and `not_run` state transitions;
- `401`/`403`, journal I/O, global project, and disk failures stop scheduling
  and produce an aborted run;
- `--ignore-errors` never masks an aborted run;
- over-3,000-character direct row makes no network call;
- missing per-row/default voice is a row failure;
- deterministic names, duplicates, unsafe names, and existing paths;
- results remain input-ordered when completion order differs;
- progress emits exactly one terminal event per row;
- journal is append-only, schema-valid, durable per completion, and excludes
  full text/tokens;
- forced termination after a journaled result resumes without another request;
- resume rejects input/options/schema mismatches before a network call;
- resume verifies request hash, byte count, and output hash before skipping;
- report is schema-valid, ordered, atomic, and excludes full text/tokens;
- dry-run performs complete validation/path planning with zero network and
  audio/journal writes;
- `--ignore-errors` changes exit status only;
- worker/task count is bounded by concurrency;
- maximum observed in-flight requests is `>1` and `<= concurrency`;
- atomic usage reservations never exceed character/request caps under
  concurrency, and retry attempts consume fresh conservative reservations;
- resume reconstructs prior hosted usage before scheduling and cannot reset a
  run budget;
- monetary caps fail before network use without an explicit supported pricing
  source, and reports distinguish estimates from provider-confirmed usage;
- cancellation closes the client, cancels workers, cleans temporary files, and
  does not manufacture row errors;
- very large fake input demonstrates bounded active tasks and acceptable
  memory behavior while results are journaled rather than collected.

Do not make wall-clock timing the main concurrency assertion. Instrument the
mock handler with an event/barrier and an in-flight counter. One broad
wall-clock smoke assertion may remain, but it must not be flaky on shared CI.

### 12.5 Required audiobook-build and dogfood tests

- the same fake TTS and transformation services run through direct interfaces
  and audiobook nodes with equivalent resolved operations, progress, errors,
  and artifact metadata;
- architecture/import tests prevent audiobook modules from importing provider
  transports, Chatterbox packages, Click commands, or backend-private models;
- Namespaced yakbox speech exclude/only/pause directives produce the expected
  normalized speech document; malformed pairs and invalid pause bounds fail
  with source locations;
- CommonMark fixtures cover headings, lists, quotes, links, images, code,
  inline directives, raw HTML, Unicode, and exact source-line diagnostics
  without regex-only parsing;
- chapter ordering, slugging, source locations, segment IDs, and hashes are
  deterministic across platforms;
- pronunciation dry-run shows every replacement, respects enabled/status
  fields, higher-priority/longest-match precedence, one-pass non-recursion, and
  detects ambiguous overlap;
- logical voice identity maps to distinct typed backend profiles without
  leaking one engine's controls into another;
- manifest/profile unknown keys, inheritance cycles, invalid capabilities,
  missing reference assets, and unsafe paths fail before model/network use;
- planning performs no network, model load, subprocess, or output mutation;
- preview, audition, production, and release outputs cannot collide;
- audition writes profile-named playable files and a schema-valid comparison
  report; profile overrides are captured in the run while the source manifest
  remains unchanged;
- explicit chunking respects backend limits and source order, preserves pauses,
  and records algorithm/version/fingerprints;
- local single-model scheduling is sequential by default; multi-process
  scheduling obeys process/thread/device budgets; hosted scheduling remains
  bounded by provider concurrency;
- fingerprint changes invalidate only affected downstream artifacts;
- run-journal serialization, target locks, atomic manifest commits, torn-tail
  recovery, mid-journal corruption detection, and unsupported-future-schema
  handling preserve a consistent artifact graph;
- inventory accounts for managed WAV/MP3/intermediate/report/cache bytes by
  run, target, stage, and retention class without trusting file extensions;
- cleanup dry-run is non-mutating, apply rejects stale plans, and reference
  protection prevents deletion of source, active-run, release, and unknown
  files;
- quarantine/restore round-trips filenames, bytes, modes where portable, and
  digests; restore collision and permanent-purge confirmation paths fail safe;
- simulated cancellation and process death leave no owned `.part` output
  committed, while ambiguous abandoned files become visible cleanup
  candidates;
- disk-budget planning warns or fails before synthesis according to policy,
  and status/explain accurately report storage and fingerprint invalidation;
- plan/run change summaries identify added, removed, changed, and reused nodes
  with the correct source/profile/backend/tool/metric reasons;
- an audiobook with mixed local/cache/hosted nodes charges usage only to actual
  hosted attempts, stops safely at its cap, and resumes without losing its
  counters;
- raw rendering, mastering, inspection, assembly, and release manifests form a
  reproducible artifact DAG with no filename-only cache hits;
- FFmpeg/FFprobe invocations use argument arrays, reject missing tools
  clearly, validate metrics, and commit outputs atomically;
- inspection rejects empty, malformed, truncated, wrong-format, and
  out-of-policy audio;
- default release output retains verified mastered WAV chapters, derives
  verified MP3 chapters from those masters with deterministic metadata, and
  never performs lossy-to-lossy release transcoding;
- configured M4B assembly preserves chapter order/markers and remains an
  additional artifact rather than replacing chapter WAV/MP3 files;
- assembly rejects missing, duplicate, stale, out-of-order, mixed-profile, or
  hash-mismatched chapters;
- shard planning is stable, and combine accepts only a complete,
  non-overlapping compatible manifest set;
- release mode rejects preview, limit, unsafe skip, partial, and keep-going
  options and produces a checksum-verified immutable manifest;
- reference-audio rights/provenance and watermark metadata appear in reports
  without copying restricted audio or secrets;
- release-check refuses missing/unknown/restricted reference-voice
  rights/consent declarations while preserving local non-release policy;
- backend-specific `uv` runtime fingerprints are stable and an incompatible
  Python/package requirement fails during planning or environment preparation,
  not halfway through a book render;
- hostile manifests cannot inject commands, indexes, URLs, dependencies,
  environment secrets, or path escapes into local worker setup;
- local worker protocol version mismatch, cancellation, non-zero exit,
  malformed events, missing artifact manifest, and forced termination produce
  safe stage failures with complete logs and no committed partial output;
- a fake audiobook with at least 20 chapters exercises plan, shard, resume,
  master, inspect, assemble, and release-check end to end without a real model
  or provider.

### 12.6 Required CLI and compatibility tests

- help output for every command;
- API key precedence and missing-key message before a network call;
- hidden prompt/keyring behavior and refusal to write plaintext credentials;
- deprecated `--api-key` and command aliases remain compatible without
  appearing in canonical examples;
- positional, `--text-file`, and stdin text sources are mutually exclusive and
  preserve multiline/SSML bytes;
- legacy cloud invocation golden tests captured before refactoring;
- stdout/stderr separation in human and JSON modes;
- every success, usage error, config error, and provider error JSON fixture
  validates against the packaged CLI schema;
- progress enabled only in a TTY-compatible context;
- `NO_COLOR`, quiet, verbose, and non-interactive behavior;
- exact exit code matrix;
- local command registration/help/tests remain unchanged;
- default installation imports and runs cloud/help without PyTorch;
- `yakbox[local]` exposes local behavior on its supported CPU test matrix;
- existing hosted-Chatterbox commands, when present, retain their documented
  behavior and have backend-isolated contract tests;
- doctor defaults to offline/read-only, redacts credentials, leaves no probe
  files, distinguishes pass/warn/fail/skipped, and returns the documented exit
  status;
- doctor network checks require `--network`, never call synthesis/mutating
  endpoints, and skip backends without a safe diagnostic endpoint;
- doctor deep local checks require `--deep` and never load a model or generate
  audio;
- canonical `yakbox` imports and any legacy shim/deprecation contract work from
  built artifacts.

### 12.7 Property-based tests

Use Hypothesis for:

- arbitrary Unicode filenames cannot escape the output directory;
- arbitrary managed artifact paths cannot escape quarantine or restore roots;
- duplicate names remain unique and deterministic;
- retry delay is non-negative and respects caps;
- batch result ordering is invariant under arbitrary completion ordering;
- audiobook artifact ordering and shard coverage are invariant under arbitrary
  completion ordering;
- arbitrary valid directive placement and Unicode source text preserve segment
  identity through normalization;
- supported serialized request values round-trip through response fixtures
  where applicable.

### 12.8 Coverage policy

- Branch coverage, not only statement coverage.
- Repository minimum: 90% line and 85% branch after the migration.
- Audiobook/speech core (`audiobook/`, `speech/services.py`, artifact state)
  and cloud reliability core (`retry.py`, `output.py`, `batch.py`): 95% line
  and 90% branch.
- Every bug fix adds a regression test.
- Coverage exclusions require an inline reason and review.
- Coverage is a gate, not evidence that behavior is correct; contract and
  invariant tests remain mandatory.

## 13. Code quality and maintainability standards

### 13.1 Automated gates

```text
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
uv audit --frozen
uv build --no-sources
```

Ruff enables at least:

```text
E, F, W, I, B, UP, ASYNC, C4, C90, DTZ, G, PERF, PIE, PL, PTH, RUF, SIM
```

Configure `lint.mccabe.max-complexity = 10`. Also enable reasonable limits for
branches, statements, arguments, and returns, with narrow documented ignores
at Click adapters or data-model constructors where the shape is inherently
wide.

### 13.2 Human review standards

- Cyclomatic complexity no greater than 10 per function unless a documented
  exception is approved.
- Prefer functions under roughly 50 logical lines; extract policy from I/O.
- No bare `except`, silent exception swallowing, mutable default arguments, or
  blocking I/O inside async code.
- Public APIs have docstrings, full annotations, and examples.
- Use `Path`, UTC-aware datetimes, immutable `slots=True` dataclasses, and
  explicit enums.
- Avoid boolean positional arguments.
- Never use internal dictionaries as an implicit long-lived schema.
- Log structured identifiers and counts, not secrets or full synthesis text.
- Complexity metrics trigger refactoring discussions; do not game metrics by
  splitting cohesive logic into meaningless wrappers.

### 13.3 Type checking

All Python is typed, not only the public API. Every function and method,
including private helpers, Click callbacks, async workers, fixtures, and test
helpers, declares parameter and return types. Module attributes whose types
cannot be inferred safely are annotated. Public packages ship `py.typed`.

Astral `ty` is the only required type checker. It runs as a blocking gate over
`src/` and tests:

```toml
[tool.ty.environment]
python-version = "3.13"
python-platform = "all"

[tool.ty.src]
include = ["src", "tests"]

[tool.ty.analysis]
strict-equality-semantics = true

[tool.ty.rules]
blanket-ignore-comment = "error"

[tool.ty.terminal]
error-on-warning = true
```

Contain unavoidable untyped provider JSON at the decoding boundary and
validate it immediately from `object`/`Mapping[str, object]` into typed models.
Use typed dataclasses/enums for domain state and `TypedDict` only for explicit
serialization boundaries. Do not spread explicit or implicit `Any` across the
application, use `cast()` as a substitute for validation, return provider
dictionaries from services, blanket-ignore unresolved imports, or disable
rules project-wide to make CI green. Third-party packages without usable type
information receive the narrowest local protocol/stub boundary possible.
Every suppression is narrow, uses `ty: ignore[rule-name]`, includes a reason,
and is covered by a runtime test when static proof is unavailable.

IDs, formats, stages, statuses, capability names, and cleanup dispositions use
dedicated enums or immutable value types instead of interchangeable strings.
Filesystem APIs accept and return `Path`; serialized paths become normalized
relative POSIX strings only at schema boundaries. Callback, iterator,
context-manager, subprocess-event, and injected clock/random/sleeper contracts
are fully parameterized—no bare collections or partially typed callables.

`ty` is beta as of this review but is recommended by Astral for motivated
production users. Pin the tested release in `uv.lock`, retain the `<0.1`
quality-group bound until its compatibility policy stabilizes, and review
scheduled upgrades through normal CI. Do not silently add a second type checker
with competing semantics.

## 14. GitHub Actions and release pipeline

### 14.1 Pull-request CI

`ci.yml` runs with least privileges, explicit job timeouts, concurrency
cancellation for superseded commits, and actions pinned to full commit SHAs.
Use Astral's official `astral-sh/setup-uv` action, pinned to a reviewed commit,
with its built-in cache. Supply the matrix's `python-version` input so uv
installs and selects Python; do not add `actions/setup-python`. Every job
installs from the committed lock with `uv sync --frozen --all-groups` before
running `uv run ...`.

Jobs:

1. `quality` on Python 3.14: Ruff format/lint, `ty check`, config validation.
2. `test-core` on Ubuntu for Python 3.13 and 3.14 with branch
   coverage, audiobook/speech/hosted dependencies, and no local extra.
3. `platform-smoke` on macOS, Windows, and Ubuntu using Python 3.14.
4. `minimum-direct`: in a temporary project copy, resolve
   `uv lock --resolution lowest-direct` and run audiobook, speech, cloud, unit,
   and API tests on Python 3.13, proving published lower bounds.
5. `local-extra`: install `yakbox[local]`, run local CPU smoke tests, and run
   direct-versus-audiobook Chatterbox conformance only on the Python/platform
   combinations supported by chatterbox and PyTorch.
6. `audiobook-build`: install FFmpeg/FFprobe on Ubuntu and run the fake
   audiobook fixture through plan, shard, resume, master, inspect, assemble,
   and release-check; backend synthesis remains fake.
7. `package`: `uv build --no-sources`, metadata check, clean wheel/sdist
   installs, default audiobook/hosted/direct-speech smoke tests,
   `credentials`-extra import tests, and local-extra resolution tests.
8. `security`: `uv audit --frozen`, `uv export --format cyclonedx1.5`,
   frontend lockfile audit/provenance checks when Phase 9 is activated, and
   dependency review for pull requests.

Upload JUnit and coverage artifacts on failure and success. Do not upload
inputs, API tokens, or synthesized audio from private fixtures.

### 14.2 Scheduled CI

Weekly:

- run the test suite against the newest lock resolution;
- optional Python 3.15 prerelease job, allowed to fail until declared
  supported;
- CodeQL analysis;
- dependency audit.

Live provider smoke tests require an explicitly protected GitHub environment,
minimal synthetic text, budget limits, and no execution on forked pull
requests.

### 14.3 Release

Release from an annotated `vX.Y.Z` tag or published GitHub Release:

1. verify tag matches the project version and GitHub Release notes;
2. rerun quality and full tests;
3. build sdist and wheel once with `uv build --no-sources`;
4. validate and smoke-test those exact artifacts;
5. generate `SHA256SUMS`, a CycloneDX SBOM, and build-provenance attestations
   for those exact artifacts;
6. upload artifacts, checksums, SBOM, and attestations to the workflow;
7. publish the same artifacts with
   `uv publish --trusted-publishing always dist/*` through PyPI Trusted
   Publishing (OIDC), with no long-lived PyPI token;
8. attach artifacts, checksums, SBOM, and provenance references to the GitHub
   Release.

Use a protected `pypi` environment with required approval. A manual
`workflow_dispatch` may publish release candidates to TestPyPI. Never rebuild
between verification and publication. Grant `id-token: write` only to the
attestation/publish jobs, keep all actions pinned to commit SHAs, and document
vulnerability reporting and supported versions in the operations guide.

## 15. Delivery phases and acceptance gates

### Phase 0 — baseline and compatibility inventory

Deliverables:

- repository/package/config/dependency inventory;
- a product capability map covering local Chatterbox, hosted Chatterbox,
  and Resemble implementations or explicitly identified gaps;
- captured `--help`, exit codes, and representative CLI behavior across those
  available capabilities;
- characterization tests for existing Resemble, Chatterbox, and local
  commands;
- confirmed Resemble request/response fixtures.

Gate:

- existing behavior is recorded well enough to distinguish a deliberate
  change from a regression;
- no code migration begins while current cloud/local flags, backend/runtime
  constraints, or existing audiobook-workflow contracts remain unknown.

### Phase 1 — modern package foundation

Deliverables:

- `src/` package metadata, `uv.lock`, Python 3.13-3.14 support;
- `uv_build`, console entry point, Ruff, `ty`, and pytest configuration;
- canonical `yakbox` namespace, optional compatibility shim, lightweight
  default dependencies, and `local`/`credentials` extras;
- typed doctor framework with offline package/config/tool checks;
- initial CI for lint, type, tests, and package smoke checks.

Gate:

- clean checkout can run `uv sync --frozen --all-groups`, all checks, build both
  artifacts, install the wheel, and execute `yakbox --help`;
- `yakbox doctor --json` runs its baseline offline checks without importing
  PyTorch or touching the network.

### Phase 2 — speech services and audiobook walking skeleton

Deliverables:

- public immutable speech operation/artifact models and exception hierarchy;
- typed TTS and transformation service protocols plus capability
  declarations and fake backends;
- minimal `yakbox.toml`, one-source normalized speech document, one-target build
  plan, artifact identity, and atomic output helpers;
- thin direct `tts` and `vc` application adapters over the same fake services
  used by the audiobook node;
- architecture/import and dogfood contract tests.

Gate:

- one fake chapter plans and builds through `yakbox build`;
- the same fake TTS/transformation operations work through direct
  interfaces with equivalent validation, progress, errors, and artifacts;
- audiobook code has no provider/model/Click imports;
- no backend migration begins until this vertical service boundary is proven.

### Phase 3 — audiobook source, manifest, and artifact foundation

Deliverables:

- full schema-versioned audiobook manifest, logical voices, typed backend
  profiles, executor/hardware separation, and capability validation;
- CommonMark normalization, speech directives, TOML pronunciations,
  deterministic chapters/segments, explicit long-text chunking, and source
  maps;
- stage-specific canonical fingerprints, build DAG, selective invalidation,
  append-only run journals, target locks, atomic manifest commits, resume, and
  exported plan/run/shard/artifact manifests;
- typed artifact inventory, storage usage, cleanup planning,
  quarantine/restore, and status/explain services;
- workspace-aware doctor checks for manifest validity, permissions, storage,
  locks, and atomic output behavior;
- `init`, `validate`, `plan`, `status`, `explain`, artifact-management, and
  dry-run CLI/Python surfaces.

Gate:

- an unchanged 20+ chapter fake book replans identically across platforms;
- changing one source/pronunciation/profile invalidates only dependent nodes;
- crash/restart, torn-journal-tail, and concurrent-build lock tests preserve
  consistent state;
- cleanup cannot touch source, unknown files, active work, or immutable
  releases, and quarantined fake artifacts restore byte-for-byte;
- planning performs no model load, network request, worker launch, or artifact
  mutation.

### Phase 4 — local Chatterbox through shared speech services

Deliverables:

- existing Chatterbox TTS and voice transformation wrapped by the typed speech
  services without duplicating generation logic;
- in-process direct execution and isolated audiobook worker execution over the
  same local backend adapter;
- model/device/profile capability reporting, worker protocol, resource budgets,
  incremental output, cleanup, and cancellation;
- opt-in deep doctor checks for local package/runtime/device/model-file
  readiness without loading a model;
- existing direct local `tts`, `vc`, and `batch` commands adapted or
  compatibility-wrapped around the shared speech services;
- existing `verify` and `models` behavior retained through typed local
  capability/model-management adapters, without forcing those operations into
  a synthesis protocol.

Gate:

- characterization tests prove no unintended local behavior regression;
- direct and audiobook local calls resolve equivalent backend operations and
  artifact metadata;
- a build never shares one PyTorch model unsafely or runs GPU work in
  `asyncio.to_thread()`;
- default/help/hosted-only installs still do not import PyTorch.

### Phase 5 — hosted speech and provider management backends

Deliverables:

- retry policy, shared server-directed cooldown, `ResembleClient` lifecycle,
  injection, safe errors, response mapping, and atomic streamed output;
- Resemble TTS adapter implementing the shared speech service;
- direct cloud `tts`, `stream`, voices, recordings, projects, and bounded batch
  commands with compatibility aliases;
- durable cloud-batch NDJSON journal, hash-verified resume, dry-run, systemic
  failure handling, and reports;
- shared hosted-usage reservation gate, character/request/estimated-spend
  limits, preflight estimates, confirmation thresholds, and journaled usage;
- opt-in safe hosted doctor checks for connectivity, authentication, and
  capability discovery without synthesis or mutation;
- remotely hosted Chatterbox adapter only when phase 0 identifies and verifies
  its actual service contract.

Gate:

- HTTP contract tests prove retries, resource closure, safe errors,
  one-client reuse, streaming rules, and bounded concurrency;
- the same hosted TTS adapter passes direct, cloud-batch, and audiobook-node
  conformance tests;
- old provider invocations remain compatible, and the obsolete Resemble
  `is_public` project field is absent;
- concurrent/retried hosted work cannot exceed a configured usage cap, and
  safe doctor network checks make no synthesis or mutating request;
- no task/client/temp-file leaks under fault injection.

### Phase 6 — production audiobook pipeline

Deliverables:

- `audition`, `build`, `inspect`, `assemble`, and `release check` application
  services and CLI adapters;
- deterministic multi-profile audition matrices, preview isolation, bounded
  local/hosted scheduling, profile-named outputs, terminal/JSON comparison,
  heartbeats, and worker logs without approval-state machinery;
- optional FFmpeg/FFprobe mastering, technical inspection, source-aligned
  audio diagnostics, ordered assembly, audiobook/container metadata,
  checksums, and provenance records;
- retained mastered WAV chapters and default delivery MP3 chapters generated
  from those masters, plus optional configured M4B assembly;
- retention-policy enforcement, disk-space preflight, verified media
  inventory, and cleanup of generated WAV/MP3/intermediate artifacts through
  the shared artifact lifecycle service;
- stable sharding and verified recombination for hosted automation;
- Chatterbox-local and Resemble-hosted target examples, with remote Chatterbox
  examples only after its contract exists.

Gate:

- the required audiobook/dogfood tests in section 12.5 pass;
- an `anima-cara`-shaped fake fixture—one manuscript, 20+ chapters, spoken-only
  substitutions, pronunciations, multiple audition profiles, sharded
  production, QA, mastering, and full-book assembly—passes end to end without
  importing the reference repository;
- the fake release retains every chapter WAV master and produces the matching
  ordered MP3 chapter set with valid metadata and digests;
- direct speech commands and audiobook nodes demonstrably share services;
- the local generate/listen/choose audition flow requires only ordinary audio
  files, a manifest, and CLI profile overrides;
- release mode cannot accept or publish an incomplete artifact graph.

### Phase 7 — CLI polish and documentation

Deliverables:

- stable JSON schema, TTY-aware Rich output, color/quiet/verbose behavior;
- safe positional/file/stdin text sources, canonical command naming, secure
  keyring profiles, and deprecated secret/command aliases;
- README audiobook quick start, manifest/source/pronunciation guides, backend
  selection, direct speech usage, cloud/batch reference, Python API examples,
  artifact storage/cleanup and recovery, troubleshooting, security/config
  documentation, WAV-master/MP3-delivery/M4B release guidance, hosted-budget
  and doctor guides, and shell completion notes;
- equal-quality local Chatterbox, hosted Chatterbox, and Resemble navigation
  and usage for every capability present in the release;
- release notes, API/CLI migration guide, and a generic script-heavy audiobook
  project migration example using yakbox-native files rather than legacy
  compatibility code.

Gate:

- documentation commands run as tests where practical;
- fresh-user and automation examples are copy/pasteable;
- `yakbox --help` leads with the audiobook lifecycle and makes direct speech
  and backend/provider management discoverable without implementation
  terminology.

### Phase 8 — hardening and release candidate

Deliverables:

- complete CI matrix, security audit, branch coverage, packaging verification;
- minimum-direct dependency, default/local-extra, and schema compatibility
  evidence;
- release workflow with Trusted Publishing;
- checksums, CycloneDX SBOM, provenance attestations, and security policy;
- performance/memory evidence for cloud batch and audiobook build targets;
- release candidate tested from built artifacts on all three operating
  systems.

Gate:

- every item in section 16 has authoritative evidence;
- no unresolved high-severity audit finding;
- no known regression in local, hosted-Chatterbox, or Resemble commands present
  in the release.

### Phase 9 — optional localhost UI (parked; explicit activation required)

Activation gate:

- phase 8 and the 1.0 definition of done are complete;
- the user explicitly asks to start **Phase 9 — localhost UI**;
- an architecture decision locks exact dependency versions, frontend package
  manager/lockfile, packaging impact, accessibility target, and threat model
  for the stack selected below before scaffolding is added.

Until all three are true, do no Phase 9 implementation or preparatory
refactoring beyond preserving the application-service boundaries already
required by the CLI.

Planned deliverables after activation:

- an optional `ui` installation extra so the default/local CLI installations
  gain no server or frontend dependency;
- a fully typed Python server using FastAPI, Pydantic at the HTTP/schema
  boundary only, and Uvicorn; all routes call existing typed application
  services rather than introducing UI-specific domain logic;
- a React frontend using strict TypeScript, Vite, and Material UI. Generate a
  typed client from the versioned OpenAPI/JSON Schema boundary and fail CI when
  generated frontend types drift from Python API schemas;
- current stable, mutually compatible frontend components selected under
  section 5.1, with the Node.js active-LTS line and package-manager version
  recorded, a committed frozen lockfile, automated update and vulnerability
  review, registry signature/provenance verification where available, and no
  runtime CDN dependencies;
- `yakbox ui [MANIFEST] --host 127.0.0.1 --port 0
  [--open/--no-open]`, binding to loopback by default and printing the exact
  URL and shutdown instructions; the local UI command refuses non-loopback
  hosts rather than becoming an accidentally exposed server;
- a local dashboard for manifest/target selection, plan/status/change
  summaries, build progress, cancellation/resume, hosted-usage budgets,
  Doctor results, QA/inspection findings, and artifact storage/cleanup;
- profile-named audition comparison with in-browser playback, resolved
  settings, duration/render metrics, and a clear manifest snippet for the
  selected profile—without introducing approval queues or hidden selection
  state;
- a versioned typed local HTTP API over the existing Python application
  services and server-sent events for one-way progress; use ordinary HTTP
  actions for mutation/cancellation rather than requiring WebSockets;
- an offline-capable packaged frontend with no runtime CDN dependency, no
  telemetry, and no network access except the selected yakbox loopback origin
  and explicitly requested provider operations performed by the Python
  service;
- fully typed Python server/routes under the existing `ty` gate and TypeScript
  `strict` mode with no unchecked `any`. Vite builds hashed static assets into
  the wheel; installing and running the UI requires no Node toolchain;
- loopback/session-token protection, strict origin/host checks, CSRF defenses,
  content-security policy, safe path mediation, secret redaction, and explicit
  confirmations for hosted spending and cleanup/purge actions;
- accessible keyboard navigation, labels, contrast, reduced-motion behavior,
  responsive layouts, and automated HTTP/browser/accessibility tests;
- first-class desktop support for the current stable Chrome, Firefox, and
  Safari releases. Run Playwright Chromium/Firefox/WebKit coverage in CI and a
  real Safari smoke/accessibility pass on macOS before a UI release; do not
  claim that WebKit emulation alone proves Safari compatibility;
- `docs/ui.md` covering startup, security, generated files, recovery, and the
  fact that closing the browser does not corrupt a journaled build, plus
  `docs/ui-design.md` documenting Material tokens, components, layouts, and
  interaction rules.

#### UI information architecture

The UI is an audiobook studio, not a generic infrastructure dashboard. Its
primary desktop workspace has four coordinated regions:

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Book / target / backend     build state     usage budget     actions │
├───────────────┬────────────────────────────────────┬────────────────┤
│ Book outline  │ Manuscript / spoken-text editor    │ Voice, render, │
│ chapters      │ current paragraph highlighted     │ QA + artifact  │
│ segments      │ source ↔ normalized comparison    │ inspector      │
├───────────────┴────────────────────────────────────┴────────────────┤
│ Persistent audio transport + waveform + markers + job progress      │
└─────────────────────────────────────────────────────────────────────┘
```

- The left outline shows chapters/segments in manuscript order with compact
  stale/rendering/ready/warning/error indicators and duration. Search filters
  headings and spoken text without hiding the current audio location.
- The center is reading-first: generous line length, typography, and whitespace
  for sustained manuscript reading. Toggle between editable source Markdown,
  derived spoken text, and a source-versus-spoken diff. Source maps keep the
  same selection visible across modes.
- The right inspector is contextual. For text it shows pronunciations,
  directives, source location, and normalization diagnostics. For audio it
  shows voice/profile, backend settings, format, duration, loudness, lineage,
  and QA findings. Controls come from typed capability/profile schemas rather
  than arbitrary provider JSON.
- The bottom transport remains visible while moving between chapters. It
  combines audio controls, waveform, segment/QA markers, current job progress,
  and an expandable job/log drawer without turning the manuscript into a
  monitoring screen.

Primary routes/views:

- **Studio:** read/edit text, select passages, play the current artifact, and
  generate selected audio;
- **Auditions:** compare profile-named variants for the same source selection;
- **Build:** plan changes, start/cancel/resume work, and inspect stage progress;
- **QA:** navigate silence, loudness, clipping, duration, and format findings
  back to exact text/audio positions;
- **Artifacts:** browse raw/mastered/release variants, storage use, lineage,
  verification, quarantine, restore, and purge;
- **Doctor/settings:** environment readiness, backend capabilities, profiles,
  usage limits, and safe configuration editing.

These are views over one workspace, not independent mini-apps. Selecting a
chapter, segment, artifact, or QA issue keeps that context when changing views.

#### Manuscript and spoken-text editing

- Use CodeMirror **6** as the strict-TypeScript source editor for canonical
  Markdown, with syntax, headings, search, undo/redo, keyboard shortcuts, and
  exact line diagnostics. CodeMirror 6 is distributed as independently
  versioned modular npm packages under `codemirror`, `@codemirror/*`, and
  `@lezer/*`; install the latest stable, mutually compatible packages actually
  imported by the application, including `@codemirror/lang-markdown`, and own
  the small React-to-`EditorView` integration rather than making an
  unmaintained wrapper authoritative. Do not install the legacy CodeMirror 5
  package or use a lossy rich-text representation as the source of truth.
- Treat `https://codemirror.net/` and the package metadata in the npm registry
  as the documentation and release authorities. CodeMirror development moved
  from the now-archived GitHub repositories to the maintainer's
  `https://code.haverbeke.berlin/codemirror/` Forgejo organization in April
  2026; an archived GitHub mirror is not evidence that CodeMirror 6 is
  unmaintained.
- Derived spoken text is never edited as an untracked blob. A proposed spoken
  change becomes a typed source directive, pronunciation entry, or explicit
  source edit, and the UI previews the resulting diff before applying it.
- Markdown saves use revision-checked text patches against the exact UTF-8
  source bytes. Preserve untouched comments, whitespace, line-ending style,
  optional BOM, and final-newline state; only explicitly edited spans may
  change. Before saving, validate directives, show the textual and normalized
  speech diff, require the source revision/ETag to match, write atomically, and
  retain a recoverable backup. An external editor change produces a merge
  conflict view instead of last-writer-wins.
- Maintain crash-recovery drafts beneath `.yakbox/ui-drafts/`, keyed by source
  digest. Drafts are never build inputs until explicitly restored and saved,
  and the artifact cleanup system can inventory them.
- Highlighting text exposes direct actions:
  `Audition selection`, `Generate chapter preview`, and, when the selected
  target permits it, `Rebuild affected production segments`. The UI shows the
  exact invalidated downstream nodes before production work begins.
- Use `tomlkit` at the optional UI editing boundary to apply minimal
  comment/order/whitespace-preserving changes to `yakbox.toml` and
  `pronunciations.toml`. Reparse the candidate bytes through yakbox's canonical
  `tomllib`-based typed loader, compare the semantic diff to the requested
  fields, show the text diff, require the source revision, back up, and commit
  atomically. If either parser cannot round-trip the requested construct
  without unrelated changes, refuse the edit and provide a copyable snippet.
- Backend/profile changes remain temporary overrides until the user previews
  and explicitly applies that typed TOML patch. The UI never rewrites an
  entire TOML document merely to change one profile field.

#### Audio playback and generation

- Serve managed WAV/MP3 with HTTP range support for immediate seek/playback.
  Never read an entire audiobook into browser memory.
- Provide play/pause, scrubbing, volume, mute, configurable skip
  backward/forward, playback speed, previous/next segment, loop selection,
  waveform zoom, and documented keyboard shortcuts. Never autoplay generated
  audio unexpectedly.
- Generate/cache compact waveform peak data as a disposable derived artifact.
  The manuscript cursor and waveform use provider timestamps or verified
  alignment when available. Clearly label approximate segment-level
  synchronization rather than presenting fabricated word precision.
- Clicking manuscript text, a segment, waveform marker, or QA issue seeks every
  other surface to the same source segment. During playback, the visible
  spoken sentence follows audio without constantly scrolling the reader.
- Audition A/B switching keeps the same source segment and nearest verified
  timestamp across variants. The comparison shows voice/profile, backend,
  resolved parameters, generation time, duration, issues, and estimated hosted
  usage. It keeps profile identities visible by default; blind labels remain a
  separately justified future option.
- Use three unmistakable generation scopes:
  **Selection audition** writes only to auditions,
  **Chapter preview** writes only to previews, and
  **Production build** mutates the target DAG. Each button shows destination,
  invalidation, estimated work/storage, hosted budget impact, and overwrite
  policy before starting.
- Existing playable audio remains available while a replacement generates.
  The new artifact appears only after atomic commit and validation; a failed or
  cancelled job never replaces the current playable version.

#### Visual and interaction design

- Use Material Design 3 as the visual and interaction foundation. The intended
  feel is a clean, polished Google-style productivity application: bright,
  calm, highly legible, spacious, and immediately understandable. Follow the
  system's component/state/accessibility behavior without copying Google
  product branding, logos, or proprietary layouts.
- Structure the desktop studio with a Material top app bar, navigation
  rail/drawer, central manuscript surface, contextual side sheet, and anchored
  audio transport. Use standard Material dialogs, menus, tabs, chips,
  text fields, tooltips, snackbars, progress indicators, and selection states
  so common interactions behave predictably.
- Use one Yakbox seed palette expressed through Material tonal roles. The
  default light theme favors clean white and soft neutral surfaces with a
  confident blue primary and a restrained complementary waveform accent. The
  dark theme uses true tonal surfaces rather than simply inverting colors.
  Error, warning, success, generating, stale, and selected states use semantic
  roles plus icons/text; color is never their only signal.
- Define a single token source for Material color roles, typography scale,
  spacing, shape, elevation, state layers, focus rings, motion, waveform, and
  semantic status. Python/schema-generated status names and frontend tokens
  must not drift into separate vocabularies.
- Use a modern sans-serif Material typography system throughout navigation,
  controls, metadata, and editing chrome. The manuscript reading surface may
  use a highly legible book-oriented serif as an intentional content treatment,
  with user-selectable text size, line height, measure, and serif/sans reading
  preference. Bundle or use licensed/system fonts locally; never require a
  Google Fonts network request.
- Choose an actively maintained Material-compatible component implementation
  during the Phase 9 architecture decision instead of rebuilding buttons,
  dialogs, focus handling, and accessibility primitives from scratch. Bundle
  Material Symbols or an equivalent licensed icon set locally and always pair
  ambiguous icons with labels/tooltips.
- Map action emphasis consistently: one filled primary action for the current
  generation/build step, tonal actions for previews/auditions, outlined/text
  actions for secondary operations, and confirmation dialogs for destructive
  purge or release-affecting changes. Do not fill every toolbar with competing
  primary buttons.
- Favor one strong reading canvas, persistent transport, and contextual
  controls over grids of decorative cards. Use progressive disclosure for
  backend internals, logs, and advanced mastering settings. Cards group
  genuinely separate objects; they are not the default container for every
  label and value.
- Material motion is short and purposeful: shared context, panel changes,
  playback position, job completion, snackbars, and artifact replacement.
  Preserve spatial continuity without delaying work. Respect reduced-motion
  preferences and avoid decorative animation during reading.
- Desktop is the primary editing/build surface. Tablet/mobile layouts retain
  reading, playback, progress, and safe cancellation. The navigation rail
  becomes a drawer/bottom navigation and inspectors become Material sheets;
  advanced editing and waveform tools collapse rather than becoming cramped.
- Design every empty, loading, generating, disconnected, stale, conflicting,
  partial-failure, and recovery state deliberately. Errors stay beside the
  affected text/job/artifact and include the next corrective action. Skeletons
  reserve real layout space, progress indicators reflect known versus
  indeterminate work honestly, and snackbars never carry information that
  disappears before it can be acted upon.

#### UI API and safety details

- Use stable typed resource models for workspace outline, source revisions,
  normalized speech, audio ranges/peaks, jobs, progress, artifacts, QA
  findings, cleanup plans, Doctor checks, and hosted budgets.
- Mutating requests carry source/plan revisions and idempotency tokens where
  yakbox owns the operation. Reject stale edits and duplicate generation
  actions before scheduling.
- Server-sent progress is resumable from an event ID; reconnecting the browser
  reconstructs truth from run journals rather than trusting client state.
- Secrets remain server-side. The browser receives backend/profile names,
  capability metadata, and redacted credential status only.
- File browsing is limited to manifest-declared sources and managed artifact
  roots. Every write, generation, cleanup, restore, and purge passes through
  the same typed path and policy services used by the CLI.

The UI process may supervise work only while it runs. Build state remains the
same plans, journals, manifests, locks, and artifacts used by the CLI. On UI
restart, it discovers and resumes that state; it does not add a database.
Multiple tabs share the same target lock and cannot start conflicting builds.
Permanent purge remains deliberately harder than quarantine and requires an
explicit confirmation.

Gate:

- the same fake backend/build conformance suite passes through CLI, public
  Python, and UI HTTP adapters with equivalent requests, progress, results,
  errors, budgets, and artifacts;
- default and `local` installations remain byte/dependency tested without the
  `ui` extra;
- the UI works with networking disabled except for loopback and makes zero
  provider requests during offline planning, browsing, and Doctor checks;
- browser refresh, tab closure, server interruption, cancellation, and resume
  leave journals and artifact commits consistent;
- source save/conflict/draft recovery tests prove byte-preserving Markdown
  edits, atomic writes, external-edit detection, and no build from unsaved
  drafts;
- TOML editing fixtures cover comments, ordering, whitespace, quoted/dotted
  keys, arrays/tables, Unicode, and concurrent external edits; every accepted
  minimal `tomlkit` patch preserves unrelated bytes and passes the canonical
  `tomllib`-based typed loader with exactly the requested semantic diff;
- range playback, waveform peaks, source/audio seeking, A/B switching,
  keyboard controls, and selection/chapter/production generation scopes work
  against representative WAV/MP3 and timestamp/no-timestamp fixtures;
- no provider secret, unrestricted path, full sensitive text, or licensed
  reference audio is exposed to browser state/logs unintentionally;
- responsive light/dark visual-regression, accessibility, keyboard-only, and
  supported-browser tests pass from the built wheel, followed by a human
  design review of the complete reading/listening/generation journey.

### Phase 10 — optional service and Kubernetes deployment (parked; separate explicit activation required)

Activation gate:

- phase 9 is complete and accepted, unless a new decision explicitly justifies
  a headless service without it;
- the user explicitly asks to start **Phase 10 — service/Kubernetes**;
- a separate architecture and security review defines tenancy, authentication,
  persistence, artifact transport, worker trust, recovery objectives, and
  expected scale.

Phase 9 approval does not authorize Phase 10. Do not add container, Helm,
authentication, remote-storage, queue, or cluster dependencies beforehand.

Planned first deployment after activation:

- an optional `service` extra and a versioned remote API reusing the same typed
  application services as CLI/UI, exposed through a distinct `yakbox serve`
  entry point that is never enabled by `yakbox ui`;
- a non-root, read-only-root-filesystem OCI image, pinned runtime dependencies,
  health/readiness probes, graceful shutdown, SBOM, signature/provenance, and
  documented resource requests/limits;
- one Helm chart with a schema-validated `values.yaml`, standard Kubernetes
  `Deployment`/`Service`/optional `Ingress`/PVC/Secret/RBAC/NetworkPolicy/Job`
  resources, TLS ingress, OIDC or another explicitly selected authentication
  mechanism, pod security settings, and no default `hostPath`;
- no Kubernetes operator, CRD, custom resource, admission webhook, or
  controller-runtime dependency. Installation, upgrade, and removal use Helm;
  the application may create ordinary `batch/v1` Jobs through narrowly scoped
  RBAC but does not install a custom reconciliation control plane;
- a single-coordinator topology first: exactly one API/coordinator replica owns
  the file-backed run journal and hosted-usage reservations on a persistent
  volume; the chart enforces this limitation rather than implying high
  availability;
- a lightweight API/coordinator/UI image separated from pinned CPU/GPU backend
  worker images so web/API pods never import model packages they do not need;
- isolated Kubernetes Jobs for authorized CPU/GPU local-model work, with
  node selectors/tolerations, explicit accelerator/memory budgets, heartbeat,
  cancellation, and signed/digest-verified artifact manifests returned to the
  coordinator;
- persistent-volume or explicitly designed object-storage artifact transport,
  backup/restore and upgrade procedures, retention/cleanup behavior, structured
  logs/metrics, and end-to-end budget enforcement across worker retries;
- the Phase 9 browser UI adapted to the authenticated remote API without
  exposing cluster paths, Kubernetes credentials, or provider/model secrets;
- remote API authorization for every workspace/target/action and stronger
  confirmations/policy for synthesis spending, cleanup, release, and purge.

Horizontal multi-coordinator scheduling, multi-user collaboration, and high
availability are a separate subphase. File locks on a shared volume are not
presented as distributed consensus. If measured requirements justify scale-out,
choose and specify a durable queue/coordination store and object-store contract
then. Any database or queue belongs to the optional service deployment and
must not leak into `yakbox`, `yakbox[local]`, or `yakbox[ui]`.

Gate:

- the built container and chart pass vulnerability, signature, policy,
  `helm lint`, rendered-manifest, install/upgrade/rollback/uninstall,
  backup/restore, network-isolation, authn/authz, secret redaction, and
  graceful-termination tests;
- installation creates no CRD or operator deployment, and the service account
  cannot mutate resources beyond its documented standard Job/workload scope;
- cluster interruption and worker duplication cannot publish an
  checksum-invalid artifact or exceed the centralized hosted-usage budget;
- GPU and hosted work respect declared resource/concurrency limits;
- the supported replica count and persistence guarantees are enforced and
  documented honestly;
- the normal local CLI, UI, and Python API conformance suites remain unchanged
  and require no cluster/service dependencies.

## 16. Definition of done

The target 1.0 release is complete only when all are proven below. Parked
phases 9 and 10 are intentionally excluded: they neither block 1.0 nor begin
automatically when this checklist passes.

- [ ] A clean audiobook manifest builds through normalize, plan, synthesize,
      master, inspect, assemble, and release-check with durable state,
      selective rebuilds, and provenance.
- [ ] Audiobook nodes and direct `tts` and `vc` interfaces use the same typed
      speech services, backend adapters, validation, errors, progress, and
      artifact contracts.
- [ ] Architecture and conformance tests prevent duplicated provider/model
      implementations or CLI-to-CLI orchestration.
- [ ] Existing cloud CLI flags and output behavior are preserved or a
      deliberate migration is documented and tested.
- [ ] The default install includes the complete lightweight audiobook planner,
      state/artifact model, direct speech interfaces, and hosted backends
      without installing PyTorch; `yakbox[local]` adds local Chatterbox.
- [ ] Existing local generation behavior remains compatible while its adapter
      is shared by direct and audiobook execution modes.
- [ ] Local Chatterbox and Resemble-hosted synthesis pass the shared
      capability/conformance suite.
- [ ] Remotely hosted Chatterbox is implemented only from a verified contract;
      if unavailable for 1.0, its typed capability slot and documented gap do
      not block other audiobook backends.
- [ ] All cloud HTTP uses HTTPX; `requests` is removed only if unused
      repository-wide.
- [ ] `yakbox` is the canonical, typed, documented, artifact-tested Python
      namespace; any required `yakbox_cli` shim is isolated, warning-backed,
      and covered by the compatibility policy.
- [ ] One shared async client is reused and always closed correctly.
- [ ] Retry classification, exponential backoff, shared `Retry-After` gate,
      exhaustion, and cancellation are deterministic and tested.
- [ ] Direct synthesis and `stream_to_file()` use atomic output files; the raw
      async stream API has explicit ownership and post-first-byte retry rules.
- [ ] Batch concurrency is bounded, actually overlaps work, preserves order,
      satisfies the memory contract, and isolates row-local failures.
- [ ] Systemic failures open the circuit breaker, stop new work, cancel safe
      in-flight work, and distinguish `error`, `skipped`, and `not_run`.
- [ ] Hosted character/request/estimated-spend guardrails are enforced through
      atomic reservations across concurrency and retries, survive resume, and
      never present an estimate as an invoice.
- [ ] Text limits are correct: direct 3,000; HTTP stream 2,000.
- [ ] `--dry-run` performs no provider request or output mutation.
- [ ] The durable NDJSON journal survives interruption, and `--resume`
      verifies schema, input, option, request, and output hashes before
      skipping any row.
- [ ] Batch reports and all `--json` success/error/usage documents conform to
      versioned, published JSON Schemas.
- [ ] Exit codes, JSON output, TTY progress, color, and interruption behavior
      satisfy the CLI contract.
- [ ] Positional text, `--text-file`, and stdin are mutually exclusive and
      work consistently across direct and streaming synthesis.
- [ ] API keys are redacted, deprecated on argv, and resolved through
      environment variables or optional OS-keyring-backed profiles without
      plaintext fallback.
- [ ] Canonical nested resource commands and boolean flag names are tested;
      compatibility aliases remain within their documented deprecation window.
- [ ] A schema-versioned audiobook manifest can plan and reproduce an
      audiobook-shaped workflow across logical voices, typed backend profiles,
      and explicit execution targets without serializing secrets.
- [ ] Markdown spoken/printed directives, explicit pauses, pronunciation
      lexicons, deterministic chapter splitting, and source-aware chunking are
      validated before synthesis.
- [ ] Audition, preview, production, processed, and release artifacts cannot
      overwrite or be mistaken for one another.
- [ ] Auditioning produces profile-named audio and typed comparison metadata,
      accepts a reproducible build-time profile override, and requires no
      account, database, review queue, or embedded playback.
- [ ] Local model scheduling respects model/thread/device budgets; provider
      scheduling respects async concurrency; neither execution strategy is
      imposed on the other.
- [ ] Audiobook resume uses complete content/profile/tool fingerprints and a
      schema-valid artifact graph rather than filename existence.
- [ ] File-backed build plans, append-only journals, atomic manifests, target
      locks, crash recovery, schema compatibility, and portable exports are
      tested on supported platforms.
- [ ] Plan and run reports automatically explain changes from the previous
      compatible successful run without adding a separate review workflow.
- [ ] Generated WAV, MP3, intermediate, audition, report, log, and cache files
      have typed inventory and retention records with accurate usage totals.
- [ ] Cleanup is preview-first, reference-aware, stale-plan-safe, and
      quarantine-based; it cannot remove source, unknown files, active work, or
      immutable releases, and supports verified restore and explicit purge.
- [ ] Storage estimation and optional workspace budgets detect clearly
      insufficient disk space before expensive synthesis.
- [ ] `yakbox doctor` provides typed offline checks plus explicit safe network
      and deep-local modes, never synthesizes audio, redacts secrets, and
      produces schema-valid human/JSON diagnostics with actionable remedies.
- [ ] Optional mastering and inspection validate technical audio properties,
      and assembly refuses incomplete, duplicate, stale, mixed-profile, or
      checksum-invalid chapter sets.
- [ ] The default release retains mastered WAV chapters and produces matching
      MP3 chapters from those masters with deterministic ordering, metadata,
      lineage, and digests; optional M4B does not replace either set.
- [ ] Stable shard manifests support hosted execution and verified
      recombination without hardcoded chapter numbers.
- [ ] Release mode rejects partial-work options and emits immutable manifests,
      checksums, backend/tool versions, watermark disclosures, and
      reference-voice provenance identifiers.
- [ ] Python 3.13 and 3.14 pass CI.
- [ ] Ruff, the documented blocking `ty` beta version, complexity, coverage,
      package, and security gates pass.
- [ ] Every Python function/method and module boundary in `src/` and tests is
      fully annotated; untyped external data is validated at the boundary and
      no unchecked `Any` escapes into domain or application services.
- [ ] Locked and minimum-direct dependency resolutions pass, and wheel/sdist
      installs are smoke-tested with default, credentials, and local extras.
- [ ] GitHub release publishes checksum-verified artifacts through Trusted
      Publishing, with a CycloneDX SBOM and provenance attestations.
- [ ] The operations guide documents reporting, support, and secret-handling
      policy.
- [ ] README, audiobook manifest/source/build guides, direct speech, backend,
      cloud/batch, Python API, and troubleshooting docs are current; top-level
      documentation leads with audiobook builds and gives every supported
      backend a clear installation and usage path.

## 17. Resolved design questions

### Batch metadata report

Yes. Write a versioned JSON report by default inside the batch output
directory, allow `--report` and `--no-report`, omit full text and cost, and
include a text hash plus documented provider timing. This supports automation
without claiming provider cost data that the API does not return. The durable
NDJSON journal is the authoritative incremental recovery record; the final
JSON report is a derived, ordered run summary.

### Durable execution and resume

Yes. Ship `--dry-run`, an append-only NDJSON journal, and hash-verified
`--resume` in 1.0. A resume may reuse only rows whose schema version, normalized
input identity, effective options, request fingerprint, output byte count, and
output digest still match. A malformed or incompatible journal fails safely
instead of guessing.

### Error scope

Classify failures as row-local or systemic. Row-local provider validation and
per-row I/O errors produce `error` while independent work continues. Auth,
global project/configuration, shared output/journal, and equivalent
run-invalidating failures open a circuit breaker; pending rows become
`not_run`, intentionally suppressed work becomes `skipped`, and the run is
`aborted`. `--ignore-errors` applies only to row-local failures.

### Hosted usage guardrails

Yes. Character and provider-request limits are enforceable without knowing a
provider's price and are core 1.0 safety features. Monetary limits are also
supported when a versioned pricing source is explicitly available. The shared
atomic reservation gate counts retries conservatively, persists usage for
resume, and stops before crossing a cap.

### Doctor diagnostics

Yes. `yakbox doctor` is a typed core command and Python API. It is offline and
read-only by default, requires explicit modes for safe network or deeper local
runtime checks, never synthesizes audio, and emits actionable schema-versioned
results without secrets.

### Default audiobook release artifacts

Retain mastered PCM WAV chapters as the lossless source of record and derive
delivery MP3 chapters from those masters by default. Protect both sets as
release artifacts. A chapter-marked M4B is an optional additional target, not a
default and not a replacement for WAV/MP3 chapters.

### Speech-to-text scope

Leave speech-to-text, transcription commands, and transcription-assisted QA out
of 1.0. Focus the product and implementation on text-to-speech, voice
transformation, and audiobook production. A future STT capability requires a
new explicit specification; do not ship placeholder protocols or dependencies.

### Phase 9 UI stack and editing

When explicitly activated, use typed FastAPI/Pydantic/Uvicorn on Python and
React with strict TypeScript, Vite, and Material UI in the browser. Support
current stable Chrome, Firefox, and Safari. Select the latest stable, mutually
compatible component releases under section 5.1 and freeze them in the
committed Python and frontend lockfiles. Use modular CodeMirror 6 npm packages
for Markdown editing and `tomlkit` for minimal
comment/order/whitespace-preserving TOML patches, validated again through the
canonical typed loader before atomic commit.

### Concurrency configuration

Yes. Store a cloud concurrency default in existing yakbox configuration when
the schema supports a compatible extension. Precedence is CLI, environment,
config, then 5. Keep it a batch concern; it does not affect local commands.

### Install profiles and namespace

Make the audiobook core, direct speech interfaces, and hosted backends the
lightweight default; move local model dependencies behind `yakbox[local]`, with
lazy imports and independent CI. Use `yakbox` as the canonical Python
namespace. Retain `yakbox_cli` only as a narrow compatibility shim if phase 0
proves that published consumers need it.

### Credentials

Keep the existing API-key option temporarily for compatibility, but deprecate
passing secrets on the command line. Resolve credentials from environment
variables or named profiles stored through the optional `keyring` extra. Never
silently fall back to a plaintext token store.

### Long text

Do not silently chunk in the low-level `cloud tts`, `cloud stream`, or
`cloud batch` commands because chunk boundaries, naming, ordering, audio
joining, and SSML validity change the meaning of a simple provider request.
Reject over-limit input there and provide correct guidance.

The explicit audiobook workflow does support long-form chunking because its plan
records the normalized speech document, chunker/version, source boundaries,
pauses, per-chunk hashes, ordered artifacts, and assembly policy before
rendering. This keeps audiobook production first-class without making a
single-request command perform hidden multi-request work.

### Retry library

Use an internal policy initially. Streaming cleanup, response closure,
provider status classification, and deterministic injected timing are clearer
as first-party domain code than as a general decorator. Reconsider only if the
implementation becomes materially more complex than this contract.

## 18. Future-compatible feature path

The architecture should permit, without promising in 1.0:

- advanced SSML-aware and language-aware chunking beyond the deterministic
  plain-text/Markdown audiobook chunker;
- provider cost metadata if officially returned;
- additional hosted synthesizers beyond core Chatterbox and Resemble
  capabilities, using a narrow protocol where semantics align;
- a separately specified speech-to-text/transcription capability and optional
  source-aligned QA stage if explicitly requested after the TTS/audiobook core
  is complete; do not pre-build its command, protocol, models, or dependencies;
- async pagination iterators;
- structured logging and OpenTelemetry hooks;
- webhook-backed asynchronous clip jobs;
- richer SSML validation and pronunciation management.

New features must extend typed models and application services; they must not
put provider dictionaries, Click contexts, or terminal rendering into the
batch core.

## 19. Later possibilities to evaluate

These are design candidates, not 1.0 commitments, implementation requirements,
or implied release gates. Promote one into the main specification only after a
short architecture decision record identifies its user need, provider
constraints, compatibility impact, security implications, and acceptance
tests.

1. **Local watch mode.** Consider an opt-in foreground
   `yakbox build --watch` loop that debounces source/manifest changes,
   replans, and rebuilds only invalidated local targets. It must remain an
   ordinary interruptible process, not install a daemon or background service.
2. **Adopting existing narration.** Consider importing already-recorded chapter
   audio into the artifact DAG after format, ordering, provenance, and digest
   validation, allowing hybrid human/TTS audiobooks without pretending the
   files were synthesized by yakbox.
3. **Blind audition review bundles.** Consider generating a static local
   HTML/JSON comparison bundle that randomizes profile labels, supports
   time-aligned A/B listening and notes, and reveals settings only after a
   reviewer records a preference. Do not build it until repeated
   multi-reviewer auditions justify more than profile-named audio files.
4. **Versioned configuration migrations.** Consider schema-versioned,
   transactional config upgrades with a backup and automatic rollback when a
   migration fails.
5. **A formal batch state machine.** Consider specifying every allowed
   row/run/journal transition and exercising it with stateful property tests
   across cancellation, process death, corruption, and resume.
6. **Compatibility-diff CI.** Consider an approval gate that detects changes
   to public Python exports, Click commands/help, flags, exit codes, and JSON
   Schemas before release.
7. **Remote archive and storage tiering.** Consider verified transfer to
   S3-compatible or cold storage, cross-workspace content deduplication,
   retention/legal-hold integration, and restore-on-demand. Keep the core 1.0
   cleanup system local and do not pursue this without demonstrated storage
   pressure. Evaluate it only within the explicit Phase 10 storage decision.
   Do not mistake quarantine for secure deletion.
8. **A distributed worker protocol.** Consider turning stable audiobook shards
   into authenticated leaseable jobs for self-hosted machines, with heartbeat,
   cancellation, artifact upload, and duplicate-execution handling. Do not
   infer this merely from the manifest format or add it without a concrete
   workload that cannot be served by local workers and hosted APIs. This is the
   scale-out subphase of Phase 10, not part of the local UI.
9. **Third-party extension governance.** If providers beyond the core
    Chatterbox and Resemble paths are opened to plugins, first consider
    protocol versioning, compatibility negotiation, trust boundaries, secret
    access, failure isolation, and support policy.

## 20. Primary references

Reviewed through 2026-07-29:

- Local design case study (not a runtime dependency):
  `/Users/pbsladek/Code/pbsladek/shorts/scifi/anima-cara`
- [Python active releases](https://www.python.org/downloads/)
- [uv project and packaging guide](https://docs.astral.sh/uv/guides/package/)
- [uv build documentation](https://docs.astral.sh/uv/concepts/projects/build/)
- [uv locking, upgrades, audits, and SBOM export](https://docs.astral.sh/uv/concepts/projects/sync/)
- [Astral `uv_build` backend](https://docs.astral.sh/uv/configuration/build-backend/)
- [Using uv in GitHub Actions](https://docs.astral.sh/uv/guides/integration/github/)
- [Ruff formatter](https://docs.astral.sh/ruff/formatter/)
- [Astral `ty` type checker](https://docs.astral.sh/ty/)
- [`ty` configuration reference](https://docs.astral.sh/ty/reference/configuration/)
- [markdown-it-py on PyPI](https://pypi.org/project/markdown-it-py/)
- [markdown-it-py token parsing](https://markdown-it-py.readthedocs.io/en/latest/using.html)
- [keyring on PyPI](https://pypi.org/project/keyring/)
- [pytest-socket on PyPI](https://pypi.org/project/pytest-socket/)
- [Dependabot-supported package ecosystems](https://docs.github.com/en/code-security/reference/supply-chain-security/supported-ecosystems-and-repositories)
- [npm package provenance verification](https://docs.npmjs.com/viewing-package-provenance/)
- [CodeMirror](https://codemirror.net/)
- [CodeMirror repository migration](https://discuss.codemirror.net/t/codemirrors-migration-to-forgejo/9706)
- [GitHub Actions Python build/test guide](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
- [PyPA Trusted Publishing workflow guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [Resemble direct synthesis](https://docs.resemble.ai/api-reference/text-to-speech/synthesize)
- [Resemble HTTP stream synthesis](https://docs.resemble.ai/api-reference/text-to-speech/stream-synthesize)
- [Resemble voices](https://docs.resemble.ai/voice-creation/voices/list)
- [Resemble recording creation](https://docs.resemble.ai/voice-creation/recordings/create)
- [Resemble projects](https://docs.resemble.ai/platform-management/projects)
