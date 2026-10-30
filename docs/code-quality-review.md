# Code quality review

> Historical document: hosted Resemble support has since been removed.
> Current behavior is local-only; see [migration](migration.md#local-models-only).

Review date: 2026-07-31

This review covers the Python package, tests, CI workflows, dependency
metadata, and developer documentation. It uses the repository's locked Ruff,
`ty`, pytest, and coverage configuration as the executable source of truth.
Import Linter and Mutmut additionally measure dependency boundaries and test
effectiveness.

## Executive result

The project has strong functional and safety foundations: typed domain models,
schema-tested machine output, atomic filesystem operations, offline-by-default
tests, and multi-platform CI. The main weakness was that complexity rules were
configured but disabled across the files that needed them most.

The review found seven production functions above the McCabe limit of 10. The
largest had complexity 29 and 96 statements. File-wide exemptions hid those
findings, along with excessive branch, statement, return, and parameter counts.
The configured lint command therefore passed without enforcing its stated
design limit.

Those exemptions have been removed. All production functions now pass the
configured complexity, branch, statement, return, total-parameter, and
positional-parameter limits. The quality configuration itself is covered by a
test so these gates cannot be silently weakened by adding a production
file-wide exemption.

## Findings and changes

### Complexity and responsibility

The cloud batch workflow mixed run setup, resume recovery, queue management,
journaling, usage accounting, interruption recovery, and reporting in one
function. It now uses typed settings and run-state objects with separate phases
for each responsibility. Resume handling separately validates identity,
indexes attempts, verifies artifacts, restores completed rows, and reconstructs
usage.

Manifest loading now separates file I/O, contract validation, book parsing,
source options, path resolution, and cross-reference checks. Markdown source
normalization separately extracts source events and builds typed chapter
items. The build CLI callback now translates Click inputs into a typed options
object and delegates selection, preflight, progress, execution, and output.

The resulting Ruff complexity audit reports no `C901`, `PLR0911`, `PLR0912`,
`PLR0913`, `PLR0915`, or unsuppressed `PLR0917` findings.

### Lint policy

The lint gate now also checks complete annotations, unused arguments, blind
exception handling, return structure, common security mistakes, and stray
`print` calls. Production file-wide exemptions are gone. Remaining line-level
exemptions are limited to:

- Click callback signatures defined by the framework;
- lazy imports that keep optional model backends optional;
- callback, worker-protocol, task, and keyring isolation boundaries;
- fixed, argument-vector subprocess calls with `shell=False`;
- third-party override annotations that require `Any`.

Every exemption names its rule and explains the boundary. Complexity, branch,
statement, and return-count rules cannot be exempted under the repository
standard.

### Exception handling

Configuration and manifest diagnostics previously caught every exception. They
now catch the domain errors they can report meaningfully. The doctor command
also catches only expected application and operating-system failures.

Seventeen broad catches remain. Nine re-raise after rollback, durable journal
recovery, or contextual error translation. Eight deliberately isolate an
external callback, queue job, worker request, or keyring operation and carry a
documented `BLE001` exemption. This distinction is enforced by Ruff.

### Tests and coverage

The default suite remains deterministic and offline. At the end of the review,
214 tests pass and four live/performance tests remain explicitly deselected.
Branch-aware coverage is 79.10%, above the enforced floor of 75%.

The lowest-covered production areas are the CLI adapters and optional backend
process paths. Their lower coverage reflects platform, credential, and model
boundaries, but they remain the best targets for the next coverage increase.
The floor should move upward as direct tests are added; it must never be lowered
to accommodate a change.

### Runtime and toolchain

Yakbox supports Python 3.14 only. Package metadata, Ruff, `ty`, the lockfile,
documentation, and CI agree on that runtime. The project and workflows
accept `uv>=0.13.0` without an upper version ceiling, and workflows install
the latest stable uv. A current local installation can run the same gates as CI.

### CLI and SDK contracts

The visible CLI now has a reviewed 42-command snapshot. Every command and
option has semantic help, meaningful defaults are visible, and common option
families live in shared decorators. JSON envelopes identify the command,
publish an exit code that matches the process, reject structurally invalid
status combinations, and use stable error codes. Unexpected failures no longer
expose exception details.

Credential resolution is centralized and tested across explicit arguments,
environment variables, keyring profiles, legacy configuration, and missing
credentials. Hosted work cannot enter through the generic local batch command.
Legacy credential flags and plaintext configuration remain compatible but emit
source-aware migration warnings.

The Python SDK now has explicit facade exports, typed request objects, shared
audio enums, validated hosted-pricing identifiers, stable exception codes, and
semantic public docstrings. Its exports, signatures, enum members, and error
codes are reviewed in `tests/public-api-v1.yaml`. CI installs the built wheel
into an isolated environment, imports every public export, and type-checks a
representative consumer without resolving the source tree.

## Residual design risks

`audiobook/build.py` and `cli.py` are still large modules. Their functions meet
the enforced limits, and CLI help plus shared option families have moved into
dedicated modules. Module-size growth is now guarded: `cli.py` may not exceed
3,000 lines, and new command families should be introduced in cohesive modules
instead of raising that cap. Release management remains the strongest
extraction candidate in `audiobook/build.py`.

The 75% coverage threshold is an initial ratchet. A sensible next milestone is
80%, reached through direct tests of CLI error paths, optional backend lifecycle
handling, cloud journal failures, and release preflight behavior—not by omitting
hard-to-test code from coverage.

### Architecture and mutation testing

Five Import Linter contracts now keep the CLI, neutral speech domain, audio
processing, concrete backends, and subpackage dependency cycles within their
documented boundaries. All five contracts pass across the current 49-module,
162-dependency import graph.

The first Mutmut baseline covers five safety-critical modules and generated
315 covered mutants. Tests killed 261 for an 82.86% mutation score; 54 survived,
and none lacked tests, timed out, appeared suspicious, were interrupted, or
crashed. Scheduled CI enforces an 80% floor and retains the machine-readable
statistics. The survivors are a prioritized queue for strengthening observable
filesystem and artifact assertions.

Live providers, local model execution, and scheduled performance baselines are
outside the default suite by design. Changes to those paths still require their
documented opt-in canaries before release.

## Performance and correctness follow-up: 2026-10-10

This follow-up reviews filesystem commits, artifact and cache cleanup, hosted
request queues, journals, runtime lifecycle, worker protocols, text chunking,
and audio processing. It uses controlled offline reproductions and local
benchmarks. It does not establish real-model narration quality or hosted
provider compatibility.

### Fixed findings

- **P1: byte writes could overwrite a competing writer.**
  `atomic_write_bytes(overwrite=False)` checked existence and then replaced the
  destination. A file created between those operations was overwritten. Byte
  writes now share the exclusive commit primitive used by media output and
  sync the parent directory after publication. Empty byte files remain valid.
- **P1: synthesis cache cleanup could delete files outside the workspace.**
  A cache-root symlink made the external directory its own containment boundary.
  Cache discovery now validates the root against the workspace and validates
  each entry against the cache root. Cleanup also revalidates every candidate
  before deleting any file, rejects regenerated or newly pinned entries, and
  retains pinned entries even when their cache metadata is corrupt.
- **P1: cancelling a journal append stopped the writer.**
  Cancellation cancelled the acknowledgement future; completing that future
  raised `InvalidStateError`. The writer now checks acknowledgement state.
  Separately, disk failure could drain the queue while a blocked producer
  resumed into a queue with no consumer. Failed writers now shut down the
  queue and wake those producers. Regression tests fill the real 100-record
  queue and exercise cancellation during a durable append.
- **P1: cancelling a hosted batch could leave provider work running.**
  Queue filling happened outside the cancellation handler. Cancellation while
  waiting for capacity therefore bypassed dispatcher cleanup. Enqueueing is
  now covered by that handler. Closing the service shuts down its queue,
  cancels active workers, and rejects later single or batch operations with a
  domain error.
- **P2: the runtime idle timer ignored later activity.**
  The original timer expired even if an authenticated request had updated
  `last_activity`. The server now recalculates the remaining idle period and
  keeps an active operation alive. Nonfinite timeout values are rejected.
- **P2: chunking copied the remaining input on every iteration.**
  Chunking now advances an offset through one string and copies bounded chunk
  windows. A comparison against the previous implementation passed 300 cases
  covering semantic boundaries and Unicode graphemes. The default suite adds
  repeated grapheme-boundary tests; the performance suite adds an 8-million-
  character baseline.

The new failure reproductions failed against the previous implementations and
passed after the corresponding fixes. The package-owned worker archive was
rebuilt because it includes the changed filesystem helper.

### Chunking measurements

The following timings use CPython 3.14.4 on the local Apple Silicon host,
500-character chunks, and repeated `word ` input. They are local samples,
separate from the performance suite's traced-memory limits.

| Characters | Chunks | Before | After |
| ---: | ---: | ---: | ---: |
| 1,000,000 | 2,000 | 0.050 s | 0.028 s |
| 4,000,000 | 8,000 | 0.471 s | 0.109 s |
| 8,000,000 | 16,000 | 1.281 s | 0.222 s |

The largest input improved by approximately 5.8 times. These measurements
cover planning work; they do not measure TTS or ASR inference.

### Open findings

**P1: persistent Chatterbox inference blocks its control plane.**
`LocalChatterboxService.synthesize_to_file()` invokes synchronous inference
inside its async method. The persistent runtime awaits that method on the
same event loop that serves status and stop requests. A controlled 250 ms
inference stub delayed a 10 ms heartbeat to 260 ms. Real inference exceeding
the five-second connection timeout can therefore make a running runtime
appear unhealthy. `ensure_local_runtime()` can then attempt another startup.
Move inference behind a responsive process supervisor and test health,
shutdown, and duplicate-start prevention with a deliberately slow worker.
Preserve the documented rule against sharing a PyTorch model across threads.

**P1 for loader upgrades: dependency pins and qualification metadata drift.**
[Dependabot PR #20](https://github.com/pbsladek/yakbox/pull/20) changes the
Parakeet worker pin to 0.5.3 and the Qwen worker pin to 0.5.8. The matching
`BUILT_IN_WORKERS` definitions and model registry still identify 0.5.2 and
0.4.8. An offline status reproduction with the upgraded installed versions
reports `backend_package_unqualified` for both families. A successful worker
handshake does not exercise model loading. Before releasing those upgraded
loaders, update their reviewed provenance and worker definitions together and
run the documented model qualification. Keep the historical qualification
record distinct from evidence for a new package version.

**P2: repair splicing expands complete audio into Python integers.**
`audio.splice._read_wave()` decodes the entire source into a tuple, and
crossfades repeatedly copy accumulated output. A 30-second mono 24 kHz PCM16
fixture occupied 1,440,044 bytes on disk and peaked at 36,173,729 bytes of
Python heap during decoding alone, approximately 25 times the file size.
Long chapters and multiple replacements increase both memory and copy work.
Stream unchanged prefixes and suffixes and decode only replacement windows
plus bounded context, using compact sample buffers. Preserve sample format,
frame coordinates, and the current splice evidence in comparison tests.

### Verification

The default offline suite passed 788 tests with 79.00% branch coverage; 20 live
and performance tests were deselected. All three performance baselines passed
when run explicitly. Ruff formatting and lint, `ty`, all eight import contracts,
workflow validation with Actionlint, and all four lockfile checks passed.

A clean temporary Git snapshot built the wheel and source distribution using
the release helper. Both passed strict Twine metadata checks. Isolated wheel
consumers exercised the public SDK and packaged worker. Isolated wheel and
source installs both printed the snapshot commit with `yakbox --version`, and
the builder restored the source metadata afterward. These checks used macOS
and CPython 3.14.4; Linux CI, real-model qualification, and live-provider tests
were not run. No release or remote repository change was made.
