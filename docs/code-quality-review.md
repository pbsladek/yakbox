# Code quality review

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

Yakbox now supports Python 3.13 and 3.14. Package metadata, Ruff, `ty`, the
lockfile, documentation, and CI agree on that range. The project and workflows
also use `uv 0.12.x`, so a standard local installation no longer fails the
repository's own uv-version check.

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
codes are reviewed in `tests/public-api-v1.json`. CI installs the built wheel
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
