# Code quality standard

Yakbox favors code that is easy to change safely over code that is merely
compact. A change is ready when its behavior is covered, its failure modes are
explicit, and the standard checks pass without broad exemptions.

## Required checks

Run these commands from the repository root:

```console
uv sync --frozen --all-groups
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run lint-imports --no-cache
uv run pytest --cov=yakbox --cov-branch
uv lock --check
uv build --no-sources
```

CI runs the same checks on Python 3.13 and 3.14 across Linux, macOS, and
Windows. Python 3.13 is the minimum supported runtime.

## Architecture contracts

Import Linter checks repository-level dependency boundaries on every pull
request and release. The contracts in `pyproject.toml` enforce that:

- application modules do not import the CLI entrypoint (`__main__` is the
  intentional exception);
- speech models, capabilities, and guardrails do not depend on concrete hosted
  or local backends;
- audio processing does not depend on audiobook orchestration, diagnostics,
  speech, or concrete backends;
- hosted and local backends do not directly depend on each other; and
- imports within the audio, audiobook, cloud, diagnostics, and speech packages
  remain acyclic.

Run `uv run lint-imports --no-cache` after moving responsibilities or changing
imports. Update a contract only when the intended architecture changes; do not
add an ignored import merely to make a dependency violation pass.

## Complexity and function design

Ruff enforces these limits on production and test code:

- cyclomatic complexity: at most 10 per function (`C901`)
- branches: at most 12 per function (`PLR0912`)
- statements: at most 50 per function (`PLR0915`)
- return statements: at most 6 per function (`PLR0911`)
- positional parameters: at most 5 (`PLR0917`)
- total parameters: at most 20 (`PLR0913`); only framework-injected CLI
  callbacks may exceed this

Treat the limits as design feedback, not targets to work around. Split parsing,
validation, orchestration, side effects, and serialization into named phases.
Use a typed options or context dataclass when a workflow carries many related
values. CLI callbacks should translate framework inputs and delegate; they
should not contain the application workflow.

Do not suppress `C901`, `PLR0912`, or `PLR0915`. Production-wide and file-wide
lint exemptions are not allowed. A line-level exemption is acceptable only
when an external framework dictates the shape of the code or a boundary must
deliberately catch or load something dynamically. Every exemption must name the
rule and include a short reason on the same line.

## Types and contracts

All functions must have complete parameter and return annotations. Keep
`object` at untrusted decoding boundaries and narrow it through validation.
Avoid `Any`; when a third-party override requires it, document that exception
at the annotation.

Use typed dataclasses, enums, and domain exceptions for values that cross
module boundaries. Machine-readable output is a public API: update its JSON
Schema, compatibility tests, examples, and documentation in the same change.
Breaking fields require a new schema version.

### Interface contract metrics

Interface coverage is enforced separately from statement coverage:

- the complete visible CLI leaf-command set is reviewed as a snapshot;
- every visible command and option has help, and every meaningful default is
  shown;
- `cli.py` is capped at 3,000 lines while command presentation lives in
  `cli_help.py`; new command families should move into cohesive modules rather
  than raising the cap;
- the public SDK's exports, signatures, enum members, and exception codes match
  `tests/public-api-v1.json`;
- every public callable and public method has a docstring, and every export is
  named in `docs/python-api.md`;
- the built wheel is installed into an isolated virtual environment, then
  imported and type-checked through `tests/package_sdk_consumer.py` without
  resolving the source tree;
- schema tests reject invalid envelopes instead of validating only
  producer-generated examples; and
- credential tests cover explicit, environment, keyring, legacy-config, and
  missing values in precedence order.

Stable machine error codes use lowercase snake case and do not depend on Python
class names. CLI JSON includes a `command` discriminator, and its declared exit
code must match the process exit code. Unexpected exceptions return a bounded
`internal_error` without implementation detail.

## Errors and boundaries

Catch the narrowest exception that can be handled. Ruff's `BLE` rules reject
blind exception handling. Catching `Exception` is reserved for isolation
boundaries such as worker protocols, callbacks, task supervisors, and
third-party plugin APIs. Those catches must either re-raise, preserve durable
failure state, or translate the error into a bounded and redacted domain
result.

Never log credentials, provider tokens, full manuscript text, or unbounded
third-party error bodies. Preserve cancellation and do not convert programmer
errors into successful or ambiguous results.

## Filesystem, subprocess, and security

Use `pathlib.Path` and the safe primitives in `yakbox._files`. Managed paths
must remain below their declared root. Durable output must use a sibling
temporary file and an atomic commit. Destructive operations need a plan,
managed-root validation, and rollback or quarantine behavior.

Subprocesses must use argument vectors with `shell=False`, bounded timeouts,
captured output, and checked or explicitly interpreted exit codes. Executable
names and path inputs must come from fixed commands or validated values. Ruff's
security rules are part of the required lint gate.

## Tests and coverage

The default suite is deterministic and offline. Provider calls use mocks, audio
builds prefer the fake backend, and socket access remains limited to loopback.
Live and performance suites require explicit markers and opt-in configuration.

Branch-aware project coverage must stay at or above 75%. This is a floor, not a
goal. New behavior and bug fixes need tests for the success path, expected
failure path, and important branches. Changed safety-critical code—artifact
commits, resume validation, cleanup, budgets, schemas, and credential
redaction—should have direct branch coverage rather than relying on an
end-to-end test to pass through it incidentally.

Tests should assert observable contracts. Avoid tests coupled to private call
order unless ordering is itself part of the behavior. Use temporary workspaces,
fixed clocks or injected randomness where needed, and bounded asynchronous
operations.

### Mutation testing

The weekly Linux assurance workflow runs Mutmut over the covered lines in the
safety-critical filesystem, runtime-contract, artifact, journal, and hosted
guardrail modules listed under `[tool.mutmut]`. It requires at least an 80%
mutation score, rejects mutants with no associated tests, and uploads the JSON
statistics for review. The initial baseline is 261 killed of 315 total mutants
(82.86%), with no untested, timed-out, suspicious, interrupted, or crashing
mutants.

Run the same scope locally on macOS or Linux with:

```console
uv run mutmut run --max-children 1
uv run mutmut results
uv run mutmut browse
```

The single child avoids pytest temporary-directory cleanup races on macOS;
increment it only after verifying the host. Mutmut requires process-fork
support, so Windows development uses WSL. Treat surviving mutants as
test-design prompts: add an observable assertion when the mutation changes
behavior, and document equivalent mutants during review. Add new
safety-critical modules to `only_mutate` as their tests become suitable, then
ratchet the score upward; do not shrink the scope to preserve the score.

## Dependencies and review

Declare dependencies in `pyproject.toml` and let `uv` update `uv.lock`. Keep
optional model dependencies lazy so importing Yakbox does not load or require a
model runtime. Dependency audits, package builds, and isolated wheel/sdist smoke
tests are release gates.

Review changes in this order:

1. Correctness, data loss, security, and compatibility risks.
2. Error handling, cancellation, resource ownership, and deterministic output.
3. Complexity, typing, duplication, naming, and module boundaries.
4. Tests, documentation, and operational diagnostics.

The author must report which required checks ran and explain any check that
could not run. Generated audio, build output, secrets, and unrelated formatting
must not appear in the change.
