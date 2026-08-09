# AGENTS.md

This file applies to the entire repository. Keep changes focused, preserve
existing behavior unless the task calls for a behavior change, and update tests
and documentation with the code they describe.

## Project overview

Yakbox is a local-first, reproducible audiobook build system for Python 3.14.
The package uses a `src` layout and is managed with `uv`.

- `src/yakbox/`: application and public Python package
- `src/yakbox/audiobook/`: manifests, planning, builds, artifacts, and cleanup
- `src/yakbox/audio/`: FFmpeg/FFprobe-backed processing
- `src/yakbox/cloud/`: hosted-provider clients and batch execution
- `src/yakbox/speech/`: backend-neutral speech contracts and workers
- `src/yakbox/schemas/`: packaged JSON Schemas for machine-readable contracts
- `tests/unit/`: fast component and contract tests
- `tests/integration/`: CLI and end-to-end tests using controlled fixtures
- `tests/live/`: explicitly opted-in provider or real-model tests
- `tests/performance/`: scheduled performance baselines
- `docs/`: user documentation and the architecture plan
- `examples/`: runnable audiobook projects

The current code, tests, and user documentation describe shipped behavior.
`docs/AUDIOBOOK_BUILD_SYSTEM_PLAN.md` also contains later-phase plans; do not
treat an unimplemented roadmap item as current functionality.
The enforceable engineering rules live in `docs/code-quality.md`.

## Development workflow

Install the locked development environment with:

```console
uv sync --frozen --all-groups
```

Run commands through `uv run`. Prefer a targeted test while iterating, then run
the relevant broader checks before handing off:

```console
uv run pytest tests/unit/test_manifest.py
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run lint-imports --no-cache
uv lock --check
```

Use `uv run ruff format .` to format code. The full packaging check is:

```console
uv build --no-sources
```

FFmpeg and FFprobe are required for audio mastering, encoding, inspection, and
assembly tests. If they are unavailable, report that limitation rather than
weakening the related tests.

## Test boundaries

The default pytest configuration excludes `live` and `performance` tests and
only permits loopback networking. Tests should remain deterministic and offline
unless they are explicitly part of one of those suites.

- Put provider calls behind mocks (RESPX where appropriate) in normal tests.
- Do not enable arbitrary sockets to make a default test pass.
- Run performance tests explicitly with
  `uv run pytest -m performance tests/performance`.
- Run live tests only when the task explicitly requires them and the documented
  credentials, consent, budget, and opt-in environment variables are present.
- Use `tests/live/test_local_chatterbox_e2e.py` for real local audiobook QA;
  preserve its generated technical report and complete and validate the bound
  `qa/listening-review.toml` artifact before claiming narration quality.
- The optional local Chatterbox dependency set is installed with
  `uv sync --frozen --all-groups --extra local`; it is large and is not needed
  for the default fake-backend suite.

Add regression coverage for bug fixes. For behavior spanning the CLI and a
service, test the service contract and the observable CLI result. Use temporary
workspaces and the fake backend for audiobook build tests whenever possible.

## Coding conventions

- Follow the Ruff and `ty` configuration in `pyproject.toml`; the line length is
  88 and the target language level is Python 3.14.
- Do not add file-wide production lint exemptions or suppress complexity,
  branch-count, or statement-count rules. Any line-level exemption must name
  the rule and state why the boundary requires it.
- Keep public boundaries typed. Prefer small dataclasses and explicit domain
  exceptions derived from `YakboxError` over unstructured dictionaries and
  generic exceptions.
- Use `pathlib.Path` for filesystem paths and preserve cross-platform behavior
  across Linux, macOS, and Windows.
- Keep backend-specific behavior behind the speech or cloud abstractions. Avoid
  importing heavy optional dependencies at module import time.
- Preserve deterministic ordering, stable fingerprints, and reproducible
  output. Do not include secrets, credentials, full input text, or provider
  tokens in logs, journals, reports, or exception messages.
- Reuse the safe filesystem helpers in `src/yakbox/_files.py`. Generated output
  must stay inside its managed root, and durable artifacts should be written
  atomically. Never replace those safeguards with direct destructive writes.

## Contracts, schemas, and CLI output

Machine-readable documents are public contracts. They carry an absolute
`$schema`, `schema_version`, Yakbox version, and UTC timestamp where applicable.
When changing one:

1. Update the implementation and the matching schema under
   `src/yakbox/schemas/`.
2. Update schema, unit, and CLI tests as applicable.
3. Update examples and documentation that show the affected fields.
4. Preserve compatibility for additive optional changes. Renaming, removing,
   changing the meaning or type of a field, or making an optional field required
   needs a new schema version and migration consideration.

Human-readable CLI wording may evolve, but automation must be able to rely on
structured output. Keep JSON output free of Rich formatting and incidental
diagnostics, use domain exceptions for expected failures, and preserve
documented exit behavior.

## Dependency and documentation changes

Edit dependencies in `pyproject.toml` and refresh `uv.lock` with `uv`; do not
hand-edit the lockfile. Preserve the security overrides and Python-version
markers for the `local` extra unless the change explicitly updates and verifies
that dependency contract.

Follow `docs/licensing.md` for all new dependencies and third-party material.
Do not add an asset without its source URL, rights basis, and checksum. Reject
unknown, noncommercial, and no-derivatives assets. Prefer MIT, Apache-2.0, BSD,
or ISC software; document and review any exception before adding it.

Keep `README.md` concise and route detailed explanations into `docs/`. When a
CLI flag, manifest field, artifact layout, backend requirement, or operational
safety rule changes, update the relevant documentation and runnable example in
the same change.

Before finishing, inspect the diff for accidental generated artifacts,
credentials, audio files, build outputs, or unrelated formatting churn, and
state which checks were run and any checks that could not be run. Versioned
reference audio explicitly approved under the licensing policy is not an
accidental generated artifact.
