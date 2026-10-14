# Installing and releasing yakbox

## Install the CLI

Yakbox supports macOS and Linux with Python 3.14. Windows is not supported.
The default installation is small and does not install PyTorch. Install the
published v0.2.0 wheel directly:

```console
uv tool install --python 3.14 \
  https://github.com/pbsladek/yakbox/releases/download/v0.2.0/yakbox-0.2.0-py3-none-any.whl
yakbox --version
yakbox doctor
```

To upgrade, use `uv tool install --force --python 3.14` with the new release's
wheel URL. Direct URL installs do not require a Yakbox project on PyPI;
dependencies still come from the configured package index.

For library use inside a Python 3.14 environment, pip also accepts the wheel URL:

```console
python -m pip install \
  https://github.com/pbsladek/yakbox/releases/download/v0.2.0/yakbox-0.2.0-py3-none-any.whl
```

Install optional extras with a named URL requirement:

```console
uv tool install --python 3.14 \
  "yakbox[credentials] @ https://github.com/pbsladek/yakbox/releases/download/v0.2.0/yakbox-0.2.0-py3-none-any.whl"
uv tool install --python 3.14 \
  "yakbox[local] @ https://github.com/pbsladek/yakbox/releases/download/v0.2.0/yakbox-0.2.0-py3-none-any.whl" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.2.0/constraints/chatterbox-security-overrides.txt
uv tool install --python 3.14 \
  "yakbox[credentials,local] @ https://github.com/pbsladek/yakbox/releases/download/v0.2.0/yakbox-0.2.0-py3-none-any.whl" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.2.0/constraints/chatterbox-security-overrides.txt
uv tool install --python 3.14 \
  "yakbox[local,alignment] @ https://github.com/pbsladek/yakbox/releases/download/v0.2.0/yakbox-0.2.0-py3-none-any.whl" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.2.0/constraints/chatterbox-security-overrides.txt
```

Local Chatterbox has a much larger machine-learning dependency graph. Install
it only on a machine that will run the model. The versioned override file
replaces vulnerable exact pins in Chatterbox 0.1.7 and requires patched urllib3
for model downloads; do not omit it. FFmpeg and
FFprobe are separate system tools required for mastering, MP3 encoding,
inspection, and M4B assembly.

The `alignment` extra installs `mlx-whisper` only on Apple Silicon macOS. It is
needed only for `short_utterances.strategy = "context_extract"`; direct
synthesis and the default CLI do not import or install it. The configured model
revision is resolved into the local Hugging Face cache before transcription.

## Verify release checksums

Each GitHub Release contains a platform-independent wheel, a source archive,
`SHA256SUMS`, four CycloneDX SBOMs, and a preflight report. Download every asset
so all entries in the checksum manifest can be verified:

```console
gh release download v0.2.0 --repo pbsladek/yakbox --dir yakbox-v0.2.0
cd yakbox-v0.2.0
shasum -a 256 -c SHA256SUMS
uv tool install --python 3.14 ./yakbox-0.2.0-py3-none-any.whl
```

On Linux, use `sha256sum -c SHA256SUMS` for the checksum step. GitHub also
produces build-provenance and SBOM attestations. Verify a downloaded wheel with:

```console
gh attestation verify yakbox-0.2.0-py3-none-any.whl --repo pbsladek/yakbox
```

## Release with one command

After these workflows have been merged to `main`, authenticate GitHub CLI with
`gh auth login` and run:

```console
make release
```

The command waits through preparation, CI, merge, and publication. It uses the
remote `main` branch; commit and merge application changes before releasing.
Local uncommitted or unpushed work is not included.

Google's [Release Please](https://github.com/googleapis/release-please-action)
maintains a release PR from Conventional Commits. Use PR titles such as
`fix: preserve cached chunks`, `feat: add an output format`, or
`perf: reduce chunking copies` when squash-merging application changes.
Release Please chooses the next version and writes the changelog,
`pyproject.toml`, source version fallback, and release manifest. Preparation
then refreshes `uv.lock` with uv and rebuilds the packaged worker. None of
these files needs a manual version edit for routine releases.

`make release` does the following:

1. Runs the pinned Release Please action to create or update the release PR.
2. Waits for CI on the exact prepared PR commit.
3. Squash-merges that commit using the caller's GitHub credentials. Required
   reviews, branch rules, and merge queues still apply; it never uses `--admin`.
4. Waits for the existing release workflow to build, audit, test, attest, and
   publish the merged commit to GitHub.

The workflows also maintain the PR on pushes to `main`. Merging that PR through
GitHub starts publication automatically. The built-in `GITHUB_TOKEN` does not
start PR-triggered CI for bot-created changes, so preparation explicitly
dispatches `ci.yml` on the prepared release branch. The manifest merge likewise
explicitly dispatches publication. No additional PAT or publishing secret is
needed for GitHub releases.

Repository Settings > Actions > General must permit GitHub Actions to create
pull requests. Workflow permissions remain read-only by default; each job
requests only its required permissions. Release Please's initial manifest
records the already published 0.2.1 release. Adding that manifest does not
republish the historical version.

If there are no releasable Conventional Commits, the command reports that no
release PR is available. CI failure or a changed PR head stops the command
before merging. A publication failure leaves the merged version pending;
running `make release` again retries that exact version and commit rather than
bumping another version. Existing tags must always point to the release commit.

`make help` lists the available settings. Automatic release preparation targets
`main`; `RELEASE_REPO` selects another repository with the same workflows, and
`UV` and `GH` select executable paths. Explicit tags retain direct workflow
dispatch for a prepared version or another ref:

```console
make release RELEASE_REF=main RELEASE_TAG=v0.2.2
make release-runs
make release-watch RUN_ID=12345
```

Replace the example tag with the prepared project version, and `12345` with a
run ID from `make release-runs`. Watching a run returns failure if it fails.
An explicit tag must match the selected commit's project version. Reusing a
published version for new changes is rejected; ordinary `make release` lets
Release Please prepare the next version instead.

The release workflow runs lock freshness, formatting, lint, typing, import,
test, and dependency-audit gates, including audits for the Whisper, Parakeet,
and Qwen worker slices. It builds distributions once, verifies isolated wheel
and source installs, and exports checksums, SBOMs, and a JSON preflight report.
Twine checks package metadata and README rendering before publication. Both
macOS and Linux test the merged source and verify the same wheel. Publication
jobs verify every downloaded asset against `SHA256SUMS`.

Both distributions embed the full tested commit. `yakbox --version` prints it
alongside the package version without needing Git:

```text
yakbox, version 0.2.2 (commit 0123456789abcdef0123456789abcdef01234567)
```

An unstamped development package prints `commit unknown`. For a local package
build from a clean checkout, use `uv run python -m yakbox._build`. To run the
complete preflight locally, use:

```console
make release-preflight
```

Local preflight derives its tag from the local project version and writes
`dist/` and `release-metadata/`. It does not prepare or merge a release PR.
Alternatively, pushing an already prepared version tag runs the publication
workflow directly. Routine releases need only `make release`.

## Optional PyPI publishing

GitHub Releases host distribution files, rather than a PyPI-compatible package
index. Direct wheel URLs work with
[pip](https://pip.pypa.io/en/stable/cli/pip_install/) and
[uv](https://docs.astral.sh/uv/guides/tools/). For installation by name with
`uv tool install yakbox` or `pip install yakbox`, publish Yakbox to PyPI.

Create a PyPI account and configure a
[pending Trusted Publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
for project `yakbox`, owner `pbsladek`, repository `yakbox`, workflow
`release.yml`, and environment `pypi`. The package name must be available or
owned by your account. Create the corresponding GitHub environment and, if
desired, require reviewer approval. This uses short-lived OIDC credentials;
no PyPI API token is needed in GitHub secrets.

After setup, publish the reviewed tagged version from your terminal:

```console
make release RELEASE_REF=v0.2.0 PUBLISH_TO=pypi
```

`PUBLISH_TO=pypi` publishes to PyPI only; it does not recreate the GitHub
Release. `PUBLISH_TO=testpypi` likewise publishes only to TestPyPI, using the
`testpypi` environment and a separately configured TestPyPI Trusted Publisher.
Package-index publication requires an existing matching tag that resolves to
the tested commit; an untagged branch cannot publish to either index.
Every destination requires the same preflight and macOS/Linux verification.
