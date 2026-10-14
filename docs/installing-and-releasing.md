# Installing and releasing yakbox

## Install the CLI

Yakbox supports macOS and Linux with Python 3.14. Windows is not supported.
The default installation is small and does not install PyTorch. Install the
latest release from PyPI:

```console
uv tool install --python 3.14 yakbox
yakbox --version
yakbox doctor
```

To upgrade, run `uv tool upgrade yakbox`. You can also install a specific
wheel URL from GitHub Releases; dependencies still come from the configured
package index.

For library use inside a Python 3.14 environment, install from PyPI:

```console
python -m pip install yakbox
```

Install optional extras by name. Local extras use the matching release's
reviewed security overrides:

<!-- x-release-please-start-version -->
```console
uv tool install --python 3.14 "yakbox[credentials]"
uv tool install --python 3.14 "yakbox[local]==0.3.2" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.3.2/constraints/chatterbox-security-overrides.txt
uv tool install --python 3.14 "yakbox[credentials,local]==0.3.2" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.3.2/constraints/chatterbox-security-overrides.txt
uv tool install --python 3.14 "yakbox[local,alignment]==0.3.2" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.3.2/constraints/chatterbox-security-overrides.txt
```
<!-- x-release-please-end -->

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

<!-- x-release-please-start-version -->
```console
gh release download v0.3.2 --repo pbsladek/yakbox --dir yakbox-release
cd yakbox-release
shasum -a 256 -c SHA256SUMS
uv tool install --python 3.14 ./yakbox-*.whl
```
<!-- x-release-please-end -->

On Linux, use `sha256sum -c SHA256SUMS` for the checksum step. GitHub also
produces build-provenance and SBOM attestations. Verify a downloaded wheel with:

```console
gh attestation verify yakbox-*.whl --repo pbsladek/yakbox
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

Release Please also updates the marked installation examples in the README and
guides, keeping local-extra pins and security override URLs on the same release.

`make release` does the following:

1. Runs the pinned Release Please action to create or update the release PR.
2. Waits for CI on the exact prepared PR commit.
3. Squash-merges that commit using the caller's GitHub credentials. Required
   reviews, branch rules, and merge queues still apply; it never uses `--admin`.
4. Waits for the existing release workflow to build, audit, test, attest, and
   publish the same wheel and source archive to GitHub and PyPI.
5. Verifies both destinations, marks the release PR as published, and resumes
   preparation for any changes merged during publication.

The workflows also maintain the PR on pushes to `main`. Merging that PR through
GitHub starts publication automatically. The built-in `GITHUB_TOKEN` does not
start PR-triggered CI for bot-created changes, so preparation explicitly
dispatches `ci.yml` on the prepared release branch. The manifest merge likewise
explicitly dispatches publication. No additional PAT or publishing secret is
needed. PyPI uses the configured Trusted Publisher described below.

Repository Settings > Actions > General must permit GitHub Actions to create
pull requests. Workflow permissions remain read-only by default; each job
requests only its required permissions. Release Please's initial manifest
records the already published 0.2.1 release. Adding that manifest does not
republish the historical version.

If there are no releasable Conventional Commits, the command reports that no
release PR is available. CI failure or a changed PR head stops the command
before merging. A publication failure leaves the merged version pending;
running `make release` again retries that version. If a tag already exists,
the retry keeps its original commit even if `main` has advanced. Before a tag
exists, a retry can use a corrected `main` commit with the same manifest version.
All publication checks run again. Published tags and distributions remain
immutable: conflicting hashes stop the retry.

Release preparation and publication start independently after a release PR
merges. While the PR carries `autorelease: pending`, preparation reports a
notice and defers the next PR; the waiting state does not fail the workflow.
`make release` waits for an existing publication or its dispatch instead of
starting a duplicate. If publication failed, it retries after any required fix
has been merged. Keep the pending label until publication succeeds.
Successful publication to both GitHub and PyPI marks the PR
`autorelease: tagged` and dispatches preparation on `main` again, so changes
merged while publication was pending can receive the next release PR.

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
jobs share a download-and-verify action that checks every asset against
`SHA256SUMS` and binds the preflight report to the release version and commit.
Python helpers in `release_workflow.py` and `release_publication.py` implement
the workflow decisions and have offline regression tests.

The workflow list shows `Prepare release · main`, `Start release · main`, and
`Release vX.Y.Z`. Release CI identifies its preparation number. The CLI uses
GitHub’s dispatch run IDs to track preparation and retries; UUIDs and full
commit hashes are kept out of run titles.

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

## PyPI Trusted Publishing

Every release publishes to GitHub and PyPI. There is no destination selector
or TestPyPI job. The `pypi` environment and Trusted Publisher are required
for a release to complete; no PyPI API token is stored in GitHub.

Create a PyPI account and configure a
[pending Trusted Publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
for project `yakbox`, owner `pbsladek`, repository `yakbox`, workflow
`release.yml`, and environment `pypi`. The package name must be available or
owned by your account. Create the corresponding GitHub environment and, if
desired, require reviewer approval. This uses short-lived OIDC credentials;
no PyPI API token is needed in GitHub secrets.

For a new project, sign in to
[PyPI account publishing settings](https://pypi.org/manage/account/publishing/)
and fill the GitHub pending-publisher form:

| Field | Value |
| --- | --- |
| PyPI Project Name | `yakbox` |
| Owner | `pbsladek` |
| Repository name | `yakbox` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Use the workflow filename, without `.github/workflows/`. If your account
already owns `yakbox`, add the publisher under that project's **Manage >
Publishing** instead. A pending publisher does not reserve the project name;
the first successful upload creates the project and converts the publisher
to a normal publisher.

The matching GitHub environment lives under
[repository Settings > Environments](https://github.com/pbsladek/yakbox/settings/environments).
Its name must be exactly `pypi`. You may add required reviewers there if you
want a separate approval before each upload. The `publish-pypi` job in
`.github/workflows/release.yml` uses that environment, grants `id-token: write`,
and invokes the pinned PyPA publishing action after all release checks pass.
Registering the publisher on PyPI is an account action; configuring GitHub
alone does not register it.

After setup, run `make release` for routine releases. To retry an already
prepared version explicitly, select its tag:

```console
make release RELEASE_REF=v0.3.2 RELEASE_TAG=v0.3.2
```

Replace the example with the version being retried. The retry verifies any
existing GitHub distributions and PyPI files before accepting them. Identical
files are retained, and a partial PyPI upload can finish without recreating the
GitHub release. A mismatched digest fails the release instead of overwriting or
silently skipping a different package.

The release PR keeps `autorelease: pending` if either destination fails. Only
the completion job, after verifying both published files on PyPI and the
GitHub release, marks it `autorelease: tagged` and starts the next preparation.
Use `make release-runs` and `make release-watch RUN_ID=...` to follow a run.

After publication, verify installation by name:

```console
uv tool install --python 3.14 yakbox
yakbox --version
```
