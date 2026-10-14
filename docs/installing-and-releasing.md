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

## Trigger a GitHub Release locally

The project version in `pyproject.toml` is authoritative. The next release
version is `0.2.1`, with tag `v0.2.1`. Commit and merge the version and workflow
changes to `main` before running the release workflow. Authenticate `gh` with
`gh auth login` if needed, then run:

```console
make release
make release-runs
make release-watch RUN_ID=12345
```

Replace `12345` with the ID returned by `make release-runs`. The workflow must
exist on the repository's default branch before it can be dispatched.
`RELEASE_REF` selects the remote branch or tag to release; uncommitted or
unpushed local changes are not included. `make release` defaults to repository
`pbsladek/yakbox`, ref `main`, and GitHub publication. An empty `RELEASE_TAG`
derives the tag from the selected
commit's project version. A supplied tag must match that version. Override these
settings when needed:

```console
make release RELEASE_REPO=pbsladek/yakbox RELEASE_REF=main RELEASE_TAG=v0.2.1
```

The equivalent GitHub CLI command is:

```console
gh workflow run release.yml --repo pbsladek/yakbox --ref main \
  -f tag=v0.2.1 -f publish_to=github
```

`make help` lists all release targets and settings; `UV` and `GH` can select
alternate executable paths. Watching a run returns a failure status if the
workflow fails.

The workflow runs the lock, formatting, lint, typing, test, and universal
dependency-audit gates, including exact audits for the Whisper, Parakeet, and
Qwen worker dependency slices. It builds the distributions once, verifies
isolated installs, and exports checksums, SBOMs, and a JSON preflight report.
Twine validates the wheel and source metadata and README before publication.
Both distributions embed the full Git commit selected for the release.
`yakbox --version` prints it alongside the package version:

```text
yakbox, version 0.2.1 (commit 0123456789abcdef0123456789abcdef01234567)
```

Preflight checks isolated installs of the wheel and source archive against
the release version and commit; the macOS and Linux release jobs also check
the installed wheel. Metadata remains embedded when the source archive is
rebuilt after installation. Version output reads the installed package and
does not require Git or inspect the caller's working directory.
An unstamped development package prints `commit unknown`. For a package build
from a clean Git checkout, use `uv run python -m yakbox._build`;
`make release-preflight` already embeds the commit and restores the source file
after packaging. Commit-stamped builds reject uncommitted changes so their
commit identifier describes the packaged source.

Both macOS and Linux then run the offline test suite and verify the same wheel.
Release test failures retain coverage and JUnit diagnostics. Every publication
job verifies the downloaded assets against `SHA256SUMS` before uploading them.
Only after those jobs pass does the GitHub job publish the tested artifacts
and create the tag at the tested commit. Existing tags must already point to
that commit. GitHub publication needs no PyPI account or publishing token.

Each new release needs a new version. For subsequent patch releases, run:

```console
uv version --bump patch
```

This updates `pyproject.toml` and refreshes `uv.lock`. Update the source-only
version fallback in `src/yakbox/__init__.py` to match and rebuild the packaged
worker, which includes that initializer:

```console
uv run python -m yakbox.speech.analysis_worker_artifact --write-packaged
```

Commit and merge the changes before running `make release`.

If preflight says a tag does not point to the release commit, the selected
version already belongs to another commit. For example, `v0.2.0` remains tied
to the published 0.2.0 source even after `main` advances. Bump the version to
release the new changes. Keep published tags at their original commits. To
publish an existing version to PyPI, select its tag as `RELEASE_REF`, as shown
below.

You can also run preflight locally from a clean worktree before publishing:

```console
make release-preflight
```

This derives the tag from the local project version; `RELEASE_TAG=v0.2.1` can
set it explicitly. It runs `uv run yakbox-release-preflight --tag v0.2.1` and
writes distributions under `dist/` and release evidence under
`release-metadata/`. `make release` runs preflight in GitHub Actions; it does
not require a separate local preflight. Alternatively, pushing a reviewed
version tag starts the GitHub release workflow automatically:

```console
git tag -a v0.2.1 -m "yakbox v0.2.1"
git push origin v0.2.1
```

Use either the manual dispatch or tag push for a new GitHub Release. A tag push
publishes to GitHub only. Releases are immutable versions; bump the project
version and refresh `uv.lock` before publishing the next one.

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
