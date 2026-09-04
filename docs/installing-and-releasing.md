# Installing and releasing yakbox

## Install the CLI

Yakbox supports Python 3.14 only. The default installation is small
and does not install PyTorch:

```console
uv tool install yakbox
yakbox --version
yakbox doctor
```

Upgrade an existing installation with:

```console
uv tool upgrade yakbox
```

Install optional secure credential storage, local Chatterbox, or both:

```console
uv tool install "yakbox[credentials]"
uv tool install "yakbox[local]" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.1.0/constraints/chatterbox-security-overrides.txt
uv tool install "yakbox[credentials,local]" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.1.0/constraints/chatterbox-security-overrides.txt
uv tool install "yakbox[local,alignment]" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.1.0/constraints/chatterbox-security-overrides.txt
```

Local Chatterbox has a much larger machine-learning dependency graph. Install
it only on a machine that will run the model. The versioned override file
replaces vulnerable exact pins in Chatterbox 0.1.7; do not omit it. FFmpeg and
FFprobe are separate system tools required for mastering, MP3 encoding,
inspection, and M4B assembly.

The `alignment` extra installs `mlx-whisper` only on Apple Silicon macOS. It is
needed only for `short_utterances.strategy = "context_extract"`; direct
synthesis and the default CLI do not import or install it. The configured model
revision is resolved into the local Hugging Face cache before transcription.

## Install a GitHub Release

Each tagged release contains a platform-independent wheel, a source archive,
and `SHA256SUMS`. Download all three files from the
[GitHub Releases](https://github.com/pbsladek/yakbox/releases) page, verify
them, and install the wheel:

```console
shasum -a 256 -c SHA256SUMS
uv tool install ./yakbox-*.whl
```

On Linux, use `sha256sum -c SHA256SUMS` for the checksum step.

## Publish a release

The project version in `pyproject.toml` is authoritative. A reviewed release
begins by running the same fail-closed preflight used by CI:

```console
uv run yakbox-release-preflight --tag vX.Y.Z
```

The command requires a clean worktree; validates the version and proposed tag;
runs the lock, formatting, lint, typing, test, and universal dependency-audit
gates—including exact audits for the Whisper, Parakeet, and Qwen worker
dependency slices; builds the wheel and source archive once; isolated-install
tests both; exports the package CycloneDX 1.5 SBOM plus one SBOM for each worker
slice; and writes `SHA256SUMS` plus a JSON preflight report under
`release-metadata/`.

After reviewing that evidence, create the tag:

```console
git tag -a vX.Y.Z -m "yakbox vX.Y.Z"
git push origin vX.Y.Z
```

The tagged workflow reruns preflight while additionally proving that the tag
points to the tested commit. It publishes through PyPI Trusted Publishing and
creates a GitHub Release containing the exact distributions, checksums, four
SBOM documents, and preflight report. GitHub produces both build-provenance and SBOM
attestations with short-lived OIDC credentials.

Consumers can verify a downloaded distribution against this repository:

```console
gh attestation verify yakbox-*.whl --repo pbsladek/yakbox
```

The workflow can also be started manually to publish the same validated
artifacts to TestPyPI. Configure the protected `pypi` and `testpypi` GitHub
environments before the first publication.
