# Licensing and third-party material

Yakbox source code is distributed under the MIT License. The repository should
remain reusable for commercial and noncommercial work without hidden asset
restrictions.

## Software

Prefer MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, or ISC for dependencies and
incorporated software. Another OSI-approved license needs explicit review for
distribution obligations and compatibility with Yakbox's MIT license. Do not
add software with an unknown, source-unavailable, noncommercial, or otherwise
proprietary license.

The lockfile, vulnerability audit, generated CycloneDX SBOM, and release
attestations cover the resolved software supply chain. A dependency update must
also include a license review; a vulnerability result is not a license result.
The required PyYAML configuration parser is MIT-licensed and is invoked only
through its safe loaders.

## Media, reference voices, and documentation assets

Bundled media must be public domain, CC0, or CC BY 4.0. Attribution required by
CC BY must remain with redistributed copies. Do not bundle material under CC
BY-NC, CC BY-ND, an unknown license, or informal permission that cannot be
verified. ShareAlike material needs explicit compatibility review and should
not be treated as a repository default.

Every third-party asset record must include:

- the local filename and SHA-256 digest;
- the creator or performer and original work;
- stable source, catalog, and rights URLs;
- the source-file digest when a local derivative is bundled;
- the transformation interval and output format when applicable; and
- a no-endorsement statement for reference voices.

Automated tests must verify that every bundled media file has exactly one
record, its bytes match the recorded digest, and its declared rights identifier
is allowed. Generated build outputs, auditions, and caches remain unversioned.

## Included LibriVox prompts

The local Chatterbox example contains twenty-five qualified 20-second prompts.
LibriVox
states that its recordings are public domain in the United States and can be
modified or used commercially. Copyright status may differ outside the United
States. The complete reader list and per-file record are in
`examples/local-chatterbox/voices/voices.toml`.

Public-domain copyright status does not imply endorsement and does not settle
every jurisdiction's personality, publicity, or synthetic-voice rules. Users
remain responsible for the laws and consent requirements that apply to their
use.

## Optional MLX Whisper alignment

The Apple Silicon `alignment` extra uses MIT-licensed MLX Whisper and OpenAI
Whisper software. Its default model is `mlx-community/whisper-large-v3-turbo`,
an MLX conversion of Whisper Large-v3-Turbo, pinned by immutable revision in the default
manifest parser. `yakbox whisper models install` downloads model files to a
user cache; builds never download them. They are not part of
Yakbox distributions. Treat a custom model as a separate licensing review even
when its loader code is MIT-licensed.

## Optional multi-model speech analysis

The speech-analysis extras add two permissively licensed runtimes:

- `parakeet-mlx` is Apache-2.0 licensed; and
- `mlx-audio` is MIT licensed. Yakbox imports only its local speech-to-text
  implementation. It does not initialize the package's server, microphone,
  text-to-speech, or speech-to-speech features.

The reviewed model set is installed only by the explicit model lifecycle
service. Model files stay in the user's platform cache and are not included in
Yakbox wheels or source distributions.

| Yakbox engine | Model | License |
| --- | --- | --- |
| `whisper` | OpenAI Whisper Large-v3-Turbo, MLX conversion | MIT |
| `parakeet` | NVIDIA Parakeet TDT 0.6B v3, MLX conversion | CC BY 4.0 |
| `qwen` | Qwen3-ASR 1.7B BF16, MLX conversion | Apache-2.0 |
| `qwen-forced` | Qwen3-ForcedAligner 0.6B BF16, MLX conversion | Apache-2.0 |

The separate qualification-only registry also pins the official MLX-community
8-bit affine, group-size-64 conversions of Qwen3-ASR 1.7B and
Qwen3-ForcedAligner 0.6B. They remain Apache-2.0, stay outside the default model
set, and cannot be selected by a book manifest. They exist only for measured
BF16 non-inferiority comparisons.

Parakeet's CC BY 4.0 terms require attribution when its weights are shared.
Preserve the model name, NVIDIA as the model provider, the license, and links
to the upstream and converted model cards. Do not imply endorsement.

The packaged registry at
`src/yakbox/data/speech-model-registry-v1.toml` is the authoritative record for
package versions and hashes, immutable model revisions, file allowlists,
checksums, source URLs, conversion recipes, and license identifiers. The
qualification candidates use the same controls in
`src/yakbox/data/speech-model-candidates-v1.toml` but carry no default-model
authority. The
qualification record in `docs/speech-analysis-qualification.md` states which
runtime and corpus gates have actually been run. A pinned artifact is not
automatically a qualified artifact.

## Optional phoneme forced alignment

The `phoneme` extra uses BSD-licensed PyTorch, Apache-2.0 Transformers, and
Meta's Apache-2.0 `facebook/wav2vec2-lv-60-espeak-cv-ft` model. The model is
pinned, downloaded only by an explicit command, and is not redistributed by
Yakbox.

Expected text is converted to IPA by an external eSpeak NG executable. eSpeak
NG is GPL-3.0-or-later. Yakbox does not vendor, link, download, or redistribute
it; the optional backend invokes a separately installed command. This is an
explicitly reviewed exception to the repository's permissive-dependency
preference. Deployments that cannot accept that runtime obligation should keep
`phoneme_alignment = false`.
