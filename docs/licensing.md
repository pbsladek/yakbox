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

The local Chatterbox example contains twenty-five 20-second prompts. LibriVox
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
