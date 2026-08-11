# Third-party notices

Yakbox's source code is licensed under the root MIT License. The following
bundled media comes from third parties and retains its own stated rights basis.

## YAML configuration parser

Yakbox uses PyYAML to safely parse user-authored YAML configuration. PyYAML is
distributed under the MIT License. Yakbox uses safe loaders and does not enable
arbitrary Python object construction.

- PyYAML license: <https://github.com/yaml/pyyaml/blob/main/LICENSE>

## LibriVox reference voice prompts

The files under `examples/local-chatterbox/voices/` are 20-second derivatives
of recordings by the twenty-five qualified LibriVox readers listed in the adjacent
`voices.toml` registry. LibriVox states that its recordings are public domain
in the United States and may be used, modified, and sold without permission.
Copyright status may differ outside the United States.

The exact reader, source work, catalog URL, recording URL, source checksum,
extraction interval, local format, and local checksum for each file are recorded
in `examples/local-chatterbox/voices/voices.toml`.

No reader, author, or LibriVox endorsement is claimed or implied.

- LibriVox public-domain policy: <https://librivox.org/pages/public-domain/>
- LibriVox about page: <https://librivox.org/pages/about-librivox/>

## Optional local alignment software and model

The optional Apple Silicon alignment path uses Apple's MLX examples package
(`mlx-whisper`) and OpenAI Whisper code, both under the MIT License. The default
`mlx-community/whisper-large-v3-turbo` weights are an MLX conversion of
OpenAI's Whisper Large-v3-Turbo model. Yakbox pins the selected model revision,
installs it only through an explicit model command, and does not bundle
or redistribute the model files; the runtime downloads them into the user's
local cache.

- MLX examples license: <https://github.com/ml-explore/mlx-examples/blob/main/LICENSE>
- OpenAI Whisper license: <https://github.com/openai/whisper/blob/main/LICENSE>
- Default MLX model card: <https://huggingface.co/mlx-community/whisper-large-v3-turbo>

## Optional phoneme forced-alignment runtime and model

The optional phoneme path uses Meta's Apache-2.0
`facebook/wav2vec2-lv-60-espeak-cv-ft` model through Apache-2.0 Transformers
and BSD-licensed PyTorch. Model files are explicitly installed to the user's
cache and are not included in Yakbox distributions.

Yakbox optionally invokes a separately installed eSpeak NG executable to
derive expected IPA phonemes. eSpeak NG is GPL-3.0-or-later and is neither
vendored nor downloaded by Yakbox.

- Phoneme model card: <https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft>
- eSpeak NG license: <https://github.com/espeak-ng/espeak-ng/blob/master/COPYING>
