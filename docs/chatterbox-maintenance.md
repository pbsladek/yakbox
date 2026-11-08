# Keeping Chatterbox usable

Yakbox can keep using the existing Chatterbox weights without training a new
model. Preserve the model files, inference code, and working environment as
separate artifacts. A library fork can maintain compatibility, fix bugs, and
improve inference while leaving the weights unchanged.

The upstream repository currently describes Chatterbox as retired research.
The [library license](https://github.com/resemble-ai/chatterbox/blob/master/LICENSE),
[original model card](https://huggingface.co/ResembleAI/chatterbox), and
[Turbo model card](https://huggingface.co/ResembleAI/chatterbox-turbo) declare
MIT licensing. Preserve the license and copyright notices with copies and
modifications. Review each additional model, conversion, tokenizer, watermark
dependency, and reference voice separately; a loader's license does not cover
every file it can load.

## Preserve a working model

Start with the exact model revision that produces the audio you want. Save
its full commit ID, model card, license, tokenizer files, configuration,
conditioning data, weights, and checksums. Preserve actual bytes rather than
Git LFS/Xet pointer files or symlinks into a cache.

Install and use the Hugging Face CLI separately. Downloading is an explicit
preparation step; replace the revision placeholder with the selected full
40-character commit:

```console
hf download ResembleAI/chatterbox --revision FULL_40_CHARACTER_COMMIT \
  --local-dir models/chatterbox-original
yakbox models --record models/chatterbox-original --variant original \
  --repository ResembleAI/chatterbox --revision FULL_40_CHARACTER_COMMIT
yakbox models --verify models/chatterbox-original --variant original
```

Recording writes `yakbox-chatterbox-model.toml` atomically. It contains a
versioned schema envelope, source repository/revision, reviewed license,
file sizes and SHA-256 digests, and a stable model fingerprint. Record only a
trusted snapshot whose license you have reviewed. Recording establishes an
inventory; it does not independently prove the source's authenticity or
establish narration quality. Existing records are not overwritten.

Configure the recorded snapshot in the book manifest:

```toml
[profiles.archived]
backend = "chatterbox-local"
voice = "narrator"
device = "cpu"
variant = "original"
model_directory = "models/chatterbox-original"
model_fingerprint = "COPY_THE_RECORDED_64_HEX_FINGERPRINT"
```

Both model fields are required together. Absolute directories are supported
for a shared local archive. The loader verifies the record, file set, and file
digests before loading from local files. A worker reuses that verification while
the file set and filesystem metadata remain unchanged; the explicit verification
command always hashes every file. Added Python files, symlinks, missing
files, changed bytes, and remote-code configuration are rejected. The
fingerprint binds generation caches to the model snapshot. Moving an identical
snapshot does not change the direct synthesis-cache identity.

The original snapshot needs `ve.safetensors`, `t3_cfg.safetensors`,
`s3gen.safetensors`, `tokenizer.json`, and `conds.pt`. Keep the other reviewed
metadata and license files as well. The legacy profile with no local model
fields retains the upstream pretrained loader for compatibility; use a
recorded snapshot for reproducible offline operation.

## Turbo and MLX variants

`variant = "turbo"` selects PyTorch Chatterbox Turbo from the existing local
extra. Its local snapshot needs `ve.safetensors`, `t3_turbo_v1.safetensors`,
`s3gen_meanflow.safetensors`, and the tokenizer configuration and vocabulary
files used by that revision. Archive from `ResembleAI/chatterbox-turbo`, then
record it with `--variant turbo`. Set a target's profile explicitly when
comparing it with original Chatterbox.

`variant = "mlx-turbo"` selects the optional Apple Silicon MLX port, installed
through the `analysis-qwen` extra. It requires a recorded local conversion,
`config.json` selecting `chatterbox_turbo`, the converted weights,
`conds.safetensors`, tokenizer configuration/vocabulary, and the auxiliary
S3TokenizerV2 weights at `s3tokenizer/model.safetensors`. Archive and review that
auxiliary model's revision and license separately before recording the combined
file set. Yakbox resolves the port's tokenizer lookup from that local file;
its loader cannot silently fall back to downloading it. Use `device = "auto"`
or `"mps"` for the MLX profile. MLX executes on Apple Silicon's Metal device.

Pass `--license MIT` or `--license Apache-2.0` explicitly when recording an MLX
snapshot, matching the reviewed conversion's model card. For example, the
[MLX community FP16 conversion](https://huggingface.co/mlx-community/chatterbox-turbo-fp16)
declares Apache-2.0. Retain its notices as well as the original Chatterbox MIT
notices and the auxiliary model's provenance. The record's license field
describes the main model; it does not replace the licenses of bundled components.

Turbo variants reject nonzero `cfg_weight` and `exaggeration` because upstream
does not implement those controls. Original and Turbo profiles can set
`temperature`, `top_p`, and `repetition_penalty`; their values participate in
synthesis-cache identity. The PyTorch variants retain the upstream PerTh
watermark implementation. The MLX port has a different implementation and
does not establish the same watermark or narration-quality guarantees.

For a direct local-snapshot test:

```console
yakbox tts "A short test." --backend chatterbox-local --variant original \
  --model-directory models/chatterbox-original \
  --model-fingerprint COPY_THE_RECORDED_64_HEX_FINGERPRINT --out test.wav
```

The new variants are opt-in integrations. Offline fixture tests cover their
contracts; release narration still requires the real local end-to-end report
and completed hash-bound listening review described in
[local narration QA](local-narration-qa.md). Switching variants, precision,
device, or loader versions requires a new comparison; a dependency update is
not narration evidence.

## Keep your own mirror and backup

Create a private model repository under your Hugging Face account or
organization. Upload the unmodified model files, original license/model card,
source revision, and checksum record. Identify it as an archival mirror and
name the original author. An upload does not require training a model.
See Hugging Face's [repository guide](https://huggingface.co/docs/hub/repositories-getting-started)
and [upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload).

For example, after creating a private repository and authenticating separately:

```console
hf upload YOUR_ACCOUNT/chatterbox-original-archive models/chatterbox-original
```

Keep the record's original provenance when mirroring. Pin the mirror's commit
when downloading it again, then verify the original record and expected
fingerprint. Keep a second copy outside Hugging Face, such as a local disk
plus an offsite archive. Test recovery with network access disabled. Private
hosting is a storage/access setting; license obligations still apply.

## Maintain the library separately

Fork the inference library in GitHub and preserve its original history and
MIT notices. Start from the revision matching the working `chatterbox-tts`
wheel, retain import compatibility, and make small changes with independent
tests. Useful maintenance work includes Python/PyTorch/Transformers
compatibility, secure dependency ranges, local-only model loading, reference
conditioning reuse, MPS/CUDA fixes, and audio I/O. Preserve the upstream
watermark implementation while maintaining the PyTorch models.

Pin Yakbox to a full fork commit or an immutable wheel with a checksum. A
maintained fork can retain the `chatterbox` import namespace. Choose a
distribution/release strategy deliberately so users do not install two
packages that both own that namespace. Keep the security overrides and rerun
dependency audits when changing the fork's dependency declarations.

Preserve a working environment alongside the code and weights: Python version,
`uv.lock`, security overrides, relevant wheels for each supported platform,
FFmpeg/FFprobe versions, and a short recovery procedure. Keep a small set of
licensed reference clips and representative sentences for names, numbers,
dialogue, and short utterances. Run synthesis, Whisper checks, and listening
review against those fixtures after each compatibility change. Updating the
inference implementation does not require retraining the frozen weights.
