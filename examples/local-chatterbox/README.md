# Local Chatterbox

This project uses one isolated local worker and conservative controls. It
includes twenty-five 20-second public-domain LibriVox reference prompts. The
sample uses generic roles: narration uses Andy Minter, `character-1` uses Caro
Davy, `character-2` uses Nick Whitley, `character-3` uses Ruth Golding,
`character-4` uses Bill Boerst, and `character-5` uses Stuart Bell. Nineteen
more generic slots expose the expanded catalog:

| Character | Profile | Gender |
| --- | --- | --- |
| `character-6` | `karen-savage` | female |
| `character-7` | `elizabeth-klett` | female |
| `character-8` | `cori-samuel` | female |
| `character-9` | `mil-nicholson` | female |
| `character-10` | `lucy-burgoyne` | female |
| `character-11` | `mark-f-smith` | male |
| `character-12` | `bob-neufeld` | male |
| `character-13` | `mark-nelson` | male |
| `character-14` | `david-barnes` | male |
| `character-15` | `adrian-praetzellis` | male |
| `character-16` | `gregg-margarite` | male |
| `character-17` | `david-clarke` | male |
| `character-18` | `martin-geeson` | male |
| `character-19` | `phil-chenevert` | male |
| `character-20` | `peter-yearsley` | male |
| `character-21` | `kara-shallenberg` | female |
| `character-22` | `kirsten-ferreri` | female |
| `character-23` | `sibella-denton` | female |
| `character-24` | `laurie-anne-walden` | female |

```console
uv tool install "yakbox[local]" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.1.0/constraints/chatterbox-security-overrides.txt
yakbox doctor yakbox.toml --backend chatterbox-local --deep
yakbox validate
yakbox plan
yakbox audition --profile caro-davy --text "A short voice check."
yakbox build
```

Compare all twenty-five voices before a long render:

```console
yakbox audition \
  --profile caro-davy \
  --profile nick-whitley \
  --profile andy-minter \
  --profile ruth-golding \
  --profile bill-boerst \
  --profile stuart-bell \
  --profile karen-savage \
  --profile elizabeth-klett \
  --profile cori-samuel \
  --profile mil-nicholson \
  --profile lucy-burgoyne \
  --profile mark-f-smith \
  --profile bob-neufeld \
  --profile mark-nelson \
  --profile david-barnes \
  --profile adrian-praetzellis \
  --profile gregg-margarite \
  --profile david-clarke \
  --profile martin-geeson \
  --profile phil-chenevert \
  --profile peter-yearsley \
  --profile kara-shallenberg \
  --profile kirsten-ferreri \
  --profile sibella-denton \
  --profile laurie-anne-walden \
  --text "A short voice check."
```

Choose a different narrator in `yakbox.toml` and keep the target profile in
sync:

```toml
[characters.narrator]
profile = "andy-minter" # or any profile declared in this manifest

[targets.default]
profile = "andy-minter"
```

Each character has its own profile, gender metadata, and optional performance
settings. Change those entries independently. Unmarked paragraphs use the
narrator; a `yakbox:speech:speaker` directive routes dialogue inside the next
paragraph to that character while the narrator reads surrounding action and
attribution tags. An unquoted routed paragraph remains entirely in the
character voice. The `gender` field documents the role. Yakbox does not infer
or alter a voice from it—the assigned profile and reference audio determine the
rendered voice.

The example defaults to CPU. Set `device = "auto"`, `"mps"`, or `"cuda"` only
when the matching runtime is available. Never run multiple workers against one
GPU merely to increase CLI concurrency.

The source, transformation, checksum, and rights record for every prompt is in
[`voices/voices.toml`](voices/voices.toml). LibriVox recordings are public
domain in the United States, but may have a different copyright status
elsewhere. The readers have not endorsed Yakbox or generated speech made with
these prompts.

For repeatable CLI, chunk-boundary, all-voice, and audiobook listening tests,
run the opt-in [local narration QA suite](../../docs/local-narration-qa.md).

On an Apple Silicon Mac, install both local model extras to try verified
context extraction for one- to three-word chunks:

```console
uv tool install "yakbox[local,alignment]" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.1.0/constraints/chatterbox-security-overrides.txt
```

Then run `yakbox whisper models install` and set
`short_utterances.strategy = "context_extract"`. Whisper Large-v3-Turbo runs
locally, rejects recognized prefix or suffix words, and supplies word timing
for the crop. Keep `require_review_for_one_word = true` for release work. See
the [design and testing plan](../../docs/short-utterance-synthesis.md) for the
hard gates and review workflow.
