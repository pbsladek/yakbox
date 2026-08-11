# Local Chatterbox

This project uses one isolated local worker and conservative controls. It
includes twenty-five qualified 20-second public-domain LibriVox reference prompts.
The sample uses generic roles: narration uses Andy Minter, `character-1` uses Caro
Davy, `character-2` uses Nick Whitley, `character-3` uses Ruth Golding,
`character-4` uses Bill Boerst, and `character-5` uses John Greenman. Nineteen
more generic slots expose the expanded catalog:

| Character | Profile | Gender |
| --- | --- | --- |
| `character-6` | `karen-savage` | female |
| `character-7` | `amanda-friday` | female |
| `character-8` | `cori-samuel` | female |
| `character-9` | `mil-nicholson` | female |
| `character-10` | `lucy-burgoyne` | female |
| `character-11` | `mark-f-smith` | male |
| `character-12` | `bob-neufeld` | male |
| `character-13` | `mark-nelson` | male |
| `character-14` | `david-barnes` | male |
| `character-15` | `simon-evers` | male |
| `character-16` | `gregg-margarite` | male |
| `character-17` | `tony-foster` | male |
| `character-18` | `martin-geeson` | male |
| `character-19` | `phil-chenevert` | male |
| `character-20` | `peter-yearsley` | male |
| `character-21` | `kirsten-ferreri` | female |
| `character-22` | `sibella-denton` | female |
| `character-23` | `laurie-anne-walden` | female |
| `character-24` | `john-burlinson` | male |

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
  --profile john-greenman \
  --profile karen-savage \
  --profile amanda-friday \
  --profile cori-samuel \
  --profile mil-nicholson \
  --profile lucy-burgoyne \
  --profile mark-f-smith \
  --profile bob-neufeld \
  --profile mark-nelson \
  --profile david-barnes \
  --profile simon-evers \
  --profile gregg-margarite \
  --profile tony-foster \
  --profile martin-geeson \
  --profile phil-chenevert \
  --profile peter-yearsley \
  --profile kirsten-ferreri \
  --profile sibella-denton \
  --profile laurie-anne-walden \
  --profile john-burlinson \
  --text "A short voice check."
```

For an objective first-pass quality gate, synthesize the same sustained test
passage with every profile, then compare the resulting audition WAVs to the
four human-approved baseline voices:

```console
yakbox audition yakbox.toml \
  --profile andy-minter \
  --profile ruth-golding \
  --profile caro-davy \
  --profile nick-whitley \
  --profile bill-boerst \
  --text-file voice-quality.txt

yakbox whisper qualify-voices artifacts/auditions/RUN_ID/audition.json \
  --expected-file voice-quality.txt \
  --baseline andy-minter \
  --baseline ruth-golding \
  --baseline caro-davy \
  --baseline nick-whitley \
  --out artifacts/auditions/RUN_ID/voice-quality.json
```

Pass all desired profiles to `audition`; the shortened list above keeps the
example readable. Qualification measures transcript accuracy, word confidence,
independent-decode agreement, Whisper segment evidence, clipping, boundary
jumps, VAD disagreement, stationary tones, and edge silence. Pitch, accent,
timbre, and speaking rate are reported but do not decide acceptance. A
`suspect` result means the voice needs investigation or a new reference clip;
it is not a claim that the reader or accent is undesirable.

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
paragraph to that character. This example enables `strip_attribution_tags`, so
the distinct voices replace tags such as `the first technician said` while the
narrator still reads surrounding action. Set it to `false` when the narrator
should read those tags. For a manuscript without inline directives, set
`dialogue.routes` to a reviewed sidecar produced by `yakbox dialogue routes
suggest`; inspect the exact result with `yakbox dialogue preview`. An unquoted
routed paragraph remains entirely in the
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

The current baseline and high-quality classifications are recorded in
[`voices/quality.toml`](voices/quality.toml). Failed reference recordings are no
longer bundled or selectable; the file retains their reason codes and replacement
names as compact audit history. John Greenman, Amanda Friday, Simon Evers, and Tony
Foster are qualified replacements. Lee Ann Howlett passed automated qualification
but was removed after manual listening review. John Burlinson is the qualified male
catalog expansion used by the Anima Street Bar Manager.

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
