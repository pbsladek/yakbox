# Speech-analysis qualification record

This record separates implemented behavior from evidence that still needs to
be collected. The architecture is described in
[`multi-model-speech-analysis-plan.md`](multi-model-speech-analysis-plan.md).
The immutable package and model facts live in
`src/yakbox/data/speech-model-registry-v1.toml`.

Status as of 2026-08-14: the internal contracts, layered evidence cache,
isolated worker scheduler, bounded multi-round candidate search, performance
protocol, migration service, and fail-closed cutover decision are implemented.
Real-model smoke, endurance, and memory-pressure checks have also passed on the
reference host. Those runs now produce digest-bound independent-evidence
attestations. The full held-out corpus, calibrated safety result, performance
benchmark, and final review artifacts remain open, so the ensemble does not
affect public build outcomes.

## Qualified platform and package graph

The current test host is an Apple Silicon M5 Mac with 64 GB of unified memory,
macOS, and CPython 3.14.4. The locked environment resolves and imports:

| Package | Version | License |
| --- | ---: | --- |
| `mlx-whisper` | 0.4.3 | MIT |
| `parakeet-mlx` | 0.5.2 | Apache-2.0 |
| `mlx-audio` | 0.4.8 | MIT |

`uv audit --frozen` reported no known vulnerabilities in the resolved
179-package graph after the repository raised `cryptography` to 50.0.0. The
registry records reviewed wheel and source hashes and the corresponding source
revisions. It also records whether each upstream project publishes CI, tests, a
security policy, signed releases, and PyPI Trusted Publishing. Missing upstream
controls raise the pinning bar; they are not silently treated as present.

Separate frozen audits also reported no known vulnerabilities in the Whisper
(86 packages), Parakeet (80 packages), and Qwen (68 packages) worker slices.
Release preflight exports a CycloneDX 1.5 SBOM for each of those slices and
includes all three documents in release checksums and metadata.

## Model provenance

All four selected model snapshots are installed at immutable Hugging Face
revisions. The registry accepts only its explicit file allowlists, verifies
every size and digest, rejects symlinks and Python model code, and rejects
configuration that requests remote code.

The Whisper conversion received the strongest check. Yakbox downloaded the
official `large-v3-turbo.pt` checkpoint, verified SHA-256
`aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a`,
ran the converter from `mlx-whisper` 0.4.0 at source revision
`8160e0c4e56df261d0c8406f68d40b42ef0a188b`, and reproduced both converted
files byte for byte. The resulting hashes are:

- `config.json`:
  `b34fc29e4e11e0a25e812775dd67f4dd16fc2c8eb43d28ae25ff7d660ecb6379`
- `weights.safetensors`:
  `951ed3fc1203e6a62467abb2144a96ce7eafca8fa77e3704fdb8635ff3e7f8a6`

Parakeet publishes the exact conversion script linked by its model card. The
Qwen model cards identify `mlx-audio` 0.3.1 as the converter. Their exact
revisions, precision policies, recipes, converted file hashes, and upstream
revisions are pinned in the registry.

On 2026-08-14, that exact converter resolved and imported under Python 3.14.
Clean BF16 conversions from the pinned upstream revisions reproduced every
allowlisted file byte for byte. The ASR weight file was 4,076,186,653 bytes with
SHA-256 `2f080a3b769ae469aeaaa2dcb9e13a94141e54c9e6d5a7aa63392e0dc5a51789`;
the forced-aligner weight file was 1,835,539,240 bytes with SHA-256
`9d0728e17e28ee122c366bfaa9748ccbbb8fc3df4c0ce09dd8a5eac8202989e8`.
The opt-in rebuild test verifies the complete output directories against the
registry, including file allowlists, sizes, Git blob identities, SHA-256
digests, and unsafe configuration checks.

The qualification-only registry pins the official 8-bit affine, group-size-64
Qwen candidates independently of the BF16 defaults. On the current smoke clip,
the 8-bit ASR candidate produced the exact BF16 lexical sequence. The 8-bit
forced aligner produced the same ordered text units and kept every start/end
boundary within 20 ms of BF16. This is load and directional evidence only; BF16
remains selected until the complete calibration and held-out corpora prove
non-inferiority for names, numbers, short speech, known defects, and boundaries.

## Real-model smoke results

The opt-in test `tests/live/test_multi_model_speech_analysis.py` uses the
versioned Karen Savage LibriVox prompt, prepares one canonical mono PCM16 16 kHz
input, and runs only locally installed models. On 2026-08-13 it produced these
provisional single-run measurements:

| Engine | Precision | Wall time | Lexical units | Peak Metal allocation |
| --- | --- | ---: | ---: | ---: |
| Whisper | FP16 | 1.84 s | 68 | 2.57 GB |
| Parakeet | FP32 weights; BF16 runtime default | 1.49 s | 68 | 2.75 GB |
| Qwen3-ASR | BF16 | 4.20 s | 68 | 5.35 GB |
| Qwen3-ForcedAligner | BF16 | 2.23 s | 68 | 3.11 GB |

These numbers are smoke evidence, not performance thresholds. The three
recognizers returned the same number of lexical units but did not return the
same lexical sequence. Against the public-domain ground truth, the observed
differences were the acoustically ambiguous forms `whys`/`why's` and
`neighbors`/`neighbor's`/`neighbours`. The harness keeps those differences
visible for bounded directional-equivalence and calibration tests.

The isolated Whisper worker also passed launch, inference, status, unload,
idle cancellation, restart, and shutdown checks. It reported 1.92 GB peak RSS
and 2.57 GB peak Metal allocation. A corrected unload releases mlx-whisper's
process-global model cache: active MLX allocation fell from about 1.62 GB to
116 KB in the measured cycle.

The endurance gate then completed 150 calls for each recognizer worker: 450
successful real-model recognition calls in 2 minutes 45 seconds. Each sequence
mixed 5-second and 20-second windows, invalid-digest requests, status probes,
two explicit unload/reload cycles, active cancellation, worker termination, and
restart. Semantic result fingerprints remained stable for each input, the last
20-call median stayed within three times the first 20-call median, sampled
active Metal allocation stayed within a 256 MiB envelope, model-load counts
matched the planned lifecycle, and every worker recovered from cancellation in
a new process. This satisfies the long-lived recognizer endurance gate on the
qualified host.

On 2026-08-14, the separate memory-pressure gate kept the Qwen and Parakeet
models resident at the same time while serializing their inference. It forced a
Qwen deadline, active cancellation, process termination, and restart. The
Parakeet worker kept the same process and returned the same semantic result
before and after those failures; its explicit unload reduced active Metal
allocation below 16 MiB. Both workers stayed within their family memory
ceilings. This qualifies two-family residency with one heavy inference at a
time; it does not claim that concurrent Metal inference is safe.

The worker environment sets Hugging Face and Transformers offline modes, uses
verified local model paths, and has no model-install operation. Before loading
an adapter, the worker installs a Python audit guard that rejects IPv4 and IPv6
socket creation and network name resolution while preserving local IPC. The
smoke run completed through that boundary without a network fetch. A
default-suite subprocess test checks the guard independently of pytest's socket
blocking.

Direct adapters and worker-backed adapters emitted identical typed results for
Whisper, Parakeet, Qwen ASR, and Qwen forced alignment on the canonical smoke
window. The test runs each direct adapter and then the corresponding isolated
worker, comparing the complete normalized domain value rather than transcript
text alone.

## Worker-topology decision

Dependency-family isolation is selected for the qualified runtime. Three clean,
frozen Python 3.14 environments were built from `uv.lock`, each with Yakbox and
only one analysis extra. Each environment ran its intended real model through
the worker protocol, rejected the other two engine packages by absence, and
reported a stable installed-environment fingerprint:

| Environment | Installed distributions | Environment fingerprint |
| --- | ---: | --- |
| Combined development environment | 106 | development-only; not release evidence |
| Whisper worker | 40 | `24eecc50bdfc767ce5cedf71f808abd82f9774ee1a4384f6c7106b8a6ab14b4e` |
| Parakeet worker | 54 | `1df9229d96134044fef78897a6bd8e5608b68a77919742cf167ac019dedc65d0` |
| Qwen worker | 41 | `d327c97e939f989484e971d063bb7a5b3f5730a73aa18dd92882321adfa914b9` |

The fingerprint covers the exact Python patch release and sorted installed
distribution names and versions; it contains no environment path. Worker status
returns that fingerprint so doctor and qualification code can detect drift.
The combined environment remains a contributor convenience, not the selected
production topology.

The installed-wheel matrix was repeated on 2026-08-14 with the built
`yakbox-0.1.0-py3-none-any.whl`. The wheel installed each packaged lock into a
separate managed root, executed the wheel-owned worker artifact, and matched
every planned handshake. The resulting install fingerprints were
`3db298bd00c0eed94102214e1325dfc21b0e8f65ce475a38620ce38ec8553cd9`
for Whisper,
`fc9052e5b7cc39831899b07b84da673e4a50468526a60141d1299f8e72eab0a5`
for Parakeet, and
`01bcb8f3ad59cd3ce3daf3efcc90f53aac9ef2bcface32e8f3cc11b11471de7d`
for Qwen. Inspection from each isolated interpreter found only its intended ML
package: `mlx-whisper` 0.4.3, `parakeet-mlx` 0.5.2, or `mlx-audio` 0.4.8. All
three used worker artifact
`ad1a5c690f25ce36e786568c3052fb36260bafdf87d3f550ec0b5064d1dda5d1`.

The built-in worker definitions bind measured peak-memory ceilings of 4 GiB for
Whisper, 4 GiB for Parakeet, and 8 GiB for the Qwen family. Real smoke and
endurance tests fail if either peak RSS or peak Metal allocation crosses the
corresponding ceiling. These ceilings are qualification limits, not permission
to keep every model resident at once; runtime scheduling still uses at most two
resident worker families.

Parakeet's documented default streaming candidate—one-second input chunks,
`context_size=(256, 256)`, `depth=1`, and local attention—produced the exact
offline lexical sequence on the smoke clip. Streaming remains non-authoritative
until the same comparison passes every frozen clip and window class; one clean
sentence is not enough to establish owned-token or decision equivalence.

On 2026-08-14, the complete selected Apple Silicon suite passed 8 tests in
38.94 seconds. It covered direct and worker-backed output equivalence, the
timing-only forced-aligner boundary, BF16 versus 8-bit Qwen smoke evidence,
Parakeet offline versus streaming output, unload and restart, two-family memory
pressure, cancellation recovery, and the installed runtime matrix. The local
JUnit report and its independent-evidence record are under
`build/speech-analysis-qualification/`. The evidence fingerprint is
`3666947ea52d55e6bd1747917213ee0ef9021d8dca37f53e40a7daddb183739f`.

The runtime-specific endurance gate then completed 150 calls per recognizer,
450 calls total, in 165.01 seconds. It mixed short and long windows, invalid
requests, unload and reload cycles, memory sampling, cancellation, and restart.
The evidence fingerprint is
`c150e9ddb1cbe2e3d7e2a6a70039084b04d9a2754804a79776d387288c09e2e9`.
Both attestations bind the report digest, test source, worker artifact, runtime
locks, installed runtime identities, model snapshots, and execution class.
They fail closed for a skipped, failed, empty, incomplete, or identity-mismatched
run.

## Preregistered safety protocol

`src/yakbox/data/speech-evaluation-protocol-v1.toml` freezes the held-out safety
margin before the full corpus is evaluated. It requires zero observed false
accepts from the candidate decision and from the complete generation and
ranking workflow. Each must have a one-sided 95% Wilson upper bound of at most
5%, both overall and within every registered defect class. The protocol also
requires no increase in false rejections, name and number accuracy no worse
than the Whisper baseline, better median and P95 forced boundaries, and no
increase in contaminated or clipped crops.

With zero failures, that bound requires 52 independent source-passage and voice
clusters per class. The protocol registers 12 classes—carrier speech, chapter
audio, clipped boundaries, damaged joins, delivery encodes, extra syllables,
isolated words, names, numbers and codes, pauses, repaired regions, and short
questions—for 624 eligible defect clusters. Several crops from one passage and
voice count as one cluster, so they cannot inflate the evidence. Yakbox rejects
an evaluation if the corpus contains an unregistered defect class or lacks the
required independent clusters.

The protocol and its semantic fingerprint are implemented and tested. The
source stage verifies the existing licensed LibriVox voice registry and the
full recording checksum behind every approved clip. The initial 75 short
windows represented only 25 independent passages, so they were not used to
claim the required sample size. Yakbox now downloads those 25 checksum-pinned
public-domain archives once, then selects three non-overlapping regions per
reader at 120, 240, and 360 seconds. Each 30-second region is trimmed to quiet
PCM boundaries.

The archive inventory contains 25 recordings totaling 350,618,433 bytes. Its
fingerprint is
`823d7c24b942d8301e92b77f196eec6903166a80f14c6b6ccb452794c504d183`.
The expanded 2026-08-14 source run produced 75 independent
source-passage/voice clusters across 25 readers. Passage duration ranged from
18.00 to 25.94 seconds, with a 22.54-second median. Its source-inventory
fingerprint is
`023f04310936b5c3c9555a701eed9b35383202e320064b336c240bbb79bba2d7`.
The local inventory and canonical passages are under
`build/speech-analysis-qualification/corpus-sources-expanded/`.

The transcript-authoring stage is also implemented. It runs Parakeet, Qwen,
and Whisper in engine-major order through their verified isolated runtimes and
checkpoints every engine/window result. A changed adapter, worker artifact, or
runtime lock selects a different checkpoint namespace. Corrupt checkpoints
are recomputed, while a stopped run resumes without repeating valid model
work. The adapter now preserves a Whisper word whose native timestamp rounds
to zero frames by marking only that timing unavailable; it does not invent a
duration or discard the lexical result.

The 2026-08-14 adapter-v3 run completed all 225 recognitions under one model
and execution identity per engine. Of the 75 passages, 23 had unanimous
normalized text, 31 had an exact two-engine majority, 21 had three-way dissent,
and none were unusable. Its draft fingerprint is
`354c97fd82f2bfd15105586edbd3a3ff7f0b2e6461c7c44b84ec40110407bfec`.
The text-free agreement report is
`build/speech-analysis-qualification/transcript-agreement-expanded.json`. The private
authoring source is
`build/speech-analysis-qualification/transcript-authoring-expanded.json`, and the
listening worksheet is
`build/speech-analysis-qualification/transcript-review-expanded.md`.

Yakbox no longer asks a reviewer to listen to all 75 clips first. It downloads
the public-domain text linked from each LibriVox catalog entry, records the
resolved URL, and checksums both the downloaded file and its UTF-8 text. Three
explicit overrides replace a stale link and two archive viewer pages. The
override file is
`examples/local-chatterbox/voices/source-text-overrides.toml`; each override
records its source and rights URL. The 25-text inventory fingerprint is
`e306fc8d84aa126acd0e92b13c94336dc22bc040cc57a3ec849ffefb69808bae`.

Source anchoring has two evidence tiers. The stronger tier requires two or
more recognizers to match one unique span in the pinned text. The second tier
requires one exact recognizer match against one unique source span, at least
20 source tokens, and all three recognizers within a five-percent bounded edit
allowance. Fuzzy comparison can corroborate that second tier, but it cannot
choose the transcript. Hyphenated and unhyphenated forms are equivalent for
matching; the accepted transcript still comes from the pinned source.

The current pass anchored 29 of 75 clips. Nineteen need no listening. The
reduced packet contains 46 exceptions and 10 deterministic audit clips, for 56
reviews instead of 75. It names the reader and voice, labels the review type,
links the exact WAV, shows the source proposal when one exists, and lists all
three recognizer transcripts. The text-free anchor report is
`build/speech-analysis-qualification/transcript-source-anchors-expanded.json`.
The review packet is
`build/speech-analysis-qualification/REVIEW-SOURCE-ANCHORED-TRANSCRIPTS.md`.

Exact recognizer agreement by itself remains a drafting aid, not transcript
truth. The approval loader rejects partial review, stale identities, altered
recognition evidence, missing reviewer fingerprints, and empty accepted text.
Until the reduced review and its audit pass, the inventory and draft remain
source material rather than held-out quality evidence.

Rebuild the source-text inventory with:

```console
uv run python -m yakbox.speech.analysis_corpus_text_sources \
  --voice-registry examples/local-chatterbox/voices/voices.toml \
  --source-overrides examples/local-chatterbox/voices/source-text-overrides.toml \
  --repository-root . \
  --output-root build/speech-analysis-qualification/corpus-text-sources \
  --inventory-output build/speech-analysis-qualification/corpus-text-sources/inventory.json
```

Then recreate the text-free evidence and reduced packet with:

```console
uv run python -m yakbox.speech.analysis_corpus_text_anchors \
  --authoring build/speech-analysis-qualification/transcript-authoring-expanded.json \
  --audio-inventory build/speech-analysis-qualification/corpus-sources-expanded/inventory.json \
  --audio-root build/speech-analysis-qualification/corpus-sources-expanded \
  --text-inventory build/speech-analysis-qualification/corpus-text-sources/inventory.json \
  --text-root build/speech-analysis-qualification/corpus-text-sources \
  --report-output build/speech-analysis-qualification/transcript-source-anchors-expanded.json \
  --review-output build/speech-analysis-qualification/REVIEW-SOURCE-ANCHORED-TRANSCRIPTS.md \
  --audio-prefix corpus-sources-expanded --audit-size 10
```

Use the internal review command instead of editing the authoring JSON. It
revalidates the inventory, audio, draft fingerprint, and current file before
each atomic update. Its output contains case IDs and counts, not transcript
text or the reviewer label.

```console
uv run python -m yakbox.speech.analysis_corpus_review \
  --authoring build/speech-analysis-qualification/transcript-authoring-expanded.json \
  --inventory build/speech-analysis-qualification/corpus-sources-expanded/inventory.json \
  --audio-root build/speech-analysis-qualification/corpus-sources-expanded \
  status
```

Put a private reviewer label in a local file. After listening to a majority or
unanimous case, approve its prefilled proposal with:

```console
uv run python -m yakbox.speech.analysis_corpus_review \
  --authoring build/speech-analysis-qualification/transcript-authoring-expanded.json \
  --inventory build/speech-analysis-qualification/corpus-sources-expanded/inventory.json \
  --audio-root build/speech-analysis-qualification/corpus-sources-expanded \
  approve CASE_ID --reviewer-label-file /path/to/reviewer-label.txt
```

For a dissent case, write the corrected transcript to a private UTF-8 file and
add `--accepted-text-file /path/to/correction.txt`. The command hashes the
reviewer label before storing it. It never prints accepted or candidate text.

The partition and boundary-review contracts are implemented but cannot create
real evidence until transcript review is complete. The partition assigns whole
readers, not individual clips: six readers and 18 passages go to calibration;
19 readers and 57 passages go to held-out evaluation. This exceeds the required
52 held-out clusters and prevents one reader from appearing in both sets. The
boundary selector then chooses 12 distinct readers, balanced six-and-six across
the partitions. Qwen supplies only the initial timing proposal. Each selected
case requires two complete human passes and an explicit accepted boundary set;
the text-free truth report records every inter-pass boundary difference.

Deterministic risk variants still need to be generated and frozen before any
case can count toward the safety result. Every required held-out risk class must
retain at least 52 independent clusters.

Once that evaluation exists, `speech.analysis_review` generates the exact
pending review list as a versioned TOML artifact. The order is randomized from
a private seed, but only the seed fingerprint is stored. Each row contains a
case ID and review purpose, never manuscript text or a reviewer name. The
loader rejects stale evaluation, corpus, policy, or protocol fingerprints, as
well as missing, extra, duplicate, or pending decisions. An approved artifact
is fingerprinted into the calibration disposition. This removes manual case
bookkeeping; it does not replace the one-time judgments required to qualify a
new policy.

## Performance and cutover gates

`src/yakbox/data/speech-performance-protocol-v1.toml` freezes the Phase 9
benchmark shape. It covers all six workflows under cold-process/cold-cache,
warm-process/cold-evidence, and fully repeated cache states. Expensive chapter
workflows require five measured runs after an unmeasured warm-up. Bounded cache
operations require 20 measured runs before Yakbox reports a P95.

The performance evaluator checks the release criteria directly: an unchanged
repeat performs no inference, localized repair analysis stays below one quarter
of full-chapter verification time, multi-repair work uses one assembly and one
mastering pass, model loads stay bounded, Qwen runs only on policy-required
windows, offline operation succeeds, and delivery verification remains a
separate measured cost. No reference-M5 performance artifact has been approved
yet.

Candidate search uses the manifest's fixed candidates-per-round and
maximum-round limits. Each round runs the model-major truth table before
ranking. The ranker receives only accepted candidates and cannot return a
rejected or invented take. Search stops after the first round with a verified
winner. If no candidate passes, it records exhaustion at the configured limit
instead of relaxing a gate or expanding work implicitly.

`speech.analysis_cutover` combines the approved quality evaluation,
evaluation-bound calibration and review, performance qualification, all three
verified runtimes, all four verified model installations, and independent
release evidence. The forced aligner has its own pinned model even though it
shares the Qwen runtime family. The independent set includes the automated suite,
installed-wheel runtime matrix, Apple Silicon real-model tests, repeated-call
endurance, listening review, dependency/SBOM audit, and package release
preflight. Missing, duplicated, failed, stale, or mismatched evidence produces
a schema-valid report with `ready = false`. It cannot silently enable the
version-2 public surface.

The generic preflight evaluates that same runtime and model coverage before a
strict analysis run. It also checks the English capability matrix, per-engine
window duration, approved execution-class calibration, required clip-class
thresholds, and declared disk and memory capacity. A failure returns stable,
schema-valid issue codes and raises before any synthesis backend is loaded.

## Commands

Install the optional runtimes and run the bounded smoke suite:

```console
uv sync --frozen --all-groups --extra alignment \
  --extra analysis-parakeet --extra analysis-qwen
YAKBOX_RUN_SPEECH_ANALYSIS_LIVE=1 \
  uv run pytest -m live tests/live/test_multi_model_speech_analysis.py
```

Run the long-lived worker gate separately. It refuses a call count below 150:

```console
YAKBOX_RUN_SPEECH_ANALYSIS_LIVE=1 \
YAKBOX_RUN_SPEECH_ANALYSIS_ENDURANCE=1 \
  uv run pytest -m live \
  tests/live/test_multi_model_speech_analysis.py::test_long_lived_workers_endure_stable_repeated_inference
```

After explicitly installing the pinned qualification candidates, run the
non-default precision and streaming smoke comparisons with:

```console
uv run python -c 'from yakbox.speech.model_registry import ModelRegistry, default_qualification_model_root, load_qualification_model_registry; registry = ModelRegistry(default_qualification_model_root(), data=load_qualification_model_registry()); [registry.install(engine) for engine in registry.engines()]'
YAKBOX_RUN_SPEECH_ANALYSIS_LIVE=1 \
YAKBOX_RUN_SPEECH_ANALYSIS_CANDIDATES=1 \
  uv run pytest -m live tests/live/test_multi_model_speech_analysis.py \
  -k 'qwen_8bit or parakeet_streaming'
```

After rebuilding both Qwen BF16 models with the registry's exact converter,
upstream revisions, and `--dtype bfloat16 --model-domain stt` recipe, verify
the output directories with:

```console
YAKBOX_QWEN_ASR_REBUILD=/path/to/qwen-asr-bf16 \
YAKBOX_QWEN_FORCED_REBUILD=/path/to/qwen-forced-bf16 \
  uv run pytest -m live \
  tests/live/test_multi_model_speech_analysis.py::test_clean_qwen_bf16_rebuilds_match_pinned_snapshots
```

## Open gates

The following evidence is required before the Phase 9 public cutover:

- turn the staged 75-cluster licensed source inventory into the frozen corpus,
  including short dialogue, names, numbers, damaged joins, clipping, pauses,
  delivery encodes, chapter windows, and real repaired regions;
- review the 46 transcript exceptions and 10 source-anchor audit cases, then
  review representative word boundaries with the required second pass;
- freeze the strict policy truth table before evaluating held-out audio; the
  risk-class sample counts and non-inferiority margins are now preregistered;
- fit independent thresholds for every engine and clip class, then run the
  held-out shadow comparison;
- run the frozen reference-M5 performance protocol and preserve its approved
  report;
- run the Qwen BF16/8-bit and Parakeet offline/streaming comparisons on the
  complete calibration and held-out corpora; the smoke comparisons pass;
- run the extended release preflight once from a clean release worktree so its
  three worker SBOMs become durable release artifacts; and
- collect reviewer approval for the deterministic source-anchor audit, every
  unresolved transcript, the final truth table, and the fitted thresholds; the
  durable, evaluation-bound review template and validator are implemented.

After those artifacts exist, Yakbox can evaluate the cutover-readiness report.
Only a report with every gate passed authorizes the manifest-v2, CLI, schema,
documentation, and public-API cutover described in the architecture plan.

Routine use will not require listening review. The remaining reviews are a
one-time qualification gate for a new policy, model revision, calibration
table, or execution class.
