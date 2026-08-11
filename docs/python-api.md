# Python SDK

Yakbox exposes typed application services for programs that need more control
than the CLI provides. Python callers should import from the package facades
listed below. Modules beneath those facades are implementation details unless a
public type explicitly names them.

The wheel includes `py.typed`. Public signatures, enum members, exception
codes, and `__all__` exports are checked against
`tests/public-api-v1.yaml`. During the 0.x series, an incompatible change may
ship only in a minor release. A deprecated public name remains functional for
at least two minor releases and 90 days. Patch releases preserve these
contracts.

## Errors

Catch `YakboxError` when an application can handle any expected Yakbox
failure. More specific exceptions let callers separate invalid input,
configuration, unavailable backends, failed builds, and unsafe artifact work.

```python
from yakbox import BackendUnavailableError, ValidationError, YakboxError

try:
    ...
except ValidationError as error:
    print(f"invalid input: {error}")
except BackendUnavailableError as error:
    print(f"backend unavailable: {error}")
except YakboxError as error:
    print(f"yakbox operation failed: {error}")
```

Expected exceptions have stable snake-case `code` values for logs and service
boundaries. Cancellation remains control flow and is not converted into an
application error.

The root `yakbox` facade exports `ArtifactError`,
`BackendUnavailableError`, `BuildError`, `ConfigurationError`,
`ValidationError`, `YakboxError`, and `__version__`.

## Planning and building audiobooks

Load manifests through `load_manifest`; it validates paths and cross-references
before returning an `AudiobookManifest`. Planning does not load models, contact
a provider, or write build artifacts.

```python
import asyncio
from pathlib import Path

from yakbox.audiobook import (
    BuildProgress,
    BuildRequest,
    load_manifest,
    normalize_sources,
    plan_audiobook,
    run_audiobook_build,
)

manifest = load_manifest(Path("yakbox.toml"))
document = normalize_sources(
    manifest.sources,
    pronunciations=manifest.pronunciations,
    max_pause_ms=manifest.max_pause_ms,
)
plan = plan_audiobook(manifest, document, target_name="default")
print(plan.fingerprint)


def progress(event: BuildProgress) -> None:
    print(event.event.value, event.stage.value, event.completed, event.total)


request = BuildRequest(target_name="default", dry_run=False, resume=True)
result = asyncio.run(run_audiobook_build(manifest, request, progress=progress))
print(result.status.value, result.run_id)
```

`BuildRequest` is the preferred evolvable call surface. The keyword-oriented
`build_audiobook` function remains public for compatibility. Hosted Python
callers pass a credential explicitly in `BuildRequest.api_key`; unlike the CLI,
the SDK does not read environment variables or keyrings implicitly.

Builds use a target lock, append-only journal, atomic file commits, and
digest-verified reuse. `BuildProgressEvent`, `BuildStage`, and `BuildStatus`
define the stable state values returned to callers.

The audiobook facade also exposes focused services for pronunciation audits,
preflight, selective rebuilds, previews, auditions, cleanup, cache management,
sharding, and immutable releases. Cleanup follows a plan/apply model:
`plan_cleanup` and `plan_cache_cleanup` are read-only; `apply_cleanup` moves
artifacts into quarantine; `purge_trash` permanently deletes them.

### Audiobook reference

Manifest and backend models: `AudiobookManifest`, `BookMetadata`,
`LogicalVoice`, `BackendOptions`, `BackendProfile`, `FakeOptions`,
`ChatterboxOptions`, `ResembleOptions`, `CharacterRole`, `DialoguePolicy`,
`BuildTarget`, `RetentionPolicy`, and `RepairPolicy`.

Normalized source and planning models: `Chapter`, `Pause`, `SpeechSegment`,
`SourceLocation`, `NormalizedDocument`, `BuildStage`, `PlanNode`, `BuildPlan`,
`ChunkRoute`, `AttributionFinding`, `BuildChangeSummary`, and `BuildPreflight`.

Localized repair models: `RepairMode`, `RepairChunk`, `RepairPlan`, `RepairTake`,
`RepairSession`, and `RepairApproval`.

Build models and callbacks: `BuildRequest`, `BuildResult`, `BuildStatus`,
`BuildProgress`, `BuildProgressEvent`, and `BuildProgressCallback`.

Artifact, cache, cleanup, release, and shard models: `ArtifactKind`,
`ArtifactRecord`, `InventoryReport`, `CacheEntry`, `CacheInventory`,
`CacheCleanupPlan`, `CleanupCandidate`, `CleanupPlan`, `ReleaseCheck`,
`ReleaseDiff`, and `ShardManifest`.

Pronunciation models: `PronunciationAudit` and `PronunciationRuleAudit`.

Application functions: `load_manifest`, `normalize_sources`,
`plan_audiobook`, `preflight_audiobook_build`, `run_audiobook_build`,
`build_audiobook`, `select_build_chapters`, `preview_audiobook`,
`audition_audiobook`, `audit_pronunciations`, `inventory_artifacts`,
`repair_artifact_metadata`, `inventory_synthesis_cache`,
`plan_cache_cleanup`, `apply_cache_cleanup`, `plan_cleanup`, `apply_cleanup`,
`restore_trash`, `purge_trash`, `assemble_release`, `check_release`,
`diff_releases`, `export_shard_manifests`, `verify_shard_manifests`,
`plan_repair`, `generate_repair_session`, `approve_repair_session`, and
`explain_synthesis_chunk`.

## Direct speech services

`open_configured_speech_backend` is the preferred factory because its typed
options object can grow without lengthening the function signature.

```python
import asyncio
from pathlib import Path

from yakbox.speech import (
    AudioFormat,
    SpeechBackendOptions,
    SpeechSynthesisRequest,
    open_configured_speech_backend,
)


async def render() -> None:
    options = SpeechBackendOptions()
    async with open_configured_speech_backend("fake", options) as service:
        artifact = await service.synthesize_to_file(
            SpeechSynthesisRequest(
                text="A short typed example.",
                voice="narrator",
                backend="fake",
                output_format=AudioFormat.WAV,
            ),
            Path("example.wav"),
        )
        print(artifact.sha256)


asyncio.run(render())
```

The legacy keyword factory `open_speech_backend` remains public.
`open_transformation_backend` opens voice-conversion services. All three are
async context managers; do not retain a service after its context exits.

### Speech reference

Request and result types: `AudioFormat`, `Precision`,
`ChatterboxSynthesisOptions`, `SpeechSynthesisRequest`,
`SpeechTransformationRequest`, `SpeechArtifact`, `BackendCapabilities`, and
`SpeechBackendOptions`.

Hosted guardrail types and functions: `CurrencyCode`, `PricingSourceId`,
`HostedUsageBudget`, `HostedUsageSnapshot`, `HostedWorkEstimate`,
`estimate_hosted_work`, `hosted_confirmation_reasons`, and
`validate_hosted_preflight`.

`CurrencyCode` strips whitespace, normalizes to three uppercase ASCII letters,
and rejects other values. `PricingSourceId` strips whitespace and rejects an
empty identifier. Construct these value types before creating a spending
budget so invalid pricing metadata fails at the application boundary.

Service protocols and implementations: `TextToSpeechService`,
`BatchTextToSpeechService`, `SpeechTransformationService`,
`HostedUsageReportingService`, `HostedUsageJournalingService`,
`HostedUsageRecorder`, and `FakeSpeechService`.

Factories: `open_configured_speech_backend`, `open_speech_backend`, and
`open_transformation_backend`.

## Resemble client and hosted batches

Construct one `ResembleClient` per operation or run. The client owns its HTTP
connection pool unless an `httpx.AsyncClient` is injected. Enter it once and
close it through `async with`.

```python
import asyncio
import os
from pathlib import Path

from yakbox.cloud import ResembleClient, SynthesisRequest


async def render_hosted() -> None:
    async with ResembleClient(os.environ["RESEMBLE_API_KEY"]) as client:
        result = await client.synthesize_to_file(
            SynthesisRequest(text="A short line.", voice_uuid="VOICE_UUID"),
            Path("hosted.wav"),
        )
        print(result.request_id, result.attempts)


asyncio.run(render_hosted())
```

`run_cloud_batch` is the public batch application service. It accepts
`BatchRow` values and an open `TextToSpeechService`; it adds bounded
concurrency, durable journals, resume validation, ordered reports, spending
reservations, and cancellation recovery. Reuse one service across every row.

```python
from pathlib import Path

from yakbox.cloud import BatchRow, run_cloud_batch
from yakbox.speech import TextToSpeechService

rows = (
    BatchRow(index=1, row_id="intro", text="Opening line."),
    BatchRow(index=2, row_id="chapter", text="Chapter one."),
)


async def render_batch(service: TextToSpeechService) -> None:
    report = await run_cloud_batch(
        rows,
        service,
        default_voice="VOICE_UUID",
        project_uuid=None,
        out_dir=Path("cloud-output"),
    )
    print(report.ok, report.failed)
```

`ResembleClient` provides `synthesize`, `synthesize_to_file`, `stream`,
`stream_to_file`, `list_voices`, `create_recording`, `list_projects`,
`create_project`, and `aclose`. When a `HostedUsageGate` is attached,
`usage_snapshot`, `set_usage_recorder`, and `restore_usage` expose the durable
accounting lifecycle used by resumed batches. `ResembleSpeechService` provides
the neutral `synthesize_to_file` and `synthesize_many_to_files` methods plus
the same usage lifecycle and `aclose`.

### Cloud reference

Client configuration and lifecycle: `ResembleClient`, `ClientOptions`,
`RetryPolicy`, `HostedUsageGate`, and `ResembleSpeechService`.

Provider requests and results: `AudioFormat`, `Precision`, `SynthesisRequest`,
`StreamRequest`, `SynthesisResult`, `FileSynthesisResult`, `Voice`, `Project`,
`Recording`, and `Page`.

Batch API: `BatchRow`, `BatchStatus`, `BatchResult`, `BatchReport`,
`ProgressCallback`, and `run_cloud_batch`.

Cloud exceptions: `CloudError`, `ClientStateError`, `ProviderError`,
`ProviderProtocolError`, `AmbiguousMutationError`, `RetryExhaustedError`,
`HostedBudgetExceeded`, `BatchJournalError`, and `ResumeMismatchError`.

## Audio processing

The `yakbox.audio` facade provides FFmpeg- and FFprobe-backed operations.
Destinations are atomic and are not replaced unless `overwrite=True`.

- `master_wav` masters a WAV file.
- `encode_mp3` writes an MP3 with optional book metadata.
- `assemble_m4b` assembles ordered chapters and metadata into an M4B.
- `inspect_audio` returns an `AudioInspection` evaluated against an optional
  `AudioQualityPolicy`.

These operations require the matching external tools on `PATH` and can raise
`ArtifactError` or an operating-system exception when a caller supplies an
unusable path.

## Diagnostics

`run_doctor` returns a typed `DoctorReport`. Each `Diagnostic` has a
`DiagnosticStatus`, `DiagnosticSeverity`, summary, optional action, timing, and
bounded evidence. Network checks are opt-in and read-only. Deep local checks do
not synthesize audio. Python callers may pass `api_key` explicitly when a
credential or network check should include a hosted backend.

```python
import asyncio
from yakbox.diagnostics import run_doctor

report = asyncio.run(run_doctor(backend="fake"))
for diagnostic in report.diagnostics:
    print(diagnostic.status.value, diagnostic.id, diagnostic.summary)
```

The diagnostics facade exports `Diagnostic`, `DiagnosticSeverity`,
`DiagnosticStatus`, `DoctorReport`, and `run_doctor`.

## Serialization and compatibility

Result models that cross process or durable-file boundaries expose `to_dict()`
and carry a schema URI and version where applicable. Treat those dictionaries
as the serialization contract. Do not serialize dataclasses with `asdict()`
unless the model explicitly documents that representation.

Public enum values and stable exception codes are compatibility-sensitive.
Callers should compare enum members or their `.value`, never class names or
human-readable exception messages. The CLI envelope is a separate contract;
Python programs should call these services instead of invoking or scraping the
CLI.
