# Python API

The CLI is a thin adapter over typed application services. Python callers do
not invoke or scrape the CLI.

## Plan and build

```python
import asyncio
from pathlib import Path

from yakbox.audiobook import (
    build_audiobook,
    load_manifest,
    normalize_sources,
    plan_audiobook,
)

manifest = load_manifest(Path("yakbox.toml"))
document = normalize_sources(
    manifest.sources,
    pronunciations=manifest.pronunciations,
    max_pause_ms=manifest.max_pause_ms,
)
plan = plan_audiobook(manifest, document, target_name="default")
print(plan.fingerprint)

result = asyncio.run(build_audiobook(manifest, target_name="default"))
print(result.run_id)
```

Planning is pure with respect to models, workers, network, and artifacts.
Build operations use target locks, append-only journals, atomic files, and
digest-verified reuse.

## Direct speech service

```python
import asyncio
from pathlib import Path

from yakbox.speech import AudioFormat, SpeechSynthesisRequest, open_speech_backend


async def render() -> None:
    async with open_speech_backend("fake") as service:
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

## Typed Resemble client

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

Construct one client per operation/run and pass it through concurrent work.
Never create one client per row. Batch callers should prefer the exported
batch application service because it adds durable journals, cancellation,
bounded concurrency, spending reservations, and ordered reports.

Expected application failures derive from `YakboxError`. Cancellation is
control flow and is not converted into a row/provider error.
