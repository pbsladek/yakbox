# Live backend canaries

Normal tests never contact a provider, download a model, or synthesize with
real Chatterbox. Live canaries are separately marked, disabled by default, and
use only the text `Hi.`.

## Resemble

The hosted canary permits exactly one provider attempt and three submitted
characters. Retries are disabled, the connection pool contains one
connection, and the whole operation has a 90-second bound.

```console
YAKBOX_RUN_RESEMBLE_LIVE=1 \
RESEMBLE_API_KEY=... \
YAKBOX_CLOUD_VOICE_UUID=... \
uv run pytest --force-enable-socket -m live \
  tests/live/test_resemble_live.py
```

The `Live backend canary` GitHub workflow runs this test through the protected
`live-resemble` environment. Configure `RESEMBLE_API_KEY` and
`RESEMBLE_VOICE_UUID` as environment secrets.

## Local Chatterbox

The local canary uses one isolated worker, one thread, one short request, and a
bounded timeout:

```console
uv sync --frozen --all-groups --extra local
YAKBOX_RUN_LOCAL_LIVE=1 \
YAKBOX_LIVE_LOCAL_DEVICE=auto \
uv run pytest --force-enable-socket -m live \
  tests/live/test_local_chatterbox_live.py
```

The GitHub job requires a deliberately configured self-hosted runner carrying
the `yakbox-chatterbox` label. It does not consume a general hosted runner or
attempt a whole chapter.

## Local narration E2E

The canary does not assess audiobook behavior or narration quality. The
separate opt-in local E2E suite runs the real CLI across validation, planning,
twenty-five-voice auditioning, a full raw/mastered/MP3 build, inspection, and cache
reuse. It preserves a technical report and human listening scorecard. See
[Local narration QA](local-narration-qa.md) for the command, corpus, metrics,
and review procedure.
