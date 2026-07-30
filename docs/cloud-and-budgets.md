# Resemble cloud, concurrency, and spending limits

## Credentials

Environment variables are best for CI and headless hosts:

```console
export RESEMBLE_API_KEY='...'
```

The optional credentials extra enables system-keyring profiles:

```console
uv tool install "yakbox[credentials]"
yakbox config auth login --profile studio
yakbox config auth status --profile studio
yakbox cloud --profile studio voices list
```

Credential precedence is deprecated `--api-key`, environment, selected
keyring profile, then a read-only legacy plaintext config token. Secrets are
not written to manifests, journals, reports, or command output.

## Direct provider operations

```console
yakbox cloud tts "A hosted line." \
  --voice-uuid VOICE_UUID --out line.wav
yakbox cloud stream "A short streamed line." \
  --voice-uuid VOICE_UUID --out stream.wav
yakbox cloud voices list
yakbox cloud voices recordings create VOICE_UUID sample.wav \
  --name "Sample" --text "Sample transcript"
yakbox cloud projects create "Audiobook Project"
```

Direct synthesis accepts at most 3,000 characters. Streaming accepts at most
2,000 and is not a long-text escape hatch. Use explicit rows/chunks or the
audiobook builder.

Management `GET` operations retry transient status and transport failures.
Project and recording mutations retry only definite rate limiting and
connection failures known to occur before sending. An ambiguous timeout or
server failure stops and asks you to verify provider state before manually
retrying, avoiding accidental duplicate records.

## Concurrent batch

The batch formats match local batch:

- `.txt`: one non-empty row per line;
- `.csv`: header plus required `text`; optional `id`, `voice_uuid`, `title`;
- `.jsonl`: one object per line using the same fields.

```console
yakbox cloud batch lines.csv \
  --voice-uuid VOICE_UUID \
  --out-dir cloud-output \
  --concurrency 5 \
  --max-submitted-characters 100000 \
  --max-provider-requests 100 \
  --yes
```

One async client and a bounded worker pool reuse connections. A shared
server-directed cooldown honors `Retry-After`. A failed row does not cancel
other rows. Journals are append-only and usage reservations are durable before
each provider attempt:

```console
yakbox cloud batch lines.csv --resume cloud-output/batch-journal.ndjson
```

Reports contain request/result hashes, ordered row status, attempts, safe
errors, usage counters, and preflight ranges. `--no-report` avoids final
`O(rows)` result materialization. `--ignore-errors` changes the exit code only
for row-local partial failure; it never hides an aborted/systemic failure.

Higher concurrency is not always faster once provider rate limits apply.
Configure a default with:

```toml
[cloud]
concurrency = 5
```

or `YAKBOX_CLOUD_CONCURRENCY`.

## Spending guardrails

Hosted commands and targets support character, provider-request, and estimated
spend limits. Monetary limits require a currency, pricing source, and explicit
price input; estimates are conservative guardrails, not invoices.

```console
yakbox build \
  --max-submitted-characters 200000 \
  --max-provider-requests 200 \
  --max-estimated-spend 25 \
  --currency USD \
  --pricing-source resemble-account-pricing \
  --price-per-character 0.0001
```

Retries consume fresh reservations because an ambiguous request may have been
billed. Resume restores those counters before any new send.
