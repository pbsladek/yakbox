# Resemble.ai

Replace `REPLACE_WITH_RESEMBLE_VOICE_UUID` in `yakbox.toml`, then provide the
API key through the environment or a configured keyring profile.

```console
export RESEMBLE_API_KEY=...
yakbox doctor yakbox.toml --backend resemble --network
yakbox validate
yakbox plan
yakbox audition --profile hosted --text "Hi."
yakbox build --yes
```

The manifest bounds provider requests and submitted characters. Planning and
validation remain offline and make no billable request.
