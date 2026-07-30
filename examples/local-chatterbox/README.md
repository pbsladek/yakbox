# Local Chatterbox

This project uses one isolated local worker and conservative controls. Keep
the first audition short; the first run may download and load model files.

```console
uv tool install "yakbox[local]" \
  --overrides https://raw.githubusercontent.com/pbsladek/yakbox/v0.1.0/constraints/chatterbox-security-overrides.txt
yakbox doctor yakbox.toml --backend chatterbox-local --deep
yakbox validate
yakbox plan
yakbox audition --profile local --text "Hi."
yakbox build
```

Change `device` to the value appropriate for your machine. Never run multiple
workers against one GPU merely to increase CLI concurrency.
