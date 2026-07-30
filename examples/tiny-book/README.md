# Tiny book

This example exercises the complete audiobook workflow without downloading a
model or contacting a hosted provider.

```console
yakbox validate
yakbox plan
yakbox audition --profile default --text "A short voice check."
yakbox build
yakbox status
yakbox release check --write-manifest
```

The fake backend creates test audio, not listening-quality narration. Replace
the profile in `yakbox.toml` with a local Chatterbox or Resemble profile when
you are ready to render a real book.
