# Tiny book

This example exercises the complete audiobook workflow without downloading a
model or contacting a hosted provider.

```console
yakbox validate
yakbox plan
yakbox audition --profile default --text "A short voice check."
yakbox build
yakbox status
yakbox artifacts list --voice narrator --limit 5
yakbox release check --write-manifest
```

To create a separate starter project with your own book metadata, run
`yakbox init ../my-book --title "My Book" --author "Author Name"`.
Add `--backend chatterbox-local` to create a local narration profile; install
its dependencies separately before generating audio.
`yakbox doctor --strict` treats diagnostic warnings as failures for automation.
Generate a completion script with `yakbox completion --shell zsh`; see the
[operations guide](../../docs/operations.md#automation-and-shell-completion)
for installation commands for Bash, Zsh, and Fish.

The fake backend creates test audio, not listening-quality narration. Replace
the profile in `yakbox.toml` with a local Chatterbox or Resemble profile when
you are ready to render a real book.

On a desktop, generate a sample and open it in your default audio player:

```console
yakbox preview --text "A short voice check." --play
yakbox audition --profile default --text "Another voice check." --play
```

While a build runs in another terminal, use `yakbox status --watch` to follow
its chapter and stage progress. Press Ctrl+C to stop watching.
