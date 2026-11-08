# Tiny book

This example exercises the complete audiobook workflow without downloading a
model or contacting a hosted provider.

```console
yakbox validate
yakbox plan
yakbox audition --profile default --text "A short voice check."
yakbox build
yakbox status --latest
yakbox --json status --target default --limit 5
yakbox artifacts list --voice narrator --limit 5
yakbox release check --write-manifest
```

To create a separate starter project with your own book metadata, run:

```console
yakbox init ../my-book --title "My Book" --author "Author Name" \
  --narrator "Narrator Name" --language en
```

Existing pronunciation and manuscript files are preserved during initialization.
Add `--backend chatterbox-local` to create a local narration profile; install
its dependencies separately before generating audio.
`yakbox doctor --strict` treats diagnostic warnings as failures for automation.
Generate a completion script with `yakbox completion --shell zsh`; see the
[operations guide](../../docs/operations.md#automation-and-shell-completion)
for installation commands for Bash, Zsh, and Fish.

The fake backend creates test audio, not listening-quality narration. Replace
the profile in `yakbox.toml` with a local Chatterbox profile when
you are ready to render a real book.

On a desktop, generate a sample and open it in your default audio player:

```console
yakbox preview --text "A short voice check." --play
yakbox audition --profile default --text "Another voice check." --play
```

While a build runs in another terminal, use `yakbox status --watch` to follow
its chapter and stage progress. Press Ctrl+C to stop watching.
Use `yakbox status --watch --interval 5 --exit-when-done` to refresh every five
seconds and exit when the latest run finishes (0 for complete, 1 otherwise).

Inspect the JSON plan for artifact IDs, then pass one to
`yakbox explain --artifact ID`. A misspelled artifact ID returns an error.
Shell completion suggests this manifest's targets and profiles. Unknown names
also offer close spelling suggestions when available.

To read sample text from a UTF-8 file, run
`yakbox preview --text-file source/book.md`. Use `--text-file -` for stdin;
unreadable files and invalid UTF-8 return a usage error with guidance.

To cancel the build itself, press Ctrl+C in its build terminal. Yakbox keeps
the target locked until media writers stop; run `yakbox build` again to resume.
See [failure recovery](../../docs/operations.md#failure-recovery) for artifact
validation and media inspection behavior.
