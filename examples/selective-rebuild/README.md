# Selective rebuild

Build the project once, then edit only the text under `# Second Stop`:

```console
yakbox build
yakbox plan
yakbox explain --chapter 0002
yakbox build
```

Only the changed chapter and its dependent mastering, MP3, and inspection
nodes are regenerated. Digest-verified artifacts for the other chapters are
reused.
