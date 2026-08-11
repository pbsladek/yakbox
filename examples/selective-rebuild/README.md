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

For a problem in one spoken passage, use the smaller repair loop:

```console
yakbox --json plan
yakbox repair plan --chunk-id CHUNK_ID
yakbox repair generate --chunk-id CHUNK_ID
yakbox repair approve REPAIR_ID --take 1
```

The final command reconstructs the chapter from the approved take and cached
chunks. It doesn't resynthesize the rest of the chapter.
