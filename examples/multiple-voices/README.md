# Multiple voice profiles

Yakbox can audition several logical voice profiles and build separate editions
from the same manuscript:

```console
yakbox validate
yakbox audition --profile narrator --profile character --text "A short test."
yakbox build --target narrator-edition
yakbox build --target character-edition
```

One production target has one narration profile. This example creates two
complete editions; it does not silently infer speakers from dialogue.
