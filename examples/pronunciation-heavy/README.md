# Pronunciation-heavy manuscript

This example keeps reviewed pronunciations in a typed TOML sidecar and uses
speech-only markup where printed and spoken text differ.

```console
yakbox validate
yakbox plan
yakbox audition --profile default --chapter 0001
yakbox build
```

Edit a rule, rerun `yakbox plan`, and use `yakbox explain --chapter 0001` to
see the affected fingerprints before rebuilding.
