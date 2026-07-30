# M4B release

This project retains mastered WAV chapters, creates delivery MP3 chapters, and
enables an additional M4B assembly:

```console
yakbox validate
yakbox build
yakbox inspect
yakbox release check --write-manifest
yakbox assemble
```

FFmpeg and FFprobe are required. The M4B is an additional release artifact;
it does not replace the WAV masters or MP3 chapter set.
