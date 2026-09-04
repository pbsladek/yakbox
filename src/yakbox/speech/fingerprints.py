"""Stable identities for backend-neutral speech generation requests."""

from __future__ import annotations

import hashlib
import json

from yakbox._files import sha256_file
from yakbox.speech.models import SpeechSynthesisRequest


def speech_request_fingerprint(request: SpeechSynthesisRequest) -> str:
    """Fingerprint only inputs that can alter synthesized audio bytes."""
    from yakbox.fingerprints import (  # noqa: PLC0415 - avoids audiobook cycle
        backend_runtime_fingerprint,
    )

    chatterbox = request.chatterbox
    payload = {
        "version": 1,
        "text": request.text,
        "voice": request.voice,
        "backend": request.backend,
        "backend_runtime": backend_runtime_fingerprint(request.backend),
        "output_format": request.output_format.value,
        "sample_rate": request.sample_rate,
        "use_hd": request.use_hd,
        "precision": request.precision,
        "apply_custom_pronunciations": request.apply_custom_pronunciations,
        "project": request.project,
        "reference_audio_sha256": (
            sha256_file(request.reference_audio) if request.reference_audio else None
        ),
        "chatterbox": (
            {
                "cfg_weight": chatterbox.cfg_weight,
                "exaggeration": chatterbox.exaggeration,
                "seed": chatterbox.seed,
            }
            if chatterbox is not None
            else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["speech_request_fingerprint"]
