"""Execute the packaged analysis worker from an installed Yakbox wheel."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from yakbox.speech.analysis_protocol import parse_worker_handshake
from yakbox.speech.analysis_runtime import BUILT_IN_WORKERS
from yakbox.speech.analysis_runtime_install import load_runtime_project
from yakbox.speech.analysis_scheduler import build_worker_handshake
from yakbox.speech.analysis_worker_artifact import verify_packaged_worker_artifact


def main() -> int:
    """Run every family handshake from the exact wheel-owned zip application."""
    artifact = verify_packaged_worker_artifact()
    environment = {
        "PATH": os.defpath,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    with tempfile.TemporaryDirectory(prefix="yakbox-wheel-worker-") as temporary:
        root = Path(temporary)
        for family, definition in BUILT_IN_WORKERS.items():
            project = load_runtime_project(family)
            completed = subprocess.run(  # noqa: S603 - wheel-owned fixed argv
                [
                    sys.executable,
                    "-I",
                    str(artifact.path),
                    "--family",
                    family,
                    "--audio-root",
                    str(root / "audio"),
                    "--model-root",
                    str(root / "models"),
                    "--calibration-fingerprint",
                    "0" * 64,
                    "--worker-artifact-digest",
                    artifact.sha256,
                    "--lock-digest",
                    project.lock_digest,
                    "--definition-fingerprint",
                    definition.fingerprint,
                ],
                input=b"",
                check=False,
                capture_output=True,
                cwd=root,
                env=environment,
                timeout=30,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{family} wheel worker failed: "
                    + completed.stderr.decode("utf-8", errors="replace")[:512]
                )
            lines = completed.stdout.splitlines()
            if len(lines) != 1:
                raise RuntimeError(f"{family} wheel worker emitted unexpected output")
            actual = parse_worker_handshake(lines[0])
            expected = build_worker_handshake(
                family=family,
                engines=definition.engines,
                worker_artifact_fingerprint=artifact.sha256,
                environment_lock_fingerprint=project.lock_digest,
                adapter_fingerprint=definition.fingerprint,
            )
            if actual != expected:
                raise RuntimeError(f"{family} wheel worker handshake differs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
