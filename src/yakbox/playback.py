"""Open generated audio in the desktop's default player."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from yakbox.errors import ArtifactError, BuildError


def play_audio(path: Path) -> None:
    """Launch the default macOS or Linux player without mixing output into JSON."""
    audio = path.resolve()
    if not audio.is_file():
        raise ArtifactError(f"Audio is missing or not a file: {audio}")
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    try:
        subprocess.run(  # noqa: S603 - fixed desktop opener and absolute file path; no shell
            (opener, str(audio)),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BuildError(
            "Could not open the default audio player. "
            f"Generated audio is available at {audio}"
        ) from error
