"""Dispatch only a new, merged manifest version, including bootstrap safety."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path


def main() -> None:
    """Emit a tag only when an existing manifest advances with the project."""
    previous = os.environ["PREVIOUS_COMMIT"]
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", previous) is None:
        raise SystemExit("Invalid previous release commit")
    manifest = ".release-please-manifest.json"
    git = shutil.which("git")
    if git is None:
        raise SystemExit("Git is required to resolve the merged release")
    result = subprocess.run(  # noqa: S603 - validated Git revision and fixed path
        [git, "show", f"{previous}:{manifest}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode:
        return  # Initial manifest adoption does not publish a historical version.
    version = json.loads(Path(manifest).read_text())["."]
    old_version = json.loads(result.stdout)["."]
    if version == old_version:
        return
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    if version != project["version"] or re.fullmatch(r"\d+\.\d+\.\d+", version) is None:
        raise SystemExit("Release manifest and project versions must match")
    if tuple(map(int, version.split("."))) <= tuple(map(int, old_version.split("."))):
        raise SystemExit("Release version must advance")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
        stream.write(f"tag=v{version}\n")


if __name__ == "__main__":
    main()
