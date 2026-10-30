"""Central help text and discoverability rules for the Click interface."""

from __future__ import annotations

import click

COMMAND_HELP = {
    "artifacts list": "List managed artifacts and their recorded metadata.",
    "artifacts usage": "Summarize managed and unknown artifact storage usage.",
    "artifacts cache list": "List reusable synthesis-cache entries.",
    "artifacts verify": "Verify artifact sizes and digests against their sidecars.",
    "artifacts clean": "Plan or apply reversible cleanup of managed artifacts.",
    "artifacts trash list": "List quarantined cleanup entries available to restore.",
    "artifacts trash restore": "Restore one quarantined cleanup entry safely.",
    "artifacts trash purge": "Permanently delete selected quarantined entries.",
    "backends list": "Report configured speech backends and availability.",
    "backends capabilities": "Show the typed capabilities of one speech backend.",
    "shards export": "Export deterministic build shards for distributed work.",
    "shards verify": "Verify complete, non-overlapping build shard manifests.",
}

OPTION_HELP = {
    "apply_changes": "Delete the cache entries in the displayed cleanup plan.",
    "apply_plan": "Move the displayed cleanup candidates into quarantine.",
    "artifact": "Explain the plan node that owns this artifact path.",
    "backend": "Select a configured speech backend by name.",
    "chapter": "Select one chapter ID, comma list, or supported chapter range.",
    "count": "Split the plan into this many deterministic shards.",
    "deep": "Inspect local runtime and device readiness without loading a model.",
    "dry_run": "Validate and report planned work without writing output.",
    "from_stage": "Start at this build stage after verifying prerequisites.",
    "keep_runs": "Protect this many recent successful build runs from cleanup.",
    "kind": "Limit artifacts to this artifact-kind value.",
    "manifest": "Use this audiobook manifest when comparing releases.",
    "max_age_days": "Select cache entries at least this many days old.",
    "max_bytes": "Reduce selected cache storage to at most this many bytes.",
    "no_progress": "Disable the interactive progress display on stderr.",
    "older_than": "Select artifacts older than this many days.",
    "out": "Write the generated audio to this path.",
    "out_dir": "Write generated files beneath this directory.",
    "output_format": "Select the generated audio container format.",
    "overwrite": "Replace an existing destination only after successful generation.",
    "profile": "Override the manifest's configured backend profile.",
    "profiles": "Render each named backend profile; the option may be repeated.",
    "reference_audio": "Use this authorized reference recording for voice cloning.",
    "resume": "Resume an interrupted build when compatible journal state exists.",
    "sample_rate": "Request this output sample rate in hertz.",
    "target": "Select the named manifest build target.",
    "text": "Use this explicit text instead of selecting manuscript content.",
    "text_file": "Read UTF-8 text from this file, or use '-' for stdin.",
    "through_stage": "Stop after this build stage.",
    "voice": "Select the logical or backend voice name.",
    "yes": "Confirm an irreversible operation.",
}

OPTION_OVERRIDES = {
    ("batch", "backend"): "Select fake or chatterbox-local.",
    ("release diff", "target"): "Select the target within the supplied manifest.",
}


def configure_cli_help(root: click.Command) -> None:
    """Apply central help and default-display policy to every visible command."""

    def visit(command: click.Command, path: tuple[str, ...]) -> None:
        command_path = " ".join(path)
        if path and not command.help:
            command.help = COMMAND_HELP.get(command_path)
        for parameter in command.params:
            if not isinstance(parameter, click.Option) or parameter.hidden:
                continue
            if not parameter.help:
                parameter.help = OPTION_OVERRIDES.get(
                    (command_path, parameter.name),
                    OPTION_HELP.get(parameter.name),
                )
            default = parameter.default
            if type(default).__name__ != "Sentinel" and default not in (
                None,
                False,
                (),
            ):
                parameter.show_default = True
        if isinstance(command, click.Group):
            for name, child in command.commands.items():
                if not child.hidden:
                    visit(child, (*path, name))

    visit(root, ())
