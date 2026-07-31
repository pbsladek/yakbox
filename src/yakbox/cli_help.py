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
    "cloud tts": "Synthesize one bounded text through Resemble.",
    "cloud stream": "Stream one bounded text from Resemble into an audio file.",
    "cloud batch": "Run guarded, resumable Resemble synthesis for a batch file.",
    "cloud voices list": "List Resemble voices with pagination.",
    "cloud voices recordings create": "Upload one recording to a Resemble voice.",
    "cloud projects list": "List Resemble projects with pagination.",
    "cloud projects create": "Create a Resemble project.",
    "config auth login": "Store a Resemble token in the operating-system keyring.",
    "config auth logout": "Remove a Resemble token from the operating-system keyring.",
    "config auth status": "Check whether a keyring credential profile exists.",
}

OPTION_HELP = {
    "active": "Create the recording as active; use --inactive to disable it.",
    "apply_changes": "Delete the cache entries in the displayed cleanup plan.",
    "apply_plan": "Move the displayed cleanup candidates into quarantine.",
    "archived": "Create the project archived or not archived.",
    "artifact": "Explain the plan node that owns this artifact path.",
    "backend": "Select a configured speech backend by name.",
    "chapter": "Select one chapter ID, comma list, or supported chapter range.",
    "collaborative": "Create the project as collaborative or private.",
    "concurrency": "Set the maximum number of concurrent provider operations.",
    "confirm_above_characters": (
        "Require --yes when estimated submitted characters exceed this count."
    ),
    "confirm_above_requests": (
        "Require --yes when estimated provider requests exceed this count."
    ),
    "count": "Split the plan into this many deterministic shards.",
    "currency": "Set the ISO-style currency code used by the spending estimate.",
    "custom_pronunciations": "Ask the provider to apply account pronunciations.",
    "deep": "Inspect local runtime and device readiness without loading a model.",
    "description": "Set the optional project description.",
    "dry_run": "Validate and report planned work without writing output.",
    "emotion": "Attach the optional provider emotion label to the recording.",
    "fill": "Request provider fill processing for the recording.",
    "from_stage": "Start at this build stage after verifying prerequisites.",
    "hd": "Request the provider's high-definition synthesis mode.",
    "ignore_errors": "Return exit 0 for row-local failures, but not systemic aborts.",
    "journal": "Write durable batch events to this NDJSON journal path.",
    "keep_runs": "Protect this many recent successful build runs from cleanup.",
    "kind": "Limit artifacts to this artifact-kind value.",
    "manifest": "Use this audiobook manifest when comparing releases.",
    "max_age_days": "Select cache entries at least this many days old.",
    "max_bytes": "Reduce selected cache storage to at most this many bytes.",
    "max_estimated_spend": "Reject work estimated above this monetary amount.",
    "max_provider_requests": "Reject work requiring more provider attempts.",
    "max_submitted_characters": "Reject work submitting more billable characters.",
    "name": "Set the provider-visible recording name.",
    "network": "Perform safe read-only provider discovery; never synthesize.",
    "no_progress": "Disable the interactive progress display on stderr.",
    "no_report": "Do not materialize or write the final batch report.",
    "older_than": "Select artifacts older than this many days.",
    "out": "Write the generated audio to this path.",
    "out_dir": "Write generated files beneath this directory.",
    "output_format": "Select the generated audio container format.",
    "overwrite": "Replace an existing destination only after successful generation.",
    "page": "Request this one-based provider result page.",
    "page_size": "Request at most this many provider results per page.",
    "precision": "Select the provider sample precision.",
    "price_per_character": "Estimate spend using this currency amount per character.",
    "pricing_source": "Identify the account pricing source used for the estimate.",
    "profile": "Override the manifest's configured backend profile.",
    "profiles": "Render each named backend profile; the option may be repeated.",
    "project_uuid": "Associate provider synthesis with this project UUID.",
    "reference_audio": "Use this authorized reference recording for voice cloning.",
    "report": "Write the final batch report to this JSON path.",
    "resume": "Resume an interrupted build when compatible journal state exists.",
    "sample_rate": "Request this output sample rate in hertz.",
    "target": "Select the named manifest build target.",
    "text": "Use this explicit text instead of selecting manuscript content.",
    "text_file": "Read UTF-8 text from this file, or use '-' for stdin.",
    "through_stage": "Stop after this build stage.",
    "title": "Set the optional provider synthesis title.",
    "token": "Store this token; omit it to be prompted without terminal echo.",
    "voice": "Select the logical or backend voice name.",
    "voice_uuid": "Use this Resemble voice UUID.",
    "yes": "Confirm reviewed hosted work or an irreversible operation.",
}

OPTION_OVERRIDES = {
    ("batch", "backend"): "Select a local backend; hosted aliases are rejected.",
    ("cloud batch", "resume"): "Resume from a compatible cloud batch journal.",
    ("config auth login", "profile"): "Name the OS-keyring credential profile.",
    ("config auth logout", "profile"): "Name the OS-keyring credential profile.",
    ("config auth status", "profile"): "Name the OS-keyring credential profile.",
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
