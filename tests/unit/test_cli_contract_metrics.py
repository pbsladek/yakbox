from __future__ import annotations

from pathlib import Path

import click

from yakbox.cli import main

ROOT = Path(__file__).parents[2]

EXPECTED_LEAF_COMMANDS = {
    "artifacts cache clean",
    "artifacts cache list",
    "artifacts clean",
    "artifacts list",
    "artifacts trash list",
    "artifacts trash purge",
    "artifacts trash restore",
    "artifacts usage",
    "artifacts verify",
    "assemble",
    "audition",
    "backends capabilities",
    "backends list",
    "batch",
    "build",
    "cloud batch",
    "cloud projects create",
    "cloud projects list",
    "cloud stream",
    "cloud tts",
    "cloud voices list",
    "cloud voices recordings create",
    "config auth login",
    "config auth logout",
    "config auth status",
    "doctor",
    "explain",
    "init",
    "inspect",
    "models",
    "plan",
    "preview",
    "pronunciations audit",
    "release check",
    "release diff",
    "shards export",
    "shards verify",
    "short-review approve",
    "short-review list",
    "short-review play",
    "short-review reject",
    "short-test",
    "status",
    "tts",
    "validate",
    "vc",
    "verify",
    "whisper calibrate",
    "whisper inspect",
    "whisper inspect-joins",
    "whisper inspect-phonemes",
    "whisper models install",
    "whisper models path",
    "whisper models status",
    "whisper models verify",
    "whisper phoneme-models install",
    "whisper phoneme-models status",
    "whisper reinspect",
    "whisper verify-manuscript",
}


def _all_commands() -> tuple[tuple[str, click.Command], ...]:
    commands: list[tuple[str, click.Command]] = []

    def visit(command: click.Command, path: tuple[str, ...]) -> None:
        commands.append((" ".join(path), command))
        if isinstance(command, click.Group):
            for name, child in command.commands.items():
                if not child.hidden:
                    visit(child, (*path, name))

    visit(main, ())
    return tuple(commands)


def _commands() -> tuple[tuple[str, click.Command], ...]:
    return tuple(
        (path, command)
        for path, command in _all_commands()
        if not isinstance(command, click.Group)
    )


def test_public_command_tree_is_an_explicit_contract() -> None:
    assert {path for path, _command in _commands()} == EXPECTED_LEAF_COMMANDS


def test_every_public_command_and_option_has_semantic_help() -> None:
    missing_commands = [
        path or "<root>" for path, command in _all_commands() if not command.help
    ]
    missing_options = [
        f"{path} --{parameter.name.replace('_', '-')}"
        for path, command in _all_commands()
        for parameter in command.params
        if isinstance(parameter, click.Option)
        and not parameter.hidden
        and not parameter.help
    ]

    assert missing_commands == []
    assert missing_options == []


def test_every_meaningful_option_default_is_visible() -> None:
    hidden_defaults = [
        f"{path} --{parameter.name.replace('_', '-')}"
        for path, command in _all_commands()
        for parameter in command.params
        if isinstance(parameter, click.Option)
        and not parameter.hidden
        and type(parameter.default).__name__ != "Sentinel"
        and parameter.default not in (None, False, ())
        and not parameter.show_default
    ]

    assert hidden_defaults == []


def test_cli_module_size_and_hosted_policy_do_not_regress() -> None:
    source = (ROOT / "src" / "yakbox" / "cli.py").read_text(encoding="utf-8")
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src" / "yakbox").rglob("*.py")
    )

    assert len(source.splitlines()) <= 3_000
    assert ".resemble_api_key" not in production
