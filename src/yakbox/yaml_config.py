"""Safe YAML loading for user-authored Yakbox configuration inputs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from yakbox.errors import ValidationError

YAML_SUFFIXES = frozenset({".yaml", ".yml"})


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe loader that rejects ambiguous duplicate mapping keys."""

    def construct_mapping(
        self,
        node: MappingNode,
        deep: bool = False,
    ) -> dict[object, object]:
        self.flatten_mapping(node)
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as error:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def load_yaml(path: Path, *, description: str) -> object:
    """Load one YAML document without enabling arbitrary object construction."""
    if path.suffix.casefold() not in YAML_SUFFIXES:
        raise ValidationError(f"{description} must use .yaml or .yml")
    try:
        return yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,  # noqa: S506 - extends SafeLoader only
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValidationError(
            f"Cannot read {description.lower()} {path}: {error}"
        ) from error


def iter_yaml_documents(
    stream: TextIO,
    *,
    description: str,
) -> Iterator[object]:
    """Stream safely loaded YAML documents with domain-level parse errors."""
    try:
        yield from yaml.load_all(stream, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise ValidationError(f"Invalid {description}: {error}") from error
