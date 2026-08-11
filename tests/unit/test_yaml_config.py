from pathlib import Path

import pytest

from yakbox.errors import ValidationError
from yakbox.yaml_config import load_yaml


def test_load_yaml_accepts_yaml_and_yml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("enabled: true\ncount: 2\n", encoding="utf-8")
    yml_path = tmp_path / "config.yml"
    yml_path.write_text("items:\n  - one\n  - two\n", encoding="utf-8")

    assert load_yaml(yaml_path, description="Test configuration") == {
        "enabled": True,
        "count": 2,
    }
    assert load_yaml(yml_path, description="Test configuration") == {
        "items": ["one", "two"]
    }


def test_load_yaml_rejects_json_extensions(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"enabled": true}', encoding="utf-8")

    with pytest.raises(ValidationError, match=r"must use \.yaml or \.yml"):
        load_yaml(path, description="Test configuration")


def test_load_yaml_does_not_construct_python_objects(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.yaml"
    path.write_text(
        "value: !!python/object/apply:builtins.str [unsafe]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Cannot read test configuration"):
        load_yaml(path, description="Test configuration")


def test_load_yaml_rejects_duplicate_mapping_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("enabled: true\nenabled: false\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="found duplicate key 'enabled'"):
        load_yaml(path, description="Test configuration")
