"""Load and validate project YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping using the complete, safe YAML parser.

    Configuration errors fail early instead of silently changing key nesting.
    """

    config_file = Path(path)
    with config_file.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"top-level YAML value must be a mapping: {config_file}")
    return payload


def project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def portable_path(path: str | Path, *, root: str | Path | None = None) -> str:
    """Serialize project-local paths without leaking a workstation prefix."""

    candidate = Path(path)
    project = Path(root or project_root())
    try:
        return candidate.resolve().relative_to(project.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def portable_project_values(value: Any, *, root: str | Path | None = None) -> Any:
    """Recursively make absolute paths inside the project repository portable."""

    project = Path(root or project_root()).resolve()
    if isinstance(value, dict):
        return {
            key: portable_project_values(item, root=project)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [portable_project_values(item, root=project) for item in value]
    if isinstance(value, tuple):
        return tuple(portable_project_values(item, root=project) for item in value)
    if isinstance(value, Path):
        return portable_path(value, root=project)
    if isinstance(value, str):
        prefix = project.as_posix().rstrip("/") + "/"
        if value.startswith(prefix):
            return value.removeprefix(prefix)
    return value


def config_path(name: str) -> Path:
    return project_root() / "configs" / name
