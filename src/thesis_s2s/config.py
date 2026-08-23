"""Load and validate project YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if text in {"true", "True", "yes"}:
        return True
    if text in {"false", "False", "no"}:
        return False
    if text in {"null", "None", "~"}:
        return None
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def _load_yaml_legacy(path: str | Path) -> dict[str, Any]:
    """Legacy restricted parser kept only to make old behavior auditable."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.split("#", 1)[0].rstrip()
        if not stripped.strip():
            i += 1
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        content = stripped.strip()
        while stack and indent <= stack[-1][0]:
            if indent == stack[-1][0] and not isinstance(stack[-1][1], list):
                break
            if indent < stack[-1][0]:
                stack.pop()
                continue
            break
        parent = stack[-1][1]
        if content.startswith("- "):
            item = content[2:].strip()
            if not isinstance(parent, list):
                raise ValueError(f"list item without list parent: {content}")
            if item.endswith(":") and ":" in item[:-1]:
                key, _, rest = item.partition(":")
                node: dict[str, Any] = {}
                parent.append(node)
                if rest.strip():
                    node[key.strip()] = _parse_scalar(rest)
                else:
                    stack.append((indent, node))
            else:
                parent.append(_parse_scalar(item))
            i += 1
            continue
        key, sep, rest = content.partition(":")
        if not sep:
            raise ValueError(f"cannot parse YAML line: {raw}")
        key = key.strip()
        rest = rest.strip()
        if rest == ">" or rest == "|":
            block: list[str] = []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip() or nxt.startswith(" " * (indent + 2)) or nxt.startswith("\t"):
                    block.append(nxt.strip())
                    i += 1
                    continue
                nxt_stripped = nxt.split("#", 1)[0]
                nxt_indent = len(nxt_stripped) - len(nxt_stripped.lstrip(" "))
                if nxt_indent > indent and nxt.strip():
                    block.append(nxt.strip())
                    i += 1
                    continue
                break
            parent[key] = " ".join(part for part in block if part)
            continue
        if rest == "":
            # Look ahead: list or mapping.
            j = i + 1
            child: Any = {}
            while j < len(lines) and not lines[j].split("#", 1)[0].strip():
                j += 1
            if j < len(lines):
                look = lines[j].split("#", 1)[0]
                look_indent = len(look) - len(look.lstrip(" "))
                if look.strip().startswith("- ") and look_indent > indent:
                    child = []
            parent[key] = child
            stack.append((indent, child))
            i += 1
            continue
        parent[key] = _parse_scalar(rest)
        i += 1
    return root


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


def config_path(name: str) -> Path:
    return project_root() / "configs" / name
