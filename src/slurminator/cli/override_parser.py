"""Parse dotted-key command-line override strings."""

from __future__ import annotations

import re
from typing import Any

import yaml

# Keep lowercase "none" as a literal string: projects may use it as an enum.
_NULL_RE = re.compile(r"^(?:[Nn][Uu][Ll][Ll]|~|None)$")


def parse_override_list(kv_items: list[str] | str) -> dict[str, Any]:
    """Convert ``KEY=VALUE`` override strings into a dictionary.

    Separators may be semicolons, commas, or whitespace at top level. Commas
    and spaces inside quoted strings, lists, tuples, or mappings are preserved.
    Values are coerced through YAML scalar parsing for booleans, numbers,
    explicit null spellings, lists, and mappings.
    """

    overrides: dict[str, Any] = {}
    if not kv_items:
        return overrides

    if isinstance(kv_items, str):
        items_list = _split_override_items(kv_items)
    elif len(kv_items) == 1 and (";" in kv_items[0] or "," in kv_items[0] or " " in kv_items[0]):
        items_list = _split_override_items(kv_items[0])
    else:
        items_list = kv_items

    for item in items_list:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be in KEY=VALUE format")

        key, raw_val = item.split("=", 1)
        try:
            value = _coerce_scalar(raw_val.strip())
        except Exception as exc:
            raise ValueError(f"Could not parse value '{raw_val}' for key '{key}': {exc}") from None
        overrides[key.strip()] = value

    return overrides


def _split_override_items(raw: str) -> list[str]:
    """Split an override string into top-level ``KEY=VALUE`` items."""
    items: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    escaped = False
    depth_sq = 0
    depth_par = 0
    depth_curly = 0

    for ch in str(raw):
        if quote is not None:
            buf.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue

        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            continue

        if ch == "[":
            depth_sq += 1
        elif ch == "]":
            depth_sq = max(0, depth_sq - 1)
        elif ch == "(":
            depth_par += 1
        elif ch == ")":
            depth_par = max(0, depth_par - 1)
        elif ch == "{":
            depth_curly += 1
        elif ch == "}":
            depth_curly = max(0, depth_curly - 1)

        is_sep = ch == ";" or ch == "," or ch.isspace()
        if is_sep and depth_sq == 0 and depth_par == 0 and depth_curly == 0:
            token = "".join(buf).strip()
            if token:
                items.append(token)
            buf = []
        else:
            buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        items.append(tail)
    return items


def _coerce_scalar(value: Any) -> Any:
    """Convert YAML/JSON scalar strings to Python values."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if _NULL_RE.match(stripped):
        return None
    try:
        if (
            stripped.startswith(("[", "{", '"', "'"))
            or stripped.endswith(("]", "}"))
            or stripped.lower() in ("true", "false")
            or stripped.replace(".", "", 1).isdigit()
            or (stripped.startswith("-") and stripped[1:].replace(".", "", 1).isdigit())
        ):
            return yaml.safe_load(stripped)
    except yaml.YAMLError:
        return value
    return value


__all__ = ["parse_override_list"]
