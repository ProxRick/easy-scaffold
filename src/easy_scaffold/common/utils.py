"""Shared helper utilities."""
from __future__ import annotations

from importlib import import_module
from typing import Any, Dict


def import_from_string(dotted_path: str) -> Any:
    """Import and return an attribute from a dotted module path."""
    module_path, attr = dotted_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, attr)


def get_from_nested_dict(data: Dict[str, Any], path: str) -> Any:
    """Retrieve a value from a nested dict/object chain using dot notation."""
    current: Any = data
    for key in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return None
    return current


def get_root_exception(exc: BaseException) -> BaseException:
    """Traverse __cause__ chain to find the originating exception."""
    current = exc
    while getattr(current, "__cause__", None) is not None:
        current = current.__cause__
    return current



