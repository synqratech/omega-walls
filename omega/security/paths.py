"""Filesystem containment helpers.

All paths derived from untrusted identifiers must be resolved through this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


class PathContainmentError(ValueError):
    """Raised when a candidate path escapes its configured root."""


def _parts_are_safe(parts: Iterable[str]) -> None:
    for raw in parts:
        value = str(raw)
        if not value or value in {".", ".."}:
            raise PathContainmentError("path component must be non-empty and cannot be dot segments")
        candidate = Path(value)
        if candidate.is_absolute() or candidate.drive:
            raise PathContainmentError("absolute paths and drive-qualified paths are forbidden")
        if any(part in {"", ".", ".."} for part in candidate.parts):
            raise PathContainmentError("path traversal components are forbidden")


def resolve_contained_path(root: str | Path, *parts: str, create_parent: bool = False) -> Path:
    """Resolve a path and prove that it remains under ``root``.

    ``Path.resolve(strict=False)`` also collapses symlinked parent components that
    already exist, preventing a symlink escape through an otherwise clean name.
    """

    root_path = Path(root).expanduser().resolve(strict=False)
    if not root_path.is_absolute():  # pragma: no cover - resolve always returns absolute
        raise PathContainmentError("root must resolve to an absolute path")
    _parts_are_safe(parts)
    candidate = root_path.joinpath(*(str(x) for x in parts)).resolve(strict=False)
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise PathContainmentError("resolved path escapes configured root") from exc
    if candidate == root_path:
        raise PathContainmentError("candidate path must not equal the root directory")
    if create_parent:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        # Re-check after directory creation in case a concurrent actor replaced a
        # component with a symlink.
        resolved_after_create = candidate.resolve(strict=False)
        try:
            resolved_after_create.relative_to(root_path)
        except ValueError as exc:
            raise PathContainmentError("resolved path escaped root after parent creation") from exc
        candidate = resolved_after_create
    return candidate
