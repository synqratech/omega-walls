"""Filesystem source loading for RAG packets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Sequence

from omega.interfaces.contracts_v1 import ContentItem


TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".log"}


def _infer_source_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".md", ".rst", ".txt", ".log"}:
        return "web"
    if ext == ".pdf":
        return "pdf"
    return "other"


def _doc_id_for_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
    return f"doc-{digest}"


def load_content_items_from_directory(
    root_dir: str,
    trust: str = "untrusted",
    include_extensions: Sequence[str] | None = None,
) -> List[ContentItem]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root_dir}")

    extensions = set(e.lower() for e in (include_extensions or TEXT_EXTENSIONS))
    items: List[ContentItem] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        items.append(
            ContentItem(
                doc_id=_doc_id_for_path(path, root),
                source_id=f"file:{path.resolve().as_posix()}",
                source_type=_infer_source_type(path),
                trust=trust,
                text=text,
                meta={"path": str(path.resolve())},
            )
        )

    return items
