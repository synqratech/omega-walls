"""Filesystem source loading for RAG packets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

from omega.interfaces.contracts_v1 import ContentItem
from omega.rag.attachment_ingestion import AttachmentChunk, extract_attachment


TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".log"}
ATTACHMENT_EXTENSIONS = {".pdf", ".docx", ".html", ".htm"}


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
    attachment_cfg: Dict[str, Any] | None = None,
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

        base_doc_id = _doc_id_for_path(path, root)
        source_id = f"file:{path.resolve().as_posix()}"
        source_type = _infer_source_type(path)

        ext = path.suffix.lower()
        if ext in ATTACHMENT_EXTENSIONS:
            extracted = extract_attachment(path=path, cfg=attachment_cfg)
            chunks = list(extracted.chunks)
            if not chunks:
                marker = "[attachment_scan_like]" if extracted.scan_like else "[attachment_text_empty]"
                chunks = [AttachmentChunk(text=marker, kind="placeholder", is_hidden=False)]
            for idx, chunk in enumerate(chunks):
                chunk_text = str(chunk.text).strip()
                if not chunk_text:
                    continue
                chunk_doc_id = f"{base_doc_id}-c{idx:03d}"
                items.append(
                    ContentItem(
                        doc_id=chunk_doc_id,
                        source_id=source_id,
                        source_type=source_type,
                        trust=trust,
                        text=chunk_text,
                        meta={
                            "path": str(path.resolve()),
                            "attachment_format": extracted.format,
                            "attachment_text_empty": bool(extracted.text_empty),
                            "attachment_scan_like": bool(extracted.scan_like),
                            "attachment_hidden_text_chars": int(extracted.hidden_text_chars),
                            "attachment_warnings": list(extracted.warnings),
                            "recommended_verdict": str(extracted.recommended_verdict),
                            "attachment_chunk_index": int(idx),
                            "attachment_chunk_kind": str(chunk.kind),
                            "attachment_chunk_hidden": bool(chunk.is_hidden),
                            "attachment_chunk_total": int(len(chunks)),
                        },
                    )
                )
            continue

        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        items.append(
            ContentItem(
                doc_id=base_doc_id,
                source_id=source_id,
                source_type=source_type,
                trust=trust,
                text=text,
                meta={"path": str(path.resolve())},
            )
        )

    return items
