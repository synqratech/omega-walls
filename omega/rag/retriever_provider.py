"""Retriever provider abstractions for production-oriented adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Protocol

from omega.interfaces.contracts_v1 import ContentItem
from omega.rag.retriever_fts import SQLiteFTSRetriever
from omega.rag.sources_fs import load_content_items_from_directory


class RetrieverProvider(Protocol):
    def search(self, query: str, k: int) -> Iterable[Dict[str, Any]]:
        ...


class ExternalRetrieverClient(Protocol):
    def search(self, query: str, k: int) -> Iterable[Any]:
        ...


def _to_dict(hit: Any) -> Dict[str, Any]:
    if isinstance(hit, dict):
        return hit
    if hasattr(hit, "__dict__"):
        return dict(hit.__dict__)
    raise TypeError("External hit must be a dict-like object")


def _infer_source_type_from_relpath(root: Path, meta_path: str, fallback: str) -> str:
    try:
        rel = Path(str(meta_path)).resolve().relative_to(root).as_posix()
    except Exception:
        return fallback
    subdir = rel.split("/", 1)[0].lower()
    if subdir == "trusted":
        return "wiki"
    if subdir == "semi_trusted":
        return "ticket"
    if subdir == "untrusted":
        return "web"
    return fallback


@dataclass
class SQLiteFTSProvider:
    retriever: SQLiteFTSRetriever

    @classmethod
    def from_directory(
        cls,
        root_dir: str,
        homoglyph_map: Dict[str, str],
        default_trust: str = "untrusted",
        include_extensions: List[str] | None = None,
        attachment_cfg: Mapping[str, Any] | None = None,
    ) -> "SQLiteFTSProvider":
        root = Path(root_dir).resolve()
        raw_items = load_content_items_from_directory(
            root_dir=root_dir,
            trust=default_trust,
            include_extensions=include_extensions,
            attachment_cfg=dict(attachment_cfg or {}),
        )
        corpus: List[ContentItem] = []
        for item in raw_items:
            source_type = item.source_type
            meta_path = (item.meta or {}).get("path")
            if meta_path:
                source_type = _infer_source_type_from_relpath(root=root, meta_path=str(meta_path), fallback=source_type)
            corpus.append(
                ContentItem(
                    doc_id=item.doc_id,
                    source_id=item.source_id,
                    source_type=source_type,
                    trust=item.trust,
                    text=item.text,
                    language=item.language,
                    meta=item.meta,
                )
            )
        return cls(retriever=SQLiteFTSRetriever(corpus=corpus, homoglyph_map=homoglyph_map))

    def search(self, query: str, k: int) -> Iterable[Dict[str, Any]]:
        rows = self.retriever.retrieve(query=query, top_k=k)
        out: List[Dict[str, Any]] = []
        for item in rows:
            out.append(
                {
                    "doc_id": item.doc_id,
                    "text": item.text,
                    "source_id": item.source_id,
                    "source_type": item.source_type,
                    "metadata": dict(item.meta or {}),
                }
            )
        return out


@dataclass
class ExternalRetrieverProvider:
    client: ExternalRetrieverClient

    def search(self, query: str, k: int) -> Iterable[Dict[str, Any]]:
        return [_to_dict(hit) for hit in self.client.search(query, k)]
