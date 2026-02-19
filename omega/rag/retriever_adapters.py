"""Thin adapters to bridge real retrievers to the Omega ContentItem contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol

from omega.interfaces.contracts_v1 import ContentItem
from omega.rag.retriever_fts import SQLiteFTSRetriever
from omega.rag.source_policy import SourceTrustPolicy
from omega.rag.sources_fs import load_content_items_from_directory


class ExternalRetrieverClient(Protocol):
    def search(self, query: str, k: int) -> Iterable[Any]:
        ...


def _to_dict(hit: Any) -> Dict[str, Any]:
    if isinstance(hit, dict):
        return hit
    if hasattr(hit, "__dict__"):
        return dict(hit.__dict__)
    raise TypeError("External hit must be a dict-like object")


def _hit_to_content_item(hit: Any, source_policy: SourceTrustPolicy) -> ContentItem:
    row = _to_dict(hit)
    content_id = str(
        row.get("content_id")
        or row.get("doc_id")
        or row.get("id")
        or row.get("chunk_id")
        or "unknown-content-id"
    )
    text = str(row.get("text") or row.get("content") or "")
    source_id = str(row.get("source_id") or row.get("source") or "unknown-source")
    source_type = str(row.get("source_type") or row.get("type") or "other")
    trust = source_policy.trust_for(source_type=source_type, source_id=source_id)
    metadata = dict(row.get("metadata") or {})
    return ContentItem(
        doc_id=content_id,
        source_id=source_id,
        source_type=source_type,
        trust=trust,
        text=text,
        meta=metadata,
    )


@dataclass
class ExternalRetrieverAdapter:
    client: ExternalRetrieverClient
    source_policy: SourceTrustPolicy

    @classmethod
    def from_config(cls, client: ExternalRetrieverClient, config: Dict[str, Any]) -> "ExternalRetrieverAdapter":
        return cls(client=client, source_policy=SourceTrustPolicy.from_config(config))

    def search(self, query: str, k: int) -> List[ContentItem]:
        return [_hit_to_content_item(hit, self.source_policy) for hit in self.client.search(query, k)]


@dataclass
class SQLiteFTSRetrieverAdapter:
    retriever: SQLiteFTSRetriever

    @classmethod
    def from_directory(
        cls,
        root_dir: str,
        config: Dict[str, Any],
        include_extensions: List[str] | None = None,
    ) -> "SQLiteFTSRetrieverAdapter":
        root = Path(root_dir).resolve()
        source_policy = SourceTrustPolicy.from_config(config)
        raw_items = load_content_items_from_directory(
            root_dir=root_dir,
            trust=source_policy.default_trust,
            include_extensions=include_extensions,
        )
        corpus: List[ContentItem] = []
        for item in raw_items:
            source_type = item.source_type
            meta_path = (item.meta or {}).get("path")
            if meta_path:
                try:
                    rel = Path(str(meta_path)).resolve().relative_to(root).as_posix()
                    subdir = rel.split("/", 1)[0].lower()
                    if subdir == "trusted":
                        source_type = "wiki"
                    elif subdir == "semi_trusted":
                        source_type = "ticket"
                    elif subdir == "untrusted":
                        source_type = "web"
                except Exception:
                    pass

            trust = source_policy.trust_for(source_type=source_type, source_id=item.source_id)
            corpus.append(
                ContentItem(
                    doc_id=item.doc_id,
                    source_id=item.source_id,
                    source_type=source_type,
                    trust=trust,
                    text=item.text,
                    language=item.language,
                    meta=item.meta,
                )
            )
        retriever = SQLiteFTSRetriever(corpus=corpus, homoglyph_map=config["pi0"]["homoglyph_map"])
        return cls(retriever=retriever)

    def search(self, query: str, k: int) -> List[ContentItem]:
        return self.retriever.retrieve(query=query, top_k=k)

    def retrieve(self, query: str, top_k: int = 4) -> List[ContentItem]:
        return self.search(query=query, k=top_k)
