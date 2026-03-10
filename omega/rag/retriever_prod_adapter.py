"""Unified production-style adapter for retriever providers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, List, Optional

from omega.interfaces.contracts_v1 import ContentItem
from omega.rag.retriever_provider import ExternalRetrieverProvider, RetrieverProvider, SQLiteFTSProvider
from omega.rag.source_policy import SourceTrustPolicy

LOGGER = logging.getLogger(__name__)


def _coerce_hit(hit: Any) -> Dict[str, Any]:
    if isinstance(hit, dict):
        return hit
    if hasattr(hit, "__dict__"):
        return dict(hit.__dict__)
    raise TypeError("Retriever hit must be dict-like")


def _to_content_item(hit: Dict[str, Any], source_policy: SourceTrustPolicy) -> Optional[ContentItem]:
    doc_id = str(
        hit.get("content_id")
        or hit.get("doc_id")
        or hit.get("id")
        or hit.get("chunk_id")
        or "unknown-content-id"
    )
    source_id = str(hit.get("source_id") or hit.get("source") or f"unknown:{doc_id}")
    source_type = str(hit.get("source_type") or hit.get("type") or "other")
    text = str(hit.get("text") or hit.get("content") or "").strip()
    if not text:
        LOGGER.warning("retriever hit dropped due to empty text", extra={"doc_id": doc_id, "source_id": source_id})
        return None

    trust = source_policy.trust_for(source_type=source_type, source_id=source_id)
    metadata = dict(hit.get("metadata") or hit.get("meta") or {})
    return ContentItem(
        doc_id=doc_id,
        source_id=source_id,
        source_type=source_type,
        trust=trust,
        text=text,
        meta=metadata,
    )


@dataclass
class RetrieverProdAdapter:
    provider: RetrieverProvider
    source_policy: SourceTrustPolicy

    @classmethod
    def from_config(cls, provider: RetrieverProvider, config: Dict[str, Any]) -> "RetrieverProdAdapter":
        return cls(provider=provider, source_policy=SourceTrustPolicy.from_config(config))

    def search(self, query: str, k: int) -> List[ContentItem]:
        out: List[ContentItem] = []
        for raw in self.provider.search(query, k):
            item = _to_content_item(_coerce_hit(raw), self.source_policy)
            if item is not None:
                out.append(item)
        return out


def build_retriever_prod_adapter(
    config: Dict[str, Any],
    source_root: Optional[str] = None,
    include_extensions: Optional[List[str]] = None,
    external_client: Any = None,
) -> RetrieverProdAdapter:
    cfg = config.get("retriever", {})
    backend = str(cfg.get("backend", "sqlite_fts")).strip().lower()

    if backend == "sqlite_fts":
        sqlite_cfg = cfg.get("sqlite_fts", {})
        root_dir = source_root or str(sqlite_cfg.get("source_root", "data/smoke_sources/domain_docs"))
        exts = include_extensions or list(sqlite_cfg.get("include_extensions", [".txt", ".md"]))
        attachment_cfg = sqlite_cfg.get("attachments", {}) if isinstance(sqlite_cfg.get("attachments", {}), dict) else {}
        source_policy = SourceTrustPolicy.from_config(config)
        provider = SQLiteFTSProvider.from_directory(
            root_dir=root_dir,
            homoglyph_map=config["pi0"]["homoglyph_map"],
            default_trust=source_policy.default_trust,
            include_extensions=exts,
            attachment_cfg=attachment_cfg,
        )
        return RetrieverProdAdapter.from_config(provider=provider, config=config)

    if backend == "external":
        if external_client is None:
            raise ValueError("retriever.backend=external requires external_client")
        provider = ExternalRetrieverProvider(client=external_client)
        return RetrieverProdAdapter.from_config(provider=provider, config=config)

    raise ValueError(f"Unsupported retriever backend: {backend}")
