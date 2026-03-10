"""Backward-compatible thin retriever adapters built on top of prod adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from omega.interfaces.contracts_v1 import ContentItem
from omega.rag.retriever_prod_adapter import RetrieverProdAdapter, build_retriever_prod_adapter
from omega.rag.retriever_provider import ExternalRetrieverProvider, SQLiteFTSProvider
from omega.rag.source_policy import SourceTrustPolicy


@dataclass
class ExternalRetrieverAdapter:
    provider: ExternalRetrieverProvider
    adapter: RetrieverProdAdapter
    source_policy: SourceTrustPolicy

    @classmethod
    def from_config(cls, client: Any, config: Dict[str, Any]) -> "ExternalRetrieverAdapter":
        provider = ExternalRetrieverProvider(client=client)
        adapter = RetrieverProdAdapter.from_config(provider=provider, config=config)
        source_policy = SourceTrustPolicy.from_config(config)
        return cls(provider=provider, adapter=adapter, source_policy=source_policy)

    def search(self, query: str, k: int) -> List[ContentItem]:
        return self.adapter.search(query=query, k=k)


@dataclass
class SQLiteFTSRetrieverAdapter:
    provider: SQLiteFTSProvider
    adapter: RetrieverProdAdapter

    @property
    def retriever(self):
        return self.provider.retriever

    @classmethod
    def from_directory(
        cls,
        root_dir: str,
        config: Dict[str, Any],
        include_extensions: List[str] | None = None,
    ) -> "SQLiteFTSRetrieverAdapter":
        adapter = build_retriever_prod_adapter(
            config=config,
            source_root=root_dir,
            include_extensions=include_extensions,
        )
        if not isinstance(adapter.provider, SQLiteFTSProvider):
            raise ValueError("SQLiteFTSRetrieverAdapter expects retriever.backend=sqlite_fts")
        return cls(provider=adapter.provider, adapter=adapter)

    def search(self, query: str, k: int) -> List[ContentItem]:
        return self.adapter.search(query=query, k=k)

    def retrieve(self, query: str, top_k: int = 4) -> List[ContentItem]:
        return self.search(query=query, k=top_k)

