"""LangChain retriever adapter -> Omega Retriever contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from omega.interfaces.contracts_v1 import ContentItem
from omega.rag.source_policy import SourceTrustPolicy
from omega.rag.sources_fs import load_content_items_from_directory

LOGGER = logging.getLogger(__name__)


def _stable_id(text: str, metadata: Dict[str, Any]) -> str:
    seed = f"{metadata.get('source', '')}|{metadata.get('path', '')}|{text[:200]}"
    return f"lc-{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:12]}"


def _metadata_from_doc(doc: Any) -> Dict[str, Any]:
    md = getattr(doc, "metadata", None)
    if isinstance(md, dict):
        return dict(md)
    if isinstance(doc, dict):
        maybe = doc.get("metadata")
        if isinstance(maybe, dict):
            return dict(maybe)
    return {}


def _text_from_doc(doc: Any) -> str:
    text = getattr(doc, "page_content", None)
    if isinstance(text, str):
        return text.strip()
    if isinstance(doc, dict):
        if isinstance(doc.get("page_content"), str):
            return str(doc["page_content"]).strip()
        if isinstance(doc.get("text"), str):
            return str(doc["text"]).strip()
    return ""


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


class _KeywordRetriever:
    def __init__(self, docs: Sequence[Any], k: int = 4) -> None:
        self._docs = list(docs)
        self.k = int(k)

    def invoke(self, query: str, config: Optional[Dict[str, Any]] = None) -> List[Any]:
        top_k = self.k
        if isinstance(config, dict):
            try:
                top_k = int(config.get("k", top_k))
            except (TypeError, ValueError):
                top_k = self.k

        query_tokens = _tokenize(query)
        scored: List[tuple[int, int, Any]] = []
        for idx, doc in enumerate(self._docs):
            text = _text_from_doc(doc)
            score = len(query_tokens & _tokenize(text))
            scored.append((score, -idx, doc))
        scored.sort(reverse=True)

        non_zero = [doc for score, _, doc in scored if score > 0]
        if non_zero:
            return non_zero[:top_k]
        return [doc for _, _, doc in scored[:top_k]]


@dataclass
class LangChainRetrieverAdapter:
    retriever: Any
    source_policy: SourceTrustPolicy

    @classmethod
    def from_config(cls, retriever: Any, config: Dict[str, Any]) -> "LangChainRetrieverAdapter":
        return cls(retriever=retriever, source_policy=SourceTrustPolicy.from_config(config))

    def _invoke(self, query: str, k: int) -> List[Any]:
        if hasattr(self.retriever, "invoke"):
            try:
                out = self.retriever.invoke(query, config={"k": int(k)})
            except TypeError:
                out = self.retriever.invoke(query)
        elif hasattr(self.retriever, "get_relevant_documents"):
            try:
                out = self.retriever.get_relevant_documents(query, k=int(k))
            except TypeError:
                out = self.retriever.get_relevant_documents(query)
        else:
            raise ValueError("LangChain retriever must support invoke(...) or get_relevant_documents(...)")
        if not isinstance(out, list):
            return []
        return out[: int(k)]

    def search(self, query: str, k: int) -> List[ContentItem]:
        docs = self._invoke(query=query, k=k)
        out: List[ContentItem] = []
        for doc in docs:
            text = _text_from_doc(doc)
            if not text:
                LOGGER.warning("langchain doc dropped due to empty text")
                continue
            metadata = _metadata_from_doc(doc)
            doc_id = str(metadata.get("doc_id") or metadata.get("id") or _stable_id(text=text, metadata=metadata))
            source_id = str(
                metadata.get("source_id")
                or metadata.get("source")
                or metadata.get("path")
                or f"langchain:{doc_id}"
            )
            source_type = str(metadata.get("source_type") or metadata.get("type") or "other")
            trust = self.source_policy.trust_for(source_type=source_type, source_id=source_id)
            out.append(
                ContentItem(
                    doc_id=doc_id,
                    source_id=source_id,
                    source_type=source_type,
                    trust=trust,
                    text=text,
                    meta=metadata,
                )
            )
        return out


def build_langchain_bm25_adapter_from_directory(
    root_dir: str,
    config: Dict[str, Any],
    *,
    include_extensions: Optional[Sequence[str]] = None,
    k: int = 4,
) -> LangChainRetrieverAdapter:
    try:
        from langchain_core.documents import Document
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "LangChain integration requires langchain/langchain-core. "
            "Install with: pip install -e .[integrations]"
        ) from exc

    source_policy = SourceTrustPolicy.from_config(config)
    items = load_content_items_from_directory(
        root_dir=root_dir,
        trust=source_policy.default_trust,
        include_extensions=include_extensions,
    )
    docs = []
    for it in items:
        metadata = dict(it.meta or {})
        metadata.setdefault("doc_id", it.doc_id)
        metadata.setdefault("source_id", it.source_id)
        metadata.setdefault("source_type", it.source_type)
        docs.append(Document(page_content=it.text, metadata=metadata))
    retriever: Any
    try:
        from langchain_community.retrievers import BM25Retriever

        retriever = BM25Retriever.from_documents(docs)
        retriever.k = int(k)
    except Exception:  # pragma: no cover - optional dependency/runtime fallback
        LOGGER.warning("BM25Retriever unavailable, using keyword fallback retriever")
        retriever = _KeywordRetriever(docs=docs, k=int(k))
    return LangChainRetrieverAdapter.from_config(retriever=retriever, config=config)
