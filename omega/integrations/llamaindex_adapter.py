"""LlamaIndex retriever adapter -> Omega Retriever contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from typing import Any, Dict, List, Optional, Sequence

from omega.interfaces.contracts_v1 import ContentItem
from omega.rag.source_policy import SourceTrustPolicy
from omega.rag.sources_fs import load_content_items_from_directory

LOGGER = logging.getLogger(__name__)


def _stable_id(text: str, metadata: Dict[str, Any]) -> str:
    seed = f"{metadata.get('source_id', '')}|{metadata.get('source', '')}|{text[:200]}"
    return f"li-{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:12]}"


def _node_and_score(hit: Any) -> tuple[Any, Optional[float]]:
    if hasattr(hit, "node"):
        score = getattr(hit, "score", None)
        return getattr(hit, "node"), float(score) if isinstance(score, (float, int)) else None
    return hit, None


def _node_text(node: Any) -> str:
    text = getattr(node, "text", None)
    if isinstance(text, str):
        return text.strip()
    if isinstance(node, dict) and isinstance(node.get("text"), str):
        return str(node["text"]).strip()
    return ""


def _node_id(node: Any, text: str, metadata: Dict[str, Any]) -> str:
    if hasattr(node, "id_"):
        id_ = getattr(node, "id_")
        if id_:
            return str(id_)
    if isinstance(node, dict):
        id_ = node.get("id_") or node.get("id")
        if id_:
            return str(id_)
    if metadata.get("doc_id"):
        return str(metadata["doc_id"])
    if metadata.get("id"):
        return str(metadata["id"])
    return _stable_id(text=text, metadata=metadata)


def _node_metadata(node: Any) -> Dict[str, Any]:
    md = getattr(node, "metadata", None)
    if isinstance(md, dict):
        return dict(md)
    if isinstance(node, dict) and isinstance(node.get("metadata"), dict):
        return dict(node["metadata"])
    return {}


@dataclass
class LlamaIndexRetrieverAdapter:
    retriever: Any
    source_policy: SourceTrustPolicy

    @classmethod
    def from_config(cls, retriever: Any, config: Dict[str, Any]) -> "LlamaIndexRetrieverAdapter":
        return cls(retriever=retriever, source_policy=SourceTrustPolicy.from_config(config))

    def search(self, query: str, k: int) -> List[ContentItem]:
        if not hasattr(self.retriever, "retrieve"):
            raise ValueError("LlamaIndex retriever must support retrieve(query)")
        hits = self.retriever.retrieve(query)
        if not isinstance(hits, list):
            return []

        out: List[ContentItem] = []
        for hit in hits[: int(k)]:
            node, score = _node_and_score(hit)
            text = _node_text(node)
            if not text:
                LOGGER.warning("llamaindex node dropped due to empty text")
                continue
            metadata = _node_metadata(node)
            if score is not None:
                metadata = dict(metadata)
                metadata.setdefault("score", score)
            doc_id = _node_id(node=node, text=text, metadata=metadata)
            source_id = str(
                metadata.get("source_id")
                or metadata.get("source")
                or metadata.get("path")
                or f"llamaindex:{doc_id}"
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


def build_llamaindex_vector_adapter_from_directory(
    root_dir: str,
    config: Dict[str, Any],
    *,
    include_extensions: Optional[Sequence[str]] = None,
    similarity_top_k: int = 4,
) -> LlamaIndexRetrieverAdapter:
    try:
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.core.embeddings.mock_embed_model import MockEmbedding
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "LlamaIndex integration requires llama-index-core. "
            "Install with: pip install -e .[integrations]"
        ) from exc

    source_policy = SourceTrustPolicy.from_config(config)
    items = load_content_items_from_directory(
        root_dir=root_dir,
        trust=source_policy.default_trust,
        include_extensions=include_extensions,
    )

    docs: List[Any] = []
    for it in items:
        metadata = dict(it.meta or {})
        metadata.setdefault("doc_id", it.doc_id)
        metadata.setdefault("source_id", it.source_id)
        metadata.setdefault("source_type", it.source_type)
        docs.append(Document(text=it.text, doc_id=it.doc_id, metadata=metadata))

    Settings.embed_model = MockEmbedding(embed_dim=128)
    index = VectorStoreIndex.from_documents(docs)
    retriever = index.as_retriever(similarity_top_k=int(similarity_top_k))
    return LlamaIndexRetrieverAdapter.from_config(retriever=retriever, config=config)
