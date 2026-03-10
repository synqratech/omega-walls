"""Lightweight semantic encoder utilities for projector enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


def mean_pool_hidden(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return cosine similarities for two embedding matrices."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("cosine_similarity_matrix expects 2D arrays")
    if a.shape[1] != b.shape[1]:
        raise ValueError("embedding dimensions must match")

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_safe = a / np.clip(a_norm, 1e-12, None)
    b_safe = b / np.clip(b_norm, 1e-12, None)
    return np.matmul(a_safe, b_safe.T)


@dataclass(frozen=True)
class SemanticEncoderConfig:
    model_path: str
    device: str = "auto"
    max_length: int = 256
    batch_size: int = 16
    normalize_embeddings: bool = True


class SemanticEncoder:
    """Transformer-backed local encoder with mean pooling."""

    def __init__(self, config: SemanticEncoderConfig) -> None:
        self.config = config
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Semantic model path does not exist: {model_path}")

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError(
                "Semantic encoder requires transformers and torch. "
                "Install with your local environment dependencies."
            ) from exc

        self._torch = torch
        self.device = self._resolve_device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        self.model.eval()
        self.model.to(self.device)

    def _resolve_device(self, device: str):
        if device != "auto":
            return self._torch.device(device)
        if self._torch.cuda.is_available():
            return self._torch.device("cuda")
        return self._torch.device("cpu")

    def _mean_pool(self, hidden_state, attention_mask):
        return mean_pool_hidden(hidden_state, attention_mask)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        rows = [str(t) for t in texts]
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)

        outputs: List[np.ndarray] = []
        bs = max(1, int(self.config.batch_size))
        with self._torch.no_grad():
            for start in range(0, len(rows), bs):
                batch = rows[start : start + bs]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=int(self.config.max_length),
                    return_tensors="pt",
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                model_out = self.model(**tokens)
                pooled = self._mean_pool(model_out.last_hidden_state, tokens["attention_mask"])
                if self.config.normalize_embeddings:
                    pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                outputs.append(pooled.detach().cpu().numpy().astype(np.float32))

        return np.vstack(outputs)
