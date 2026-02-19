"""Text normalization utilities for pi0 projector."""

from __future__ import annotations

import re
from typing import Dict, List

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
NON_WORD_RE = re.compile(r"[^a-z0-9_<>:/\.\-\s]+")
MULTISPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, homoglyph_map: Dict[str, str]) -> str:
    lowered = text.lower()
    stripped = ZERO_WIDTH_RE.sub("", lowered)
    mapped = "".join(homoglyph_map.get(ch, ch) for ch in stripped)
    filtered = NON_WORD_RE.sub(" ", mapped)
    return MULTISPACE_RE.sub(" ", filtered).strip()


def nospace_text(text: str) -> str:
    return text.replace(" ", "")


def tokenize(text: str) -> List[str]:
    return [tok for tok in text.split(" ") if tok]
