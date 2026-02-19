"""Redaction utilities for audit events."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import List, Tuple


SECRET_PATTERNS = [
    re.compile(r"(?i)authorization:\\s*bearer\\s+[A-Za-z0-9._-]+"),
    re.compile(r"(?i)(api[_-]?key|token|password)\\s*[:=]\\s*[^\\s]+"),
    re.compile(r"[A-Za-z0-9+/]{24,}={0,2}"),
    re.compile(r"-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY-----[\\s\\S]*?-----END (RSA|EC|OPENSSH) PRIVATE KEY-----"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"),
    re.compile(r"\\+?\\d[\\d\\s()-]{7,}"),
]


@dataclass
class RedactionResult:
    redacted: str
    text_sha256: str
    redaction_hits: int



def redact_text(text: str, max_chars: int = 800) -> RedactionResult:
    candidate = text[:max_chars]
    redaction_hits = 0
    for pattern in SECRET_PATTERNS:
        candidate, count = pattern.subn("<REDACTED>", candidate)
        redaction_hits += count

    return RedactionResult(
        redacted=candidate,
        text_sha256=sha256(text.encode("utf-8")).hexdigest(),
        redaction_hits=redaction_hits,
    )
