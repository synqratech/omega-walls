"""Redaction utilities for audit events.

Redaction is applied before truncation so a secret crossing the excerpt boundary
cannot leak a prefix.  The original text is represented only by SHA-256.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Pattern


@dataclass(frozen=True)
class _RedactionPattern:
    label: str
    pattern: Pattern[str]
    replacement: str = "<REDACTED>"


_SECRET_PATTERNS = [
    _RedactionPattern(
        "private_key",
        re.compile(
            r"-----BEGIN\s+(?:RSA|EC|DSA|OPENSSH|PGP)?\s*PRIVATE KEY-----.*?"
            r"-----END\s+(?:RSA|EC|DSA|OPENSSH|PGP)?\s*PRIVATE KEY-----",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ),
    # Fail-safe for a private-key header whose footer is outside a bounded input.
    _RedactionPattern(
        "private_key_unterminated",
        re.compile(
            r"-----BEGIN\s+(?:RSA|EC|DSA|OPENSSH|PGP)?\s*PRIVATE KEY-----[\s\S]*$",
            flags=re.IGNORECASE,
        ),
    ),
    _RedactionPattern(
        "authorization_bearer",
        re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]+"),
    ),
    _RedactionPattern(
        "cookie_header",
        re.compile(r"(?i)\b(?:set-cookie|cookie)\s*:\s*[^\r\n]+"),
    ),
    _RedactionPattern(
        "secret_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|"
            r"token|password|passwd|secret|client[_-]?secret|session[_-]?id)\b"
            r"\s*[:=]\s*[\"']?[^\s,;\"']+"
        ),
    ),
    _RedactionPattern("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}\b")),
    _RedactionPattern("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    _RedactionPattern(
        "github_token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    _RedactionPattern("slack_token", re.compile(r"\b(?:xox[baprs]-|xapp-)[A-Za-z0-9-]{10,}\b")),
    _RedactionPattern(
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
    _RedactionPattern("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    _RedactionPattern("phone", re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")),
    # Kept last to avoid hiding structure needed by more specific patterns.
    _RedactionPattern("long_secret_like", re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{32,}={0,2}(?![A-Za-z0-9+/])")),
]

# Backward-compatible public constant used by downstream integrations.
SECRET_PATTERNS = [entry.pattern for entry in _SECRET_PATTERNS]


@dataclass
class RedactionResult:
    redacted: str
    text_sha256: str
    redaction_hits: int


def redact_text(text: str, max_chars: int = 800) -> RedactionResult:
    source = str(text or "")
    max_chars_value = int(max_chars)
    if max_chars_value <= 0:
        raise ValueError("max_chars must be > 0")

    candidate = source
    redaction_hits = 0
    for entry in _SECRET_PATTERNS:
        candidate, count = entry.pattern.subn(entry.replacement, candidate)
        redaction_hits += int(count)

    # Truncate only after all substitutions.  This prevents boundary-prefix leaks.
    candidate = candidate[:max_chars_value]
    return RedactionResult(
        redacted=candidate,
        text_sha256=sha256(source.encode("utf-8")).hexdigest(),
        redaction_hits=redaction_hits,
    )
