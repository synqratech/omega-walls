"""Dependency-free secret scan used by source-archive and CI release gates.

This scanner is intentionally conservative around executable/configuration files
and explicitly recognizes common synthetic benchmark placeholders.  It is not a
replacement for provider-side rotation or an enterprise scanner, but it makes
secret-free source packaging deterministic and blocking.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable, Sequence


TEXT_SUFFIXES = {
    ".cfg", ".conf", ".css", ".env", ".example", ".html", ".ini", ".js",
    ".json", ".jsonl", ".md", ".ps1", ".py", ".rst", ".sh", ".toml",
    ".ts", ".txt", ".xml", ".yaml", ".yml",
}
SKIP_PARTS = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", "artifacts", "build", "dist", "node_modules", "tmp_codex_pytest", "_tmp"}
FORBIDDEN_NAMES = {".env", ".env.local", ".env.prod", ".env.production", ".env.development"}
SYNTHETIC_PATH_PREFIXES = ("tests/data/",)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    fingerprint: str


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {char: value.count(char) for char in set(value)}
    size = len(value)
    return -sum((count / size) * math.log2(count / size) for count in counts.values())


def _looks_synthetic(value: str, *, rel_path: str) -> bool:
    low = value.lower()
    if rel_path.replace("\\", "/").startswith(SYNTHETIC_PATH_PREFIXES):
        return True
    markers = (
        "example", "placeholder", "redacted", "your_key", "your-key", "fake",
        "replace-with", "quickstart", "dummy", "sample", "changeme", "change-me", "abcdef", "0123456789",
        "sk-...", "<redacted>", "***", "test-", "_test_",
    )
    if any(marker in low for marker in markers):
        return True
    # Obvious alphabet/counting fixtures.
    compact = re.sub(r"[^a-z0-9]", "", low)
    return "abcdefghijklmnopqrstuvwxyz" in compact or len(set(compact)) <= 4


def _fingerprint(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_PARTS for part in rel_parts):
            continue
        yield path


def scan_tree(root: Path) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    assignment_re = re.compile(
        r"(?i)\b(?:OPENAI_API_KEY|ANTHROPIC_API_KEY|API_KEY|SECRET_KEY|ACCESS_TOKEN|"
        r"AUTH_TOKEN|PASSWORD|CLIENT_SECRET|AWS_SECRET_ACCESS_KEY)\b\s*[:=]\s*[\"']?([^\s\"',;]+)"
    )
    token_patterns = [
        ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
        ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{24,}|github_pat_[A-Za-z0-9_]{24,})\b")),
        ("slack_token", re.compile(r"\b(?:xox[baprs]-|xapp-)[A-Za-z0-9-]{16,}\b")),
        ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ]
    private_key_re = re.compile(
        r"-----BEGIN\s+(?:RSA|EC|DSA|OPENSSH|PGP)?\s*PRIVATE KEY-----([\s\S]*?)"
        r"-----END\s+(?:RSA|EC|DSA|OPENSSH|PGP)?\s*PRIVATE KEY-----",
        re.IGNORECASE,
    )

    for path in _iter_files(root):
        rel = path.relative_to(root).as_posix()
        if path.name in FORBIDDEN_NAMES or (path.name.startswith(".env.") and path.name != ".env.example"):
            findings.append(Finding(rel, 1, "forbidden_env_file", _fingerprint(rel)))
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"Dockerfile", "Makefile"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for match in private_key_re.finditer(text):
            body = re.sub(r"\s+", "", match.group(1))
            if len(body) >= 80 and not _looks_synthetic(body, rel_path=rel):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(Finding(rel, line, "private_key", _fingerprint(body)))

        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in assignment_re.finditer(line):
                value = match.group(1).strip()
                expression_prefixes = ("str(", "os.", "self.", "_", "getenv(", "environ", "none", "null")
                if value.lower().startswith(expression_prefixes):
                    continue
                if len(value) >= 12 and not _looks_synthetic(value, rel_path=rel) and _entropy(value) >= 3.2:
                    findings.append(Finding(rel, line_no, "secret_assignment", _fingerprint(value)))
            for rule, pattern in token_patterns:
                for match in pattern.finditer(line):
                    value = match.group(0)
                    if rule == "aws_access_key" and value == "AKIAIOSFODNN7EXAMPLE":
                        continue
                    if not _looks_synthetic(value, rel_path=rel) and _entropy(value) >= 3.2:
                        findings.append(Finding(rel, line_no, rule, _fingerprint(value)))

    unique = {(f.path, f.line, f.rule, f.fingerprint): f for f in findings}
    return sorted(unique.values(), key=lambda item: (item.path, item.line, item.rule))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan a source tree for high-confidence secrets")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    findings = scan_tree(root)
    report = {
        "event": "omega_secret_scan_v1",
        "root": str(root),
        "status": "fail" if findings else "ok",
        "findings": [finding.__dict__ for finding in findings],
    }
    rendered = json.dumps(report, ensure_ascii=True, indent=2)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
