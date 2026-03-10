"""Attachment ingestion helpers for PDF/DOCX/HTML with bounded chunking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


DEFAULT_MAX_FILE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_EXTRACTED_CHARS = 200_000
DEFAULT_MAX_CHUNK_CHARS = 2_000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_HIDDEN_PREFIX = "[hidden_html] "

MIME_TO_FORMAT = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/html": "html",
}

HIDDEN_STYLE_RE = re.compile(r"(display\s*:\s*none|visibility\s*:\s*hidden)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class AttachmentChunk:
    text: str
    kind: str
    is_hidden: bool = False


@dataclass(frozen=True)
class AttachmentExtractResult:
    text: str
    chunks: List[AttachmentChunk]
    format: str
    text_empty: bool
    scan_like: bool
    hidden_text_chars: int
    warnings: List[str]
    recommended_verdict: str


@dataclass(frozen=True)
class AttachmentIngestionConfig:
    enabled: bool = True
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_extracted_chars: int = DEFAULT_MAX_EXTRACTED_CHARS
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    html_include_hidden: bool = True
    hidden_chunk_prefix: str = DEFAULT_HIDDEN_PREFIX
    scan_like_min_chars_per_page: int = 25
    scan_like_min_alpha_ratio: float = 0.30
    zip_enabled: bool = False
    zip_max_files: int = 100
    zip_max_depth: int = 5
    zip_max_total_bytes: int = 20 * 1024 * 1024
    zip_allow_encrypted: bool = False

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "AttachmentIngestionConfig":
        data = dict(cfg or {})
        zip_cfg = data.get("zip", {}) if isinstance(data.get("zip", {}), dict) else {}
        max_chunk = int(data.get("max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS))
        overlap = int(data.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP))
        if max_chunk <= 0:
            max_chunk = DEFAULT_MAX_CHUNK_CHARS
        if overlap < 0:
            overlap = DEFAULT_CHUNK_OVERLAP
        if overlap >= max_chunk:
            overlap = max_chunk - 1
        return cls(
            enabled=bool(data.get("enabled", True)),
            max_file_bytes=int(data.get("max_file_bytes", DEFAULT_MAX_FILE_BYTES)),
            max_extracted_chars=int(data.get("max_extracted_chars", DEFAULT_MAX_EXTRACTED_CHARS)),
            max_chunk_chars=max_chunk,
            chunk_overlap=overlap,
            html_include_hidden=bool(data.get("html_include_hidden", True)),
            hidden_chunk_prefix=str(data.get("hidden_chunk_prefix", DEFAULT_HIDDEN_PREFIX)),
            scan_like_min_chars_per_page=int(data.get("scan_like_min_chars_per_page", 25)),
            scan_like_min_alpha_ratio=float(data.get("scan_like_min_alpha_ratio", 0.30)),
            zip_enabled=bool(zip_cfg.get("enabled", False)),
            zip_max_files=int(zip_cfg.get("max_files", 100)),
            zip_max_depth=int(zip_cfg.get("max_depth", 5)),
            zip_max_total_bytes=int(zip_cfg.get("max_total_bytes", 20 * 1024 * 1024)),
            zip_allow_encrypted=bool(zip_cfg.get("allow_encrypted", False)),
        )


def _detect_format(*, path: Path | None, filename: str | None, mime: str | None) -> str:
    if mime:
        mime_l = str(mime).strip().lower()
        if mime_l in MIME_TO_FORMAT:
            return MIME_TO_FORMAT[mime_l]
    candidates: List[str] = []
    if path is not None:
        candidates.append(path.suffix.lower())
    if filename:
        candidates.append(Path(filename).suffix.lower())
    ext = next((x for x in candidates if x), "")
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext in {".html", ".htm"}:
        return "html"
    if ext == ".zip":
        return "zip"
    return "text"


def _load_raw_bytes(*, path: Path | None, content_bytes: bytes | None) -> bytes:
    if content_bytes is not None:
        return content_bytes
    if path is None:
        raise ValueError("either path or content_bytes must be provided")
    return path.read_bytes()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _chunk_text(text: str, *, max_chunk_chars: int, chunk_overlap: int) -> List[str]:
    text_norm = _normalize_text(text)
    if not text_norm:
        return []
    if len(text_norm) <= max_chunk_chars:
        return [text_norm]

    step = max(1, max_chunk_chars - chunk_overlap)
    chunks: List[str] = []
    i = 0
    while i < len(text_norm):
        j = min(len(text_norm), i + max_chunk_chars)
        chunk = text_norm[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= len(text_norm):
            break
        i += step
    return chunks


def _clip_text(text: str, *, max_chars: int, warnings: List[str]) -> str:
    if len(text) <= max_chars:
        return text
    warnings.append("max_extracted_chars_truncated")
    return text[:max_chars]


def _missing_dependency_error(feature: str, dependency: str) -> RuntimeError:
    return RuntimeError(
        f"{feature} ingestion requires optional dependency '{dependency}'. "
        "Install with: pip install -e .[attachments]"
    )


def _extract_pdf_text_from_bytes(raw: bytes) -> Tuple[str, int]:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - dependency gate
        raise _missing_dependency_error("PDF", "pypdf") from exc

    reader = PdfReader(io.BytesIO(raw))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages), len(reader.pages)


def _extract_docx_text_from_bytes(raw: bytes) -> str:
    try:
        import docx  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency gate
        raise _missing_dependency_error("DOCX", "python-docx") from exc

    document = docx.Document(io.BytesIO(raw))
    parts: List[str] = []

    for p in document.paragraphs:
        t = _normalize_text(p.text)
        if t:
            parts.append(t)

    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                t = _normalize_text(cell.text)
                if t:
                    parts.append(t)

    for section in document.sections:
        for p in section.header.paragraphs:
            t = _normalize_text(p.text)
            if t:
                parts.append(t)
        for p in section.footer.paragraphs:
            t = _normalize_text(p.text)
            if t:
                parts.append(t)

    return "\n".join(parts)


def _element_is_hidden(tag: Any) -> bool:
    style = str(tag.attrs.get("style", "") or "")
    if HIDDEN_STYLE_RE.search(style):
        return True
    if tag.has_attr("hidden"):
        return True
    if str(tag.attrs.get("aria-hidden", "")).lower() == "true":
        return True
    return False


def _extract_html_texts(raw: bytes) -> Tuple[str, str]:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:  # pragma: no cover - dependency gate
        raise _missing_dependency_error("HTML", "beautifulsoup4") from exc

    parser = "lxml"
    try:
        __import__("lxml")
    except Exception:
        parser = "html.parser"

    html = raw.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, parser)

    hidden_texts: List[str] = []
    hidden_tags = [t for t in soup.find_all(True) if _element_is_hidden(t)]
    for tag in hidden_tags:
        txt = _normalize_text(tag.get_text(" ", strip=True))
        if txt:
            hidden_texts.append(txt)
        tag.decompose()

    visible = _normalize_text(soup.get_text(" ", strip=True))
    hidden = _normalize_text(" ".join(hidden_texts))
    return visible, hidden


def _is_scan_like_pdf(text: str, pages_count: int, cfg: AttachmentIngestionConfig) -> bool:
    if pages_count <= 0:
        return False
    text_norm = _normalize_text(text)
    if not text_norm:
        return True
    chars_per_page = len(text_norm) / float(pages_count)
    alpha_chars = sum(1 for ch in text_norm if ch.isalpha())
    alpha_ratio = float(alpha_chars) / float(max(1, len(text_norm)))
    return chars_per_page < float(cfg.scan_like_min_chars_per_page) or alpha_ratio < float(cfg.scan_like_min_alpha_ratio)


def extract_attachment(
    *,
    path: str | Path | None = None,
    content_bytes: bytes | None = None,
    filename: str | None = None,
    mime: str | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> AttachmentExtractResult:
    parsed_cfg = AttachmentIngestionConfig.from_cfg(cfg)
    if not parsed_cfg.enabled:
        return AttachmentExtractResult(
            text="",
            chunks=[],
            format="disabled",
            text_empty=True,
            scan_like=False,
            hidden_text_chars=0,
            warnings=["attachments_disabled"],
            recommended_verdict="allow",
        )

    path_obj = Path(path) if path is not None else None
    fmt = _detect_format(path=path_obj, filename=filename, mime=mime)
    raw = _load_raw_bytes(path=path_obj, content_bytes=content_bytes)
    warnings: List[str] = []

    if len(raw) > parsed_cfg.max_file_bytes:
        raise ValueError(
            f"attachment exceeds max_file_bytes={parsed_cfg.max_file_bytes}: got={len(raw)} format={fmt}"
        )

    if fmt == "zip":
        warnings.append("zip_deferred_runtime")
        return AttachmentExtractResult(
            text="",
            chunks=[],
            format="zip",
            text_empty=True,
            scan_like=False,
            hidden_text_chars=0,
            warnings=warnings,
            recommended_verdict="quarantine",
        )

    text = ""
    scan_like = False
    hidden_text = ""

    if fmt == "pdf":
        text, pages = _extract_pdf_text_from_bytes(raw)
        scan_like = _is_scan_like_pdf(text=text, pages_count=pages, cfg=parsed_cfg)
    elif fmt == "docx":
        text = _extract_docx_text_from_bytes(raw)
    elif fmt == "html":
        visible, hidden = _extract_html_texts(raw)
        text = visible
        hidden_text = hidden if parsed_cfg.html_include_hidden else ""
    else:
        text = raw.decode("utf-8", errors="ignore")

    text = _clip_text(text, max_chars=parsed_cfg.max_extracted_chars, warnings=warnings)
    hidden_text = _clip_text(hidden_text, max_chars=parsed_cfg.max_extracted_chars, warnings=warnings)

    chunks: List[AttachmentChunk] = []
    for c in _chunk_text(text, max_chunk_chars=parsed_cfg.max_chunk_chars, chunk_overlap=parsed_cfg.chunk_overlap):
        chunks.append(AttachmentChunk(text=c, kind="visible", is_hidden=False))

    hidden_chars = len(_normalize_text(hidden_text))
    if hidden_text:
        warnings.append("hidden_text_present")
        pref = parsed_cfg.hidden_chunk_prefix
        for c in _chunk_text(hidden_text, max_chunk_chars=parsed_cfg.max_chunk_chars, chunk_overlap=parsed_cfg.chunk_overlap):
            chunks.append(AttachmentChunk(text=f"{pref}{c}".strip(), kind="hidden", is_hidden=True))

    full_text = _normalize_text(" ".join(ch.text for ch in chunks))
    text_empty = len(full_text) == 0
    if text_empty:
        warnings.append("text_empty")
    if scan_like:
        warnings.append("scan_like")

    verdict = "quarantine" if (text_empty or scan_like) else "allow"
    return AttachmentExtractResult(
        text=full_text,
        chunks=chunks,
        format=fmt,
        text_empty=text_empty,
        scan_like=bool(scan_like),
        hidden_text_chars=hidden_chars,
        warnings=sorted(set(warnings)),
        recommended_verdict=verdict,
    )


def extract_text_payload(*, text: str, cfg: Mapping[str, Any] | None = None) -> AttachmentExtractResult:
    parsed_cfg = AttachmentIngestionConfig.from_cfg(cfg)
    warnings: List[str] = []
    clipped = _clip_text(str(text or ""), max_chars=parsed_cfg.max_extracted_chars, warnings=warnings)
    chunks = [
        AttachmentChunk(text=chunk, kind="visible", is_hidden=False)
        for chunk in _chunk_text(
            clipped,
            max_chunk_chars=parsed_cfg.max_chunk_chars,
            chunk_overlap=parsed_cfg.chunk_overlap,
        )
    ]
    full_text = _normalize_text(" ".join(c.text for c in chunks))
    text_empty = len(full_text) == 0
    if text_empty:
        warnings.append("text_empty")
    verdict = "quarantine" if text_empty else "allow"
    return AttachmentExtractResult(
        text=full_text,
        chunks=chunks,
        format="text",
        text_empty=text_empty,
        scan_like=False,
        hidden_text_chars=0,
        warnings=sorted(set(warnings)),
        recommended_verdict=verdict,
    )
