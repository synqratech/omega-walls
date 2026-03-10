"""Text normalization and bounded anti-obfuscation utilities for pi0 projector."""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
NON_WORD_RE = re.compile(r"[^a-z0-9_<>:/\.\-\s]+")
MULTISPACE_RE = re.compile(r"\s+")
SPACE_SPLIT_RE = re.compile(r"(\s+)")
BASE64_LITE_RE = re.compile(r"\[base64-lite\](.*?)\[/base64-lite\]", flags=re.IGNORECASE | re.DOTALL)
URL_LITE_RE = re.compile(r"url-lite://q=([^\s]+)", flags=re.IGNORECASE)
MARKDOWN_FENCE_RE = re.compile(r"```(?:[^\n`]*)\n(.*?)```", flags=re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`\n]{1,220})`")
HTML_TAG_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)[^>]*>(.*?)</\1>", flags=re.IGNORECASE | re.DOTALL)
ROLE_TAGS = {"system", "developer", "assistant", "user"}
OBFUSCATED_TOKEN_RE = re.compile(r"\b(?:[a-zA-Z0-9]\s*[.\-_\s]){3,}[a-zA-Z0-9]\b")


@dataclass(frozen=True)
class PreprocessContext:
    kind: str
    raw_text: str
    normalized_text: str


@dataclass(frozen=True)
class PreprocessConfig:
    enabled: bool = True
    nfkc: bool = True
    remove_zero_width: bool = True
    leet_homoglyph_limited: bool = True
    max_replacements_per_token: int = 6
    max_token_len_for_map: int = 32
    max_join_sequences: int = 8
    max_decode_segments: int = 2
    max_payload_chars: int = 256
    max_decoded_chars: int = 256
    decode_depth: int = 1
    max_context_blocks: int = 4
    max_context_chars_per_block: int = 220
    max_context_total_chars: int = 500
    printable_ratio_threshold: float = 0.85

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any] | None) -> "PreprocessConfig":
        data = cfg or {}
        bounded_decode = data.get("bounded_decode", {}) if isinstance(data.get("bounded_decode", {}), dict) else {}
        context_extraction = (
            data.get("context_extraction", {}) if isinstance(data.get("context_extraction", {}), dict) else {}
        )
        return cls(
            enabled=bool(data.get("enabled", True)),
            nfkc=bool(data.get("nfkc", True)),
            remove_zero_width=bool(data.get("remove_zero_width", True)),
            leet_homoglyph_limited=bool(data.get("leet_homoglyph_limited", True)),
            max_replacements_per_token=int(data.get("max_replacements_per_token", 6)),
            max_token_len_for_map=int(data.get("max_token_len_for_map", 32)),
            max_join_sequences=int(data.get("max_join_sequences", 8)),
            max_decode_segments=int(bounded_decode.get("max_decode_segments", 2)),
            max_payload_chars=int(bounded_decode.get("max_payload_chars", 256)),
            max_decoded_chars=int(bounded_decode.get("max_decoded_chars", 256)),
            decode_depth=int(bounded_decode.get("decode_depth", 1)),
            max_context_blocks=int(context_extraction.get("max_context_blocks", 4)),
            max_context_chars_per_block=int(context_extraction.get("max_context_chars_per_block", 220)),
            max_context_total_chars=int(context_extraction.get("max_context_total_chars", 500)),
            printable_ratio_threshold=float(data.get("printable_ratio_threshold", 0.85)),
        )


@dataclass(frozen=True)
class PreprocessResult:
    primary_text: str
    contexts: List[PreprocessContext]
    flags: Dict[str, bool]
    decoded_segments_count: int
    joined_obfuscation_sequences_count: int


def _collapse_spaces(text: str) -> str:
    return MULTISPACE_RE.sub(" ", text).strip()


def _is_printable_ratio_ok(text: str, threshold: float) -> bool:
    if not text:
        return False
    printable = sum(1 for ch in text if ch.isprintable())
    return (float(printable) / float(len(text))) >= threshold


def _is_probable_id_or_hash(token: str) -> bool:
    alnum = [ch for ch in token if ch.isalnum()]
    if not alnum:
        return False
    if len(alnum) <= 2:
        return False
    digits = sum(1 for ch in alnum if ch.isdigit())
    letters = sum(1 for ch in alnum if ch.isalpha())
    if letters > 0:
        return False
    return (float(digits) / float(len(alnum))) > 0.6


def _apply_limited_homoglyph_map(text: str, homoglyph_map: Dict[str, str], cfg: PreprocessConfig) -> str:
    if not cfg.leet_homoglyph_limited or not homoglyph_map:
        return "".join(homoglyph_map.get(ch, ch) for ch in text)

    parts = SPACE_SPLIT_RE.split(text)
    out_parts: List[str] = []
    for part in parts:
        if not part or part.isspace():
            out_parts.append(part)
            continue
        if len(part) > cfg.max_token_len_for_map or _is_probable_id_or_hash(part):
            out_parts.append(part)
            continue
        replaceable = sum(1 for ch in part if ch in homoglyph_map)
        if replaceable > cfg.max_replacements_per_token:
            out_parts.append(part)
            continue
        chars: List[str] = []
        for ch in part:
            if ch in homoglyph_map:
                chars.append(homoglyph_map[ch])
            else:
                chars.append(ch)
        out_parts.append("".join(chars))
    return "".join(out_parts)


def _normalize_core(text: str, homoglyph_map: Dict[str, str], cfg: PreprocessConfig) -> str:
    out = text
    if cfg.nfkc:
        out = unicodedata.normalize("NFKC", out)
    out = out.lower()
    if cfg.remove_zero_width:
        out = ZERO_WIDTH_RE.sub("", out)
    out = _apply_limited_homoglyph_map(out, homoglyph_map, cfg)
    out = NON_WORD_RE.sub(" ", out)
    return _collapse_spaces(out)


def _extract_wrapped_decoded_contexts(text: str, cfg: PreprocessConfig) -> List[PreprocessContext]:
    if cfg.decode_depth <= 0:
        return []

    contexts: List[PreprocessContext] = []
    seen = set()
    decoded_segments = 0

    for match in BASE64_LITE_RE.finditer(text):
        if decoded_segments >= cfg.max_decode_segments:
            break
        payload = match.group(1).strip()
        if not payload or len(payload) > cfg.max_payload_chars:
            continue
        try:
            raw = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error):
            continue
        if len(raw) > cfg.max_decoded_chars:
            continue
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not _is_printable_ratio_ok(decoded, cfg.printable_ratio_threshold):
            continue
        key = ("decoded_base64_lite", decoded)
        if key in seen:
            continue
        seen.add(key)
        decoded_segments += 1
        contexts.append(PreprocessContext(kind="decoded_base64_lite", raw_text=decoded, normalized_text=""))

    for match in URL_LITE_RE.finditer(text):
        if decoded_segments >= cfg.max_decode_segments:
            break
        payload = match.group(1).strip()
        if not payload or len(payload) > cfg.max_payload_chars:
            continue
        decoded = urllib.parse.unquote(payload)
        if len(decoded) > cfg.max_decoded_chars:
            continue
        if not _is_printable_ratio_ok(decoded, cfg.printable_ratio_threshold):
            continue
        key = ("decoded_url_lite", decoded)
        if key in seen:
            continue
        seen.add(key)
        decoded_segments += 1
        contexts.append(PreprocessContext(kind="decoded_url_lite", raw_text=decoded, normalized_text=""))
    return contexts


def _extract_joined_obfuscation_sequences(text: str, cfg: PreprocessConfig) -> List[str]:
    matches: List[str] = []
    seen = set()
    for match in OBFUSCATED_TOKEN_RE.finditer(text):
        joined = re.sub(r"[^a-zA-Z0-9]+", "", match.group(0))
        if len(joined) < 4:
            continue
        key = joined.lower()
        if key in seen:
            continue
        seen.add(key)
        matches.append(joined)
        if len(matches) >= cfg.max_join_sequences:
            break
    return matches


def _extract_markup_contexts(text: str) -> List[PreprocessContext]:
    out: List[PreprocessContext] = []
    for block in MARKDOWN_FENCE_RE.findall(text):
        out.append(PreprocessContext(kind="markdown_fence", raw_text=block, normalized_text=""))
    for block in INLINE_CODE_RE.findall(text):
        out.append(PreprocessContext(kind="inline_code", raw_text=block, normalized_text=""))
    for tag, inner in HTML_TAG_RE.findall(text):
        tag_l = tag.lower()
        if tag_l in ROLE_TAGS:
            out.append(PreprocessContext(kind=f"html_role_tag:{tag_l}", raw_text=inner, normalized_text=""))
        else:
            out.append(PreprocessContext(kind="html_inner_text", raw_text=inner, normalized_text=""))
    return out


def _bounded_contexts(
    contexts: Sequence[PreprocessContext],
    *,
    homoglyph_map: Dict[str, str],
    cfg: PreprocessConfig,
) -> List[PreprocessContext]:
    out: List[PreprocessContext] = []
    seen_norm = set()
    used_chars = 0
    for ctx in contexts:
        if len(out) >= cfg.max_context_blocks:
            break
        raw = _collapse_spaces(str(ctx.raw_text))
        if not raw:
            continue
        if len(raw) > cfg.max_context_chars_per_block:
            raw = raw[: cfg.max_context_chars_per_block]
        norm = _normalize_core(raw, homoglyph_map, cfg)
        if not norm:
            continue
        if norm in seen_norm:
            continue
        if used_chars + len(norm) > cfg.max_context_total_chars:
            break
        seen_norm.add(norm)
        used_chars += len(norm)
        out.append(PreprocessContext(kind=ctx.kind, raw_text=raw, normalized_text=norm))
    return out


def preprocess_text(text: str, *, homoglyph_map: Dict[str, str], cfg: Dict[str, Any] | None) -> PreprocessResult:
    pp_cfg = PreprocessConfig.from_cfg(cfg)
    if not pp_cfg.enabled:
        primary = _normalize_core(text, homoglyph_map, pp_cfg)
        return PreprocessResult(
            primary_text=primary,
            contexts=[],
            flags={"enabled": False, "nfkc": pp_cfg.nfkc, "remove_zero_width": pp_cfg.remove_zero_width},
            decoded_segments_count=0,
            joined_obfuscation_sequences_count=0,
        )

    primary = _normalize_core(text, homoglyph_map, pp_cfg)
    contexts_raw: List[PreprocessContext] = []

    decoded = _extract_wrapped_decoded_contexts(text, pp_cfg)
    contexts_raw.extend(decoded)

    for joined in _extract_joined_obfuscation_sequences(text, pp_cfg):
        contexts_raw.append(PreprocessContext(kind="joined_obfuscation", raw_text=joined, normalized_text=""))

    contexts_raw.extend(_extract_markup_contexts(text))
    contexts = _bounded_contexts(contexts_raw, homoglyph_map=homoglyph_map, cfg=pp_cfg)

    return PreprocessResult(
        primary_text=primary,
        contexts=contexts,
        flags={
            "enabled": True,
            "nfkc": pp_cfg.nfkc,
            "remove_zero_width": pp_cfg.remove_zero_width,
            "leet_homoglyph_limited": pp_cfg.leet_homoglyph_limited,
        },
        decoded_segments_count=sum(
            1 for ctx in contexts if ctx.kind in {"decoded_base64_lite", "decoded_url_lite"}
        ),
        joined_obfuscation_sequences_count=sum(1 for ctx in contexts if ctx.kind == "joined_obfuscation"),
    )


def normalize_text(text: str, homoglyph_map: Dict[str, str]) -> str:
    result = preprocess_text(text, homoglyph_map=homoglyph_map, cfg=None)
    return result.primary_text


def nospace_text(text: str) -> str:
    return text.replace(" ", "")


def tokenize(text: str) -> List[str]:
    return [tok for tok in text.split(" ") if tok]
