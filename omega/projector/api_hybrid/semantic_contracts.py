"""Strict provider-agnostic multimodal semantic contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import copy
import re
from typing import Any, Dict, Mapping, Optional, Tuple

from omega.interfaces.contracts_v1 import WALLS_V1

_WALLS = tuple(str(x) for x in WALLS_V1)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BLOB_REF_RE = re.compile(r"^blob://[A-Za-z0-9._:-]{1,160}/[A-Za-z0-9._:-]{8,160}$")
_MIME_RE = re.compile(r"^image/(png|jpeg|webp|gif)$")
_ALLOWED_IMAGE_ROLES = {
    "untrusted_visual_content",
    "full_page_context",
    "zoomed_region",
    "ocr_target_crop",
}
_ALLOWED_SEMANTIC_STATUSES = {
    "semantic_active",
    "vision_semantic_active",
    "vision_unsupported",
    "vision_redaction_blocked",
    "semantic_failed",
    "rule_based_only",
}


def _strict_bool(value: Any, *, field_name: str) -> bool:
    if type(value) is not bool:  # bool subclasses int, so use exact type.
        raise ValueError(f"{field_name} must be boolean")
    return value


def _strict_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field_name} must be finite")
    return out


def _find_raw_media(value: Any, *, path: str = "source_meta") -> Tuple[str, ...]:
    findings: list[str] = []
    forbidden_keys = {
        "image_bytes_b64",
        "image_variants",
        "bytes_b64",
        "raw_bytes",
        "file_bytes",
        "image_bytes",
    }
    if isinstance(value, (bytes, bytearray, memoryview)):
        findings.append(f"{path}:raw_bytes_value")
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            key_s = str(key)
            if key_s in forbidden_keys:
                findings.append(f"{path}.{key_s}")
            findings.extend(_find_raw_media(nested, path=f"{path}.{key_s}"))
    elif isinstance(value, (list, tuple)):
        for idx, nested in enumerate(value):
            findings.extend(_find_raw_media(nested, path=f"{path}[{idx}]"))
    return tuple(findings)


@dataclass(frozen=True)
class ProviderCapabilities:
    text: bool = True
    image: bool = False
    supported_image_mime_types: Tuple[str, ...] = ()
    max_image_bytes: int = 20 * 1024 * 1024
    max_images: int = 1

    def __post_init__(self) -> None:
        _strict_bool(self.text, field_name="capabilities.text")
        _strict_bool(self.image, field_name="capabilities.image")
        if int(self.max_image_bytes) <= 0:
            raise ValueError("capabilities.max_image_bytes must be > 0")
        if int(self.max_images) <= 0:
            raise ValueError("capabilities.max_images must be > 0")
        normalized = tuple(
            sorted(
                {
                    str(x).strip().lower()
                    for x in self.supported_image_mime_types
                    if str(x).strip()
                }
            )
        )
        for mime in normalized:
            if not _MIME_RE.fullmatch(mime):
                raise ValueError(f"unsupported image mime capability: {mime}")
        if self.image and not normalized:
            raise ValueError(
                "image-capable provider must declare supported_image_mime_types"
            )
        if not self.image and normalized:
            raise ValueError("text-only provider cannot declare image mime types")
        object.__setattr__(self, "supported_image_mime_types", normalized)
        object.__setattr__(self, "max_image_bytes", int(self.max_image_bytes))
        object.__setattr__(self, "max_images", int(self.max_images))

    def supports_input(self, semantic_input: "SemanticInput") -> bool:
        if semantic_input.text_parts and not self.text:
            return False
        if semantic_input.image_parts:
            if not self.image or len(semantic_input.image_parts) > self.max_images:
                return False
            return all(
                part.mime in self.supported_image_mime_types
                for part in semantic_input.image_parts
            )
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": bool(self.text),
            "image": bool(self.image),
            "supported_image_mime_types": list(self.supported_image_mime_types),
            "max_image_bytes": int(self.max_image_bytes),
            "max_images": int(self.max_images),
        }


@dataclass(frozen=True)
class SemanticTextPart:
    text: str
    role: str = "untrusted_text_content"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("SemanticTextPart.text must be string")
        if not str(self.role).strip():
            raise ValueError("SemanticTextPart.role must be non-empty")


@dataclass(frozen=True)
class SemanticImagePart:
    mime: str
    bytes_ref: str
    sha256: str
    role: str = "untrusted_visual_content"
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        mime = str(self.mime).strip().lower()
        ref = str(self.bytes_ref).strip()
        digest = str(self.sha256).strip().lower()
        role = str(self.role).strip()
        if not _MIME_RE.fullmatch(mime):
            raise ValueError(f"unsupported SemanticImagePart.mime: {mime}")
        if not _BLOB_REF_RE.fullmatch(ref):
            raise ValueError(
                "SemanticImagePart.bytes_ref must be an opaque blob:// handle"
            )
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError("SemanticImagePart.sha256 must be 64 lowercase hex chars")
        if role not in _ALLOWED_IMAGE_ROLES:
            raise ValueError(f"unsupported SemanticImagePart.role: {role}")
        for name, value in (
            ("width", self.width),
            ("height", self.height),
            ("size_bytes", self.size_bytes),
        ):
            if value is not None and int(value) <= 0:
                raise ValueError(f"SemanticImagePart.{name} must be > 0 when provided")
        object.__setattr__(self, "mime", mime)
        object.__setattr__(self, "bytes_ref", ref)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "role", role)
        if self.width is not None:
            object.__setattr__(self, "width", int(self.width))
        if self.height is not None:
            object.__setattr__(self, "height", int(self.height))
        if self.size_bytes is not None:
            object.__setattr__(self, "size_bytes", int(self.size_bytes))


@dataclass(frozen=True)
class SemanticInput:
    text_parts: Tuple[SemanticTextPart, ...] = field(default_factory=tuple)
    image_parts: Tuple[SemanticImagePart, ...] = field(default_factory=tuple)
    source_meta: Mapping[str, Any] = field(default_factory=dict)
    redaction_mode: str = "none"
    trace_hints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text_parts = tuple(self.text_parts)
        image_parts = tuple(self.image_parts)
        if not all(isinstance(x, SemanticTextPart) for x in text_parts):
            raise ValueError("SemanticInput.text_parts must contain SemanticTextPart")
        if not all(isinstance(x, SemanticImagePart) for x in image_parts):
            raise ValueError("SemanticInput.image_parts must contain SemanticImagePart")
        source_meta = dict(self.source_meta or {})
        leaked = sorted(set(_find_raw_media(source_meta)))
        if leaked:
            raise ValueError(
                f"SemanticInput.source_meta contains raw media fields: {','.join(leaked)}"
            )
        redaction_mode = str(self.redaction_mode).strip().lower()
        if redaction_mode not in {"none", "redacted"}:
            raise ValueError("SemanticInput.redaction_mode must be none|redacted")
        object.__setattr__(self, "text_parts", text_parts)
        object.__setattr__(self, "image_parts", image_parts)
        object.__setattr__(self, "source_meta", source_meta)
        object.__setattr__(self, "trace_hints", dict(self.trace_hints or {}))
        object.__setattr__(self, "redaction_mode", redaction_mode)


@dataclass(frozen=True)
class SemanticResult:
    pressure_signed: Dict[str, float]
    directive_intent: Dict[str, bool]
    defensive_context: bool
    confidence: float
    semantic_status: str
    provider_meta: Dict[str, Any] = field(default_factory=dict)
    vision_meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.pressure_signed) != set(_WALLS):
            raise ValueError(
                "semantic pressure_signed must contain exactly the v1 walls"
            )
        if set(self.directive_intent) != set(_WALLS):
            raise ValueError(
                "semantic directive_intent must contain exactly the v1 walls"
            )
        normalized_pressure: Dict[str, float] = {}
        normalized_intent: Dict[str, bool] = {}
        for wall in _WALLS:
            value = _strict_number(
                self.pressure_signed[wall], field_name=f"pressure_signed.{wall}"
            )
            if value < -1.0 or value > 1.0:
                raise ValueError(f"pressure_signed.{wall} must be in [-1,1]")
            normalized_pressure[wall] = value
            normalized_intent[wall] = _strict_bool(
                self.directive_intent[wall], field_name=f"directive_intent.{wall}"
            )
        defensive = _strict_bool(self.defensive_context, field_name="defensive_context")
        confidence = _strict_number(self.confidence, field_name="confidence")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("semantic confidence must be in [0,1]")
        status = str(self.semantic_status).strip().lower()
        if status not in _ALLOWED_SEMANTIC_STATUSES:
            raise ValueError(f"unsupported semantic_status: {status}")
        object.__setattr__(self, "pressure_signed", normalized_pressure)
        object.__setattr__(self, "directive_intent", normalized_intent)
        object.__setattr__(self, "defensive_context", defensive)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "semantic_status", status)
        object.__setattr__(self, "provider_meta", dict(self.provider_meta or {}))
        object.__setattr__(self, "vision_meta", dict(self.vision_meta or {}))

    @classmethod
    def from_payload(
        cls,
        *,
        payload: Mapping[str, Any],
        semantic_status: str,
        provider_meta: Optional[Mapping[str, Any]] = None,
        vision_meta: Optional[Mapping[str, Any]] = None,
    ) -> "SemanticResult":
        required = {
            "pressure_signed",
            "directive_intent",
            "defensive_context",
            "confidence",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(
                f"semantic payload missing required fields: {','.join(missing)}"
            )
        pressure_raw = payload.get("pressure_signed")
        intent_raw = payload.get("directive_intent")
        if not isinstance(pressure_raw, Mapping) or not isinstance(intent_raw, Mapping):
            raise ValueError(
                "semantic pressure_signed and directive_intent must be mappings"
            )
        return cls(
            pressure_signed=dict(pressure_raw),
            directive_intent=dict(intent_raw),
            defensive_context=payload.get("defensive_context"),
            confidence=payload.get("confidence"),
            semantic_status=semantic_status,
            provider_meta=dict(provider_meta or {}),
            vision_meta=dict(vision_meta or {}),
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": "api_hybrid_v2",
            "pressure_signed": dict(self.pressure_signed),
            "directive_intent": dict(self.directive_intent),
            "defensive_context": bool(self.defensive_context),
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class ProviderSemanticResponse:
    result: SemanticResult
    response_id: str = ""
    retries_used: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.result, SemanticResult):
            raise ValueError("ProviderSemanticResponse.result must be SemanticResult")
        if int(self.retries_used) < 0:
            raise ValueError("ProviderSemanticResponse.retries_used must be >= 0")
        object.__setattr__(self, "response_id", str(self.response_id or ""))
        object.__setattr__(self, "retries_used", int(self.retries_used))


@dataclass
class SemanticExecutionTrace:
    semantic_input_kind: str = "text_only"
    vision_attempted: bool = False
    vision_provider_supported: bool = False
    vision_failure_policy: str = "degrade"
    vision_fallback_used: bool = False
    vision_semantic_status: str = "none"
    provider_call_count: int = 0
    retry_count: int = 0
    cache_hit: bool = False
    semantic_latency_ms: Optional[float] = None
    first_pass_latency_ms: Optional[float] = None
    second_pass_latency_ms: Optional[float] = None
    second_pass_attempted: bool = False
    second_pass_result: str = "not_attempted"
    region_trigger_reason: str = "none"
    region_variant_count: int = 0
    token_usage: Dict[str, Any] = field(default_factory=dict)
    zero_mode: str = "none"
    semantic_status: str = "semantic_active"
    schema_valid: Optional[bool] = None
    error: Optional[str] = None
    provider: str = ""
    provider_id: str = ""
    provider_capabilities: Dict[str, Any] = field(default_factory=dict)
    provider_route: list[Dict[str, Any]] = field(default_factory=list)
    fallback_level: str = "none"
    fallback_reason: Optional[str] = None
    llm_fallback_active: bool = False
    health_state: str = "healthy"
    quota_signal: Optional[str] = None
    redaction: Dict[str, Any] = field(default_factory=dict)
    tenant_id: str = ""
    data_region: str = "unspecified"
    visual_egress_decision: str = "not_evaluated"
    visual_egress_reason: str = "none"
    provider_processing_region: str = ""

    def clone(self) -> "SemanticExecutionTrace":
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_input_kind": str(self.semantic_input_kind),
            "vision_attempted": bool(self.vision_attempted),
            "vision_provider_supported": bool(self.vision_provider_supported),
            "vision_failure_policy": str(self.vision_failure_policy),
            "vision_fallback_used": bool(self.vision_fallback_used),
            "vision_semantic_status": str(self.vision_semantic_status),
            "provider_call_count": int(self.provider_call_count),
            "retry_count": int(self.retry_count),
            "cache_hit": bool(self.cache_hit),
            "semantic_latency_ms": self.semantic_latency_ms,
            "first_pass_latency_ms": self.first_pass_latency_ms,
            "second_pass_latency_ms": self.second_pass_latency_ms,
            "second_pass_attempted": bool(self.second_pass_attempted),
            "second_pass_result": str(self.second_pass_result),
            "region_trigger_reason": str(self.region_trigger_reason),
            "region_variant_count": int(self.region_variant_count),
            "token_usage": dict(self.token_usage),
            "zero_mode": str(self.zero_mode),
            "semantic_status": str(self.semantic_status),
            "schema_valid": self.schema_valid,
            "error": self.error,
            "provider": str(self.provider),
            "provider_id": str(self.provider_id),
            "provider_capabilities": dict(self.provider_capabilities),
            "provider_route": [dict(x) for x in self.provider_route],
            "fallback_level": str(self.fallback_level),
            "fallback_reason": self.fallback_reason,
            "llm_fallback_active": bool(self.llm_fallback_active),
            "health_state": str(self.health_state),
            "quota_signal": self.quota_signal,
            "redaction": dict(self.redaction),
            "tenant_id": str(self.tenant_id),
            "data_region": str(self.data_region),
            "visual_egress_decision": str(self.visual_egress_decision),
            "visual_egress_reason": str(self.visual_egress_reason),
            "provider_processing_region": str(self.provider_processing_region),
        }
