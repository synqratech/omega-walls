"""Typed OpenAPI contracts for the attachment scanning surface.

The runtime parser intentionally remains content-type aware and streaming-safe. These
models describe the public additive contract without forcing FastAPI to buffer an
UploadFile before the request-size middleware has enforced its limits.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


ImageMime = Literal["image/png", "image/jpeg", "image/webp", "image/gif"]
RuntimeMode = Literal["stateless", "stateful"]


class AttachmentScanJSONRequest(BaseModel):
    # The runtime parser is additive/forward-compatible and ignores unknown fields.
    # Reflect that behavior truthfully in generated OpenAPI.
    model_config = ConfigDict(extra="allow")

    tenant_id: str = Field(min_length=1, description="Tenant isolation key.")
    request_id: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Caller request id. Required when HMAC authentication is enabled.",
    )
    session_id: Optional[str] = Field(default=None, min_length=1)
    actor_id: Optional[str] = Field(default=None, min_length=1)
    runtime_mode: Optional[RuntimeMode] = None
    data_region: Optional[str] = Field(
        default="unspecified",
        pattern=r"^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$",
        description="Requested data-residency region used by tenant visual-egress policy.",
    )
    filename: Optional[str] = None
    mime: Optional[str] = Field(
        default=None,
        description=(
            "Declared media type. Images are first-class inputs on this endpoint; "
            "supported image types are image/png, image/jpeg, image/webp and image/gif."
        ),
    )
    file_base64: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Base64-encoded attachment bytes. Raw media is never returned in traces or logs.",
    )
    extracted_text: Optional[str] = Field(
        default=None,
        description="Optional caller-provided extracted text. May be supplied with or without a file.",
    )


class VisionProviderCapabilitiesTrace(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: bool = True
    image: bool = False
    supported_image_mime_types: List[str] = Field(default_factory=list)
    max_image_bytes: Optional[int] = None
    max_images: Optional[int] = None


class ProviderRouteTrace(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider_id: str
    provider_type: str
    model: Optional[str] = None
    key_slot: Optional[str] = None
    selected: bool = False
    status: str
    error: Optional[str] = None
    capabilities: VisionProviderCapabilitiesTrace


class OCRQualityTrace(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_spans: int = 0
    kept_spans: int = 0
    dropped_empty: int = 0
    dropped_low_confidence: int = 0
    dropped_invalid_confidence: int = 0
    dropped_invalid_geometry: int = 0
    dropped_over_limit: int = 0
    clipped_span_texts: int = 0
    mean_confidence: Optional[float] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    geometry_coverage_ratio: float = 0.0
    status: str = "none"


class PolicyTraceResponse(BaseModel):
    """Stable security fields plus an additive extension area.

    Omega has a deliberately rich trace. `extra="allow"` keeps v1 clients forward
    compatible while making the multimodal fields explicit in generated OpenAPI.
    """

    model_config = ConfigDict(extra="allow")

    trace_id: str = ""
    decision_id: str = ""
    control_outcome: str = "allow"
    off: bool = False
    severity: str = "L1"
    walls_triggered: List[str] = Field(default_factory=list)
    action_types: List[str] = Field(default_factory=list)
    semantic_failure_status: str = "none"
    semantic_failure_policy: str = "none"
    semantic_failure_policy_branch: str = "none"
    vision_attempted: bool = False
    vision_provider_supported: bool = False
    vision_failure_policy: str = "none"
    vision_fallback_used: bool = False
    vision_semantic_status: str = "none"
    semantic_input_kind: str = "text_only"
    visual_status: str = "none"
    visual_asset_count: int = 0
    visual_asset_manifest: List[Dict[str, Any]] = Field(default_factory=list)
    data_region: str = "unspecified"
    visual_egress_decision: str = "not_evaluated"
    visual_egress_reason: str = "none"
    provider_processing_region: str = ""
    ocr_status: str = "none"
    ocr_provider: Optional[str] = None
    ocr_text_chars: int = 0
    ocr_quality: OCRQualityTrace = Field(default_factory=OCRQualityTrace)
    ocr_modality_present: bool = False
    ocr_active_walls: List[str] = Field(default_factory=list)
    vision_active_walls: List[str] = Field(default_factory=list)
    ocr_vision_agreement: bool = False
    ocr_gate_applied: bool = False
    ocr_gate_reason: str = "none"
    ocr_adjudication_status: str = "not_needed"
    ocr_adjudication_result: str = "none"
    image_region_pass_enabled: bool = False
    second_pass_attempted: bool = False
    second_pass_result: str = "not_attempted"
    region_trigger_reason: str = "none"
    region_variant_count: int = 0
    provider_capabilities: Optional[VisionProviderCapabilitiesTrace] = None
    provider_route: List[ProviderRouteTrace] = Field(default_factory=list)
    provider_call_count: int = 0
    retry_count: int = 0
    cache_hit_last_request: bool = False
    semantic_latency_ms: Optional[float] = None
    token_usage: Dict[str, Any] = Field(default_factory=dict)


class EvidenceSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    walls_triggered: List[str] = Field(default_factory=list)
    rule_ids: List[str] = Field(default_factory=list)
    chunk_ids: List[str] = Field(default_factory=list)
    top_chunk_ids: List[str] = Field(default_factory=list)
    text_included: bool = False
    control_outcome: str
    trace_id: str
    decision_id: str


class AttachmentScanResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: str
    trace_id: str = ""
    decision_id: str = ""
    tenant_id: str
    risk_score: int = Field(ge=0, le=100)
    verdict: str
    content_verdict: str = "allow"
    effective_verdict: str = "allow"
    control_outcome: str = "allow"
    reasons: List[str] = Field(default_factory=list)
    evidence_id: str
    evidence: Optional[EvidenceSummaryResponse] = None
    policy_trace: PolicyTraceResponse = Field(default_factory=PolicyTraceResponse)
    response_constraints: Dict[str, Any] = Field(default_factory=dict)


class ScanErrorResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    detail: Any


ATTACHMENT_SCAN_REQUEST_BODY = {
    "required": True,
    "content": {
        "application/json": {
            "schema": AttachmentScanJSONRequest.model_json_schema(),
            "examples": {
                "image_base64": {
                    "summary": "Image attachment using the existing attachment route",
                    "value": {
                        "tenant_id": "tenant-1",
                        "request_id": "req-1",
                        "filename": "page.png",
                        "mime": "image/png",
                        "file_base64": "iVBORw0KGgo...",
                    },
                },
                "text_and_image": {
                    "summary": "Image plus caller-provided text",
                    "value": {
                        "tenant_id": "tenant-1",
                        "request_id": "req-2",
                        "filename": "screenshot.webp",
                        "mime": "image/webp",
                        "file_base64": "UklGRiIAAABXRUJQ...",
                        "extracted_text": "Optional OCR or surrounding text",
                    },
                },
            },
        },
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "required": ["tenant_id"],
                "properties": {
                    "tenant_id": {"type": "string", "minLength": 1},
                    "request_id": {"type": "string"},
                    "session_id": {"type": "string"},
                    "actor_id": {"type": "string"},
                    "runtime_mode": {
                        "type": "string",
                        "enum": ["stateless", "stateful"],
                    },
                    "filename": {"type": "string"},
                    "mime": {
                        "type": "string",
                        "description": "Declared MIME; verified against attachment magic bytes.",
                    },
                    "extracted_text": {"type": "string"},
                    "file": {
                        "type": "string",
                        "format": "binary",
                        "description": (
                            "Attachment. Image semantic availability depends on the selected provider. "
                            "Unsupported or redaction-blocked vision follows the configured semantic failure policy; "
                            "it is never silently treated as a confident clean result."
                        ),
                    },
                },
                "anyOf": [{"required": ["file"]}, {"required": ["extracted_text"]}],
            }
        },
    },
}
