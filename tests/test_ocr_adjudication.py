from __future__ import annotations

import io

import numpy as np

from omega.api.ocr_adjudication import (
    build_ocr_adjudication_items,
    build_ocr_adjudication_matrix_items,
    interpret_ocr_adjudication_projection,
)
from omega.interfaces.contracts_v1 import ProjectionEvidence, ProjectionResult
from omega.projector.api_hybrid.blob_store import ImageBlobStore
from omega.rag.attachment_ingestion import OCRSpan


def _png_bytes(width: int = 100, height: int = 100) -> bytes:
    from PIL import Image  # type: ignore

    out = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 255, 255)).save(out, format="PNG")
    return out.getvalue()



def _blob_registrar(scope_id: str = "test"):
    store = ImageBlobStore(max_total_bytes=8 * 1024 * 1024, max_records=32)

    def register(**payload):
        raw = bytes(payload["raw"])
        mime = str(payload["mime"])
        sha256 = str(payload["sha256"])
        return {
            "mime": mime,
            "sha256": sha256,
            "bytes_ref": store.put(scope_id=scope_id, data=raw, mime=mime, expected_sha256=sha256),
            "size_bytes": len(raw),
            "role": str(payload["role"]),
            "width": int(payload["width"]),
            "height": int(payload["height"]),
        }

    return register, store

def test_build_ocr_adjudication_items_keeps_local_tiles():
    raw = _png_bytes(width=400, height=200)
    register_image_payload, _store = _blob_registrar("items")
    span_lookup = {
        f"s{idx}": OCRSpan(
            span_id=f"s{idx}",
            text=f"token {idx}",
            confidence=0.99,
            polygon_px=((float(idx * 5), 10.0), (float(idx * 5 + 4), 10.0), (float(idx * 5 + 4), 16.0), (float(idx * 5), 16.0)),
            image_width=400,
            image_height=200,
            provider_order=idx,
        )
        for idx in range(10)
    }
    items, trace = build_ocr_adjudication_items(
        request_id="req",
        source_id="src",
        source_type="image",
        trust="untrusted",
        file_bytes=raw,
        triggered_span_ids=[f"s{idx}" for idx in range(10)],
        supporting_span_ids=[f"s{idx}" for idx in range(10)],
        source_walls=["override_instructions"],
        span_lookup=span_lookup,
        source_image_meta={"width": 400, "height": 200},
        register_image_payload=register_image_payload,
        crop_strategy="contextual",
        context_span_radius=2,
        max_context_spans=5,
        max_tiles=2,
        max_crop_area_ratio=0.35,
    )
    assert len(items) == 2
    assert trace["tile_count"] == 2
    assert max(len(tile["context_span_ids"]) for tile in trace["tiles"]) <= 5
    assert max(float(tile["crop_area_ratio"]) for tile in trace["tiles"]) < 0.35
    for item in items:
        image_meta = item.meta["semantic_image"]
        assert str(image_meta["bytes_ref"]).startswith("blob://")
        assert not {"bytes_b64", "raw_bytes", "file_bytes", "image_bytes"}.intersection(image_meta)


def test_build_ocr_adjudication_items_contextual_crop_expands_beyond_single_span():
    raw = _png_bytes(width=400, height=200)
    register_image_payload, _store = _blob_registrar("context")
    span_lookup = {
        "s0": OCRSpan(
            span_id="s0",
            text="Footer",
            confidence=0.99,
            polygon_px=((20.0, 100.0), (55.0, 100.0), (55.0, 118.0), (20.0, 118.0)),
            image_width=400,
            image_height=200,
            provider_order=0,
        ),
        "s1": OCRSpan(
            span_id="s1",
            text="Reply",
            confidence=0.99,
            polygon_px=((60.0, 100.0), (95.0, 100.0), (95.0, 118.0), (60.0, 118.0)),
            image_width=400,
            image_height=200,
            provider_order=1,
        ),
        "s2": OCRSpan(
            span_id="s2",
            text="Actions",
            confidence=0.99,
            polygon_px=((100.0, 100.0), (150.0, 100.0), (150.0, 118.0), (100.0, 118.0)),
            image_width=400,
            image_height=200,
            provider_order=2,
        ),
    }
    items, trace = build_ocr_adjudication_items(
        request_id="req",
        source_id="src",
        source_type="image",
        trust="untrusted",
        file_bytes=raw,
        triggered_span_ids=["s1"],
        supporting_span_ids=["s0", "s1", "s2"],
        source_walls=["override_instructions"],
        span_lookup=span_lookup,
        source_image_meta={"width": 400, "height": 200},
        register_image_payload=register_image_payload,
        crop_strategy="contextual",
        context_span_radius=1,
        max_context_spans=3,
        min_crop_width_px=160.0,
        min_crop_height_px=72.0,
    )
    assert len(items) == 1
    assert trace["variant_id"] == "contextual_image_text"
    assert trace["crop_strategy"] == "contextual"
    assert trace["tiles"][0]["context_span_ids"] == ["s0", "s1", "s2"]
    rect = trace["tiles"][0]["crop_rect_px"]
    assert float(rect["x_max"]) - float(rect["x_min"]) >= 160.0
    assert float(rect["y_max"]) - float(rect["y_min"]) >= 72.0


def test_build_ocr_adjudication_matrix_items_emits_four_variants():
    raw = _png_bytes(width=300, height=150)
    register_image_payload, _store = _blob_registrar("matrix")
    span_lookup = {
        "s0": OCRSpan(
            span_id="s0",
            text="Reply",
            confidence=0.99,
            polygon_px=((40.0, 40.0), (90.0, 40.0), (90.0, 58.0), (40.0, 58.0)),
            image_width=300,
            image_height=150,
            provider_order=0,
        )
    }
    items, trace = build_ocr_adjudication_matrix_items(
        request_id="req",
        source_id="src",
        source_type="image",
        trust="untrusted",
        file_bytes=raw,
        triggered_span_ids=["s0"],
        matched_span_ids=["s0"],
        supporting_span_ids=["s0"],
        source_walls=["override_instructions"],
        span_lookup=span_lookup,
        source_image_meta={"width": 300, "height": 150},
        register_image_payload=register_image_payload,
    )
    variant_ids = [str(v.get("variant_id", "")) for v in trace["variants"]]
    assert variant_ids == [
        "tiny_image_only",
        "tiny_image_text",
        "contextual_image_only",
        "contextual_image_text",
    ]
    assert len(items) == 4
    text_by_variant = {
        str(item.meta.get("ocr_adjudication_target", {}).get("variant_id", "")): str(item.text)
        for item in items
    }
    assert text_by_variant["tiny_image_only"] == ""
    assert text_by_variant["contextual_image_only"] == ""
    assert text_by_variant["tiny_image_text"] == "Reply"
    assert text_by_variant["contextual_image_text"] == "Reply"


def test_build_ocr_adjudication_matrix_items_omits_candidate_text_without_exact_match():
    raw = _png_bytes(width=300, height=150)
    register_image_payload, _store = _blob_registrar("matrix-no-match")
    span_lookup = {
        "s0": OCRSpan(
            span_id="s0",
            text="Reply",
            confidence=0.99,
            polygon_px=((40.0, 40.0), (90.0, 40.0), (90.0, 58.0), (40.0, 58.0)),
            image_width=300,
            image_height=150,
            provider_order=0,
        )
    }
    items, trace = build_ocr_adjudication_matrix_items(
        request_id="req",
        source_id="src",
        source_type="image",
        trust="untrusted",
        file_bytes=raw,
        triggered_span_ids=["s0"],
        matched_span_ids=[],
        supporting_span_ids=["s0"],
        source_walls=["override_instructions"],
        span_lookup=span_lookup,
        source_image_meta={"width": 300, "height": 150},
        register_image_payload=register_image_payload,
    )
    assert all(not bool(variant.get("exact_attribution", False)) for variant in trace["variants"])
    assert len(items) == 4
    assert all(str(item.text) == "" for item in items)


def test_interpret_ocr_adjudication_requires_source_wall_alignment():
    proj = ProjectionResult(
        doc_id="d1",
        v=np.array([0.0, 0.0, 0.0, 0.8], dtype=float),
        evidence=ProjectionEvidence(
            polarity=[0, -1, 0, 1],
            debug_scores_raw=[0.0, -0.8, 0.0, 0.8],
            matches={"api_perception": {"confidence": 0.9, "defensive_context": True, "vision_semantic_status": "vision_semantic_active"}},
        ),
    )
    out = interpret_ocr_adjudication_projection(
        projection=proj,
        source_walls=["override_instructions"],
        threshold=0.12,
    )
    assert out["result"] == "quoted_or_defensive"


def test_interpret_ocr_adjudication_benign_needs_high_confidence_on_source_wall():
    proj = ProjectionResult(
        doc_id="d1",
        v=np.zeros(4, dtype=float),
        evidence=ProjectionEvidence(
            polarity=[-1, 0, 0, 0],
            debug_scores_raw=[-0.4, 0.0, 0.0, 0.0],
            matches={"api_perception": {"confidence": 0.6, "defensive_context": True, "vision_semantic_status": "vision_semantic_active"}},
        ),
    )
    out = interpret_ocr_adjudication_projection(
        projection=proj,
        source_walls=["override_instructions"],
        threshold=0.12,
    )
    assert out["result"] == "insufficient_context"


def test_interpret_ocr_adjudication_malicious_needs_source_wall_match():
    proj = ProjectionResult(
        doc_id="d1",
        v=np.array([0.0, 0.8, 0.0, 0.0], dtype=float),
        evidence=ProjectionEvidence(
            polarity=[0, 1, 0, 0],
            debug_scores_raw=[0.0, 0.8, 0.0, 0.0],
            matches={"api_perception": {"confidence": 0.95, "defensive_context": False, "vision_semantic_status": "vision_semantic_active"}},
        ),
    )
    out = interpret_ocr_adjudication_projection(
        projection=proj,
        source_walls=["override_instructions"],
        threshold=0.12,
    )
    assert out["result"] == "insufficient_context"


def test_interpret_ocr_adjudication_recognizes_benign_ui_without_directive_intent():
    proj = ProjectionResult(
        doc_id="d1",
        v=np.zeros(4, dtype=float),
        evidence=ProjectionEvidence(
            polarity=[0, 0, 0, 0],
            debug_scores_raw=[0.0, 0.0, 0.0, 0.0],
            matches={
                "api_perception": {
                    "confidence": 0.93,
                    "defensive_context": False,
                    "directive_intent": {"override_instructions": False},
                    "vision_semantic_status": "vision_semantic_active",
                }
            },
        ),
    )
    out = interpret_ocr_adjudication_projection(
        projection=proj,
        source_walls=["override_instructions"],
        threshold=0.12,
    )
    assert out["result"] == "benign_ui"


def test_interpret_ocr_adjudication_recognizes_live_attack_with_directive_intent():
    proj = ProjectionResult(
        doc_id="d1",
        v=np.array([0.8, 0.0, 0.0, 0.0], dtype=float),
        evidence=ProjectionEvidence(
            polarity=[1, 0, 0, 0],
            debug_scores_raw=[0.8, 0.0, 0.0, 0.0],
            matches={
                "api_perception": {
                    "confidence": 0.95,
                    "defensive_context": False,
                    "directive_intent": {"override_instructions": True},
                    "vision_semantic_status": "vision_semantic_active",
                }
            },
        ),
    )
    out = interpret_ocr_adjudication_projection(
        projection=proj,
        source_walls=["override_instructions"],
        threshold=0.12,
    )
    assert out["result"] == "live_attack"
