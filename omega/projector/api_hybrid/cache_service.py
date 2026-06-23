"""Cache/error logging helpers for APIPerceptionProjector."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from omega.interfaces.contracts_v1 import ContentItem, WALLS_V1

from . import normalization as norm

WALLS = list(WALLS_V1)


class APIPerceptionCacheService:
    @staticmethod
    def load_cache(projector: Any) -> Dict[str, Dict[str, Any]]:
        cache_path = projector.cache_path
        if not cache_path.exists():
            return {}
        loaded: Dict[str, Dict[str, Any]] = {}
        try:
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                ln = line.strip()
                if not ln:
                    continue
                obj = json.loads(ln)
                if not isinstance(obj, Mapping):
                    continue
                key = str(obj.get("key", "")).strip()
                if not key:
                    continue
                try:
                    payload = norm.normalize_api_payload(obj)
                except Exception:  # noqa: BLE001
                    continue
                loaded[key] = {
                    "schema_version": str(payload["schema_version"]),
                    "pressure_signed": dict(payload["pressure_signed"]),
                    "directive_intent": dict(payload["directive_intent"]),
                    "defensive_context": bool(payload["defensive_context"]),
                    "confidence": float(payload["confidence"]),
                    "scores": dict(payload["scores"]),
                    "response_id": str(obj.get("response_id", "")),
                    "provider": str(obj.get("provider", projector.provider)),
                }
        except Exception:  # noqa: BLE001
            return {}
        return loaded

    @staticmethod
    def append_cache(*, projector: Any, key: str, payload: Mapping[str, Any], response_id: str) -> None:
        projector.cache_path.parent.mkdir(parents=True, exist_ok=True)
        pressure = {w: float((payload.get("pressure_signed") or {}).get(w, 0.0)) for w in WALLS}
        directive_intent = {w: bool((payload.get("directive_intent") or {}).get(w, False)) for w in WALLS}
        scores = {w: max(0.0, float((payload.get("scores") or {}).get(w, pressure[w]))) for w in WALLS}
        row = {
            "created_at_utc": norm.utc_now(),
            "key": str(key),
            "schema_version": str(payload.get("schema_version", norm.LEGACY_SCHEMA_COMPAT)),
            "pressure_signed": pressure,
            "directive_intent": directive_intent,
            "defensive_context": bool(payload.get("defensive_context", False)),
            "confidence": float(payload.get("confidence", norm.DEFAULT_CONFIDENCE)),
            "scores": scores,
            "response_id": str(response_id),
            "model": projector.model,
            "provider": projector.provider,
            "provider_id": (str(projector._active_provider_id) if projector._active_provider_id else projector.provider),
            "health_state": str(projector._active_health_state),
            "fallback_level": str(projector._active_fallback_level),
            "prompt_version": projector.prompt_version,
        }
        with projector.cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def log_error(
        *,
        projector: Any,
        item: ContentItem,
        cache_key: str,
        error: str,
        normalized_payload: Mapping[str, Any] | None = None,
    ) -> None:
        projector.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(normalized_payload or {})
        pressure = {w: float((payload.get("pressure_signed") or {}).get(w, 0.0)) for w in WALLS}
        directive_intent = {w: bool((payload.get("directive_intent") or {}).get(w, False)) for w in WALLS}
        scores = {w: max(0.0, float((payload.get("scores") or {}).get(w, pressure[w]))) for w in WALLS}
        text_raw = str(getattr(item, "text", "") or "")
        row = {
            "created_at_utc": norm.utc_now(),
            "doc_id": item.doc_id,
            "source_id": item.source_id,
            "cache_key": cache_key,
            "error": str(error),
            "text_sha256": norm.sha256_text(text_raw),
            "text_length": int(len(text_raw)),
            "schema_version": str(payload.get("schema_version", norm.LEGACY_SCHEMA_COMPAT)),
            "pressure_signed": pressure,
            "directive_intent": directive_intent,
            "defensive_context": bool(payload.get("defensive_context", False)),
            "confidence": float(payload.get("confidence", norm.DEFAULT_CONFIDENCE)),
            "scores": scores,
            "model": projector.model,
            "provider": projector.provider,
            "provider_id": (str(projector._active_provider_id) if projector._active_provider_id else projector.provider),
            "health_state": str(projector._active_health_state),
            "fallback_level": str(projector._active_fallback_level),
            "prompt_version": projector.prompt_version,
        }
        with projector.error_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

