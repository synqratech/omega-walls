"""API-backed perception projector and hybrid combiner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np

from omega.interfaces.contracts_v1 import ContentItem, ProjectionEvidence, ProjectionResult, WALLS_V1

WALLS = list(WALLS_V1)
API_HYBRID_SCHEMA_V2 = "api_hybrid_v2"
LEGACY_SCHEMA_COMPAT = "v1_compat"
DEFAULT_CONFIDENCE = 0.5


class APIRequestError(RuntimeError):
    def __init__(self, *, code: int, body: str):
        self.code = int(code)
        self.body = str(body)
        super().__init__(f"HTTP {self.code}: {self.body}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _zero_projection(item: ContentItem, reason: str) -> ProjectionResult:
    return ProjectionResult(
        doc_id=item.doc_id,
        v=np.zeros(4, dtype=float),
        evidence=ProjectionEvidence(
            polarity=[0, 0, 0, 0],
            debug_scores_raw=[0.0, 0.0, 0.0, 0.0],
            matches={"api_perception": {"active": False, "reason": reason}},
        ),
    )


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _contains_any_marker(text: str, markers: Tuple[str, ...]) -> bool:
    t = str(text or "")
    return any(m and (m in t) for m in markers)


def _sha256_text(value: str) -> str:
    h = hashlib.sha256()
    h.update(str(value).encode("utf-8"))
    return h.hexdigest()


def _post_json(*, url: str, payload: Mapping[str, Any], headers: Mapping[str, str], timeout_sec: float) -> Dict[str, Any]:
    data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(url=url, data=data, headers=dict(headers), method="POST")
    try:
        with urlrequest.urlopen(req, timeout=float(timeout_sec)) as resp:
            raw = resp.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            body = str(exc)
        raise APIRequestError(code=int(exc.code), body=body) from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"url_error: {exc}") from exc
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("api response is not a JSON object")
    return parsed


def _extract_output_text(resp: Mapping[str, Any]) -> str:
    output = resp.get("output")
    if isinstance(output, list):
        parts = []
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, Mapping):
                    continue
                ctype = str(c.get("type", "")).strip().lower()
                if ctype in {"output_text", "text"}:
                    txt = c.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
        if parts:
            return "\n".join(parts).strip()

    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            msg = first.get("message")
            if isinstance(msg, Mapping):
                content = msg.get("content")
                if isinstance(content, str):
                    return content.strip()
    return ""


def _validate_api_pressure_signed(obj: Mapping[str, Any]) -> Dict[str, float]:
    pressure_obj = obj.get("pressure_signed")
    if isinstance(pressure_obj, Mapping):
        out: Dict[str, float] = {}
        for wall in WALLS:
            if wall not in pressure_obj:
                raise ValueError(f"schema_error: missing {wall}")
            value = float(pressure_obj[wall])
            if value < -1.0 or value > 1.0:
                raise ValueError(f"schema_error: {wall} out of [-1,1]")
            out[wall] = value
        return out

    # Backward compatibility: old adapters may emit positive-only scores.
    scores_obj = obj.get("scores")
    if isinstance(scores_obj, Mapping):
        out = {}
        for wall in WALLS:
            if wall not in scores_obj:
                raise ValueError(f"schema_error: missing {wall}")
            value = float(scores_obj[wall])
            if value < 0.0 or value > 1.0:
                raise ValueError(f"schema_error: {wall} out of [0,1]")
            out[wall] = value
        return out

    if all(wall in obj for wall in WALLS):
        out = {}
        for wall in WALLS:
            value = float(obj[wall])
            if value < -1.0 or value > 1.0:
                raise ValueError(f"schema_error: {wall} out of [-1,1]")
            out[wall] = value
        return out

    raise ValueError("schema_error: pressure_signed or scores object required")


def _validate_api_scores(obj: Mapping[str, Any]) -> Dict[str, float]:
    # Backward-compatible alias used by older tests/callers.
    return _validate_api_pressure_signed(obj)


def _validate_directive_intent(obj: Any, *, pressure_signed: Mapping[str, float]) -> Dict[str, bool]:
    if isinstance(obj, Mapping):
        out: Dict[str, bool] = {}
        for wall in WALLS:
            if wall not in obj:
                raise ValueError(f"schema_error: missing directive_intent.{wall}")
            out[wall] = bool(obj[wall])
        return out
    # Compat fallback for legacy payloads.
    return {wall: float(pressure_signed.get(wall, 0.0)) > 0.0 for wall in WALLS}


def _normalize_api_payload(obj: Mapping[str, Any]) -> Dict[str, Any]:
    pressure_signed = _validate_api_pressure_signed(obj)
    schema_version_raw = str(obj.get("schema_version", "")).strip()
    is_v2 = schema_version_raw == API_HYBRID_SCHEMA_V2 and isinstance(obj.get("pressure_signed"), Mapping)

    if is_v2:
        confidence = float(obj.get("confidence", DEFAULT_CONFIDENCE))
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("schema_error: confidence out of [0,1]")
        defensive_context = bool(obj.get("defensive_context", False))
        directive_intent = _validate_directive_intent(obj.get("directive_intent"), pressure_signed=pressure_signed)
        schema_version = API_HYBRID_SCHEMA_V2
    else:
        confidence = DEFAULT_CONFIDENCE
        defensive_context = False
        directive_intent = _validate_directive_intent(None, pressure_signed=pressure_signed)
        schema_version = LEGACY_SCHEMA_COMPAT

    scores = {wall: max(0.0, float(pressure_signed[wall])) for wall in WALLS}
    return {
        "schema_version": schema_version,
        "pressure_signed": {wall: float(pressure_signed[wall]) for wall in WALLS},
        "directive_intent": {wall: bool(directive_intent[wall]) for wall in WALLS},
        "defensive_context": bool(defensive_context),
        "confidence": float(confidence),
        "scores": scores,
    }


def _is_transient_api_error(err: str) -> bool:
    t = str(err or "").lower()
    if "api_call_failed:" not in t:
        return False
    return (
        ("http 5" in t)
        or ("http 429" in t)
        or ("http 409" in t)
        or ("http 408" in t)
        or ("url_error" in t)
        or ("timed out" in t)
        or ("timeout" in t)
        or ("connection reset" in t)
        or ("temporar" in t)
    )


@dataclass
class APIPerceptionProjector:
    config: Dict[str, Any]

    def __post_init__(self) -> None:
        projector_cfg = self.config.get("projector", {}) or {}
        api_cfg = projector_cfg.get("api_perception", {}) or {}
        self.enabled_mode = str(api_cfg.get("enabled", "auto")).lower()
        self.strict = bool(api_cfg.get("strict", False))
        self.model = str(api_cfg.get("model", "gpt-5"))
        self.base_url = str(api_cfg.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        self.api_key_env = str(api_cfg.get("api_key_env", "OPENAI_API_KEY"))
        self.timeout_sec = float(api_cfg.get("timeout_sec", 30.0))
        self.max_retries = int(api_cfg.get("max_retries", 2))
        self.backoff_sec = float(api_cfg.get("backoff_sec", 0.75))
        self.retry_backoff_max_sec = float(api_cfg.get("retry_backoff_max_sec", 2.0))
        self.request_deadline_sec = float(api_cfg.get("request_deadline_sec", 20.0))
        self.long_text_threshold_chars = int(api_cfg.get("long_text_threshold_chars", 3000))
        self.long_text_max_retries = int(api_cfg.get("long_text_max_retries", 1))
        self.short_text_threshold_chars = int(api_cfg.get("short_text_threshold_chars", 1200))
        self.short_prefer_chat_completions = bool(api_cfg.get("short_prefer_chat_completions", True))
        self.short_chat_only = bool(api_cfg.get("short_chat_only", True))
        self.short_fast_path_enabled = bool(api_cfg.get("short_fast_path_enabled", True))
        self.short_fast_path_skip_on_pi0_hard = bool(api_cfg.get("short_fast_path_skip_on_pi0_hard", True))
        self.short_fast_path_skip_on_pi0_clean = bool(api_cfg.get("short_fast_path_skip_on_pi0_clean", True))
        self.short_fast_path_hard_min_score = float(api_cfg.get("short_fast_path_hard_min_score", 0.55))
        self.short_fast_path_clean_max_score = float(api_cfg.get("short_fast_path_clean_max_score", 0.0))
        self.transient_error_ttl_sec = float(api_cfg.get("transient_error_ttl_sec", 90.0))
        self.responses_cooldown_sec = float(api_cfg.get("responses_cooldown_sec", 60.0))
        self.prewarm_on_init = bool(api_cfg.get("prewarm_on_init", True))
        self.prompt_version = str(api_cfg.get("prompt_version", "api_hybrid_v1"))
        self.cache_path = Path(str(api_cfg.get("cache_path", "artifacts/projector_api/cache.jsonl")))
        self.error_log_path = Path(str(api_cfg.get("error_log_path", "artifacts/projector_api/errors.jsonl")))
        benign_task_cfg = api_cfg.get("benign_task_guard", {}) if isinstance(api_cfg.get("benign_task_guard", {}), Mapping) else {}
        marker_rows = benign_task_cfg.get("marker_phrases", []) if isinstance(benign_task_cfg.get("marker_phrases", []), list) else []
        attack_rows = benign_task_cfg.get("attack_cues", []) if isinstance(benign_task_cfg.get("attack_cues", []), list) else []
        self.benign_task_guard_markers = tuple(
            sorted(
                {
                    _normalize_text(str(x)).lower()
                    for x in marker_rows
                    if _normalize_text(str(x))
                }
            )
        )
        self.benign_task_guard_attack_cues = tuple(
            sorted(
                {
                    _normalize_text(str(x)).lower()
                    for x in attack_rows
                    if _normalize_text(str(x))
                }
            )
        )
        self.benign_task_guard_enabled = bool(
            benign_task_cfg.get("enabled", False)
            and bool(self.benign_task_guard_markers)
        )
        self.benign_task_guard_require_pi0_hard_absent = bool(
            benign_task_cfg.get("require_pi0_hard_absent", True)
        )

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._schema_errors = 0
        self._active = False
        self._runtime_error: Optional[str] = None
        self._last_error: Optional[str] = None
        self._last_schema_valid: Optional[bool] = None
        self._api_key: str = ""
        self._auth_headers: Dict[str, str] = {}
        self._responses_url: str = ""
        self._chat_url: str = ""
        self._prewarmed: bool = False
        self._transient_error_cache: Dict[str, Tuple[float, str]] = {}
        self._responses_degraded_until: float = 0.0

        self._load_cache()
        self._init_runtime()
        if self.enabled_mode == "true" and not self._active:
            raise RuntimeError(self._runtime_error or "api adapter inactive")

    def _init_runtime(self) -> None:
        if self.enabled_mode == "false":
            self._runtime_error = "api_adapter_disabled"
            self._active = False
            return
        api_key = str(os.getenv(self.api_key_env, "")).strip()
        if not api_key:
            self._runtime_error = f"missing_env:{self.api_key_env}"
            self._active = False
            return
        self._api_key = api_key
        self._auth_headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        self._responses_url = self.base_url + "/responses"
        self._chat_url = self.base_url + "/chat/completions"
        self._runtime_error = None
        self._active = True
        if self.prewarm_on_init:
            self._prewarm_runtime()

    def _prewarm_runtime(self) -> None:
        # Keep prewarm side-effect free (no network call).
        # We only materialize request primitives once at startup.
        if self._prewarmed:
            return
        _ = (self._auth_headers, self._responses_url, self._chat_url)
        self._prewarmed = True

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
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
                    payload = _normalize_api_payload(obj)
                except Exception:  # noqa: BLE001
                    continue
                self._cache[key] = {
                    "schema_version": str(payload["schema_version"]),
                    "pressure_signed": dict(payload["pressure_signed"]),
                    "directive_intent": dict(payload["directive_intent"]),
                    "defensive_context": bool(payload["defensive_context"]),
                    "confidence": float(payload["confidence"]),
                    "scores": dict(payload["scores"]),
                    "response_id": str(obj.get("response_id", "")),
                }
        except Exception:  # noqa: BLE001
            self._cache = {}

    def _append_cache(self, *, key: str, payload: Mapping[str, Any], response_id: str) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        pressure = {w: float((payload.get("pressure_signed") or {}).get(w, 0.0)) for w in WALLS}
        directive_intent = {w: bool((payload.get("directive_intent") or {}).get(w, False)) for w in WALLS}
        scores = {w: max(0.0, float((payload.get("scores") or {}).get(w, pressure[w]))) for w in WALLS}
        row = {
            "created_at_utc": _utc_now(),
            "key": str(key),
            "schema_version": str(payload.get("schema_version", LEGACY_SCHEMA_COMPAT)),
            "pressure_signed": pressure,
            "directive_intent": directive_intent,
            "defensive_context": bool(payload.get("defensive_context", False)),
            "confidence": float(payload.get("confidence", DEFAULT_CONFIDENCE)),
            "scores": scores,
            "response_id": str(response_id),
            "model": self.model,
            "prompt_version": self.prompt_version,
        }
        with self.cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _log_error(
        self,
        *,
        item: ContentItem,
        cache_key: str,
        error: str,
        raw_text: str = "",
        normalized_payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(normalized_payload or {})
        pressure = {w: float((payload.get("pressure_signed") or {}).get(w, 0.0)) for w in WALLS}
        directive_intent = {w: bool((payload.get("directive_intent") or {}).get(w, False)) for w in WALLS}
        scores = {w: max(0.0, float((payload.get("scores") or {}).get(w, pressure[w]))) for w in WALLS}
        row = {
            "created_at_utc": _utc_now(),
            "doc_id": item.doc_id,
            "source_id": item.source_id,
            "cache_key": cache_key,
            "error": str(error),
            "raw_text": str(raw_text)[:2000],
            "schema_version": str(payload.get("schema_version", LEGACY_SCHEMA_COMPAT)),
            "pressure_signed": pressure,
            "directive_intent": directive_intent,
            "defensive_context": bool(payload.get("defensive_context", False)),
            "confidence": float(payload.get("confidence", DEFAULT_CONFIDENCE)),
            "scores": scores,
            "model": self.model,
            "prompt_version": self.prompt_version,
        }
        with self.error_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def ensure_api_adapter_active(self) -> bool:
        return bool(self._active)

    def api_perception_status(self) -> Dict[str, Any]:
        total = int(self._cache_hits + self._cache_misses)
        hit_rate = float(self._cache_hits) / float(total) if total > 0 else 0.0
        err = self._runtime_error if self._runtime_error else self._last_error
        return {
            "api_adapter_active": bool(self._active),
            "api_adapter_error": err,
            "schema_valid": self._last_schema_valid,
            "model": self.model,
            "cache_hit_rate": hit_rate,
            "cache_hits": int(self._cache_hits),
            "cache_misses": int(self._cache_misses),
            "schema_errors": int(self._schema_errors),
        }

    def semantic_status(self) -> Dict[str, Any]:
        status = self.api_perception_status()
        return {
            "enabled_mode": "n/a",
            "active": False,
            "attempted": False,
            "model_path": None,
            "error": None,
            "docs_total": 0,
            "docs_with_boost": 0,
            "docs_polarity_promoted": 0,
            "docs_guard_suppressed": 0,
            "docs_with_boost_rate": 0.0,
            "promoted_polarity_rate": 0.0,
            "guard_suppression_rate": 0.0,
            "api_adapter_active": bool(status["api_adapter_active"]),
            "api_adapter_error": status["api_adapter_error"],
            "schema_valid": status["schema_valid"],
            "model": status["model"],
            "cache_hit_rate": float(status["cache_hit_rate"]),
        }

    def pitheta_conversion_status(self) -> Dict[str, Any]:
        return {
            "active": False,
            "error": "not_supported_for_api_hybrid",
            "api_perception": self.api_perception_status(),
        }

    def _build_messages(self, *, text: str) -> Tuple[str, str]:
        system_prompt = (
            "Return strict JSON only with this exact schema:\n"
            "{"
            "\"schema_version\":\"api_hybrid_v2\","
            "\"pressure_signed\":{\"override_instructions\":number,\"secret_exfiltration\":number,"
            "\"tool_or_action_abuse\":number,\"policy_evasion\":number},"
            "\"directive_intent\":{\"override_instructions\":boolean,\"secret_exfiltration\":boolean,"
            "\"tool_or_action_abuse\":boolean,\"policy_evasion\":boolean},"
            "\"defensive_context\":boolean,"
            "\"confidence\":number"
            "}\n"
            "Constraints: pressure_signed in [-1,1], confidence in [0,1]. "
            "Interpretation: +1 directive malicious pressure, -1 defensive/protective pressure, 0 neutral. "
            "No markdown, no prose, no extra keys."
        )
        user_prompt = (
            "Analyze the text below for prompt-injection pressure and return JSON only.\n\n"
            f"TEXT:\n{text}"
        )
        return system_prompt, user_prompt

    def _call_api_scores(self, *, text: str) -> Tuple[Dict[str, Any], str]:
        if not self._api_key:
            raise RuntimeError("missing_api_key")
        system_prompt, user_prompt = self._build_messages(text=text)
        headers = dict(self._auth_headers) if self._auth_headers else {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        responses_url = self._responses_url or (self.base_url + "/responses")
        chat_url = self._chat_url or (self.base_url + "/chat/completions")
        use_temperature = True
        last_exc: Optional[Exception] = None
        started_at = time.monotonic()
        is_short_text = len(str(text or "")) <= int(self.short_text_threshold_chars)
        is_long_text = len(str(text or "")) >= int(self.long_text_threshold_chars)
        short_force_chat = bool(is_short_text and self.short_chat_only)
        short_prefer_chat = bool(is_short_text and self.short_prefer_chat_completions)
        effective_max_retries = int(self.max_retries)
        if is_long_text:
            effective_max_retries = min(effective_max_retries, int(self.long_text_max_retries))

        def _build_payloads() -> Tuple[Dict[str, Any], Dict[str, Any]]:
            responses_payload: Dict[str, Any] = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
                ],
                "metadata": {"prompt_version": self.prompt_version},
            }
            chat_payload: Dict[str, Any] = {
                "model": self.model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "metadata": {"prompt_version": self.prompt_version},
            }
            if use_temperature:
                responses_payload["temperature"] = 0
                chat_payload["temperature"] = 0
            return responses_payload, chat_payload

        def _is_temperature_unsupported(msg: str) -> bool:
            t = str(msg or "").lower()
            return ("temperature" in t) and ("unsupported" in t or "does not support" in t)

        def _is_retryable_http(code: int) -> bool:
            c = int(code)
            return c in {408, 409, 429} or c >= 500

        def _remaining_timeout_sec() -> float:
            if self.request_deadline_sec <= 0.0:
                return float(self.timeout_sec)
            elapsed = time.monotonic() - started_at
            remaining = float(self.request_deadline_sec) - float(elapsed)
            if remaining <= 0.0:
                raise TimeoutError("request_deadline_exceeded")
            return min(float(self.timeout_sec), remaining)

        def _call_with_timeout(*, url: str, payload: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
            timeout_sec = _remaining_timeout_sec()
            resp = _post_json(url=url, payload=payload, headers=headers, timeout_sec=timeout_sec)
            txt = _extract_output_text(resp)
            parsed = json.loads(txt) if txt else {}
            if not isinstance(parsed, Mapping):
                raise ValueError("schema_error: top-level JSON object required")
            return _normalize_api_payload(parsed), str(resp.get("id", ""))

        for attempt in range(effective_max_retries + 1):
            responses_payload, chat_payload = _build_payloads()
            retryable = False
            try:
                prefer_chat = short_force_chat or short_prefer_chat or (time.monotonic() < float(self._responses_degraded_until))
                if not prefer_chat:
                    try:
                        return _call_with_timeout(url=responses_url, payload=responses_payload)
                    except APIRequestError as exc:
                        if use_temperature and _is_temperature_unsupported(exc.body):
                            use_temperature = False
                            last_exc = exc
                            continue
                        retryable = _is_retryable_http(exc.code)
                        if retryable and self.responses_cooldown_sec > 0.0:
                            self._responses_degraded_until = max(
                                float(self._responses_degraded_until),
                                float(time.monotonic() + float(self.responses_cooldown_sec)),
                            )
                        if exc.code not in {400, 404, 405, 415, 422} and not retryable:
                            last_exc = exc
                            if attempt >= effective_max_retries:
                                break
                            continue
                    except Exception as exc:  # noqa: BLE001
                        last_exc = exc
                        retryable = True

                # Fallback or preferred path.
                try:
                    return _call_with_timeout(url=chat_url, payload=chat_payload)
                except APIRequestError as chat_exc:
                    if use_temperature and _is_temperature_unsupported(chat_exc.body):
                        use_temperature = False
                        last_exc = chat_exc
                        continue
                    retryable = retryable or _is_retryable_http(chat_exc.code)
                    last_exc = chat_exc
                except Exception as chat_exc:  # noqa: BLE001
                    retryable = True
                    last_exc = chat_exc

                if (not retryable) or attempt >= effective_max_retries:
                    break
                sleep_sec = min(
                    float(self.retry_backoff_max_sec),
                    float(self.backoff_sec) * float(2**attempt),
                )
                if self.request_deadline_sec > 0.0:
                    remaining_after = float(self.request_deadline_sec) - float(time.monotonic() - started_at)
                    sleep_sec = min(sleep_sec, max(0.0, remaining_after - 0.01))
                if sleep_sec > 0.0:
                    time.sleep(sleep_sec)
            except TimeoutError as exc:
                last_exc = exc
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= effective_max_retries:
                    break
                sleep_sec = min(
                    float(self.retry_backoff_max_sec),
                    float(self.backoff_sec) * float(2**attempt),
                )
                if sleep_sec > 0.0:
                    time.sleep(sleep_sec)
        raise RuntimeError(f"api_call_failed: {last_exc}")

    def project(self, item: ContentItem) -> ProjectionResult:
        if not self._active:
            if self.enabled_mode == "true" or self.strict:
                raise RuntimeError(self._runtime_error or "api_adapter_inactive")
            return _zero_projection(item, reason=self._runtime_error or "api_adapter_inactive")

        text = _normalize_text(item.text)
        cache_key = _sha256_text(f"{text}|{self.model}|{self.prompt_version}")
        now_mono = time.monotonic()
        transient_cached = self._transient_error_cache.get(cache_key)
        if transient_cached is not None:
            expire_ts, transient_reason = transient_cached
            if now_mono < float(expire_ts):
                self._last_error = str(transient_reason)
                self._last_schema_valid = False
                return _zero_projection(item, reason=str(transient_reason))
            self._transient_error_cache.pop(cache_key, None)
        cache_hit = cache_key in self._cache
        if cache_hit:
            self._cache_hits += 1
            entry = self._cache[cache_key]
            payload = _normalize_api_payload(
                {
                    "schema_version": str(entry.get("schema_version", LEGACY_SCHEMA_COMPAT)),
                    "pressure_signed": dict(entry.get("pressure_signed", {})),
                    "directive_intent": dict(entry.get("directive_intent", {})),
                    "defensive_context": bool(entry.get("defensive_context", False)),
                    "confidence": float(entry.get("confidence", DEFAULT_CONFIDENCE)),
                    "scores": dict(entry.get("scores", {})),
                }
            )
            response_id = str(entry.get("response_id", ""))
            self._last_error = None
            self._last_schema_valid = True
            self._transient_error_cache.pop(cache_key, None)
        else:
            self._cache_misses += 1
            response_id = ""
            try:
                raw_payload, response_id = self._call_api_scores(text=text)
                payload = _normalize_api_payload(raw_payload)
                self._cache[cache_key] = {
                    "schema_version": str(payload["schema_version"]),
                    "pressure_signed": dict(payload["pressure_signed"]),
                    "directive_intent": dict(payload["directive_intent"]),
                    "defensive_context": bool(payload["defensive_context"]),
                    "confidence": float(payload["confidence"]),
                    "scores": dict(payload["scores"]),
                    "response_id": response_id,
                }
                self._append_cache(key=cache_key, payload=payload, response_id=response_id)
                self._last_error = None
                self._last_schema_valid = True
                self._transient_error_cache.pop(cache_key, None)
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
                if "schema_error:" in err:
                    self._schema_errors += 1
                self._last_error = err
                self._last_schema_valid = False
                self._log_error(item=item, cache_key=cache_key, error=err)
                if _is_transient_api_error(err) and self.transient_error_ttl_sec > 0.0:
                    self._transient_error_cache[cache_key] = (
                        float(time.monotonic() + float(self.transient_error_ttl_sec)),
                        str(err),
                    )
                # In strict mode, fail hard on contract/config issues.
                # For transient upstream outages (HTTP 5xx/429/timeouts),
                # continue with zero API contribution so long runs don't abort.
                if self.strict and not _is_transient_api_error(err):
                    raise
                return _zero_projection(item, reason=err)

        pressure_signed = {w: float(payload["pressure_signed"][w]) for w in WALLS}
        directive_intent = {w: bool(payload["directive_intent"][w]) for w in WALLS}
        raw_signed = [float(pressure_signed[w]) for w in WALLS]
        v = np.array([max(0.0, float(x)) for x in raw_signed], dtype=float)
        polarity = [1 if x > 0.0 else (-1 if x < 0.0 else 0) for x in raw_signed]
        return ProjectionResult(
            doc_id=item.doc_id,
            v=v,
            evidence=ProjectionEvidence(
                polarity=polarity,
                debug_scores_raw=raw_signed,
                matches={
                    "api_perception": {
                        "active": True,
                        "schema_valid": True,
                        "model": self.model,
                        "cache_hit": bool(cache_hit),
                        "cache_key": cache_key,
                        "response_id": response_id,
                        "schema_version": str(payload["schema_version"]),
                        "pressure_signed": {w: float(pressure_signed[w]) for w in WALLS},
                        "directive_intent": directive_intent,
                        "defensive_context": bool(payload["defensive_context"]),
                        "confidence": float(payload["confidence"]),
                        "scores": {w: max(0.0, float(pressure_signed[w])) for w in WALLS},
                    }
                },
            ),
        )

    def fit(self, items, y) -> None:  # pragma: no cover - compatibility
        _ = (items, y)
        raise NotImplementedError("APIPerceptionProjector runtime is inference-only.")


@dataclass
class HybridAPIProjector:
    pi0_projector: Any
    api_projector: APIPerceptionProjector

    def __post_init__(self) -> None:
        api_cfg = (
            (self.api_projector.config or {}).get("projector", {}).get("api_perception", {})
            if isinstance(getattr(self.api_projector, "config", {}), Mapping)
            else {}
        )
        cfg = (api_cfg or {}).get("deescalation", {})
        soft_gate_cfg = (api_cfg or {}).get("hybrid_soft_gate", {})
        benign_stabilizer_cfg = (api_cfg or {}).get("benign_stabilizer", {})
        self.deesc_confidence_min = float((cfg or {}).get("confidence_min", 0.75))
        self.deesc_p_strong = float((cfg or {}).get("p_strong", 0.35))
        self.soft_gate_enabled = bool((soft_gate_cfg or {}).get("enabled", True))
        self.soft_confirm_min = float((soft_gate_cfg or {}).get("soft_confirm_min", 0.10))
        self.require_api_for_soft = bool((soft_gate_cfg or {}).get("require_api_for_soft", True))
        self.benign_stabilizer_enabled = bool((benign_stabilizer_cfg or {}).get("enabled", True))
        self.benign_stabilizer_confidence_min = float((benign_stabilizer_cfg or {}).get("confidence_min", 0.90))
        self.benign_stabilizer_nonmal_max = float(
            (benign_stabilizer_cfg or {}).get("nonmal_max_positive_pressure", 0.10)
        )
        self.benign_task_guard_enabled = bool(getattr(self.api_projector, "benign_task_guard_enabled", False))
        self.benign_task_guard_markers = tuple(
            str(x).strip().lower()
            for x in list(getattr(self.api_projector, "benign_task_guard_markers", ()))
            if str(x).strip()
        )
        self.benign_task_guard_attack_cues = tuple(
            str(x).strip().lower()
            for x in list(getattr(self.api_projector, "benign_task_guard_attack_cues", ()))
            if str(x).strip()
        )
        self.benign_task_guard_require_pi0_hard_absent = bool(
            getattr(self.api_projector, "benign_task_guard_require_pi0_hard_absent", True)
        )
        self.short_fast_path_enabled = bool((api_cfg or {}).get("short_fast_path_enabled", True))
        self.short_fast_path_threshold_chars = int((api_cfg or {}).get("short_text_threshold_chars", 1200))
        self.short_fast_path_skip_on_pi0_hard = bool((api_cfg or {}).get("short_fast_path_skip_on_pi0_hard", True))
        self.short_fast_path_skip_on_pi0_clean = bool((api_cfg or {}).get("short_fast_path_skip_on_pi0_clean", True))
        self.short_fast_path_hard_min_score = float((api_cfg or {}).get("short_fast_path_hard_min_score", 0.55))
        self.short_fast_path_clean_max_score = float((api_cfg or {}).get("short_fast_path_clean_max_score", 0.0))

    def ensure_semantic_active(self) -> bool:
        return bool(getattr(self.pi0_projector, "ensure_semantic_active", lambda: False)())

    def ensure_api_adapter_active(self) -> bool:
        return bool(getattr(self.api_projector, "ensure_api_adapter_active", lambda: False)())

    def api_perception_status(self) -> Dict[str, Any]:
        return dict(getattr(self.api_projector, "api_perception_status", lambda: {})())

    def semantic_status(self) -> Dict[str, Any]:
        base = dict(getattr(self.pi0_projector, "semantic_status", lambda: {})())
        api_status = self.api_perception_status()
        base["api_adapter_active"] = bool(api_status.get("api_adapter_active", False))
        base["api_adapter_error"] = api_status.get("api_adapter_error")
        base["schema_valid"] = api_status.get("schema_valid")
        base["api_model"] = api_status.get("model")
        base["cache_hit_rate"] = api_status.get("cache_hit_rate", 0.0)
        return base

    def pitheta_conversion_status(self) -> Dict[str, Any]:
        return {
            "active": False,
            "error": "not_supported_for_api_hybrid",
            "api_perception": self.api_perception_status(),
        }

    def _extract_pi0_rule_tier(self, p0_matches: Mapping[str, Any]) -> Dict[str, Any]:
        raw_tier = p0_matches.get("pi0_rule_tier", {})
        walls_raw = raw_tier.get("walls", {}) if isinstance(raw_tier, Mapping) else {}
        walls: Dict[str, Dict[str, Any]] = {}
        hard_any = False
        soft_any = False
        for idx, wall in enumerate(WALLS):
            wall_raw = walls_raw.get(wall, {}) if isinstance(walls_raw, Mapping) else {}
            hard_hit = bool((wall_raw or {}).get("hard_hit", False))
            soft_hit = bool((wall_raw or {}).get("soft_hit", False))
            hard_signals = list((wall_raw or {}).get("hard_signals", [])) if isinstance(wall_raw, Mapping) else []
            soft_signals = list((wall_raw or {}).get("soft_signals", [])) if isinstance(wall_raw, Mapping) else []
            try:
                raw_score = float((wall_raw or {}).get("raw_score", 0.0))
            except Exception:  # noqa: BLE001
                raw_score = 0.0
            walls[wall] = {
                "hard_hit": hard_hit,
                "soft_hit": soft_hit,
                "hard_signals": [str(x) for x in hard_signals if str(x)],
                "soft_signals": [str(x) for x in soft_signals if str(x)],
                "raw_score": raw_score,
                "wall_index": idx,
            }
            hard_any = hard_any or hard_hit
            soft_any = soft_any or soft_hit
        hard_any = bool(raw_tier.get("hard_any", hard_any)) if isinstance(raw_tier, Mapping) else hard_any
        soft_any = bool(raw_tier.get("soft_any", soft_any)) if isinstance(raw_tier, Mapping) else soft_any
        return {"walls": walls, "hard_any": bool(hard_any), "soft_any": bool(soft_any)}

    def _short_fast_path_decision(
        self,
        *,
        item: ContentItem,
        p0: ProjectionResult,
        pi0_rule_tier: Mapping[str, Any],
    ) -> Tuple[bool, str]:
        if not self.short_fast_path_enabled:
            return False, "disabled"
        text_len = len(_normalize_text(getattr(item, "text", "")))
        if text_len > int(self.short_fast_path_threshold_chars):
            return False, "not_short"
        hard_any = bool(pi0_rule_tier.get("hard_any", False))
        soft_any = bool(pi0_rule_tier.get("soft_any", False))
        max_p0 = float(np.max(np.asarray(getattr(p0, "v", np.zeros(4, dtype=float)), dtype=float)))
        if (
            self.short_fast_path_skip_on_pi0_hard
            and hard_any
            and max_p0 >= float(self.short_fast_path_hard_min_score)
        ):
            return True, "pi0_hard_high_confidence"
        if (
            self.short_fast_path_skip_on_pi0_clean
            and (not hard_any)
            and (not soft_any)
            and max_p0 <= float(self.short_fast_path_clean_max_score)
        ):
            return True, "pi0_clean_high_confidence"
        return False, "ambiguous"

    def project(self, item: ContentItem) -> ProjectionResult:
        p0 = self.pi0_projector.project(item)
        p0_matches = dict(getattr(p0.evidence, "matches", {}) or {})
        pi0_rule_tier = self._extract_pi0_rule_tier(p0_matches)
        short_fast_path_applied, short_fast_path_reason = self._short_fast_path_decision(
            item=item,
            p0=p0,
            pi0_rule_tier=pi0_rule_tier,
        )
        if short_fast_path_applied:
            ap = _zero_projection(item, reason=f"short_fast_path:{short_fast_path_reason}")
        else:
            ap = self.api_projector.project(item)
        api_match = {}
        if isinstance(getattr(ap.evidence, "matches", {}), Mapping):
            apm = ap.evidence.matches.get("api_perception", {})
            if isinstance(apm, Mapping):
                api_match = dict(apm)
        pressure_signed = {
            w: float((api_match.get("pressure_signed") or {}).get(w, float(ap.evidence.debug_scores_raw[idx])))
            for idx, w in enumerate(WALLS)
        }
        directive_intent = {w: bool((api_match.get("directive_intent") or {}).get(w, False)) for w in WALLS}
        defensive_context = bool(api_match.get("defensive_context", False))
        confidence = float(api_match.get("confidence", DEFAULT_CONFIDENCE))
        text_norm = _normalize_text(getattr(item, "text", "")).lower()

        benign_task_guard_marker_hit = bool(
            self.benign_task_guard_enabled
            and _contains_any_marker(text_norm, self.benign_task_guard_markers)
        )
        benign_task_guard_attack_cue_hit = bool(
            self.benign_task_guard_enabled
            and _contains_any_marker(text_norm, self.benign_task_guard_attack_cues)
        )
        benign_task_guard_applied = False
        benign_task_guard_reason = ""
        if self.benign_task_guard_enabled and benign_task_guard_marker_hit and (not benign_task_guard_attack_cue_hit):
            pi0_hard_any = bool(pi0_rule_tier.get("hard_any", False))
            if (not self.benign_task_guard_require_pi0_hard_absent) or (not pi0_hard_any):
                pressure_signed = {w: 0.0 for w in WALLS}
                directive_intent = {w: False for w in WALLS}
                benign_task_guard_applied = True
                benign_task_guard_reason = "benign_workflow_marker_without_attack_cues"
            else:
                benign_task_guard_reason = "blocked_by_pi0_hard_signal"
        elif self.benign_task_guard_enabled and benign_task_guard_marker_hit and benign_task_guard_attack_cue_hit:
            benign_task_guard_reason = "attack_cue_present"
        elif self.benign_task_guard_enabled:
            benign_task_guard_reason = "marker_not_present"

        api_positive = np.array([max(0.0, float(pressure_signed[w])) for w in WALLS], dtype=float)
        max_api_positive = float(np.max(api_positive)) if len(api_positive) else 0.0
        directive_any = any(bool(v) for v in directive_intent.values())
        soft_confirmed_any = bool(directive_any or (max_api_positive >= self.soft_confirm_min))
        suppressed_walls: list[str] = []
        confirmation_walls: list[str] = []
        pi0_effective = np.array([float(x) for x in p0.v], dtype=float)
        if self.soft_gate_enabled and self.require_api_for_soft:
            for i, wall in enumerate(WALLS):
                tier = (pi0_rule_tier.get("walls", {}) or {}).get(wall, {})
                hard_hit = bool((tier or {}).get("hard_hit", False))
                soft_hit = bool((tier or {}).get("soft_hit", False))
                if not soft_hit or hard_hit:
                    continue
                if soft_confirmed_any:
                    if float(pi0_effective[i]) > 0.0:
                        confirmation_walls.append(wall)
                    continue
                if float(pi0_effective[i]) > 0.0:
                    pi0_effective[i] = 0.0
                    suppressed_walls.append(wall)
        deescalation_applied = False
        deescalation_reason = ""
        if defensive_context and (not directive_any) and confidence >= self.deesc_confidence_min and max_api_positive < self.deesc_p_strong:
            api_positive = np.zeros(4, dtype=float)
            deescalation_applied = True
            deescalation_reason = "defensive_context_zero_api_boost"
        else:
            reasons = []
            if not defensive_context:
                reasons.append("not_defensive_context")
            if directive_any:
                reasons.append("directive_intent_present")
            if confidence < self.deesc_confidence_min:
                reasons.append("confidence_below_threshold")
            if max_api_positive >= self.deesc_p_strong:
                reasons.append("strong_api_positive_pressure")
            deescalation_reason = ",".join(reasons) if reasons else "not_applicable"

        benign_stabilizer_applied = False
        benign_stabilizer_walls: list[str] = []
        if (
            self.benign_stabilizer_enabled
            and (not directive_any)
            and max_api_positive < self.benign_stabilizer_nonmal_max
            and confidence >= self.benign_stabilizer_confidence_min
        ):
            for i, wall in enumerate(WALLS):
                tier = (pi0_rule_tier.get("walls", {}) or {}).get(wall, {})
                hard_hit = bool((tier or {}).get("hard_hit", False))
                soft_hit = bool((tier or {}).get("soft_hit", False))
                if hard_hit or (not soft_hit):
                    continue
                if float(pi0_effective[i]) > 0.0:
                    pi0_effective[i] = 0.0
                    benign_stabilizer_walls.append(wall)
            benign_stabilizer_applied = bool(benign_stabilizer_walls)
        if benign_stabilizer_applied:
            benign_stabilizer_reason = "api_non_malicious_soft_only_pi0_suppressed"
        else:
            benign_reasons = []
            if not self.benign_stabilizer_enabled:
                benign_reasons.append("disabled")
            if directive_any:
                benign_reasons.append("directive_intent_present")
            if max_api_positive >= self.benign_stabilizer_nonmal_max:
                benign_reasons.append("api_positive_pressure_not_low")
            if confidence < self.benign_stabilizer_confidence_min:
                benign_reasons.append("confidence_below_threshold")
            benign_stabilizer_reason = ",".join(benign_reasons) if benign_reasons else "eligible_no_soft_nonhard_signal"

        v = np.maximum(pi0_effective, api_positive)
        polarity = []
        raw = []
        for i in range(4):
            if float(api_positive[i]) > float(pi0_effective[i]):
                raw_signed = float(pressure_signed[WALLS[i]])
                polarity.append(1 if raw_signed > 0.0 else (-1 if raw_signed < 0.0 else 0))
                raw.append(raw_signed)
            elif float(pi0_effective[i]) > 0.0:
                polarity.append(int(p0.evidence.polarity[i]))
                raw.append(float(p0.evidence.debug_scores_raw[i]))
            else:
                polarity.append(0)
                raw.append(0.0)
        api_match_out = dict(api_match)
        api_match_out["directive_intent"] = directive_intent
        api_match_out["defensive_context"] = defensive_context
        api_match_out["confidence"] = confidence
        api_match_out["schema_version"] = str(api_match.get("schema_version", LEGACY_SCHEMA_COMPAT))
        api_match_out["pressure_signed"] = {w: float(pressure_signed[w]) for w in WALLS}
        api_match_out["scores"] = {w: max(0.0, float(pressure_signed[w])) for w in WALLS}
        api_match_out["deescalation_applied"] = bool(deescalation_applied)
        api_match_out["deescalation_reason"] = deescalation_reason
        api_match_out["short_fast_path_applied"] = bool(short_fast_path_applied)
        api_match_out["short_fast_path_reason"] = str(short_fast_path_reason)

        matches = {
            "hybrid_api": {
                "mode": "max",
                "walls": WALLS,
                "deescalation_confidence_min": float(self.deesc_confidence_min),
                "deescalation_p_strong": float(self.deesc_p_strong),
                "deescalation_applied": bool(deescalation_applied),
                "soft_gate_enabled": bool(self.soft_gate_enabled),
                "require_api_for_soft": bool(self.require_api_for_soft),
                "soft_confirm_min": float(self.soft_confirm_min),
                "soft_confirmed_any": bool(soft_confirmed_any),
                "soft_suppressed_any": bool(suppressed_walls),
                "short_fast_path_applied": bool(short_fast_path_applied),
                "short_fast_path_reason": str(short_fast_path_reason),
                "short_fast_path_threshold_chars": int(self.short_fast_path_threshold_chars),
                "suppressed_walls": list(suppressed_walls),
                "confirmation_walls": list(confirmation_walls),
                "benign_stabilizer_enabled": bool(self.benign_stabilizer_enabled),
                "benign_stabilizer_confidence_min": float(self.benign_stabilizer_confidence_min),
                "benign_stabilizer_nonmal_max": float(self.benign_stabilizer_nonmal_max),
                "benign_stabilizer_applied": bool(benign_stabilizer_applied),
                "benign_stabilizer_reason": str(benign_stabilizer_reason),
                "benign_stabilizer_walls": list(benign_stabilizer_walls),
                "pi0_hard_any": bool(pi0_rule_tier.get("hard_any", False)),
                "pi0_soft_any": bool(pi0_rule_tier.get("soft_any", False)),
                "api_directive_intent_any": bool(directive_any),
                "api_max_positive_pressure": float(max_api_positive),
                "benign_task_guard_enabled": bool(self.benign_task_guard_enabled),
                "benign_task_guard_marker_hit": bool(benign_task_guard_marker_hit),
                "benign_task_guard_attack_cue_hit": bool(benign_task_guard_attack_cue_hit),
                "benign_task_guard_applied": bool(benign_task_guard_applied),
                "benign_task_guard_reason": str(benign_task_guard_reason),
            },
            "pi0": p0_matches,
            "api_perception": api_match_out,
        }
        return ProjectionResult(
            doc_id=item.doc_id,
            v=v,
            evidence=ProjectionEvidence(
                polarity=polarity,
                debug_scores_raw=raw,
                matches=matches,
            ),
        )

    def __getattr__(self, name: str):
        return getattr(self.pi0_projector, name)
