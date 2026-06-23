"""Normalization and provider-agnostic helpers for API hybrid projector."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest
import ipaddress


from omega.interfaces.contracts_v1 import WALLS_V1

WALLS = list(WALLS_V1)
DEFAULT_API_PROVIDER = "openai"
SUPPORTED_API_PROVIDERS = {"openai", "anthropic", "openai_compat", "local_vision"}
SUPPORTED_SEMANTIC_FAILURE_POLICIES = {"degrade", "escalate", "fail_closed"}
SUPPORTED_SEMANTIC_MODES = {"rules_only", "hybrid_cloud", "hybrid_redacted", "local_semantic", "rules_plus_ocr"}
LEGACY_SCHEMA_COMPAT = "v1_compat"
DEFAULT_CONFIDENCE = 0.5



_PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "openrouter": {
        "provider": "openai_compat",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "allowed_base_urls": ["https://openrouter.ai/api/v1"],
        "extra_headers": {"X-OpenRouter-Title": "Omega Walls"},
    },
    "litellm": {
        "provider": "openai_compat",
        "base_url": "http://litellm.internal:4000/v1",
        "api_key_env": "LITELLM_API_KEY",
        "allowed_base_urls": ["http://litellm.internal:4000/v1"],
        "allow_http_private_gateway": True,
    },
}
_RESERVED_HEADER_NAMES = {"authorization", "content-type", "host", "content-length"}


def supported_provider_presets() -> Tuple[str, ...]:
    return tuple(sorted(_PROVIDER_PRESETS))


def normalize_provider_preset(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if raw not in _PROVIDER_PRESETS:
        raise ValueError(
            "projector.api_perception.provider_preset must be "
            + "|".join(supported_provider_presets())
        )
    return raw


def provider_preset_defaults(preset: Any) -> Dict[str, Any]:
    raw = normalize_provider_preset(preset)
    if not raw:
        return {}
    return dict(_PROVIDER_PRESETS[raw])


def apply_provider_preset(api_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve the thin OSS provider preset layer into openai_compat config.

    Presets intentionally do not create a new provider runtime. They only fill
    transport defaults unless the caller supplied a non-default custom value.
    """
    cfg = dict(api_cfg or {})
    preset = normalize_provider_preset(cfg.get("provider_preset", ""))
    if not preset:
        return cfg
    defaults = provider_preset_defaults(preset)
    current_provider = str(cfg.get("provider", "") or "").strip().lower()
    if not current_provider or current_provider == DEFAULT_API_PROVIDER:
        cfg["provider"] = defaults.get("provider", "openai_compat")
    current_base = str(cfg.get("base_url", "") or "").strip().rstrip("/")
    if (not current_base) or current_base == default_base_url_for_provider(DEFAULT_API_PROVIDER):
        cfg["base_url"] = defaults.get("base_url", current_base)
    current_key_env = str(cfg.get("api_key_env", "") or "").strip()
    if (not current_key_env) or current_key_env == default_api_key_env_for_provider(DEFAULT_API_PROVIDER):
        cfg["api_key_env"] = defaults.get("api_key_env", current_key_env)
    for key in ("allowed_base_urls", "allow_http_private_gateway"):
        if key not in cfg and key in defaults:
            cfg[key] = defaults[key]
    provider_options = dict(defaults.get("provider_options", {}) or {})
    provider_options.update(dict(cfg.get("provider_options", {}) or {}))
    if provider_options:
        cfg["provider_options"] = provider_options
    extra_headers = dict(defaults.get("extra_headers", {}) or {})
    extra_headers.update(dict(cfg.get("extra_headers", {}) or {}))
    if extra_headers:
        cfg["extra_headers"] = extra_headers
    return cfg


def canonical_base_url(value: Any) -> str:
    raw = str(value or "").strip().rstrip("/")
    if not raw:
        return ""
    parsed = urlparse.urlparse(raw)
    scheme = str(parsed.scheme or "").lower()
    host = str(parsed.hostname or "").lower()
    if scheme not in {"http", "https", "local"}:
        raise ValueError("provider base_url scheme must be http|https|local")
    if scheme == "local":
        if raw != "local://vision":
            raise ValueError("local provider base_url must be local://vision")
        return raw
    if not host:
        raise ValueError("provider base_url must include a host")
    port = f":{parsed.port}" if parsed.port is not None else ""
    path = str(parsed.path or "").rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{host}{port}{path}{query}"


def _is_private_http_host(hostname: str) -> bool:
    host = str(hostname or "").strip().lower()
    if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".internal"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local)


def normalize_allowed_base_urls(value: Any) -> Tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("projector.api_perception.allowed_base_urls must be a list")
    out = []
    for row in value:
        url = canonical_base_url(row)
        if url:
            out.append(url)
    return tuple(sorted(set(out)))


def enforce_provider_endpoint_policy(
    *,
    base_url: str,
    allowed_base_urls: Tuple[str, ...] = (),
    allow_http_private_gateway: bool = False,
    allow_loopback_http: bool = False,
) -> None:
    base = canonical_base_url(base_url)
    if not base:
        raise ValueError("projector.api_perception.base_url must be non-empty")
    parsed = urlparse.urlparse(base)
    scheme = str(parsed.scheme or "").lower()
    if scheme == "local":
        return
    if allowed_base_urls and base not in allowed_base_urls:
        raise ValueError("provider base_url is not in allowed_base_urls")
    if scheme == "https":
        return
    if scheme != "http":
        raise ValueError("provider base_url scheme must be http|https")
    hostname = str(parsed.hostname or "")
    if bool(allow_loopback_http):
        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError:
            ip = None
        if hostname.lower() in {"localhost"} or (ip is not None and ip.is_loopback):
            return
    if not bool(allow_http_private_gateway):
        raise ValueError("HTTP provider endpoints require allow_http_private_gateway=true")
    if not _is_private_http_host(hostname):
        raise ValueError("HTTP provider endpoints must use loopback/private/.internal host")
    if not allowed_base_urls:
        raise ValueError("HTTP provider endpoints must be explicitly listed in allowed_base_urls")


def normalize_extra_headers(value: Any) -> Dict[str, str]:
    if value in (None, ""):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("projector.api_perception.extra_headers must be a mapping")
    out: Dict[str, str] = {}
    for key, raw_value in dict(value).items():
        name = str(key or "").strip()
        if not name:
            raise ValueError("projector.api_perception.extra_headers contains empty header name")
        if "\n" in name or "\r" in name or ":" in name:
            raise ValueError("projector.api_perception.extra_headers contains invalid header name")
        if name.lower() in _RESERVED_HEADER_NAMES:
            raise ValueError(f"projector.api_perception.extra_headers cannot set reserved header {name}")
        val = str(raw_value or "").strip()
        if "\n" in val or "\r" in val:
            raise ValueError(f"projector.api_perception.extra_headers.{name} contains newline")
        if val:
            out[name] = val
    return out


def resolve_api_key_from_file(path_value: Any, *, provider: str) -> str:
    path_raw = str(path_value or "").strip()
    if not path_raw:
        return ""
    path = os.path.expandvars(os.path.expanduser(path_raw))
    if not os.path.isfile(path):
        raise RuntimeError("api_key_file_missing")
    if os.path.getsize(path) > 64 * 1024:
        raise RuntimeError("api_key_file_too_large")
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read().strip()
    return normalize_api_key_value(raw, provider=provider)


def normalize_provider(provider: str) -> str:
    raw = str(provider or DEFAULT_API_PROVIDER).strip().lower()
    return raw if raw in SUPPORTED_API_PROVIDERS else DEFAULT_API_PROVIDER


def default_base_url_for_provider(provider: str) -> str:
    p = normalize_provider(provider)
    if p == "anthropic":
        return "https://api.anthropic.com/v1"
    if p == "local_vision":
        return "local://vision"
    return "https://api.openai.com/v1"


def default_api_key_env_for_provider(provider: str) -> str:
    p = normalize_provider(provider)
    if p == "anthropic":
        return "ANTHROPIC_API_KEY"
    if p == "local_vision":
        return "OMEGA_LOCAL_VISION"
    return "OPENAI_API_KEY"


def normalize_api_key_value(value: Any, *, provider: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    p = normalize_provider(provider)
    if p in {"openai", "openai_compat"}:
        match = re.search(r"(sk-[A-Za-z0-9_\-]+)", raw)
        if match:
            return str(match.group(1))
    return raw


def normalize_semantic_failure_policy(policy: str) -> str:
    raw = str(policy or "degrade").strip().lower()
    return raw if raw in SUPPORTED_SEMANTIC_FAILURE_POLICIES else "degrade"


def normalize_semantic_mode(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    if raw in SUPPORTED_SEMANTIC_MODES:
        return raw
    return None


class APIRequestError(RuntimeError):
    def __init__(self, *, code: int, body: str):
        self.code = int(code)
        self.body = str(body)
        super().__init__(f"HTTP {self.code}: {self.body}")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def contains_any_marker(text: str, markers: Tuple[str, ...]) -> bool:
    t = str(text or "")
    return any(m and (m in t) for m in markers)


def sha256_text(value: str) -> str:
    h = hashlib.sha256()
    h.update(str(value).encode("utf-8"))
    return h.hexdigest()


def sanitize_semantic_input_text(text: str) -> Tuple[str, Dict[str, Any]]:
    src = str(text or "")
    out = src
    counters = {
        "api_key_like": 0,
        "aws_access_key_like": 0,
        "email_like": 0,
        "github_token_like": 0,
        "ipv4_like": 0,
        "jwt_like": 0,
        "path_like": 0,
        "phone_like": 0,
        "slack_token_like": 0,
        "url_like": 0,
    }
    out, c1 = re.subn(r"\b(sk-[A-Za-z0-9_\-]{12,})\b", "<redacted_token>", out)
    counters["api_key_like"] += int(c1)
    out, c1b = re.subn(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", "<redacted_token>", out)
    counters["aws_access_key_like"] += int(c1b)
    out, c1c = re.subn(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b", "<redacted_token>", out)
    counters["github_token_like"] += int(c1c)
    out, c1d = re.subn(r"\b(?:xox[baprs]-[A-Za-z0-9-]{10,}|xapp-[A-Za-z0-9-]{10,})\b", "<redacted_token>", out)
    counters["slack_token_like"] += int(c1d)
    out, c1e = re.subn(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b", "<redacted_token>", out)
    counters["jwt_like"] += int(c1e)
    out, c2 = re.subn(r"\b([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b", "<redacted_email>", out)
    counters["email_like"] += int(c2)
    out, c3 = re.subn(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b", "<redacted_ip>", out)
    counters["ipv4_like"] += int(c3)
    out, c5 = re.subn(r"\bhttps?://[^\s]+", "<redacted_url>", out, flags=re.IGNORECASE)
    counters["url_like"] += int(c5)
    out, c4 = re.subn(r"([A-Za-z]:\\[^\s]+|/[A-Za-z0-9._\-/]{3,})", "<redacted_path>", out)
    counters["path_like"] += int(c4)
    out, c6 = re.subn(r"(?<!\w)(?:\+?\d[\d\-\s().]{8,}\d)(?!\w)", "<redacted_phone>", out)
    counters["phone_like"] += int(c6)

    compact = " ".join(out.split())
    max_chars = 4000
    truncated = False
    if len(compact) > max_chars:
        compact = compact[:max_chars]
        truncated = True

    redacted_count = int(sum(int(v) for v in counters.values()))
    meta = {
        "applied": bool(redacted_count > 0 or truncated),
        "truncated": bool(truncated),
        "max_chars": int(max_chars),
        "replacement_counts": counters,
        "original_text_length": int(len(src)),
        "sanitized_text_length": int(len(compact)),
    }
    return compact, meta


def _should_try_windows_curl_fallback(exc: urlerror.URLError) -> bool:
    if os.name != "nt":
        return False
    text = str(getattr(exc, "reason", exc) or "")
    return "winerror 10013" in text.lower()


def _parse_curl_http_dump(raw: str) -> Tuple[int, Dict[str, str], str]:
    text = str(raw or "")
    if not text:
        raise RuntimeError("curl_empty_response")
    normalized = text.replace("\r\n", "\n")
    blocks = [block for block in normalized.split("\n\n") if block.strip()]
    if len(blocks) < 2:
        raise RuntimeError("curl_response_parse_error")
    header_block = blocks[-2]
    body = blocks[-1]
    header_lines = [line.strip() for line in header_block.split("\n") if line.strip()]
    if not header_lines or not header_lines[0].upper().startswith("HTTP/"):
        raise RuntimeError("curl_response_missing_status")
    parts = header_lines[0].split()
    if len(parts) < 2:
        raise RuntimeError("curl_response_missing_status_code")
    status_code = int(parts[1])
    headers: Dict[str, str] = {}
    for line in header_lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[str(key).strip().lower()] = str(value).strip()
    return status_code, headers, body


def _post_json_via_windows_curl(
    *,
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_sec: float,
) -> Dict[str, Any]:
    curl_path = shutil.which("curl.exe") or shutil.which("curl")
    if not curl_path:
        raise RuntimeError("curl_unavailable")
    cmd = [
        str(curl_path),
        "-sS",
        "--show-error",
        "-D",
        "-",
        "-X",
        "POST",
        str(url),
        "--max-time",
        str(max(1, int(round(float(timeout_sec))))),
        "--data-binary",
        "@-",
    ]
    for key, value in dict(headers).items():
        cmd.extend(["-H", f"{key}: {value}"])
    proc = subprocess.run(
        cmd,
        input=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if int(proc.returncode) != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"curl_error: {stderr.strip() or proc.returncode}")
    status_code, resp_headers, body = _parse_curl_http_dump(proc.stdout.decode("utf-8", errors="replace"))
    if status_code >= 400:
        raise APIRequestError(code=status_code, body=f"{body} | headers={json.dumps(resp_headers, ensure_ascii=False)}")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("api response is not a JSON object")
    parsed["_headers"] = resp_headers
    return parsed


class _NoRedirectHandler(urlrequest.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None

    def _blocked(self, req, fp, code, msg, headers):  # type: ignore[no-untyped-def]
        _ = (req, fp, code, msg)
        location = str(headers.get("Location", "") or headers.get("location", ""))
        raise RuntimeError(f"provider_redirect_blocked:{location}")

    http_error_301 = _blocked
    http_error_302 = _blocked
    http_error_303 = _blocked
    http_error_307 = _blocked
    http_error_308 = _blocked


def post_json(*, url: str, payload: Mapping[str, Any], headers: Mapping[str, str], timeout_sec: float, allow_redirects: bool = False) -> Dict[str, Any]:
    data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(url=url, data=data, headers=dict(headers), method="POST")
    opener = (
        urlrequest.build_opener()
        if bool(allow_redirects)
        else urlrequest.build_opener(_NoRedirectHandler)
    )
    try:
        with opener.open(req, timeout=float(timeout_sec)) as resp:
            raw = resp.read().decode("utf-8")
            resp_headers = {str(k).lower(): str(v) for k, v in dict(getattr(resp, "headers", {})).items()}
    except urlerror.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            body = str(exc)
        hdrs = {str(k).lower(): str(v) for k, v in dict(getattr(exc, "headers", {})).items()}
        if int(exc.code) in {301, 302, 303, 307, 308} and not bool(allow_redirects):
            location = hdrs.get("location", "")
            raise RuntimeError(f"provider_redirect_blocked:{location}") from exc
        raise APIRequestError(code=int(exc.code), body=f"{body} | headers={json.dumps(hdrs, ensure_ascii=False)}") from exc
    except urlerror.URLError as exc:
        if _should_try_windows_curl_fallback(exc):
            return _post_json_via_windows_curl(url=url, payload=payload, headers=headers, timeout_sec=timeout_sec)
        raise RuntimeError(f"url_error: {exc}") from exc
    except OSError as exc:
        if not bool(allow_redirects):
            raise RuntimeError("provider_redirect_blocked:transport_abort") from exc
        raise RuntimeError(f"url_error: {exc}") from exc
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("api response is not a JSON object")
    parsed["_headers"] = resp_headers
    return parsed


def extract_output_text(resp: Mapping[str, Any]) -> str:
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


def validate_api_pressure_signed(obj: Mapping[str, Any]) -> Dict[str, float]:
    pressure_obj = obj.get("pressure_signed")
    if isinstance(pressure_obj, Mapping):
        out: Dict[str, float] = {}
        for wall in WALLS:
            if wall not in pressure_obj:
                raise ValueError(f"schema_error: missing {wall}")
            raw_value = pressure_obj[wall]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"schema_error: {wall} must be numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"schema_error: {wall} must be finite")
            if value < -1.0 or value > 1.0:
                raise ValueError(f"schema_error: {wall} out of [-1,1]")
            out[wall] = value
        return out

    scores_obj = obj.get("scores")
    if isinstance(scores_obj, Mapping):
        out = {}
        for wall in WALLS:
            if wall not in scores_obj:
                raise ValueError(f"schema_error: missing {wall}")
            raw_value = scores_obj[wall]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"schema_error: {wall} must be numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"schema_error: {wall} must be finite")
            if value < 0.0 or value > 1.0:
                raise ValueError(f"schema_error: {wall} out of [0,1]")
            out[wall] = value
        return out

    if all(wall in obj for wall in WALLS):
        out = {}
        for wall in WALLS:
            raw_value = obj[wall]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"schema_error: {wall} must be numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"schema_error: {wall} must be finite")
            if value < -1.0 or value > 1.0:
                raise ValueError(f"schema_error: {wall} out of [-1,1]")
            out[wall] = value
        return out

    raise ValueError("schema_error: pressure_signed or scores object required")


def validate_api_scores(obj: Mapping[str, Any]) -> Dict[str, float]:
    return {w: max(0.0, float(v)) for w, v in validate_api_pressure_signed(obj).items()}


def validate_directive_intent(obj: Any, *, pressure_signed: Mapping[str, float]) -> Dict[str, bool]:
    if isinstance(obj, Mapping):
        out: Dict[str, bool] = {}
        for wall in WALLS:
            if wall not in obj:
                raise ValueError(f"schema_error: missing directive_intent.{wall}")
            if type(obj[wall]) is not bool:
                raise ValueError(f"schema_error: directive_intent.{wall} must be boolean")
            out[wall] = obj[wall]
        return out
    return {wall: float(pressure_signed[wall]) > 0.0 for wall in WALLS}


def normalize_api_payload(obj: Mapping[str, Any]) -> Dict[str, Any]:
    pressure_signed = validate_api_pressure_signed(obj)
    schema_version_raw = str(obj.get("schema_version", "")).strip()
    is_v2 = schema_version_raw == "api_hybrid_v2" and isinstance(obj.get("pressure_signed"), Mapping)

    if is_v2:
        confidence_raw = obj.get("confidence", DEFAULT_CONFIDENCE)
        if isinstance(confidence_raw, bool) or not isinstance(confidence_raw, (int, float)):
            raise ValueError("schema_error: confidence must be numeric")
        confidence = float(confidence_raw)
        if not math.isfinite(confidence):
            raise ValueError("schema_error: confidence must be finite")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("schema_error: confidence out of [0,1]")
        defensive_raw = obj.get("defensive_context", False)
        if type(defensive_raw) is not bool:
            raise ValueError("schema_error: defensive_context must be boolean")
        defensive_context = defensive_raw
        directive_intent = validate_directive_intent(obj.get("directive_intent"), pressure_signed=pressure_signed)
        schema_version = "api_hybrid_v2"
    else:
        confidence = DEFAULT_CONFIDENCE
        defensive_context = False
        directive_intent = validate_directive_intent(None, pressure_signed=pressure_signed)
        schema_version = LEGACY_SCHEMA_COMPAT

    scores = {w: max(0.0, float(pressure_signed[w])) for w in WALLS}
    return {
        "schema_version": schema_version,
        "pressure_signed": pressure_signed,
        "directive_intent": directive_intent,
        "defensive_context": defensive_context,
        "confidence": confidence,
        "scores": scores,
    }


def is_transient_api_error(err: str) -> bool:
    t = str(err or "").lower()
    if "orchestrator_rule_only:" in t:
        return True
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


def quota_signal_from_headers(headers: Mapping[str, Any]) -> Optional[str]:
    if not isinstance(headers, Mapping):
        return None

    def _f(name: str) -> Optional[float]:
        raw = headers.get(name)
        if raw is None:
            return None
        try:
            return float(str(raw).strip())
        except Exception:  # noqa: BLE001
            return None

    rr = _f("x-ratelimit-remaining-requests")
    rl = _f("x-ratelimit-limit-requests")
    tr = _f("x-ratelimit-remaining-tokens")
    tl = _f("x-ratelimit-limit-tokens")
    ratios: list[float] = []
    if rr is not None and rl is not None and rl > 0:
        ratios.append(float(rr) / float(rl))
    if tr is not None and tl is not None and tl > 0:
        ratios.append(float(tr) / float(tl))
    if ratios and min(ratios) < 0.10:
        return "low_remaining"
    return None
