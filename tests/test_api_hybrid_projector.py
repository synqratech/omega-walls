from __future__ import annotations

import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import pytest

from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import ContentItem
import omega.projector.api_hybrid_projector as api_hybrid_module
from omega.projector.api_hybrid_projector import (
    APIPerceptionProjector,
    HybridAPIProjector,
    _normalize_api_payload,
    _validate_api_pressure_signed,
)
from omega.projector.factory import build_projector
from scripts import smoke_projector_status


def _mk_local_tmp(name: str) -> Path:
    root = Path("tests/_tmp")
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{name}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _mk_cfg(tmp_path: Path, *, strict: bool = False) -> dict:
    return {
        "projector": {
            "api_perception": {
                "enabled": "true",
                "strict": bool(strict),
                "model": "gpt-5",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
                "cache_path": str((tmp_path / "cache.jsonl").as_posix()),
                "error_log_path": str((tmp_path / "errors.jsonl").as_posix()),
            }
        }
    }


def test_validate_api_pressure_signed_ok_and_fail():
    good = _validate_api_pressure_signed(
        {
            "pressure_signed": {
                "override_instructions": 0.1,
                "secret_exfiltration": -0.2,
                "tool_or_action_abuse": 0.0,
                "policy_evasion": 0.4,
            }
        }
    )
    assert set(good.keys()) == {
        "override_instructions",
        "secret_exfiltration",
        "tool_or_action_abuse",
        "policy_evasion",
    }
    with pytest.raises(ValueError):
        _validate_api_pressure_signed({"pressure_signed": {"override_instructions": 2.0}})


def test_validate_api_pressure_signed_backcompat_scores():
    good = _validate_api_pressure_signed(
        {
            "scores": {
                "override_instructions": 0.1,
                "secret_exfiltration": 0.2,
                "tool_or_action_abuse": 0.3,
                "policy_evasion": 0.4,
            }
        }
    )
    assert set(good.keys()) == {
        "override_instructions",
        "secret_exfiltration",
        "tool_or_action_abuse",
        "policy_evasion",
    }


def test_normalize_api_payload_v2_and_legacy():
    v2 = _normalize_api_payload(
        {
            "schema_version": "api_hybrid_v2",
            "pressure_signed": {
                "override_instructions": 0.2,
                "secret_exfiltration": -0.4,
                "tool_or_action_abuse": 0.1,
                "policy_evasion": 0.0,
            },
            "directive_intent": {
                "override_instructions": True,
                "secret_exfiltration": False,
                "tool_or_action_abuse": False,
                "policy_evasion": False,
            },
            "defensive_context": True,
            "confidence": 0.9,
        }
    )
    assert v2["schema_version"] == "api_hybrid_v2"
    assert bool(v2["directive_intent"]["override_instructions"]) is True
    assert bool(v2["defensive_context"]) is True
    assert float(v2["confidence"]) == pytest.approx(0.9)

    legacy = _normalize_api_payload(
        {
            "scores": {
                "override_instructions": 0.1,
                "secret_exfiltration": 0.2,
                "tool_or_action_abuse": 0.3,
                "policy_evasion": 0.4,
            }
        }
    )
    assert legacy["schema_version"] == "v1_compat"
    assert bool(legacy["defensive_context"]) is False
    assert float(legacy["confidence"]) == pytest.approx(0.5)
    assert bool(legacy["directive_intent"]["policy_evasion"]) is True

    with pytest.raises(ValueError):
        _normalize_api_payload(
            {
                "schema_version": "api_hybrid_v2",
                "pressure_signed": {
                    "override_instructions": 0.2,
                    "secret_exfiltration": -0.4,
                    "tool_or_action_abuse": 0.1,
                    "policy_evasion": 0.0,
                },
                "directive_intent": {
                    "override_instructions": True,
                    "secret_exfiltration": False,
                    "tool_or_action_abuse": False,
                    "policy_evasion": False,
                },
                "defensive_context": False,
                "confidence": 1.5,
            }
        )


def test_api_projector_cache_determinism(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-cache")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    call_count = {"n": 0}

    def _fake_call(self, *, text: str):
        _ = text
        call_count["n"] += 1
        return (
            {
                "override_instructions": 0.11,
                "secret_exfiltration": -0.22,
                "tool_or_action_abuse": 0.33,
                "policy_evasion": 0.44,
            },
            "resp_1",
        )

    monkeypatch.setattr(APIPerceptionProjector, "_call_api_scores", _fake_call)
    projector = APIPerceptionProjector(config=_mk_cfg(tmp_path, strict=True))
    item = ContentItem(doc_id="d1", source_id="s1", source_type="other", trust="untrusted", text="test text")

    out1 = projector.project(item)
    out2 = projector.project(item)
    assert call_count["n"] == 1
    assert np.allclose(out1.v, out2.v)
    assert float(out1.v[1]) == 0.0
    assert out1.evidence.polarity[1] == -1
    assert float(out1.evidence.debug_scores_raw[1]) < 0.0
    api_match = out1.evidence.matches.get("api_perception", {})
    assert float(api_match["pressure_signed"]["secret_exfiltration"]) < 0.0
    assert float(api_match["scores"]["secret_exfiltration"]) == 0.0
    st = projector.api_perception_status()
    assert st["api_adapter_active"] is True
    assert st["schema_valid"] is True
    assert st["cache_hits"] >= 1
    cache_rows = [x for x in (tmp_path / "cache.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(cache_rows) == 1


def test_api_projector_legacy_scores_compat(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-legacy-compat")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _fake_call(self, *, text: str):
        _ = text
        return (
            {
                "scores": {
                    "override_instructions": 0.05,
                    "secret_exfiltration": 0.0,
                    "tool_or_action_abuse": 0.15,
                    "policy_evasion": 0.1,
                }
            },
            "resp_legacy",
        )

    monkeypatch.setattr(APIPerceptionProjector, "_call_api_scores", _fake_call)
    projector = APIPerceptionProjector(config=_mk_cfg(tmp_path, strict=True))
    out = projector.project(
        ContentItem(doc_id="d-legacy", source_id="s-legacy", source_type="other", trust="untrusted", text="legacy compat")
    )
    ap = out.evidence.matches.get("api_perception", {})
    assert ap["schema_version"] == "v1_compat"
    assert bool(ap["defensive_context"]) is False
    assert float(ap["confidence"]) == pytest.approx(0.5)
    assert bool(ap["directive_intent"]["tool_or_action_abuse"]) is True


def test_api_projector_schema_error_non_strict(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-schema-nonstrict")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _fake_bad(self, *, text: str):
        _ = text
        raise ValueError("schema_error: missing tool_or_action_abuse")

    monkeypatch.setattr(APIPerceptionProjector, "_call_api_scores", _fake_bad)
    projector = APIPerceptionProjector(config=_mk_cfg(tmp_path, strict=False))
    out = projector.project(
        ContentItem(doc_id="d2", source_id="s2", source_type="other", trust="untrusted", text="test text")
    )
    assert float(out.v.sum()) == 0.0
    st = projector.api_perception_status()
    assert st["api_adapter_active"] is True
    assert st["schema_valid"] is False
    assert (tmp_path / "errors.jsonl").exists()


def test_api_projector_schema_error_strict(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-schema-strict")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _fake_bad(self, *, text: str):
        _ = text
        raise ValueError("schema_error: malformed")

    monkeypatch.setattr(APIPerceptionProjector, "_call_api_scores", _fake_bad)
    projector = APIPerceptionProjector(config=_mk_cfg(tmp_path, strict=True))
    with pytest.raises(Exception):
        projector.project(
            ContentItem(doc_id="d3", source_id="s3", source_type="other", trust="untrusted", text="test text")
        )


def test_api_projector_transient_error_does_not_abort_strict(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-strict-transient")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _fake_fail(self, *, text: str):
        _ = text
        raise RuntimeError("api_call_failed: HTTP 500: server_error")

    monkeypatch.setattr(APIPerceptionProjector, "_call_api_scores", _fake_fail)
    projector = APIPerceptionProjector(config=_mk_cfg(tmp_path, strict=True))
    out = projector.project(
        ContentItem(doc_id="d4", source_id="s4", source_type="other", trust="untrusted", text="test text")
    )
    assert float(out.v.sum()) == 0.0
    assert out.evidence.matches.get("api_perception", {}).get("active", True) is False
    st = projector.api_perception_status()
    assert st["schema_valid"] is False
    assert (tmp_path / "errors.jsonl").exists()


def test_api_projector_fallback_to_chat_on_server_error(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-chat-fallback")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _fake_post_json(*, url, payload, headers, timeout_sec):
        _ = (payload, headers, timeout_sec)
        if str(url).endswith("/responses"):
            raise api_hybrid_module.APIRequestError(code=500, body='{"error":{"type":"server_error"}}')
        return {
            "id": "resp_chat_ok",
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"pressure_signed":{"override_instructions":0.5,'
                            '"secret_exfiltration":-0.3,"tool_or_action_abuse":0.2,'
                            '"policy_evasion":0.0}}'
                        )
                    }
                }
            ],
        }

    monkeypatch.setattr(api_hybrid_module, "_post_json", _fake_post_json)
    projector = APIPerceptionProjector(config=_mk_cfg(tmp_path, strict=True))
    out = projector.project(
        ContentItem(doc_id="d-chat", source_id="s-chat", source_type="other", trust="untrusted", text="fallback test")
    )
    assert float(out.v[0]) == 0.5
    assert float(out.v[1]) == 0.0
    assert out.evidence.polarity[1] == -1
    assert out.evidence.matches["api_perception"]["response_id"] == "resp_chat_ok"


def test_api_projector_long_text_retry_cap(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-long-retry-cap")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = {"responses": 0, "chat": 0}

    def _always_retryable(*, url, payload, headers, timeout_sec):
        _ = (payload, headers, timeout_sec)
        if str(url).endswith("/responses"):
            calls["responses"] += 1
        else:
            calls["chat"] += 1
        raise api_hybrid_module.APIRequestError(code=500, body='{"error":{"type":"server_error"}}')

    monkeypatch.setattr(api_hybrid_module, "_post_json", _always_retryable)
    cfg = _mk_cfg(tmp_path, strict=False)
    cfg["projector"]["api_perception"].update(
        {
            "max_retries": 4,
            "long_text_threshold_chars": 50,
            "long_text_max_retries": 0,
            "request_deadline_sec": 10.0,
            "short_chat_only": False,
            "short_prefer_chat_completions": False,
        }
    )
    projector = APIPerceptionProjector(config=cfg)
    with pytest.raises(RuntimeError, match="api_call_failed"):
        projector._call_api_scores(text="x" * 500)
    # One attempt only for long texts: responses + chat fallback.
    assert calls["responses"] == 1
    assert calls["chat"] == 1


def test_api_projector_transient_error_ttl_cache(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-transient-ttl")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    call_count = {"n": 0}

    def _transient_fail(self, *, text: str):
        _ = text
        call_count["n"] += 1
        raise RuntimeError("api_call_failed: HTTP 500: server_error")

    monkeypatch.setattr(APIPerceptionProjector, "_call_api_scores", _transient_fail)
    cfg = _mk_cfg(tmp_path, strict=True)
    cfg["projector"]["api_perception"]["transient_error_ttl_sec"] = 60.0
    projector = APIPerceptionProjector(config=cfg)
    item = ContentItem(doc_id="d-ttl", source_id="s-ttl", source_type="other", trust="untrusted", text="same text")
    out1 = projector.project(item)
    out2 = projector.project(item)
    assert float(out1.v.sum()) == 0.0
    assert float(out2.v.sum()) == 0.0
    # second call should be served from transient cooldown cache
    assert call_count["n"] == 1


def test_api_projector_responses_cooldown_skips_primary_after_failure(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-responses-cooldown")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = {"responses": 0, "chat": 0}

    def _post_with_responses_failure(*, url, payload, headers, timeout_sec):
        _ = (payload, headers, timeout_sec)
        if str(url).endswith("/responses"):
            calls["responses"] += 1
            raise api_hybrid_module.APIRequestError(code=500, body='{"error":{"type":"server_error"}}')
        calls["chat"] += 1
        return {
            "id": f"resp_chat_{calls['chat']}",
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"pressure_signed":{"override_instructions":0.1,'
                            '"secret_exfiltration":0.0,"tool_or_action_abuse":0.0,'
                            '"policy_evasion":0.0}}'
                        )
                    }
                }
            ],
        }

    monkeypatch.setattr(api_hybrid_module, "_post_json", _post_with_responses_failure)
    cfg = _mk_cfg(tmp_path, strict=True)
    cfg["projector"]["api_perception"].update(
        {
            "responses_cooldown_sec": 120.0,
            "max_retries": 0,
            "request_deadline_sec": 10.0,
            "short_chat_only": False,
            "short_prefer_chat_completions": False,
        }
    )
    projector = APIPerceptionProjector(config=cfg)
    item1 = ContentItem(doc_id="d-cd-1", source_id="s-cd-1", source_type="other", trust="untrusted", text="text one")
    item2 = ContentItem(doc_id="d-cd-2", source_id="s-cd-2", source_type="other", trust="untrusted", text="text two")
    out1 = projector.project(item1)
    out2 = projector.project(item2)
    assert float(out1.v[0]) > 0.0
    assert float(out2.v[0]) > 0.0
    # second request should bypass /responses because cooldown is active
    assert calls["responses"] == 1
    assert calls["chat"] == 2


def test_api_projector_short_chat_only_skips_responses(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-short-chat-only")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = {"responses": 0, "chat": 0}

    def _post_short_chat(*, url, payload, headers, timeout_sec):
        _ = (payload, headers, timeout_sec)
        if str(url).endswith("/responses"):
            calls["responses"] += 1
            raise AssertionError("short chat-only path must not call /responses")
        calls["chat"] += 1
        return {
            "id": "resp_chat_short",
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"pressure_signed":{"override_instructions":0.2,'
                            '"secret_exfiltration":0.0,"tool_or_action_abuse":0.0,'
                            '"policy_evasion":0.0}}'
                        )
                    }
                }
            ],
        }

    monkeypatch.setattr(api_hybrid_module, "_post_json", _post_short_chat)
    cfg = _mk_cfg(tmp_path, strict=True)
    cfg["projector"]["api_perception"].update(
        {
            "short_chat_only": True,
            "short_prefer_chat_completions": True,
            "short_text_threshold_chars": 300,
            "max_retries": 0,
            "request_deadline_sec": 10.0,
        }
    )
    projector = APIPerceptionProjector(config=cfg)
    out = projector.project(
        ContentItem(doc_id="d-short", source_id="s-short", source_type="other", trust="untrusted", text="short text")
    )
    assert float(out.v[0]) > 0.0
    assert calls["responses"] == 0
    assert calls["chat"] == 1
    assert projector._prewarmed is True


def test_hybrid_api_short_fast_path_skips_api_on_pi0_clean(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-short-fast-clean")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _should_not_call_api(self, item):  # pragma: no cover - assertion by exception
        _ = item
        raise AssertionError("API projector must be skipped on short PI0-clean fast path")

    class _Pi0:
        def project(self, item):
            _ = item
            return api_hybrid_module.ProjectionResult(
                doc_id="pi0_doc",
                v=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
                evidence=api_hybrid_module.ProjectionEvidence(
                    polarity=[0, 0, 0, 0],
                    debug_scores_raw=[0.0, 0.0, 0.0, 0.0],
                    matches={
                        "pi0_rule_tier": {
                            "hard_any": False,
                            "soft_any": False,
                            "walls": {},
                        }
                    },
                ),
            )

    monkeypatch.setattr(APIPerceptionProjector, "project", _should_not_call_api)
    cfg = _mk_cfg(tmp_path, strict=True)
    cfg["projector"]["api_perception"].update(
        {
            "short_fast_path_enabled": True,
            "short_text_threshold_chars": 500,
            "short_fast_path_skip_on_pi0_clean": True,
            "short_fast_path_skip_on_pi0_hard": True,
            "short_fast_path_clean_max_score": 0.0,
            "short_fast_path_hard_min_score": 0.55,
        }
    )
    api_proj = APIPerceptionProjector(config=cfg)
    hybrid = HybridAPIProjector(pi0_projector=_Pi0(), api_projector=api_proj)
    out = hybrid.project(ContentItem(doc_id="d-clean", source_id="s-clean", source_type="other", trust="untrusted", text="hello"))
    hm = out.evidence.matches.get("hybrid_api", {})
    ap = out.evidence.matches.get("api_perception", {})
    assert bool(hm.get("short_fast_path_applied")) is True
    assert str(hm.get("short_fast_path_reason")) == "pi0_clean_high_confidence"
    assert bool(ap.get("short_fast_path_applied")) is True
    assert str(ap.get("short_fast_path_reason")) == "pi0_clean_high_confidence"
    assert float(out.v.sum()) == 0.0


def test_factory_builds_hybrid_api(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-factory")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _fake_call(self, *, text: str):
        _ = text
        return (
            {
                "override_instructions": 0.05,
                "secret_exfiltration": -0.05,
                "tool_or_action_abuse": 0.05,
                "policy_evasion": 0.05,
            },
            "resp_factory",
        )

    monkeypatch.setattr(APIPerceptionProjector, "_call_api_scores", _fake_call)
    snapshot = load_resolved_config(
        profile="dev",
        cli_overrides={
            "projector": {
                "mode": "hybrid_api",
                "fallback_to_pi0": False,
                "api_perception": {
                    "enabled": "true",
                    "strict": True,
                    "cache_path": str((tmp_path / "cache_factory.jsonl").as_posix()),
                    "error_log_path": str((tmp_path / "errors_factory.jsonl").as_posix()),
                },
            }
        },
    )
    projector = build_projector(snapshot.resolved)
    assert projector.__class__.__name__ == "HybridAPIProjector"
    st = projector.api_perception_status()
    assert st["api_adapter_active"] is True


def test_hybrid_api_deescalation_zero_boost_applies(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-deesc-apply")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _fake_call(self, *, text: str):
        _ = text
        return (
            {
                "schema_version": "api_hybrid_v2",
                "pressure_signed": {
                    "override_instructions": 0.2,
                    "secret_exfiltration": 0.0,
                    "tool_or_action_abuse": 0.0,
                    "policy_evasion": 0.0,
                },
                "directive_intent": {
                    "override_instructions": False,
                    "secret_exfiltration": False,
                    "tool_or_action_abuse": False,
                    "policy_evasion": False,
                },
                "defensive_context": True,
                "confidence": 0.95,
            },
            "resp_deesc",
        )

    class _Pi0:
        def project(self, item):
            _ = item
            return api_hybrid_module.ProjectionResult(
                doc_id="pi0_doc",
                v=np.array([0.1, 0.0, 0.0, 0.0], dtype=float),
                evidence=api_hybrid_module.ProjectionEvidence(
                    polarity=[1, 0, 0, 0],
                    debug_scores_raw=[0.1, 0.0, 0.0, 0.0],
                    matches={"pi0_mock": True},
                ),
            )

    monkeypatch.setattr(APIPerceptionProjector, "_call_api_scores", _fake_call)
    cfg = _mk_cfg(tmp_path, strict=True)
    cfg["projector"]["api_perception"]["deescalation"] = {"confidence_min": 0.75, "p_strong": 0.35}
    api_proj = APIPerceptionProjector(config=cfg)
    hybrid = HybridAPIProjector(pi0_projector=_Pi0(), api_projector=api_proj)
    out = hybrid.project(ContentItem(doc_id="d", source_id="s", source_type="other", trust="untrusted", text="x"))
    assert float(out.v[0]) == pytest.approx(0.1)
    ap = out.evidence.matches.get("api_perception", {})
    assert bool(ap["deescalation_applied"]) is True


def test_hybrid_api_deescalation_no_trigger_with_directive(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-deesc-no")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _fake_call(self, *, text: str):
        _ = text
        return (
            {
                "schema_version": "api_hybrid_v2",
                "pressure_signed": {
                    "override_instructions": 0.2,
                    "secret_exfiltration": 0.0,
                    "tool_or_action_abuse": 0.0,
                    "policy_evasion": 0.0,
                },
                "directive_intent": {
                    "override_instructions": True,
                    "secret_exfiltration": False,
                    "tool_or_action_abuse": False,
                    "policy_evasion": False,
                },
                "defensive_context": True,
                "confidence": 0.95,
            },
            "resp_no_deesc",
        )

    class _Pi0:
        def project(self, item):
            _ = item
            return api_hybrid_module.ProjectionResult(
                doc_id="pi0_doc",
                v=np.array([0.05, 0.0, 0.0, 0.0], dtype=float),
                evidence=api_hybrid_module.ProjectionEvidence(
                    polarity=[1, 0, 0, 0],
                    debug_scores_raw=[0.05, 0.0, 0.0, 0.0],
                    matches={"pi0_mock": True},
                ),
            )

    monkeypatch.setattr(APIPerceptionProjector, "_call_api_scores", _fake_call)
    cfg = _mk_cfg(tmp_path, strict=True)
    cfg["projector"]["api_perception"]["deescalation"] = {"confidence_min": 0.75, "p_strong": 0.35}
    api_proj = APIPerceptionProjector(config=cfg)
    hybrid = HybridAPIProjector(pi0_projector=_Pi0(), api_projector=api_proj)
    out = hybrid.project(ContentItem(doc_id="d2", source_id="s2", source_type="other", trust="untrusted", text="y"))
    assert float(out.v[0]) == pytest.approx(0.2)
    ap = out.evidence.matches.get("api_perception", {})
    assert bool(ap["deescalation_applied"]) is False


def test_hybrid_api_soft_only_without_api_confirmation_is_suppressed(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-soft-suppressed")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _fake_call(self, *, text: str):
        _ = text
        return (
            {
                "schema_version": "api_hybrid_v2",
                "pressure_signed": {
                    "override_instructions": 0.0,
                    "secret_exfiltration": 0.0,
                    "tool_or_action_abuse": 0.0,
                    "policy_evasion": 0.0,
                },
                "directive_intent": {
                    "override_instructions": False,
                    "secret_exfiltration": False,
                    "tool_or_action_abuse": False,
                    "policy_evasion": False,
                },
                "defensive_context": False,
                "confidence": 0.9,
            },
            "resp_soft_suppressed",
        )

    class _Pi0:
        def project(self, item):
            _ = item
            return api_hybrid_module.ProjectionResult(
                doc_id="pi0_doc",
                v=np.array([0.4, 0.0, 0.0, 0.0], dtype=float),
                evidence=api_hybrid_module.ProjectionEvidence(
                    polarity=[1, 0, 0, 0],
                    debug_scores_raw=[0.4, 0.0, 0.0, 0.0],
                    matches={
                        "pi0_rule_tier": {
                            "hard_any": False,
                            "soft_any": True,
                            "walls": {
                                "override_instructions": {
                                    "hard_hit": False,
                                    "soft_hit": True,
                                    "hard_signals": [],
                                    "soft_signals": ["soft_directive_guard_blocked"],
                                    "raw_score": 0.4,
                                }
                            },
                        }
                    },
                ),
            )

    monkeypatch.setattr(APIPerceptionProjector, "_call_api_scores", _fake_call)
    cfg = _mk_cfg(tmp_path, strict=True)
    cfg["projector"]["api_perception"]["hybrid_soft_gate"] = {
        "enabled": True,
        "soft_confirm_min": 0.10,
        "require_api_for_soft": True,
    }
    api_proj = APIPerceptionProjector(config=cfg)
    hybrid = HybridAPIProjector(pi0_projector=_Pi0(), api_projector=api_proj)
    out = hybrid.project(ContentItem(doc_id="d-soft", source_id="s-soft", source_type="other", trust="untrusted", text="x"))
    assert float(out.v[0]) == pytest.approx(0.0)
    hm = out.evidence.matches.get("hybrid_api", {})
    assert bool(hm.get("soft_suppressed_any", False)) is True
    assert "override_instructions" in list(hm.get("suppressed_walls", []))
    assert np.all(out.v >= 0.0)


def test_hybrid_api_soft_only_with_api_confirmation_passes(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-soft-confirmed")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _fake_call(self, *, text: str):
        _ = text
        return (
            {
                "schema_version": "api_hybrid_v2",
                "pressure_signed": {
                    "override_instructions": 0.2,
                    "secret_exfiltration": 0.0,
                    "tool_or_action_abuse": 0.0,
                    "policy_evasion": 0.0,
                },
                "directive_intent": {
                    "override_instructions": False,
                    "secret_exfiltration": False,
                    "tool_or_action_abuse": False,
                    "policy_evasion": False,
                },
                "defensive_context": False,
                "confidence": 0.9,
            },
            "resp_soft_confirmed",
        )

    class _Pi0:
        def project(self, item):
            _ = item
            return api_hybrid_module.ProjectionResult(
                doc_id="pi0_doc",
                v=np.array([0.4, 0.0, 0.0, 0.0], dtype=float),
                evidence=api_hybrid_module.ProjectionEvidence(
                    polarity=[1, 0, 0, 0],
                    debug_scores_raw=[0.4, 0.0, 0.0, 0.0],
                    matches={
                        "pi0_rule_tier": {
                            "hard_any": False,
                            "soft_any": True,
                            "walls": {
                                "override_instructions": {
                                    "hard_hit": False,
                                    "soft_hit": True,
                                    "hard_signals": [],
                                    "soft_signals": ["soft_directive_guard_blocked"],
                                    "raw_score": 0.4,
                                }
                            },
                        }
                    },
                ),
            )

    monkeypatch.setattr(APIPerceptionProjector, "_call_api_scores", _fake_call)
    cfg = _mk_cfg(tmp_path, strict=True)
    cfg["projector"]["api_perception"]["hybrid_soft_gate"] = {
        "enabled": True,
        "soft_confirm_min": 0.10,
        "require_api_for_soft": True,
    }
    api_proj = APIPerceptionProjector(config=cfg)
    hybrid = HybridAPIProjector(pi0_projector=_Pi0(), api_projector=api_proj)
    out = hybrid.project(
        ContentItem(doc_id="d-soft-confirm", source_id="s-soft-confirm", source_type="other", trust="untrusted", text="x")
    )
    assert float(out.v[0]) == pytest.approx(0.4)
    hm = out.evidence.matches.get("hybrid_api", {})
    assert bool(hm.get("soft_confirmed_any", False)) is True
    assert "override_instructions" in list(hm.get("confirmation_walls", []))
    assert np.all(out.v >= 0.0)


def test_hybrid_api_hard_signal_is_not_suppressed_without_api_confirmation(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-hard-no-suppress")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _fake_call(self, *, text: str):
        _ = text
        return (
            {
                "schema_version": "api_hybrid_v2",
                "pressure_signed": {
                    "override_instructions": 0.0,
                    "secret_exfiltration": 0.0,
                    "tool_or_action_abuse": 0.0,
                    "policy_evasion": 0.0,
                },
                "directive_intent": {
                    "override_instructions": False,
                    "secret_exfiltration": False,
                    "tool_or_action_abuse": False,
                    "policy_evasion": False,
                },
                "defensive_context": False,
                "confidence": 0.9,
            },
            "resp_hard_kept",
        )

    class _Pi0:
        def project(self, item):
            _ = item
            return api_hybrid_module.ProjectionResult(
                doc_id="pi0_doc",
                v=np.array([0.4, 0.0, 0.0, 0.0], dtype=float),
                evidence=api_hybrid_module.ProjectionEvidence(
                    polarity=[1, 0, 0, 0],
                    debug_scores_raw=[0.4, 0.0, 0.0, 0.0],
                    matches={
                        "pi0_rule_tier": {
                            "hard_any": True,
                            "soft_any": False,
                            "walls": {
                                "override_instructions": {
                                    "hard_hit": True,
                                    "soft_hit": False,
                                    "hard_signals": ["real_override_intent"],
                                    "soft_signals": [],
                                    "raw_score": 0.4,
                                }
                            },
                        }
                    },
                ),
            )

    monkeypatch.setattr(APIPerceptionProjector, "_call_api_scores", _fake_call)
    cfg = _mk_cfg(tmp_path, strict=True)
    cfg["projector"]["api_perception"]["hybrid_soft_gate"] = {
        "enabled": True,
        "soft_confirm_min": 0.10,
        "require_api_for_soft": True,
    }
    api_proj = APIPerceptionProjector(config=cfg)
    hybrid = HybridAPIProjector(pi0_projector=_Pi0(), api_projector=api_proj)
    out = hybrid.project(ContentItem(doc_id="d-hard", source_id="s-hard", source_type="other", trust="untrusted", text="x"))
    assert float(out.v[0]) == pytest.approx(0.4)
    hm = out.evidence.matches.get("hybrid_api", {})
    assert bool(hm.get("soft_suppressed_any", False)) is False
    assert np.all(out.v >= 0.0)


def test_hybrid_api_benign_stabilizer_suppresses_non_hard_soft(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-benign-stabilizer-soft")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _fake_call(self, *, text: str):
        _ = text
        return (
            {
                "schema_version": "api_hybrid_v2",
                "pressure_signed": {
                    "override_instructions": 0.05,
                    "secret_exfiltration": 0.0,
                    "tool_or_action_abuse": 0.0,
                    "policy_evasion": 0.0,
                },
                "directive_intent": {
                    "override_instructions": False,
                    "secret_exfiltration": False,
                    "tool_or_action_abuse": False,
                    "policy_evasion": False,
                },
                "defensive_context": False,
                "confidence": 0.95,
            },
            "resp_benign_stabilizer_soft",
        )

    class _Pi0:
        def project(self, item):
            _ = item
            return api_hybrid_module.ProjectionResult(
                doc_id="pi0_doc",
                v=np.array([0.4, 0.0, 0.0, 0.0], dtype=float),
                evidence=api_hybrid_module.ProjectionEvidence(
                    polarity=[1, 0, 0, 0],
                    debug_scores_raw=[0.4, 0.0, 0.0, 0.0],
                    matches={
                        "pi0_rule_tier": {
                            "hard_any": False,
                            "soft_any": True,
                            "walls": {
                                "override_instructions": {
                                    "hard_hit": False,
                                    "soft_hit": True,
                                    "hard_signals": [],
                                    "soft_signals": ["semantic_boost"],
                                    "raw_score": 0.4,
                                }
                            },
                        }
                    },
                ),
            )

    monkeypatch.setattr(APIPerceptionProjector, "_call_api_scores", _fake_call)
    cfg = _mk_cfg(tmp_path, strict=True)
    cfg["projector"]["api_perception"]["hybrid_soft_gate"] = {
        "enabled": False,
        "soft_confirm_min": 0.10,
        "require_api_for_soft": False,
    }
    api_proj = APIPerceptionProjector(config=cfg)
    hybrid = HybridAPIProjector(pi0_projector=_Pi0(), api_projector=api_proj)
    out = hybrid.project(ContentItem(doc_id="d-benign-stab", source_id="s-benign-stab", source_type="other", trust="untrusted", text="x"))
    assert float(out.v[0]) == pytest.approx(0.05)
    hm = out.evidence.matches.get("hybrid_api", {})
    assert bool(hm.get("benign_stabilizer_applied", False)) is True
    assert hm.get("benign_stabilizer_walls", []) == ["override_instructions"]
    assert np.all(out.v >= 0.0)


def test_hybrid_api_benign_stabilizer_keeps_hard_signal(monkeypatch: pytest.MonkeyPatch):
    tmp_path = _mk_local_tmp("api-hybrid-benign-stabilizer-hard")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _fake_call(self, *, text: str):
        _ = text
        return (
            {
                "schema_version": "api_hybrid_v2",
                "pressure_signed": {
                    "override_instructions": 0.05,
                    "secret_exfiltration": 0.0,
                    "tool_or_action_abuse": 0.0,
                    "policy_evasion": 0.0,
                },
                "directive_intent": {
                    "override_instructions": False,
                    "secret_exfiltration": False,
                    "tool_or_action_abuse": False,
                    "policy_evasion": False,
                },
                "defensive_context": False,
                "confidence": 0.95,
            },
            "resp_benign_stabilizer_hard",
        )

    class _Pi0:
        def project(self, item):
            _ = item
            return api_hybrid_module.ProjectionResult(
                doc_id="pi0_doc",
                v=np.array([0.4, 0.0, 0.0, 0.0], dtype=float),
                evidence=api_hybrid_module.ProjectionEvidence(
                    polarity=[1, 0, 0, 0],
                    debug_scores_raw=[0.4, 0.0, 0.0, 0.0],
                    matches={
                        "pi0_rule_tier": {
                            "hard_any": True,
                            "soft_any": True,
                            "walls": {
                                "override_instructions": {
                                    "hard_hit": True,
                                    "soft_hit": True,
                                    "hard_signals": ["real_override_intent"],
                                    "soft_signals": ["semantic_boost"],
                                    "raw_score": 0.4,
                                }
                            },
                        }
                    },
                ),
            )

    monkeypatch.setattr(APIPerceptionProjector, "_call_api_scores", _fake_call)
    cfg = _mk_cfg(tmp_path, strict=True)
    cfg["projector"]["api_perception"]["hybrid_soft_gate"] = {
        "enabled": False,
        "soft_confirm_min": 0.10,
        "require_api_for_soft": False,
    }
    api_proj = APIPerceptionProjector(config=cfg)
    hybrid = HybridAPIProjector(pi0_projector=_Pi0(), api_projector=api_proj)
    out = hybrid.project(ContentItem(doc_id="d-benign-stab-hard", source_id="s-benign-stab-hard", source_type="other", trust="untrusted", text="x"))
    assert float(out.v[0]) == pytest.approx(0.4)
    hm = out.evidence.matches.get("hybrid_api", {})
    assert bool(hm.get("benign_stabilizer_applied", False)) is False
    assert hm.get("benign_stabilizer_walls", []) == []
    assert np.all(out.v >= 0.0)


def test_smoke_projector_status_active(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _fake_call(self, *, text: str):
        _ = text
        return (
            {
                "override_instructions": 0.01,
                "secret_exfiltration": 0.02,
                "tool_or_action_abuse": -0.03,
                "policy_evasion": 0.04,
            },
            "resp_smoke",
        )

    monkeypatch.setattr(APIPerceptionProjector, "_call_api_scores", _fake_call)
    argv = [
        "smoke_projector_status.py",
        "--profile",
        "dev",
        "--mode",
        "hybrid_api",
        "--api-model",
        "gpt-5",
    ]
    old_argv = list(sys.argv)
    try:
        sys.argv = argv
        rc = smoke_projector_status.main()
    finally:
        sys.argv = old_argv
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["status"] in {"active", "calibrated(error-schema)"}
    assert payload["api_adapter_active"] is True
