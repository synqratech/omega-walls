from __future__ import annotations

import json
from pathlib import Path
import uuid

import pytest

from omega.interfaces.contracts_v1 import ContentItem
from omega.orchestrator.provider_runtime import OrchestratorConfig, OrchestratorRuntime, ProviderCandidate
from omega.projector.api_hybrid_projector import APIPerceptionProjector


def _tmp_dir(name: str) -> Path:
    base = Path("artifacts/state").resolve()
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"orch_{name}_{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()


def test_orchestrator_key_encrypted_at_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp = _tmp_dir("orch_enc")
    db_path = tmp / "orch.db"
    monkeypatch.setenv("OMEGA_MASTER_KEY", "hex:" + "ab" * 32)
    cfg = OrchestratorConfig(enabled=True, sqlite_path=str(db_path))
    runtime = OrchestratorRuntime(config=cfg, actor="test")
    runtime.add_primary_key(provider_id="openai-prod", key="sk-plain-secret-1234")

    raw = db_path.read_bytes()
    assert b"sk-plain-secret-1234" not in raw
    keys = runtime.vault.list_keys()
    assert keys and keys[0]["key_mask"].endswith("1234")


def test_orchestrator_state_machine_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp = _tmp_dir("orch_state")
    monkeypatch.setenv("OMEGA_MASTER_KEY", "hex:" + "cd" * 32)
    cfg = OrchestratorConfig(
        enabled=True,
        sqlite_path=str(tmp / "orch.db"),
        threshold_errors=3,
        threshold_window_sec=60,
    )
    runtime = OrchestratorRuntime(config=cfg, actor="test")
    runtime.vault.put_key(provider_id="p1", slot="primary", key_ref="p1", key_plain="sk-1111")

    s1 = runtime.mark_error(provider_id="p1", slot="primary", error="api_call_failed: HTTP 429: too_many_requests")
    assert s1["health_state"] in {"warning", "fallback_active"}
    s2 = runtime.mark_error(provider_id="p1", slot="primary", error="api_call_failed: HTTP 429: too_many_requests")
    s3 = runtime.mark_error(provider_id="p1", slot="primary", error="api_call_failed: HTTP 429: too_many_requests")
    assert s3["health_state"] == "fallback_active"
    assert s3["fallback_level"] == "backup_provider"
    runtime.mark_success(provider_id="p1", slot="primary")
    s_ok = runtime.vault.get_state(provider_id="p1")
    assert s_ok["health_state"] == "healthy"


def test_api_projector_orchestrator_backup_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp = _tmp_dir("orch_proj")
    monkeypatch.setenv("OMEGA_MASTER_KEY", "hex:" + "ef" * 32)
    cfg = {
        "projector": {
            "mode": "hybrid_api",
            "fallback_to_pi0": False,
            "api_perception": {
                "enabled": "true",
                "strict": False,
                "provider": "openai",
                "model": "gpt-5.4-mini",
                "base_url": "https://api.openai.com/v1",
                "cache_path": str(tmp / "api_cache.jsonl"),
                    "orchestrator": {
                        "enabled": True,
                        "master_key_env": "OMEGA_MASTER_KEY",
                        "store": {"sqlite_path": str(tmp / "orch.db")},
                        "fallback": {"threshold": {"errors": 1, "window_sec": 60}},
                        "providers": [
                        {
                            "id": "openai-main",
                            "type": "openai",
                            "model": "gpt-5.4-mini",
                            "base_url": "https://api.openai.com/v1",
                            "primary_ref": "openai-main",
                            "backup_ref": "openai-main",
                        }
                    ],
                },
            },
        }
    }
    projector = APIPerceptionProjector(config=cfg)
    assert projector._orchestrator is not None  # noqa: SLF001
    projector._orchestrator.add_primary_key(provider_id="openai-main", key="sk-primary-0001")  # noqa: SLF001
    projector._orchestrator.set_backup_key(provider_id="openai-main", key="sk-backup-0002")  # noqa: SLF001

    calls = {"primary": 0, "backup": 0}

    def _fake_call(self, *, candidate, api_key, text):  # type: ignore[no-untyped-def]
        _ = text
        if str(candidate.key_slot) == "primary":
            calls["primary"] += 1
            raise RuntimeError("api_call_failed: HTTP 429: exhausted")
        calls["backup"] += 1
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
                "defensive_context": False,
                "confidence": 0.8,
                "scores": {
                    "override_instructions": 0.2,
                    "secret_exfiltration": 0.0,
                    "tool_or_action_abuse": 0.0,
                    "policy_evasion": 0.0,
                },
            },
            "resp_backup",
        )

    monkeypatch.setattr(APIPerceptionProjector, "_call_candidate_scores", _fake_call)
    out = projector.project(
        ContentItem(
            doc_id="d-orch",
            source_id="s-orch",
            source_type="other",
            trust="untrusted",
            text=f"test fallback {uuid.uuid4().hex}",
        )
    )
    assert calls["primary"] == 1
    assert calls["backup"] == 1
    st = projector.api_perception_status()
    assert st["llm_fallback_active"] is True
    assert st["fallback_level"] == "backup_provider"
    ap = dict(out.evidence.matches.get("api_perception", {}))
    assert ap.get("provider_id") == "openai-main"
    assert ap.get("fallback_level") == "backup_provider"
