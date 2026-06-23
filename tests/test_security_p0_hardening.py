from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import sqlite3
import zipfile

import numpy as np
import pytest

from omega.api.runtime_factory import ApiAuth, _parse_api_keys, _validate_prod_runtime_secrets
from omega.api.session_store import ApiSessionStore
from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import (
    ContentItem,
    OmegaState,
    ProjectionEvidence,
    ProjectionResult,
    ToolRequest,
)
from omega.policy.cross_session_state import CrossSessionStateManager
from omega.projector.api_hybrid.normalization import normalize_api_payload
from omega.telemetry.redaction import redact_text
from omega.tools.adapters import build_default_tool_registry
from omega.tools.tool_gateway import ToolGatewayV1
from scripts.build_clean_source_archive import build_archive
from scripts.secret_scan import scan_tree


ROOT = Path(__file__).resolve().parents[1]


def _evidence() -> ProjectionEvidence:
    return ProjectionEvidence(
        polarity=[0, 0, 0, 0],
        debug_scores_raw=[0.0, 0.0, 0.0, 0.0],
        matches={},
    )


def test_redaction_removes_secret_and_pii_patterns_before_truncation() -> None:
    openai_key = "sk-" + "pR0dSecretTokenValue1234567890"
    bearer = "Bearer-" + "HeaderValue987654321"
    email = "owner" + "@example.com"
    phone = "+90 555 123 45 67"
    private_key = "-----BEGIN PRIVATE KEY-----\n" + ("AbCdEf0123456789" * 8) + "\n-----END PRIVATE KEY-----"
    assignment = "api" + "_key=" + openai_key
    source = (
        "A" * 48
        + f" Authorization: Bearer {bearer}; {assignment}; contact={email}; tel={phone}; {private_key}"
    )

    result = redact_text(source, max_chars=120)

    assert result.redaction_hits >= 5
    assert openai_key not in result.redacted
    assert bearer not in result.redacted
    assert email not in result.redacted
    assert phone not in result.redacted
    assert "BEGIN PRIVATE KEY" not in result.redacted
    assert len(result.redacted) <= 120
    assert len(result.text_sha256) == 64


@pytest.mark.parametrize(
    "vector",
    [
        [float("nan"), 0.0, 0.0, 0.0],
        [float("inf"), 0.0, 0.0, 0.0],
        [-0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [[0.0, 0.0, 0.0, 0.0]],
    ],
)
def test_projection_contract_rejects_nonfinite_negative_or_wrong_shape(vector) -> None:
    with pytest.raises(ValueError):
        ProjectionResult(doc_id="doc", v=np.asarray(vector, dtype=float), evidence=_evidence())


@pytest.mark.parametrize(
    "state_vector",
    [
        [float("nan"), 0.0, 0.0, 0.0],
        [float("-inf"), 0.0, 0.0, 0.0],
        [-0.01, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
)
def test_state_contract_rejects_nonfinite_negative_or_wrong_shape(state_vector) -> None:
    with pytest.raises(ValueError):
        OmegaState(session_id="session", m=np.asarray(state_vector, dtype=float), step=0)


def test_core_revalidates_mutated_projection_without_committing_state(omega_core) -> None:
    state = OmegaState(session_id="session", m=np.zeros(4, dtype=float), step=0)
    projection = ProjectionResult(doc_id="doc", v=np.zeros(4, dtype=float), evidence=_evidence())
    projection.v[0] = float("nan")  # simulate an unsafe post-construction mutation/provider boundary
    item = ContentItem(
        doc_id="doc",
        source_id="source",
        source_type="web",
        trust="untrusted",
        text="benign",
    )

    with pytest.raises(ValueError, match="finite"):
        omega_core.step(state, [item], [projection])

    assert state.step == 0
    assert np.array_equal(state.m, np.zeros(4, dtype=float))


def test_session_store_rejects_nonfinite_write_and_corrupt_json(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    store = ApiSessionStore(sqlite_path=db_path)

    with pytest.raises(ValueError, match="finite"):
        store.save_session_state(
            tenant_id="tenant",
            session_id="session",
            actor_id="actor",
            m=np.asarray([float("nan"), 0.0, 0.0, 0.0]),
            step=1,
        )

    now = 2_000_000_000
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO session_state(tenant_id, session_id, actor_id, m_json, step, updated_at_ts, expires_at_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("tenant", "session", "actor", "[NaN, 0, 0, 0]", 1, now, now + 1000),
        )

    with pytest.raises(ValueError, match="strict JSON"):
        store.load_session_state(tenant_id="tenant", session_id="session")


def test_semantic_payload_rejects_nonfinite_values() -> None:
    payload = {
        "schema_version": "api_hybrid_v2",
        "pressure_signed": {
            "override_instructions": float("nan"),
            "secret_exfiltration": 0.0,
            "tool_or_action_abuse": 0.0,
            "policy_evasion": 0.0,
        },
        "confidence": 0.5,
    }
    with pytest.raises(ValueError, match="finite"):
        normalize_api_payload(payload)

    payload["pressure_signed"]["override_instructions"] = 0.0
    payload["confidence"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        normalize_api_payload(payload)


def test_default_registry_omits_side_effecting_adapters() -> None:
    registry = build_default_tool_registry()
    assert registry.has("summarize")
    assert registry.has("retrieval_readonly")
    assert registry.has("echo")
    assert not registry.has("write_file")
    assert not registry.has("network_post")


def test_legacy_side_effecting_builtins_cannot_be_reenabled_by_config(resolved_config: dict) -> None:
    config = deepcopy(resolved_config)
    config["tools"]["capabilities"]["network_post"]["enabled"] = True
    gateway = ToolGatewayV1(config)
    request = ToolRequest(
        tool_name="network_post",
        args={"url": "https://example.com", "payload": "x", "human_approved": True},
        session_id="s",
        step=1,
    )
    decision = gateway.enforce(request, [])
    assert decision.allowed is False
    assert decision.reason == "TOOL_DISABLED_BY_CONFIG"


def test_disabled_dangerous_tools_cannot_be_enabled_by_request_flag(resolved_config: dict) -> None:
    gateway = ToolGatewayV1(resolved_config)
    for tool_name, args in (
        ("write_file", {"filename": "ok.txt", "content": "x", "human_approved": True}),
        ("network_post", {"url": "https://example.com", "payload": "x", "human_approved": True}),
    ):
        decision = gateway.enforce(ToolRequest(tool_name=tool_name, args=args, session_id="s", step=1), [])
        assert decision.allowed is False
        assert decision.reason == "TOOL_DISABLED_BY_CONFIG"


def test_prod_credentials_fail_closed_and_accept_only_strong_runtime_secret(monkeypatch) -> None:
    config = load_resolved_config(profile="prod").resolved
    env_name = config["api"]["auth"]["api_key_env"]
    monkeypatch.delenv(env_name, raising=False)

    with pytest.raises(RuntimeError, match="requires credentials"):
        _parse_api_keys(config)

    monkeypatch.setenv(env_name, "dev-api-key")
    with pytest.raises(RuntimeError, match="refuses development"):
        _parse_api_keys(config)

    strong_key = "prod_" + "A9z" * 12
    monkeypatch.setenv(env_name, strong_key)
    assert _parse_api_keys(config) == [strong_key]
    assert config["api"]["runtime"]["mode"] == "stateful"
    assert config["api"]["runtime"]["allow_request_override"] is False
    assert config["api"]["auth"]["require_hmac"] is True


def test_prod_runtime_requires_separate_strong_hmac_secret(monkeypatch) -> None:
    config = load_resolved_config(profile="prod").resolved
    api_env = config["api"]["auth"]["api_key_env"]
    hmac_env = config["api"]["auth"]["hmac_secret_env"]
    api_key = "prod_" + "K7m" * 12
    monkeypatch.setenv(api_env, api_key)
    api_keys = _parse_api_keys(config)
    auth = ApiAuth.from_cfg(config["api"]["auth"])

    monkeypatch.delenv(hmac_env, raising=False)
    with pytest.raises(RuntimeError, match="requires HMAC secret"):
        _validate_prod_runtime_secrets(config, auth, api_keys)

    monkeypatch.setenv(hmac_env, "short-secret")
    with pytest.raises(RuntimeError, match="short HMAC"):
        _validate_prod_runtime_secrets(config, auth, api_keys)

    monkeypatch.setenv(hmac_env, api_key)
    with pytest.raises(RuntimeError, match="separate"):
        _validate_prod_runtime_secrets(config, auth, api_keys)

    monkeypatch.setenv(hmac_env, "hmac_" + "Z8q" * 16)
    _validate_prod_runtime_secrets(config, auth, api_keys)


def test_cross_session_store_rejects_nonfinite_or_wrong_shape_state(tmp_path: Path) -> None:
    config = load_resolved_config(profile="dev").resolved
    config["off_policy"]["cross_session"]["enabled"] = True
    config["off_policy"]["cross_session"]["sqlite_path"] = str(tmp_path / "cross.db")
    manager = CrossSessionStateManager.from_config(config)
    manager.hydrate_actor_state("actor", "session")

    with sqlite3.connect(manager.sqlite_path) as conn:
        conn.execute(
            "UPDATE actor_state SET scars_json = ? WHERE actor_id = ?",
            ("[NaN, 0, 0, 0]", "actor"),
        )

    with pytest.raises(ValueError, match="strict JSON"):
        manager.hydrate_actor_state("actor", "session-2")

    with sqlite3.connect(manager.sqlite_path) as conn:
        conn.execute(
            "UPDATE actor_state SET scars_json = ? WHERE actor_id = ?",
            ("[0, 0, 0]", "actor"),
        )

    with pytest.raises(ValueError, match="shape"):
        manager.hydrate_actor_state("actor", "session-3")


def test_secret_scanner_blocks_env_file(tmp_path: Path) -> None:
    secret_value = "sk-" + "actualLookingSecret1234567890ABCDE"
    (tmp_path / ".env").write_text("OPENAI_API_KEY=" + secret_value + "\n", encoding="utf-8")
    findings = scan_tree(tmp_path)
    assert any(finding.rule == "forbidden_env_file" for finding in findings)


def test_clean_source_archive_is_secret_scanned_and_reproducible(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    (source_root / ".github" / "workflows").mkdir(parents=True)
    (source_root / ".github" / "workflows" / "security-release.yml").write_text("name: security\n", encoding="utf-8")
    (source_root / "package").mkdir()
    (source_root / "package" / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source_root / ".env").write_text("SHOULD_NOT_SHIP=hidden\n", encoding="utf-8")
    (source_root / "package" / "__pycache__").mkdir()
    (source_root / "package" / "__pycache__" / "main.pyc").write_bytes(b"cache")

    first = tmp_path / "source-1.zip"
    second = tmp_path / "source-2.zip"
    report_first = build_archive(root=source_root, output=first, prefix="OmegaWalls")
    report_second = build_archive(root=source_root, output=second, prefix="OmegaWalls")

    assert report_first["secret_scan"] == "passed"
    assert report_second["secret_scan"] == "passed"
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
        assert "OmegaWalls/SOURCE_ARCHIVE_MANIFEST.json" in names
        assert "OmegaWalls/.github/workflows/security-release.yml" in names
        assert "OmegaWalls/package/main.py" in names
        assert all("/__pycache__/" not in name for name in names)
        assert all(not name.endswith((".pyc", ".pyo")) for name in names)
        assert all(Path(name).name not in {".env", ".env.local", ".env.prod", ".env.production"} for name in names)
        manifest = json.loads(archive.read("OmegaWalls/SOURCE_ARCHIVE_MANIFEST.json"))
        assert manifest["secret_scan"] == "passed"
        assert manifest["file_count"] == 2
