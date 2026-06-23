from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import ContentItem, WALLS_V1
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2
from tests.helpers import load_jsonl

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "tests" / "data" / "refactor_characterization" / "pi0_rulepath_golden_v1.json"


def _projector_rule_only() -> Pi0IntentAwareV2:
    cfg = load_resolved_config(profile="dev").resolved
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("pi0", {}).setdefault("semantic", {})["enabled"] = "false"
    return Pi0IntentAwareV2(cfg)


def _round_walls(v: Any) -> Dict[str, float]:
    return {wall: round(float(v[idx]), 6) for idx, wall in enumerate(WALLS_V1)}


def _pick_rows() -> List[Dict[str, Any]]:
    corpus: List[Dict[str, Any]] = []
    corpus.extend(load_jsonl("tests/data/redteam_pos_20.jsonl")[:6])
    corpus.extend(load_jsonl("tests/data/redteam_obf_20.jsonl")[:6])
    corpus.extend(load_jsonl("tests/data/hard_negatives_50.jsonl")[:8])
    corpus.extend(load_jsonl("tests/data/whitebox_regression_v1.jsonl")[:8])
    return corpus


def _build_snapshot() -> Dict[str, Any]:
    projector = _projector_rule_only()
    rows = _pick_rows()
    out_rows: List[Dict[str, Any]] = []
    interesting_keys = [
        "real_override_intent",
        "secret_direct_print_hit",
        "tool_parameter_injection_intent",
        "agent_chain_triggered",
        "goal_hijack_priority_intent",
        "promptshield_precedence_hijack_intent",
        "promptshield_wrapper_attack_intent",
        "promptshield_secret_emit_intent",
        "fuzzy_scan_strategy",
        "pi0_rule_tier",
        "guard_block_reason",
        "guard_block_reasons",
    ]
    for row in rows:
        doc_id = str(row.get("id", ""))
        text = str(row.get("text", ""))
        proj = projector.project(
            ContentItem(
                doc_id=doc_id,
                source_id=f"src:{doc_id}",
                source_type="other",
                trust="untrusted",
                text=text,
            )
        )
        matches = dict(proj.evidence.matches)
        selected = {k: matches.get(k) for k in interesting_keys}
        out_rows.append(
            {
                "id": doc_id,
                "v": _round_walls(proj.v),
                "polarity": [int(x) for x in proj.evidence.polarity],
                "matches": selected,
            }
        )

    return {
        "schema_version": "pi0_rulepath_golden_v1",
        "semantic_enabled": "false",
        "rows": out_rows,
    }


def _assert_close(expected: Any, actual: Any, path: str = "root") -> None:
    if isinstance(expected, float) and isinstance(actual, (float, int)):
        assert abs(float(expected) - float(actual)) <= 1e-6, f"{path}: {expected} != {actual}"
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        missing = sorted(set(expected.keys()) - set(actual.keys()))
        assert not missing, f"{path}: missing keys {missing}"
        for k in expected:
            _assert_close(expected[k], actual[k], f"{path}.{k}")
        return
    if isinstance(expected, list) and isinstance(actual, list):
        assert len(expected) == len(actual), f"{path}: len mismatch"
        for i, (e, a) in enumerate(zip(expected, actual)):
            _assert_close(e, a, f"{path}[{i}]")
        return
    assert expected == actual, f"{path}: {expected!r} != {actual!r}"


def test_pi0_refactor_rulepath_golden_snapshot() -> None:
    assert GOLDEN_PATH.exists(), f"missing golden snapshot: {GOLDEN_PATH}"
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    actual = _build_snapshot()
    _assert_close(expected, actual)
