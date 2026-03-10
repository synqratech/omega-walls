from __future__ import annotations

from copy import deepcopy

from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import WALLS_V1
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2
from tests.helpers import mk_item


def _idx() -> dict[str, int]:
    return {w: i for i, w in enumerate(WALLS_V1)}


def test_per_wall_guard_suppression_does_not_globally_block():
    cfg = deepcopy(load_resolved_config(profile="dev").resolved)
    cfg["pi0"]["semantic"]["enabled"] = "true"
    cfg["pi0"]["semantic"]["guard_apply_by_wall"] = {
        "override_instructions": ["tutorial"],
        "secret_exfiltration": ["protect"],
        "tool_or_action_abuse": ["tutorial"],
        "policy_evasion": ["negation"],
    }
    projector = Pi0IntentAwareV2(cfg)
    projector._semantic_ready = True
    projector._semantic_signals = lambda _t: (  # type: ignore[method-assign]
        {
            "override_instructions": 0.95,
            "secret_exfiltration": 0.95,
            "tool_or_action_abuse": 0.05,
            "policy_evasion": 0.05,
        },
        {"negation": 0.10, "protect": 0.95, "tutorial": 0.10},
    )

    out = projector.project(mk_item("guard-mode", "act as interviewer for this role"))
    i = _idx()
    assert out.v[i["override_instructions"]] > 0.0
    assert out.v[i["secret_exfiltration"]] == 0.0
    assert out.evidence.matches["semantic_guard_hit"] is True
    hits = out.evidence.matches["semantic_guard_hits_by_wall"]
    assert hits["override_instructions"] == []
    assert "protect" in hits["secret_exfiltration"]
