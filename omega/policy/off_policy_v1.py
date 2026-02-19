"""Off policy v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from omega.interfaces.contracts_v1 import ContentItem, OffAction, OffDecision, OmegaStepResult


@dataclass
class OffPolicyV1:
    config: Dict

    def _participating_walls(self, step_result: OmegaStepResult) -> List[str]:
        walls = self.config["omega"]["walls"]
        out: List[str] = []
        for idx, wall in enumerate(walls):
            if step_result.p[idx] > 0.0 or step_result.m_next[idx] > 0.0:
                out.append(wall)
        return out

    def _severity(self, walls_participating: List[str]) -> str:
        rules = self.config["off_policy"]["severity"]["rules"]
        if any(w in walls_participating for w in rules["L3_if_walls_any"]):
            return "L3"
        if len(walls_participating) >= int(rules["L3_if_walls_count_gte"]):
            return "L3"
        if any(w in walls_participating for w in rules["L2_if_walls_any"]):
            return "L2"
        return str(rules.get("default", "L1"))

    def select_actions(self, step_result: OmegaStepResult, items: List[ContentItem]) -> OffDecision:
        if not step_result.off:
            return OffDecision(off=False, severity="L1", actions=[])

        cfg = self.config["off_policy"]
        walls = self._participating_walls(step_result)
        severity = self._severity(walls)

        actions: List[OffAction] = []
        actions.append(
            OffAction(
                type="SOFT_BLOCK",
                target=cfg["block"].get("target", "DOC"),
                doc_ids=list(step_result.top_docs),
            )
        )

        tool_wall = "tool_or_action_abuse"
        exfil_wall = "secret_exfiltration"

        if cfg["tool_freeze"].get("enabled", True) and tool_wall in walls:
            tf = cfg["tool_freeze"]
            actions.append(
                OffAction(
                    type="TOOL_FREEZE",
                    target="TOOLS",
                    tool_mode=tf.get("mode", "TOOLS_DISABLED"),
                    allowlist=tf.get("allowlist"),
                    horizon_steps=int(tf.get("horizon_steps", 20)),
                )
            )

        if cfg["escalate"].get("enabled", True):
            escalate = False
            if cfg["escalate"].get("always_on_exfil", True) and exfil_wall in walls:
                escalate = True
            if cfg["escalate"].get("on_three_plus_walls", True) and len(walls) >= 3:
                escalate = True
            if escalate:
                actions.append(
                    OffAction(
                        type="HUMAN_ESCALATE",
                        target="AGENT",
                        incident_packet={
                            "session_id": step_result.session_id,
                            "step": step_result.step,
                            "top_docs": step_result.top_docs,
                        },
                    )
                )

        if cfg["source_quarantine"].get("enabled", True):
            source_ids = sorted({item.source_id for item in items if item.doc_id in set(step_result.top_docs)})
            if source_ids:
                actions.append(
                    OffAction(
                        type="SOURCE_QUARANTINE",
                        target="SOURCE",
                        source_ids=source_ids,
                        horizon_steps=int(cfg["source_quarantine"].get("duration_hours", 24)),
                    )
                )

        return OffDecision(off=True, severity=severity, actions=actions)
