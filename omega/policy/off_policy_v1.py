"""Off policy v1."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List

from omega.interfaces.contracts_v1 import ContentItem, OffAction, OffDecision, OmegaStepResult
from omega.policy.control_outcome import control_outcome_from_actions

LOGGER = logging.getLogger(__name__)


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

    def _warn_action(self, step_result: OmegaStepResult) -> OffAction | None:
        cfg = ((self.config.get("off_policy", {}) or {}).get("control_outcome", {}) or {}).get("warn", {})
        if not bool(cfg.get("enabled", False)):
            return None
        max_p = float(max(step_result.p)) if len(step_result.p) else 0.0
        sum_m = float(sum(step_result.m_next)) if len(step_result.m_next) else 0.0
        max_p_thr = float(cfg.get("max_p_gte", 1.1))
        sum_m_thr = float(cfg.get("sum_m_next_gte", 1.1))
        if max_p >= max_p_thr or sum_m >= sum_m_thr:
            return OffAction(type="WARN", target=str(cfg.get("target", "SESSION")))
        return None

    def _require_approval_action(
        self,
        *,
        off: bool,
        warn_active: bool,
    ) -> OffAction | None:
        cfg = ((self.config.get("off_policy", {}) or {}).get("control_outcome", {}) or {}).get("require_approval", {})
        if not bool(cfg.get("enabled", False)):
            return None
        on_off = bool(cfg.get("on_off", True))
        on_warn = bool(cfg.get("on_warn", True))
        if not ((off and on_off) or (warn_active and on_warn)):
            return None
        allowlist = [str(x) for x in list(cfg.get("tools", [])) if str(x).strip()]
        horizon_steps = int(cfg.get("horizon_steps", 0))
        return OffAction(
            type="REQUIRE_APPROVAL",
            target="TOOLS",
            allowlist=allowlist or None,
            horizon_steps=max(0, horizon_steps),
        )

    def select_actions(self, step_result: OmegaStepResult, items: List[ContentItem]) -> OffDecision:
        cfg = self.config["off_policy"]
        walls = self._participating_walls(step_result)
        severity = self._severity(walls) if step_result.off else "L1"

        actions: List[OffAction] = []
        warn_action = self._warn_action(step_result)
        if warn_action is not None:
            actions.append(warn_action)

        if step_result.off:
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
                    sq_cfg = cfg["source_quarantine"]
                    if "duration_steps" in sq_cfg:
                        horizon_steps = int(sq_cfg.get("duration_steps", 24))
                    else:
                        horizon_steps = int(sq_cfg.get("duration_hours", 24))
                        if "duration_hours" in sq_cfg:
                            LOGGER.warning(
                                "off_policy.source_quarantine.duration_hours is interpreted as step horizon; "
                                "prefer duration_steps (deprecated field)."
                            )
                    actions.append(
                        OffAction(
                            type="SOURCE_QUARANTINE",
                            target="SOURCE",
                            source_ids=source_ids,
                            horizon_steps=horizon_steps,
                        )
                    )

        require_approval = self._require_approval_action(
            off=bool(step_result.off),
            warn_active=warn_action is not None,
        )
        if require_approval is not None:
            actions.append(require_approval)

        return OffDecision(
            off=bool(step_result.off),
            severity=severity,
            actions=actions,
            control_outcome=control_outcome_from_actions(actions),
        )
