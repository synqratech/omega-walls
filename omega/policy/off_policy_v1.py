"""Off policy v1."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Tuple

from omega.interfaces.contracts_v1 import ContentItem, OffAction, OffDecision, OmegaStepResult
from omega.policy.control_outcome import control_outcome_from_actions

LOGGER = logging.getLogger(__name__)


@dataclass
class OffPolicyV1:
    config: Dict
    _tool_freeze_state: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def _participating_walls(self, step_result: OmegaStepResult) -> List[str]:
        walls = self.config["omega"]["walls"]
        cfg = ((self.config.get("off_policy", {}) or {}).get("wall_participation", {}) or {})
        current_p_gte = max(0.0, float(cfg.get("current_p_gte", 0.01)))
        current_v_gte = max(0.0, float(cfg.get("current_v_gte", 0.10)))
        scar_m_gte = max(0.0, float(cfg.get("scar_m_gte", 0.05)))
        multi_theta = float(((self.config.get("omega", {}) or {}).get("off_thresholds", {}) or {}).get("theta", 0.4))
        scar_fraction = max(0.0, float(cfg.get("scar_fraction_of_multi_threshold", 0.25)))
        effective_scar_gte = max(scar_m_gte, scar_fraction * max(0.0, multi_theta))

        out: List[str] = []
        for idx, wall in enumerate(walls):
            current_active = bool(
                float(step_result.p[idx]) >= current_p_gte
                or float(step_result.v_total[idx]) >= current_v_gte
            )
            scar_active = bool(float(step_result.m_next[idx]) >= effective_scar_gte)
            if current_active or scar_active:
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

    def _autonomy_cfg(self) -> Dict:
        cfg = (self.config.get("off_policy", {}) or {}).get("autonomy_soft", {}) or {}
        return cfg if isinstance(cfg, dict) else {}

    def _next_freeze_stage(self, *, session_id: str, step: int, severity: str, window: int) -> Tuple[int, str]:
        state = self._tool_freeze_state.get(session_id, {"count": 0, "last_step": -10**9})
        count = int(state.get("count", 0))
        last_step = int(state.get("last_step", -10**9))

        if step - last_step <= max(1, int(window)):
            count += 1
        else:
            count = 1

        stage = max(1, min(3, count))
        reason = "first_trigger" if stage == 1 else ("repeat_within_window" if stage == 2 else "third_trigger_within_window")
        if str(severity).upper() == "L3":
            stage = 3
            reason = "sev_high"

        self._tool_freeze_state[session_id] = {"count": count, "last_step": step}
        return stage, reason

    def _tool_freeze_action(self, *, step_result: OmegaStepResult, walls: List[str], severity: str) -> OffAction | None:
        cfg = self.config.get("off_policy", {})
        tf = cfg.get("tool_freeze", {}) if isinstance(cfg.get("tool_freeze", {}), dict) else {}
        if not bool(tf.get("enabled", True)):
            return None
        if "tool_or_action_abuse" not in walls:
            return None

        autonomy = self._autonomy_cfg()
        pf = autonomy.get("progressive_freeze", {}) if isinstance(autonomy.get("progressive_freeze", {}), dict) else {}
        if not bool(pf.get("enabled", False)):
            return OffAction(
                type="TOOL_FREEZE",
                target="TOOLS",
                tool_mode=tf.get("mode", "TOOLS_DISABLED"),
                allowlist=tf.get("allowlist"),
                horizon_steps=int(tf.get("horizon_steps", 20)),
                risk_level="high",
            )

        read_safe_allowlist = [
            str(x)
            for x in list(
                pf.get(
                    "read_safe_allowlist",
                    ["retrieval_readonly", "summarize", "echo"],
                )
            )
            if str(x).strip()
        ]
        stage_window = int(pf.get("repeat_window_steps", 20))
        stage, stage_reason = self._next_freeze_stage(
            session_id=str(step_result.session_id),
            step=int(step_result.step),
            severity=str(severity),
            window=stage_window,
        )
        stage_cfg = pf.get(f"stage{stage}", {}) if isinstance(pf.get(f"stage{stage}", {}), dict) else {}

        if stage == 1:
            mode = str(stage_cfg.get("mode", "TOOLS_ALLOWLIST"))
            allowlist = list(stage_cfg.get("allowlist", read_safe_allowlist))
            horizon = int(stage_cfg.get("horizon_steps", 6))
        elif stage == 2:
            mode = str(stage_cfg.get("mode", "TOOLS_ALLOWLIST"))
            allowlist = list(stage_cfg.get("allowlist", read_safe_allowlist))
            horizon = int(stage_cfg.get("horizon_steps", 12))
        else:
            mode = str(stage_cfg.get("mode", "TOOLS_DISABLED"))
            allowlist = list(stage_cfg.get("allowlist", []))
            horizon = int(stage_cfg.get("horizon_steps", 20))

        return OffAction(
            type="TOOL_FREEZE",
            target="TOOLS",
            tool_mode=mode,
            allowlist=allowlist if mode == "TOOLS_ALLOWLIST" else None,
            horizon_steps=max(0, horizon),
            freeze_stage=int(stage),
            stage_reason=str(stage_reason),
            escalation_required=bool(stage == 3),
            risk_level="high",
        )

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
            if step_result.top_docs:
                actions.append(
                    OffAction(
                        type="SOFT_BLOCK",
                        target=cfg["block"].get("target", "DOC"),
                        doc_ids=list(step_result.top_docs),
                    )
                )
            else:
                actions.append(
                    OffAction(
                        type="SOFT_BLOCK",
                        target="SESSION",
                        incident_packet={"attribution_mode": str(step_result.attribution_mode)},
                    )
                )

            exfil_wall = "secret_exfiltration"

            tool_freeze_action = self._tool_freeze_action(step_result=step_result, walls=walls, severity=severity)
            if tool_freeze_action is not None:
                actions.append(tool_freeze_action)

            if cfg["escalate"].get("enabled", True):
                escalate = False
                if cfg["escalate"].get("always_on_exfil", True) and exfil_wall in walls:
                    escalate = True
                if cfg["escalate"].get("on_three_plus_walls", True) and len(walls) >= 3:
                    escalate = True
                if any(bool(a.escalation_required) for a in actions if str(a.type) == "TOOL_FREEZE"):
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
                                "attribution_mode": str(step_result.attribution_mode),
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
