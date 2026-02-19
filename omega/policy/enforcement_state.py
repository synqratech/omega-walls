"""Persistent enforcement state for freeze/quarantine actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from omega.interfaces.contracts_v1 import OffAction


@dataclass
class EnforcementStateManager:
    strikes_to_quarantine: int
    default_source_horizon_steps: int
    source_strikes: Dict[str, int] = field(default_factory=dict)
    quarantined_sources_until: Dict[str, int] = field(default_factory=dict)
    tool_freeze_until_step: int = 0
    tool_mode: str = "TOOLS_DISABLED"
    tool_allowlist: List[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: Dict) -> "EnforcementStateManager":
        sq_cfg = config.get("off_policy", {}).get("source_quarantine", {})
        return cls(
            strikes_to_quarantine=int(sq_cfg.get("strikes_to_quarantine", 2)),
            default_source_horizon_steps=int(sq_cfg.get("duration_hours", 24)),
        )

    def reset(self) -> None:
        self.source_strikes.clear()
        self.quarantined_sources_until.clear()
        self.tool_freeze_until_step = 0
        self.tool_mode = "TOOLS_DISABLED"
        self.tool_allowlist = []

    def record_policy_actions(self, actions: List[OffAction], step: int) -> None:
        for action in actions:
            if action.type == "TOOL_FREEZE":
                self._record_tool_freeze(action, step)
            elif action.type == "SOURCE_QUARANTINE":
                self._record_source_quarantine_signal(action, step)

    def active_actions(self, step: int) -> List[OffAction]:
        self._prune_expired(step)
        actions: List[OffAction] = []

        if self.tool_freeze_until_step >= step:
            actions.append(
                OffAction(
                    type="TOOL_FREEZE",
                    target="TOOLS",
                    tool_mode=self.tool_mode,
                    allowlist=list(self.tool_allowlist) if self.tool_mode == "TOOLS_ALLOWLIST" else None,
                    horizon_steps=max(0, self.tool_freeze_until_step - step),
                )
            )

        active_sources = sorted(
            source_id
            for source_id, until_step in self.quarantined_sources_until.items()
            if until_step >= step
        )
        if active_sources:
            max_until = max(self.quarantined_sources_until[sid] for sid in active_sources)
            actions.append(
                OffAction(
                    type="SOURCE_QUARANTINE",
                    target="SOURCE",
                    source_ids=active_sources,
                    horizon_steps=max(0, max_until - step),
                )
            )

        return actions

    def is_source_quarantined(self, source_id: str, step: int) -> bool:
        self._prune_expired(step)
        return self.quarantined_sources_until.get(source_id, -1) >= step

    def snapshot(self, step: int) -> Dict[str, Any]:
        self._prune_expired(step)
        freeze_active = self.tool_freeze_until_step >= step
        quarantined = []
        for source_id, until_step in sorted(self.quarantined_sources_until.items()):
            if until_step < step:
                continue
            quarantined.append(
                {
                    "source_id": source_id,
                    "until_step": int(until_step),
                    "remaining_horizon": int(max(0, until_step - step)),
                }
            )
        return {
            "freeze": {
                "active": bool(freeze_active),
                "mode": self.tool_mode,
                "allowlist": list(self.tool_allowlist) if self.tool_mode == "TOOLS_ALLOWLIST" else [],
                "freeze_until_step": int(self.tool_freeze_until_step) if freeze_active else None,
                "remaining_horizon": int(max(0, self.tool_freeze_until_step - step)),
            },
            "quarantine": {
                "active": bool(len(quarantined) > 0),
                "quarantined_sources": quarantined,
                "total_quarantined": len(quarantined),
            },
        }

    def _record_tool_freeze(self, action: OffAction, step: int) -> None:
        horizon = max(0, int(action.horizon_steps or 0))
        until_step = step + horizon
        if until_step < self.tool_freeze_until_step:
            return
        self.tool_freeze_until_step = until_step
        self.tool_mode = action.tool_mode or "TOOLS_DISABLED"
        self.tool_allowlist = list(action.allowlist or [])

    def _record_source_quarantine_signal(self, action: OffAction, step: int) -> None:
        source_ids = action.source_ids or []
        if not source_ids:
            return

        horizon = max(0, int(action.horizon_steps or self.default_source_horizon_steps))
        until_step = step + horizon

        for source_id in source_ids:
            self.source_strikes[source_id] = self.source_strikes.get(source_id, 0) + 1
            if self.source_strikes[source_id] < self.strikes_to_quarantine:
                continue
            prev_until = self.quarantined_sources_until.get(source_id, -1)
            if until_step > prev_until:
                self.quarantined_sources_until[source_id] = until_step

    def _prune_expired(self, step: int) -> None:
        expired = [source_id for source_id, until in self.quarantined_sources_until.items() if until < step]
        for source_id in expired:
            del self.quarantined_sources_until[source_id]
