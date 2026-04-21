"""Local append-only monitor event collector."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from omega.log_contract import make_log_event
from omega.monitoring.models import MonitorEvent
from omega.structured_logging import StructuredLogEmitter, build_structured_emitter_from_config, engine_version

LOGGER = logging.getLogger(__name__)


def _utc_today_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


@dataclass(frozen=True)
class _RotationConfig:
    mode: str
    max_bytes: int


class MonitorEventCollector:
    def __init__(
        self,
        *,
        enabled: bool,
        events_path: str,
        rotation_mode: str = "none",
        rotation_max_bytes: int = 100 * 1024 * 1024,
        structured_emitter: Optional[StructuredLogEmitter] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.events_path = Path(str(events_path)).resolve()
        self._rotation = _RotationConfig(
            mode=str(rotation_mode or "none").strip().lower(),
            max_bytes=max(1, int(rotation_max_bytes)),
        )
        self._events_total = 0
        self._write_failures = 0
        self._last_error: Optional[str] = None
        self._last_event_ts: Optional[str] = None
        self._daily_token = _utc_today_token()
        self.structured_emitter = structured_emitter or StructuredLogEmitter(enabled=False, validate=False, logger=None)

    def _ensure_parent(self) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def _rotate_if_needed(self) -> None:
        if not self.events_path.exists():
            return
        mode = self._rotation.mode
        if mode == "daily":
            token = _utc_today_token()
            if token == self._daily_token:
                return
            archived = self.events_path.with_name(f"{self.events_path.stem}_{self._daily_token}{self.events_path.suffix}")
            try:
                self.events_path.replace(archived)
            except Exception as exc:  # noqa: BLE001
                self._write_failures += 1
                self._last_error = f"rotation_failed:{exc}"
                LOGGER.warning("monitor collector rotation failed: %s", exc)
                return
            self._daily_token = token
            return
        if mode == "size":
            try:
                size_now = int(self.events_path.stat().st_size)
            except Exception:
                return
            if size_now < int(self._rotation.max_bytes):
                return
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archived = self.events_path.with_name(f"{self.events_path.stem}_{ts}{self.events_path.suffix}")
            try:
                self.events_path.replace(archived)
            except Exception as exc:  # noqa: BLE001
                self._write_failures += 1
                self._last_error = f"rotation_failed:{exc}"
                LOGGER.warning("monitor collector size rotation failed: %s", exc)

    def emit(self, event: MonitorEvent) -> None:
        if not self.enabled:
            return
        self._ensure_parent()
        self._rotate_if_needed()
        try:
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            # Fail-open by design.
            self._write_failures += 1
            self._last_error = str(exc)
            LOGGER.warning("monitor collector write failed: %s", exc)
            return
        self._events_total += 1
        self._last_event_ts = str(event.ts)
        self._last_error = None
        if self.structured_emitter.enabled:
            payload = make_log_event(
                event="monitor_event",
                session_id=str(event.session_id),
                mode=str(event.mode),
                engine_version=engine_version(),
                risk_score=float(event.risk_score),
                intended_action_native=str(event.intended_action),
                actual_action_native=str(event.actual_action),
                action_types=[],
                triggered_rules=list(event.triggered_rules),
                attribution_rows=list(event.attribution),
                fragments=list(event.fragments),
                fp_hint=(str(event.false_positive_hint) if event.false_positive_hint else None),
                ts=str(event.ts),
                trace_id=str(event.trace_id),
                decision_id=str(event.decision_id),
                surface=str(event.surface),
                input_type=str((event.metadata or {}).get("input_type", "context_chunk")),
                input_length=(
                    int((event.metadata or {}).get("input_length", 0))
                    if (event.metadata or {}).get("input_length") is not None
                    else None
                ),
                source_type=str((event.metadata or {}).get("source_type", "")) or None,
            )
            self.structured_emitter.emit(payload)

    def health_snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "events_path": str(self.events_path),
            "events_total": int(self._events_total),
            "write_failures": int(self._write_failures),
            "last_error": self._last_error,
            "last_event_ts": self._last_event_ts,
            "rotation_mode": str(self._rotation.mode),
            "structured_logging": self.structured_emitter.health_snapshot(),
        }


def build_monitor_collector_from_config(
    *,
    config: Mapping[str, Any],
    force_enable: bool = False,
) -> MonitorEventCollector:
    mon = (config.get("monitoring", {}) or {}) if isinstance(config.get("monitoring", {}), Mapping) else {}
    export_cfg = (mon.get("export", {}) or {}) if isinstance(mon.get("export", {}), Mapping) else {}
    rotation_mode = str(export_cfg.get("rotation", "none")).strip().lower()
    rotation_size_mb = int(export_cfg.get("rotation_size_mb", 100))
    enabled = bool(mon.get("enabled", False) or force_enable)
    events_path = str(export_cfg.get("path", "artifacts/monitor/monitor_events.jsonl")).strip() or "artifacts/monitor/monitor_events.jsonl"
    return MonitorEventCollector(
        enabled=enabled,
        events_path=events_path,
        rotation_mode=rotation_mode,
        rotation_max_bytes=max(1, rotation_size_mb) * 1024 * 1024,
        structured_emitter=build_structured_emitter_from_config(config=config, logger_name="omega.monitor"),
    )
