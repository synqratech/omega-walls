"""Notification dispatcher with async background delivery and approval lifecycle."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging
import os
import queue
import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from omega.notifications.interfaces import ApprovalStore, Notifier
from omega.notifications.models import (
    ActionRequestEvent,
    ApprovalDecision,
    ApprovalRecord,
    RiskEvent,
    new_approval_id,
    utc_now_iso,
)
from omega.notifications.providers import SlackNotifier, TelegramNotifier
from omega.notifications.store import InMemoryApprovalStore, SQLiteApprovalStore

LOGGER = logging.getLogger(__name__)


def _normalize_trigger_set(values: Iterable[Any]) -> set[str]:
    out = {str(x).strip().upper() for x in values if str(x).strip()}
    return out


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def infer_major_triggers(*, control_outcome: str, action_types: Iterable[str], fallback_active: bool = False) -> List[str]:
    action_set = _normalize_trigger_set(action_types)
    outcome = str(control_outcome or "").strip().upper() or "ALLOW"
    triggers: List[str] = []
    if outcome in {"SOFT_BLOCK", "SOURCE_QUARANTINE", "TOOL_FREEZE", "HUMAN_ESCALATE"}:
        triggers.append("BLOCK")
    if outcome in {"WARN"}:
        triggers.append("WARN")
    if outcome not in {"ALLOW", "WARN"}:
        triggers.append(outcome)
    if "REQUIRE_APPROVAL" in action_set:
        triggers.append("REQUIRE_APPROVAL")
    if fallback_active:
        triggers.append("FALLBACK")
    return sorted({str(x) for x in triggers if str(x).strip()})


class NotificationDispatcher:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        store: ApprovalStore,
        providers: Mapping[str, Notifier],
    ) -> None:
        self.config = dict(config or {})
        self.store = store
        self.providers = {str(k): v for k, v in dict(providers).items()}
        self._queue: "queue.Queue[tuple[str, str, Any]]" = queue.Queue(maxsize=2000)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        if self.providers:
            self._worker = threading.Thread(target=self._worker_loop, name="omega-notifier", daemon=True)
            self._worker.start()
        self._metrics_lock = threading.Lock()
        self._metrics = defaultdict(int)
        self._throttle_lock = threading.Lock()
        self._last_sent: Dict[tuple[str, str, str], datetime] = {}
        self._nonce_cache: Dict[str, int] = {}
        self._startup_lock = threading.Lock()
        self._startup_sent: set[str] = set()

    def close(self) -> None:
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self.store.close()

    @property
    def nonce_cache(self) -> Dict[str, int]:
        return self._nonce_cache

    def metrics_snapshot(self) -> Dict[str, int]:
        with self._metrics_lock:
            return {str(k): int(v) for k, v in self._metrics.items()}

    def get_approval(self, approval_id: str) -> Optional[ApprovalRecord]:
        self.expire_timeouts()
        return self.store.get(approval_id)

    def resolve_approval(self, *, approval_id: str, decision: ApprovalDecision) -> Optional[ApprovalRecord]:
        out = self.store.resolve(approval_id, decision)
        if out is not None:
            with self._metrics_lock:
                self._metrics["approvals_resolved"] += 1
        return out

    def latest_approval_for_session(self, *, tenant_id: str, session_id: str) -> Optional[ApprovalRecord]:
        self.expire_timeouts()
        return self.store.get_latest_for_session(tenant_id=tenant_id, session_id=session_id)

    def create_action_request(self, *, risk_event: RiskEvent, required_action: str, timeout_sec: int) -> ApprovalRecord:
        latest = self.store.get_latest_for_session(
            tenant_id=str(risk_event.tenant_id or ""),
            session_id=str(risk_event.session_id or ""),
        )
        if latest is not None and str(latest.status) == "pending" and str(latest.required_action) == str(required_action):
            return latest
        now = _parse_iso(utc_now_iso())
        expires_at = now + timedelta(seconds=max(10, int(timeout_sec)))
        approval = ApprovalRecord(
            approval_id=new_approval_id(),
            status="pending",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            expires_at=expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            required_action=str(required_action),
            tenant_id=str(risk_event.tenant_id or ""),
            session_id=str(risk_event.session_id or ""),
            actor_id=str(risk_event.actor_id or ""),
            trace_id=str(risk_event.trace_id or ""),
            decision_id=str(risk_event.decision_id or ""),
            control_outcome=str(risk_event.control_outcome or "ALLOW"),
            channels=[],
            callback_ids={},
            resolution=None,
        )
        self.store.create(approval)
        with self._metrics_lock:
            self._metrics["approvals_pending"] += 1
            self._metrics["action_requests_created"] += 1

        action_event = ActionRequestEvent(
            approval_id=approval.approval_id,
            risk_event=risk_event,
            required_action=str(required_action),
            timeout_sec=max(10, int(timeout_sec)),
        )
        self._enqueue_event(kind="action_request", event=action_event)
        return approval

    def emit_risk_event(self, event: RiskEvent) -> None:
        self.expire_timeouts()
        self._enqueue_event(kind="alert", event=event)

    def emit_startup_event(self, event: RiskEvent, *, startup_kind: str, once_per_process: bool = True) -> bool:
        self.expire_timeouts()
        key = str(startup_kind or "").strip().lower() or "startup"
        if once_per_process:
            with self._startup_lock:
                if key in self._startup_sent:
                    with self._metrics_lock:
                        self._metrics["startup_messages_skipped_dedup"] += 1
                    return False
                self._startup_sent.add(key)
        provider_cfgs = self._provider_configs()
        if not provider_cfgs:
            return False
        sent_any = False
        for provider_name in provider_cfgs:
            try:
                self._queue.put_nowait(("startup", str(provider_name), event))
                sent_any = True
            except queue.Full:
                with self._metrics_lock:
                    self._metrics["notifications_dropped_queue_full"] += 1
                    self._metrics["startup_messages_failed"] += 1
                LOGGER.warning("notification queue is full; dropping startup event provider=%s", provider_name)
        return sent_any

    def mark_callback_id(self, *, approval_id: str, channel: str, callback_id: str) -> Optional[ApprovalRecord]:
        return self.store.mark_callback_id(approval_id, channel, callback_id)

    def expire_timeouts(self) -> List[ApprovalRecord]:
        now_iso = utc_now_iso()
        expired = self.store.expire_pending(now_iso=now_iso)
        if expired:
            with self._metrics_lock:
                self._metrics["approvals_timeout_auto_deny"] += int(len(expired))
        return expired

    def _enqueue_event(self, *, kind: str, event: Any) -> None:
        provider_cfgs = self._provider_configs()
        if not provider_cfgs:
            return
        for provider_name, cfg in provider_cfgs.items():
            if not self._provider_wants_event(provider_cfg=cfg, event=event):
                continue
            if self._throttled(provider_name=provider_name, provider_cfg=cfg, event=event):
                with self._metrics_lock:
                    self._metrics["notifications_throttled"] += 1
                continue
            try:
                self._queue.put_nowait((kind, provider_name, event))
            except queue.Full:
                with self._metrics_lock:
                    self._metrics["notifications_dropped_queue_full"] += 1
                LOGGER.warning("notification queue is full; dropping event kind=%s provider=%s", kind, provider_name)

    def _provider_configs(self) -> Dict[str, Dict[str, Any]]:
        cfg = dict((self.config.get("notifications", {}) or {}))
        out: Dict[str, Dict[str, Any]] = {}
        for name in ("slack", "telegram"):
            provider_cfg = cfg.get(name, {}) if isinstance(cfg.get(name, {}), Mapping) else {}
            if not bool(provider_cfg.get("enabled", False)):
                continue
            if name not in self.providers:
                continue
            out[name] = dict(provider_cfg)
        return out

    @staticmethod
    def _provider_wants_event(*, provider_cfg: Mapping[str, Any], event: Any) -> bool:
        triggers_cfg = _normalize_trigger_set(provider_cfg.get("triggers", []))
        event_triggers = _normalize_trigger_set(getattr(event, "triggers", []))
        if triggers_cfg and not (event_triggers & triggers_cfg):
            return False
        min_risk = provider_cfg.get("min_risk_score")
        risk_score = getattr(event, "risk_score", None)
        if min_risk is not None and risk_score is not None:
            try:
                if float(risk_score) < float(min_risk):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def _throttled(self, *, provider_name: str, provider_cfg: Mapping[str, Any], event: Any) -> bool:
        buckets = provider_cfg.get("throttle_windows_sec", {}) if isinstance(provider_cfg.get("throttle_windows_sec", {}), Mapping) else {}
        outcome = str(getattr(event, "control_outcome", "")).upper()
        bucket = "WARN" if outcome == "WARN" else "BLOCK"
        window_sec = int(buckets.get(bucket, 0))
        if window_sec <= 0:
            return False
        dedup_key = str(getattr(event, "session_id", "")) or str(getattr(event, "trace_id", ""))
        trigger_key = ",".join(sorted(_normalize_trigger_set(getattr(event, "triggers", []))))
        now = datetime.now(timezone.utc)
        key = (str(provider_name), str(trigger_key), str(dedup_key))
        with self._throttle_lock:
            prev = self._last_sent.get(key)
            if prev is not None and (now - prev).total_seconds() < float(window_sec):
                return True
            self._last_sent[key] = now
        return False

    def _worker_loop(self) -> None:
        if not self.providers:
            return
        while not self._stop_event.is_set():
            try:
                kind, provider_name, event = self._queue.get(timeout=0.5)
            except queue.Empty:
                self.expire_timeouts()
                continue
            notifier = self.providers.get(str(provider_name))
            if notifier is None:
                with self._metrics_lock:
                    self._metrics["notifications_failed"] += 1
                continue
            try:
                if kind == "alert":
                    callback_id = asyncio.run(notifier.send_alert(event))
                elif kind == "startup":
                    callback_id = asyncio.run(notifier.send_alert(event))
                else:
                    callback_id = asyncio.run(notifier.send_action_request(event))
                with self._metrics_lock:
                    self._metrics["notifications_sent"] += 1
                    if kind == "action_request":
                        self._metrics["action_requests_sent"] += 1
                    if kind == "startup":
                        self._metrics["startup_messages_sent"] += 1
                if kind == "action_request":
                    self.mark_callback_id(
                        approval_id=str(event.approval_id),
                        channel=str(provider_name),
                        callback_id=str(callback_id or ""),
                    )
            except Exception as exc:  # noqa: BLE001
                with self._metrics_lock:
                    self._metrics["notifications_failed"] += 1
                    if kind == "startup":
                        self._metrics["startup_messages_failed"] += 1
                LOGGER.warning("notifier delivery failed provider=%s kind=%s err=%s", provider_name, kind, exc)
            finally:
                self._queue.task_done()


def build_dispatcher_from_config(*, config: Mapping[str, Any], store_sqlite_path: Optional[str] = None) -> NotificationDispatcher:
    cfg = dict((config.get("notifications", {}) or {}))
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return NotificationDispatcher(config={"notifications": cfg}, store=InMemoryApprovalStore(), providers={})

    approvals_cfg = cfg.get("approvals", {}) if isinstance(cfg.get("approvals", {}), Mapping) else {}
    backend = str(approvals_cfg.get("backend", "memory")).strip().lower()
    sqlite_path = str(approvals_cfg.get("sqlite_path") or store_sqlite_path or "artifacts/state/notification_approvals.db")
    if backend == "sqlite":
        store: ApprovalStore = SQLiteApprovalStore(sqlite_path=sqlite_path)
    else:
        store = InMemoryApprovalStore()

    providers: Dict[str, Notifier] = {}
    slack_cfg = cfg.get("slack", {}) if isinstance(cfg.get("slack", {}), Mapping) else {}
    if bool(slack_cfg.get("enabled", False)):
        token_env = str(slack_cfg.get("bot_token_env", "SLACK_BOT_TOKEN")).strip()
        channel_env = str(slack_cfg.get("channel_env", "SLACK_ALERT_CHANNEL")).strip()
        token = str(os.environ.get(token_env, "")).strip()
        channel = str(os.environ.get(channel_env, str(slack_cfg.get("channel", "")))).strip()
        if token and channel:
            providers["slack"] = SlackNotifier(bot_token=token, channel=channel, base_url=str(slack_cfg.get("base_url", "https://slack.com/api")))
        else:
            LOGGER.warning("slack notifier disabled due to missing token/channel env")

    tg_cfg = cfg.get("telegram", {}) if isinstance(cfg.get("telegram", {}), Mapping) else {}
    if bool(tg_cfg.get("enabled", False)):
        token_env = str(tg_cfg.get("bot_token_env", "TG_BOT_TOKEN")).strip()
        chat_env = str(tg_cfg.get("chat_id_env", "TG_ADMIN_CHAT_ID")).strip()
        token = str(os.environ.get(token_env, "")).strip()
        chat_id = str(os.environ.get(chat_env, str(tg_cfg.get("chat_id", "")))).strip()
        if token and chat_id:
            providers["telegram"] = TelegramNotifier(bot_token=token, chat_id=chat_id, base_url=str(tg_cfg.get("base_url", "https://api.telegram.org")))
        else:
            LOGGER.warning("telegram notifier disabled due to missing bot token/chat id env")

    merged_cfg = {"notifications": cfg}
    return NotificationDispatcher(config=merged_cfg, store=store, providers=providers)
