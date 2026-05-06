"""Slack and Telegram notifier providers."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from omega.notifications.interfaces import Notifier
from omega.notifications.models import ActionRequestEvent, RiskEvent


def _http_post_json(*, url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    req = Request(
        str(url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **dict(headers or {})},
        method="POST",
    )
    try:
        with urlopen(req, timeout=8.0) as resp:  # noqa: S310
            data = resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:  # pragma: no cover - network environment dependent
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"http_error:{exc.code}:{body}") from exc
    except URLError as exc:  # pragma: no cover - network environment dependent
        raise RuntimeError(f"url_error:{exc}") from exc
    if not data.strip():
        return {}
    try:
        return dict(json.loads(data))
    except json.JSONDecodeError:
        return {"raw": data}


def _event_brief(event: RiskEvent) -> str:
    reasons = ", ".join(list(event.reasons)[:4]) or "n/a"
    actions = ", ".join(list(event.action_types)[:4]) or "n/a"
    return (
        f"[Omega] {event.control_outcome} at {event.surface} "
        f"(session={event.session_id or 'n/a'}, step={event.step}) "
        f"reasons={reasons} actions={actions} trace={event.trace_id} decision={event.decision_id}"
    )


def _startup_text(event: RiskEvent) -> str:
    raw = ((event.payload_redacted or {}).get("startup_text", "") if isinstance(event.payload_redacted, dict) else "")
    text = str(raw).strip()
    if text:
        return text
    kind = str(event.event_kind or "").strip().lower()
    if kind == "startup_outreach":
        return "[Omega Startup] onboarding message"
    return "[Omega Startup] preflight message"


def _alert_text(event: RiskEvent) -> str:
    kind = str(event.event_kind or "").strip().lower()
    if kind.startswith("startup_"):
        return _startup_text(event)
    return _event_brief(event)


def _webhook_payload(*, event: RiskEvent) -> Dict[str, Any]:
    red = dict(event.payload_redacted or {})
    return {
        "event_id": str(event.event_id),
        "event_kind": str(event.event_kind or "risk_event"),
        "timestamp": str(event.timestamp),
        "surface": str(event.surface),
        "control_outcome": str(event.control_outcome),
        "triggers": list(event.triggers),
        "reasons": list(event.reasons),
        "action_types": list(event.action_types),
        "trace_id": str(event.trace_id),
        "decision_id": str(event.decision_id),
        "tenant_id": str(event.tenant_id or ""),
        "session_id": str(event.session_id or ""),
        "actor_id": str(event.actor_id or ""),
        "step": int(event.step),
        "severity": str(event.severity or ""),
        "risk_score": event.risk_score,
        "message": _alert_text(event),
        "payload_redacted": red,
    }


class SlackNotifier(Notifier):
    def __init__(self, *, bot_token: str, channel: str, base_url: str = "https://slack.com/api") -> None:
        self.bot_token = str(bot_token).strip()
        self.channel = str(channel).strip()
        self.base_url = str(base_url).rstrip("/")
        if not self.bot_token or not self.channel:
            raise ValueError("Slack notifier requires bot token and channel")

    async def send_alert(self, event: RiskEvent) -> str:
        text = _alert_text(event)
        event_type = "omega_startup" if str(event.event_kind or "").startswith("startup_") else "omega_alert"
        payload = {
            "channel": self.channel,
            "text": text,
            "metadata": {"event_type": event_type, "event_payload": {"trace_id": event.trace_id, "decision_id": event.decision_id}},
        }
        if str(event.event_kind or "").startswith("startup_"):
            payload["blocks"] = [
                {"type": "section", "text": {"type": "mrkdwn", "text": text[:2900]}},
            ]
        out = await asyncio.to_thread(
            _http_post_json,
            url=f"{self.base_url}/chat.postMessage",
            payload=payload,
            headers={"Authorization": f"Bearer {self.bot_token}"},
        )
        return str(out.get("ts", out.get("message", {}).get("ts", "")))

    async def send_action_request(self, event: ActionRequestEvent) -> str:
        value_approve = json.dumps({"approval_id": event.approval_id, "decision": "approved"}, ensure_ascii=True)
        value_deny = json.dumps({"approval_id": event.approval_id, "decision": "denied"}, ensure_ascii=True)
        payload = {
            "channel": self.channel,
            "text": _event_brief(event.risk_event),
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": _event_brief(event.risk_event)}},
                {
                    "type": "actions",
                    "elements": [
                        {"type": "button", "text": {"type": "plain_text", "text": "Approve"}, "style": "primary", "value": value_approve, "action_id": "omega_approve"},
                        {"type": "button", "text": {"type": "plain_text", "text": "Deny"}, "style": "danger", "value": value_deny, "action_id": "omega_deny"},
                    ],
                },
            ],
            "metadata": {
                "event_type": "omega_action_request",
                "event_payload": {"approval_id": event.approval_id, "trace_id": event.risk_event.trace_id},
            },
        }
        out = await asyncio.to_thread(
            _http_post_json,
            url=f"{self.base_url}/chat.postMessage",
            payload=payload,
            headers={"Authorization": f"Bearer {self.bot_token}"},
        )
        return str(out.get("ts", out.get("message", {}).get("ts", "")))


class TelegramNotifier(Notifier):
    def __init__(self, *, bot_token: str, chat_id: str, base_url: str = "https://api.telegram.org") -> None:
        self.bot_token = str(bot_token).strip()
        self.chat_id = str(chat_id).strip()
        self.base_url = str(base_url).rstrip("/")
        if not self.bot_token or not self.chat_id:
            raise ValueError("Telegram notifier requires bot token and chat_id")

    @property
    def _api_base(self) -> str:
        return f"{self.base_url}/bot{self.bot_token}"

    async def send_alert(self, event: RiskEvent) -> str:
        payload = {"chat_id": self.chat_id, "text": _alert_text(event), "disable_web_page_preview": True}
        out = await asyncio.to_thread(_http_post_json, url=f"{self._api_base}/sendMessage", payload=payload)
        msg = out.get("result", {}) if isinstance(out.get("result"), dict) else {}
        return str(msg.get("message_id", ""))


class WebhookNotifier(Notifier):
    def __init__(self, *, url: str, allowed_types: Sequence[str], max_retries: int = 3) -> None:
        self.url = str(url).strip()
        self.allowed_types = {str(x).strip() for x in list(allowed_types) if str(x).strip()}
        self.max_retries = max(1, int(max_retries))
        if not self.url:
            raise ValueError("Webhook notifier requires url")

    def _should_send(self, event: RiskEvent) -> bool:
        if not self.allowed_types:
            return True
        kind = str(event.event_kind or "").strip()
        return kind in self.allowed_types

    async def send_alert(self, event: RiskEvent) -> str:
        if not self._should_send(event):
            return "filtered"
        payload = _webhook_payload(event=event)
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt < self.max_retries:
            attempt += 1
            try:
                out = await asyncio.to_thread(_http_post_json, url=self.url, payload=payload)
                return str(out.get("id", out.get("event_id", "ok")))
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(0.2 * float(2 ** (attempt - 1)))
        raise RuntimeError(f"webhook_delivery_failed:{last_exc}")

    async def send_action_request(self, event: ActionRequestEvent) -> str:
        payload = {
            "event_type": "approval_request",
            "approval_id": str(event.approval_id),
            "required_action": str(event.required_action),
            "timeout_sec": int(event.timeout_sec),
            "risk_event": _webhook_payload(event=event.risk_event),
        }
        out = await asyncio.to_thread(_http_post_json, url=self.url, payload=payload)
        return str(out.get("id", out.get("event_id", event.approval_id)))

    async def send_action_request(self, event: ActionRequestEvent) -> str:
        payload = {
            "chat_id": self.chat_id,
            "text": _event_brief(event.risk_event),
            "reply_markup": {
                "inline_keyboard": [
                    [
                        {"text": "Approve", "callback_data": f"omega:approved:{event.approval_id}"},
                        {"text": "Deny", "callback_data": f"omega:denied:{event.approval_id}"},
                    ]
                ]
            },
            "disable_web_page_preview": True,
        }
        out = await asyncio.to_thread(_http_post_json, url=f"{self._api_base}/sendMessage", payload=payload)
        msg = out.get("result", {}) if isinstance(out.get("result"), dict) else {}
        return str(msg.get("message_id", ""))
