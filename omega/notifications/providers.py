"""Slack, Telegram, and webhook notification providers."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from omega.notifications.interfaces import Notifier
from omega.notifications.models import ActionRequestEvent, RiskEvent
from omega.security.network import OutboundURLPolicy, validate_outbound_url


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise HTTPError(req.full_url, code, "redirects_disabled", headers, fp)


def _host_from_url(url: str) -> str:
    return str(urlsplit(str(url)).hostname or "").strip().lower()


def _http_post_json(
    *,
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    allowed_hosts: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    parsed = urlsplit(str(url))
    scheme = str(parsed.scheme or "").lower()
    port = int(parsed.port or (443 if scheme == "https" else 80))
    host_rules = list(allowed_hosts or [])
    validate_outbound_url(
        str(url),
        policy=OutboundURLPolicy(
            allowed_schemes=("https",),
            allowed_hosts=tuple(host_rules),
            allowed_ports=(port,),
            allow_ip_literals=False,
            resolve_dns=True,
        ),
    )
    req = Request(
        str(url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **dict(headers or {})},
        method="POST",
    )
    opener = build_opener(_NoRedirect())
    try:
        with opener.open(req, timeout=8.0) as resp:  # noqa: S310
            data = resp.read(1_048_577)
            if len(data) > 1_048_576:
                raise RuntimeError("notification_response_too_large")
            text = data.decode("utf-8", errors="replace")
    except HTTPError as exc:  # pragma: no cover - network environment dependent
        body = exc.read(8192).decode("utf-8", errors="replace") if getattr(exc, "fp", None) else ""
        raise RuntimeError(f"http_error:{exc.code}:{body}") from exc
    except URLError as exc:  # pragma: no cover - network environment dependent
        raise RuntimeError(f"url_error:{exc}") from exc
    if not text.strip():
        return {}
    try:
        obj = json.loads(text)
        return dict(obj) if isinstance(obj, dict) else {"value": obj}
    except json.JSONDecodeError:
        return {"raw": text}


def _event_brief(event: RiskEvent) -> str:
    reasons = ", ".join(list(event.reasons)[:4]) or "n/a"
    actions = ", ".join(list(event.action_types)[:4]) or "n/a"
    return (
        f"[Omega] {event.control_outcome} at {event.surface} "
        f"(tenant={event.tenant_id or 'n/a'}, session={event.session_id or 'n/a'}, step={event.step}) "
        f"reasons={reasons} actions={actions} trace={event.trace_id} decision={event.decision_id}"
    )


def _startup_text(event: RiskEvent) -> str:
    raw = ((event.payload_redacted or {}).get("startup_text", "") if isinstance(event.payload_redacted, dict) else "")
    text = str(raw).strip()
    if text:
        return text
    return "[Omega Startup] onboarding message" if str(event.event_kind or "").strip().lower() == "startup_outreach" else "[Omega Startup] preflight message"


def _alert_text(event: RiskEvent) -> str:
    return _startup_text(event) if str(event.event_kind or "").strip().lower().startswith("startup_") else _event_brief(event)


def _webhook_payload(*, event: RiskEvent) -> Dict[str, Any]:
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
        "payload_redacted": dict(event.payload_redacted or {}),
    }


def _action_request_brief(event: ActionRequestEvent) -> str:
    base = _event_brief(event.risk_event)
    if str(event.approval_scope) == "tool_intent":
        return f"{base}\nExact tool approval: tool={event.tool_name} intent={event.tool_intent_id}"
    return base


class SlackNotifier(Notifier):
    def __init__(self, *, bot_token: str, channel: str, base_url: str = "https://slack.com/api") -> None:
        self.bot_token = str(bot_token).strip()
        self.channel = str(channel).strip()
        self.base_url = str(base_url).rstrip("/")
        self.allowed_hosts = (_host_from_url(self.base_url),)
        if not self.bot_token or not self.channel or not self.allowed_hosts[0]:
            raise ValueError("Slack notifier requires bot token, channel, and valid base URL")

    async def send_alert(self, event: RiskEvent) -> str:
        text = _alert_text(event)
        event_type = "omega_startup" if str(event.event_kind or "").startswith("startup_") else "omega_alert"
        payload: Dict[str, Any] = {
            "channel": self.channel,
            "text": text,
            "metadata": {"event_type": event_type, "event_payload": {"trace_id": event.trace_id, "decision_id": event.decision_id}},
        }
        if str(event.event_kind or "").startswith("startup_"):
            payload["blocks"] = [{"type": "section", "text": {"type": "mrkdwn", "text": text[:2900]}}]
        out = await asyncio.to_thread(
            _http_post_json,
            url=f"{self.base_url}/chat.postMessage",
            payload=payload,
            headers={"Authorization": f"Bearer {self.bot_token}"},
            allowed_hosts=self.allowed_hosts,
        )
        return str(out.get("ts", (out.get("message", {}) or {}).get("ts", "")))

    async def send_action_request(self, event: ActionRequestEvent) -> str:
        value_approve = json.dumps({"approval_id": event.approval_id, "decision": "approved"}, ensure_ascii=True)
        value_deny = json.dumps({"approval_id": event.approval_id, "decision": "denied"}, ensure_ascii=True)
        text = _action_request_brief(event)
        payload = {
            "channel": self.channel,
            "text": text,
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": text[:2900]}},
                {"type": "actions", "elements": [
                    {"type": "button", "text": {"type": "plain_text", "text": "Approve"}, "style": "primary", "value": value_approve, "action_id": "omega_approve"},
                    {"type": "button", "text": {"type": "plain_text", "text": "Deny"}, "style": "danger", "value": value_deny, "action_id": "omega_deny"},
                ]},
            ],
            "metadata": {"event_type": "omega_action_request", "event_payload": {"approval_id": event.approval_id, "trace_id": event.risk_event.trace_id, "tool_intent_id": event.tool_intent_id}},
        }
        out = await asyncio.to_thread(
            _http_post_json,
            url=f"{self.base_url}/chat.postMessage",
            payload=payload,
            headers={"Authorization": f"Bearer {self.bot_token}"},
            allowed_hosts=self.allowed_hosts,
        )
        return str(out.get("ts", (out.get("message", {}) or {}).get("ts", "")))


class TelegramNotifier(Notifier):
    def __init__(self, *, bot_token: str, chat_id: str, base_url: str = "https://api.telegram.org") -> None:
        self.bot_token = str(bot_token).strip()
        self.chat_id = str(chat_id).strip()
        self.base_url = str(base_url).rstrip("/")
        self.allowed_hosts = (_host_from_url(self.base_url),)
        if not self.bot_token or not self.chat_id or not self.allowed_hosts[0]:
            raise ValueError("Telegram notifier requires bot token, chat_id, and valid base URL")

    @property
    def _api_base(self) -> str:
        return f"{self.base_url}/bot{self.bot_token}"

    async def send_alert(self, event: RiskEvent) -> str:
        payload = {"chat_id": self.chat_id, "text": _alert_text(event), "disable_web_page_preview": True}
        out = await asyncio.to_thread(_http_post_json, url=f"{self._api_base}/sendMessage", payload=payload, allowed_hosts=self.allowed_hosts)
        msg = out.get("result", {}) if isinstance(out.get("result"), dict) else {}
        return str(msg.get("message_id", ""))

    async def send_action_request(self, event: ActionRequestEvent) -> str:
        payload = {
            "chat_id": self.chat_id,
            "text": _action_request_brief(event),
            "reply_markup": {"inline_keyboard": [[
                {"text": "Approve", "callback_data": f"omega:approved:{event.approval_id}"},
                {"text": "Deny", "callback_data": f"omega:denied:{event.approval_id}"},
            ]]},
            "disable_web_page_preview": True,
        }
        out = await asyncio.to_thread(_http_post_json, url=f"{self._api_base}/sendMessage", payload=payload, allowed_hosts=self.allowed_hosts)
        msg = out.get("result", {}) if isinstance(out.get("result"), dict) else {}
        return str(msg.get("message_id", ""))


class WebhookNotifier(Notifier):
    def __init__(self, *, url: str, allowed_types: Sequence[str], max_retries: int = 3, allowed_hosts: Optional[Sequence[str]] = None) -> None:
        self.url = str(url).strip()
        self.allowed_types = {str(x).strip() for x in list(allowed_types) if str(x).strip()}
        self.max_retries = max(1, int(max_retries))
        self.allowed_hosts = tuple(str(x).strip().lower() for x in (allowed_hosts or [_host_from_url(self.url)]) if str(x).strip())
        if not self.url or not self.allowed_hosts:
            raise ValueError("Webhook notifier requires URL and explicit host allowlist")

    def _should_send(self, event: RiskEvent) -> bool:
        return not self.allowed_types or str(event.event_kind or "").strip() in self.allowed_types

    async def _send_with_retries(self, payload: Dict[str, Any]) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                out = await asyncio.to_thread(_http_post_json, url=self.url, payload=payload, allowed_hosts=self.allowed_hosts)
                return str(out.get("id", out.get("event_id", "ok")))
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(0.2 * float(2 ** (attempt - 1)))
        raise RuntimeError(f"webhook_delivery_failed:{last_exc}")

    async def send_alert(self, event: RiskEvent) -> str:
        if not self._should_send(event):
            return "filtered"
        return await self._send_with_retries(_webhook_payload(event=event))

    async def send_action_request(self, event: ActionRequestEvent) -> str:
        payload = {
            "event_type": "approval_request",
            "approval_id": str(event.approval_id),
            "required_action": str(event.required_action),
            "approval_scope": str(event.approval_scope),
            "tool_name": str(event.tool_name),
            "tool_args_sha256": str(event.tool_args_sha256),
            "tool_intent_id": str(event.tool_intent_id),
            "timeout_sec": int(event.timeout_sec),
            "risk_event": _webhook_payload(event=event.risk_event),
        }
        return await self._send_with_retries(payload)
