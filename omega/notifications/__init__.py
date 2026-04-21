"""Notification and approval subsystem for Omega runtime surfaces."""

from omega.notifications.dispatcher import NotificationDispatcher
from omega.notifications.interfaces import Notifier
from omega.notifications.models import (
    ActionRequestEvent,
    ApprovalDecision,
    ApprovalRecord,
    RiskEvent,
)
from omega.notifications.providers import SlackNotifier, TelegramNotifier
from omega.notifications.security import (
    verify_internal_hmac,
    verify_slack_signature,
    verify_telegram_secret_token,
)
from omega.notifications.startup import build_preflight_checklist, render_preflight_text
from omega.notifications.startup_flow import run_startup_notifications
from omega.notifications.store import ApprovalStore, InMemoryApprovalStore, SQLiteApprovalStore

__all__ = [
    "Notifier",
    "RiskEvent",
    "ActionRequestEvent",
    "ApprovalDecision",
    "ApprovalRecord",
    "ApprovalStore",
    "InMemoryApprovalStore",
    "SQLiteApprovalStore",
    "NotificationDispatcher",
    "SlackNotifier",
    "TelegramNotifier",
    "build_preflight_checklist",
    "render_preflight_text",
    "run_startup_notifications",
    "verify_internal_hmac",
    "verify_slack_signature",
    "verify_telegram_secret_token",
]
