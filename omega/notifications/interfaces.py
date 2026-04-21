"""Interfaces for notification and approval components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from omega.notifications.models import ActionRequestEvent, ApprovalDecision, ApprovalRecord, RiskEvent


class Notifier(ABC):
    @abstractmethod
    async def send_alert(self, event: RiskEvent) -> str:
        """Send a non-interactive alert. Returns provider message id."""

    @abstractmethod
    async def send_action_request(self, event: ActionRequestEvent) -> str:
        """Send interactive request (approve/deny). Returns interaction id."""


class ApprovalStore(ABC):
    @abstractmethod
    def create(self, record: ApprovalRecord) -> ApprovalRecord:
        ...

    @abstractmethod
    def get(self, approval_id: str) -> Optional[ApprovalRecord]:
        ...

    @abstractmethod
    def get_latest_for_session(self, *, tenant_id: str, session_id: str) -> Optional[ApprovalRecord]:
        ...

    @abstractmethod
    def resolve(self, approval_id: str, decision: ApprovalDecision) -> Optional[ApprovalRecord]:
        ...

    @abstractmethod
    def mark_callback_id(self, approval_id: str, channel: str, callback_id: str) -> Optional[ApprovalRecord]:
        ...

    @abstractmethod
    def expire_pending(self, *, now_iso: str) -> List[ApprovalRecord]:
        ...

    @abstractmethod
    def clear_session(self, *, tenant_id: str, session_id: str) -> int:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

