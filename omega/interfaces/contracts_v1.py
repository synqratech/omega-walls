"""Public contracts for Omega Walls v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import numpy as np


K_V1 = 4
WALLS_V1 = [
    "override_instructions",
    "secret_exfiltration",
    "tool_or_action_abuse",
    "policy_evasion",
]


@dataclass(frozen=True)
class ContentItem:
    doc_id: str
    source_id: str
    source_type: str
    trust: str
    text: str
    language: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class ProjectionEvidence:
    polarity: List[int]
    debug_scores_raw: List[float]
    matches: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectionResult:
    doc_id: str
    v: np.ndarray
    evidence: ProjectionEvidence


@dataclass
class OmegaParams:
    walls: List[str]
    epsilon: float
    alpha: float
    beta: float
    lam: float
    S: np.ndarray
    off_tau: float
    off_Theta: float
    off_Sigma: float
    off_theta: float
    off_N: int
    attrib_gamma: float


@dataclass
class OmegaState:
    session_id: str
    m: np.ndarray
    step: int = 0


@dataclass
class OmegaOffReasons:
    reason_spike: bool
    reason_wall: bool
    reason_sum: bool
    reason_multi: bool


@dataclass
class DocContribution:
    doc_id: str
    source_id: str
    v: np.ndarray
    e: np.ndarray
    c: float
    evidence: ProjectionEvidence


@dataclass
class OmegaStepResult:
    session_id: str
    step: int
    v_total: np.ndarray
    p: np.ndarray
    m_prev: np.ndarray
    m_next: np.ndarray
    off: bool
    reasons: OmegaOffReasons
    top_docs: List[str]
    contribs: List[DocContribution]


@dataclass
class OffAction:
    type: str
    target: str
    doc_ids: Optional[List[str]] = None
    source_ids: Optional[List[str]] = None
    tool_mode: Optional[str] = None
    allowlist: Optional[List[str]] = None
    horizon_steps: Optional[int] = None
    incident_packet: Optional[Dict[str, Any]] = None


@dataclass
class OffDecision:
    off: bool
    severity: str
    actions: List[OffAction]
    control_outcome: str = "ALLOW"


@dataclass
class ToolRequest:
    tool_name: str
    args: Dict[str, Any]
    session_id: str
    step: int


@dataclass
class ToolDecision:
    allowed: bool
    mode: str
    reason: str
    logged: bool = True
    validation_status: str = "not_checked"
    validation_reason: Optional[str] = None


@dataclass
class OffEvent:
    event: str
    schema_version: str
    timestamp: str
    session_id: str
    step: int
    reasons: List[str]
    walls_triggered: List[str]
    v_total: List[float]
    p: List[float]
    m_prev: List[float]
    m_next: List[float]
    actions: List[Dict[str, Any]]
    top_docs: List[Dict[str, Any]]
    config_refs: Dict[str, str]
    thresholds: Dict[str, Any]
    walls: List[str]
    control_outcome: str = "ALLOW"
    trace_id: Optional[str] = None
    decision_id: Optional[str] = None


@dataclass
class OmegaStepEvent:
    event: str
    schema_version: str
    timestamp: str
    session_id: str
    step: int
    v_total: List[float]
    p: List[float]
    m_prev: List[float]
    m_next: List[float]
    off: bool
    trace_id: Optional[str] = None
    decision_id: Optional[str] = None


@dataclass
class EnforcementStepEvent:
    event: str
    schema_version: str
    timestamp: str
    session_id: str
    step: int
    freeze: Dict[str, Any]
    quarantine: Dict[str, Any]
    active_actions: List[Dict[str, Any]]
    control_outcome: str = "ALLOW"
    cross_session: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    decision_id: Optional[str] = None


@dataclass
class ToolGatewayStepEvent:
    event: str
    schema_version: str
    timestamp: str
    session_id: str
    step: int
    request: Dict[str, Any]
    decision: Dict[str, Any]
    capability: Dict[str, Any]
    approval: Dict[str, Any]
    execution: Dict[str, Any]
    trace: Dict[str, Any]
    control_outcome: str = "ALLOW"
    trace_id: Optional[str] = None
    decision_id: Optional[str] = None


class Projector(Protocol):
    def project(self, item: ContentItem) -> ProjectionResult:
        ...


class TrainableProjector(Projector, Protocol):
    def fit(self, items: List[ContentItem], y: np.ndarray) -> None:
        ...


class OmegaCore(Protocol):
    def step(
        self,
        state: OmegaState,
        items: List[ContentItem],
        projections: List[ProjectionResult],
    ) -> OmegaStepResult:
        ...


class OffPolicy(Protocol):
    def select_actions(self, step_result: OmegaStepResult, items: List[ContentItem]) -> OffDecision:
        ...


class ToolGateway(Protocol):
    def enforce(self, request: ToolRequest, current_actions: List[OffAction]) -> ToolDecision:
        ...
