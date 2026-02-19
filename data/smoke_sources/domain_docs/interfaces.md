# Omega Walls: Interfaces & Contracts (v1)

This document defines the **canonical API** for Omega Walls v1.  
It is the source of truth for module boundaries and backward compatibility.

> Language note: interfaces are written in a **language-agnostic** way, with concrete
> reference signatures in **Python typing**. Implementations may be in Go/TS/etc,
> but must preserve the same semantics and JSON shapes.

---

## 1) Versioning and compatibility rules

### 1.1 Semantic versioning
- **v1.x**: backward compatible changes (additive)
- **v2.0**: breaking changes (shape or semantics)

### 1.2 Contract stability rules (hard)
1) **Do not rename** fields in public structures.
2) **Do not change** meaning/units of a field.
3) **Additive only** in v1:
   - you may add new optional fields with defaults.
4) Arrays are **ordered** and order is part of the contract:
   - `walls` order defines vector indices.
5) Unknown fields must be **ignored** by readers (forward compatibility).
6) All IDs are treated as opaque strings.

### 1.3 Wall order (v1)
`K=4` and wall index order is fixed:
1. `override_instructions`
2. `secret_exfiltration`
3. `tool_or_action_abuse`
4. `policy_evasion`

This order applies to `v`, `p`, `m`, and all wall-indexed vectors.

---

## 2) Core data structures

### 2.1 ContentItem
Atomic unit of projection, attribution, and blocking.

**JSON schema (v1)**
```json
{
  "doc_id": "doc-7",
  "source_id": "web:example.com/page",
  "source_type": "web|email|pdf|ticket|chat|other",
  "trust": "untrusted|semi|trusted",
  "language": "en|ru|... (optional)",
  "text": "..."
}
```

**Constraints**
- `doc_id`: unique within a packet (step).
- `source_id`: stable identifier for a logical source (domain / sender / file / ticket).
- `trust` default: `untrusted`.

---

### 2.2 ProjectionEvidence
Explains how the projector produced `v`.

**JSON schema (v1)**
```json
{
  "polarity": [1, 0, 1, 0],
  "debug_scores_raw": [1.12, 0.0, 0.81, 0.0],
  "matches": {
    "anchors": ["ignore"],
    "phrases": ["ignore previous instructions"],
    "struct": ["system:"],
    "windows": [{"a":"reveal","b":"token","dist":2}],
    "negations": []
  }
}
```

**Constraints**
- `polarity[k]` ∈ {-1, 0, +1}.
- `debug_scores_raw[k]` may be any real number; production uses `max(0,score)` and gating.

---

### 2.3 ProjectionResult
Output of a projector for one ContentItem.

**JSON schema (v1)**
```json
{
  "doc_id": "doc-7",
  "v": [0.18, 0.00, 0.55, 0.00],
  "evidence": { ...ProjectionEvidence... }
}
```

**Constraints**
- `v[k]` must be ≥ 0.
- `v` length must equal `K`.

---

### 2.4 OmegaParams
Parameters for Ω-core (math.md).

**JSON schema (v1)**
```json
{
  "walls": ["override_instructions","secret_exfiltration","tool_or_action_abuse","policy_evasion"],
  "epsilon": 0.10,
  "alpha": 1.0,
  "beta": 0.5,
  "lambda": 0.85,
  "S": [
    [0, 0.6, 0.4, 0.3],
    [0.6, 0, 0.7, 0.2],
    [0.4, 0.7, 0, 0.2],
    [0.3, 0.2, 0.2, 0]
  ],
  "off_thresholds": {
    "tau": 0.9,
    "Theta": 0.8,
    "Sigma": 0.9,
    "theta": 0.4,
    "N": 2
  },
  "attribution": { "gamma": 0.7 }
}
```

**Constraints**
- `S` is K×K, nonnegative, diagonal is 0.
- `0 < lambda < 1`, `alpha>0`, `beta>=0`, `epsilon>=0`.
- Threshold semantics are defined in `math.md`.

---

### 2.5 OmegaState
Session-scoped Ω state.

**JSON schema (v1)**
```json
{
  "session_id": "sess-123",
  "m": [0.0, 0.0, 0.0, 0.0],
  "step": 0
}
```

**Constraints**
- `m` length is K.
- `step` counts Ω updates in this session.

---

### 2.6 OmegaOffReasons
Reason flags for `Off`.

**JSON schema (v1)**
```json
{
  "reason_spike": true,
  "reason_wall": false,
  "reason_sum": false,
  "reason_multi": true
}
```

---

### 2.7 DocContribution
Per-doc attribution object.

**JSON schema (v1)**
```json
{
  "doc_id": "doc-7",
  "source_id": "web:example.com/page",
  "v": [0.18, 0.00, 0.55, 0.00],
  "e": [0.02, 0.00, 0.50, 0.00],
  "c": 0.52,
  "evidence": { ...ProjectionEvidence... }
}
```

**Constraints**
- `e = v ⊙ p` computed using packet-level `p` from Ω-core.
- `c = sum(e)` (L1 norm).

---

### 2.8 OmegaStepResult
Output of Ω-core for one packet.

**JSON schema (v1)**
```json
{
  "session_id": "sess-123",
  "step": 12,

  "v_total": [0.30, 0.10, 1.05, 0.00],
  "p":       [0.12, 0.05, 0.91, 0.00],
  "m_prev":  [0.22, 0.09, 0.31, 0.02],
  "m_next":  [0.28, 0.11, 0.43, 0.02],

  "off": true,
  "reasons": { "reason_spike": true, "reason_wall": false, "reason_sum": false, "reason_multi": true },

  "top_docs": ["doc-7","doc-8"],
  "contribs": [ ...DocContribution... ]
}
```

**Notes**
- `contribs` SHOULD include all docs or at least those above the γ-rule threshold.
- `top_docs` is derived from `contribs` and γ-rule.

---

### 2.9 OffAction (policy output)
A single action to be taken by the application.

**JSON schema (v1)**
```json
{
  "type": "SOFT_BLOCK|SOURCE_QUARANTINE|TOOL_FREEZE|HUMAN_ESCALATE",
  "target": "DOC|SOURCE|TOOLS|AGENT",
  "doc_ids": ["doc-7","doc-8"],
  "source_ids": ["web:example.com/page"],
  "tool_mode": "TOOLS_DISABLED|TOOLS_ALLOWLIST",
  "allowlist": ["retrieval_readonly","summarize"],
  "horizon_steps": 20,
  "incident_packet": { "...": "..." }
}
```

**Constraints**
- Only fields relevant to `type` must be present.
- Readers must ignore unknown fields.

---

### 2.10 OffDecision
Output of the policy engine.

**JSON schema (v1)**
```json
{
  "off": true,
  "severity": "L1|L2|L3",
  "actions": [ ...OffAction... ]
}
```

---

### 2.11 ToolRequest / ToolDecision (Tool Gateway)
All tool calls must go through the gateway.

**ToolRequest (v1)**
```json
{
  "tool_name": "network_post",
  "args": { "url": "https://example.com", "payload": "<REDACTED>" },
  "session_id": "sess-123",
  "step": 12
}
```

**ToolDecision (v1)**
```json
{
  "allowed": false,
  "mode": "TOOLS_DISABLED|TOOLS_ALLOWLIST",
  "reason": "TOOL_FREEZE_ACTIVE|NOT_IN_ALLOWLIST|POLICY_BLOCK",
  "logged": true
}
```

---

### 2.12 OffEvent (telemetry/audit)
Structured audit record emitted on each Off.

**JSON schema (v1)**
```json
{
  "event": "omega_off_v1",
  "timestamp": "2026-02-17T12:00:00Z",
  "session_id": "sess-123",
  "step": 12,

  "reasons": ["reason_multi","reason_spike"],
  "walls_triggered": ["override_instructions","tool_or_action_abuse"],

  "v_total": [0.30, 0.10, 1.05, 0.00],
  "p":       [0.12, 0.05, 0.91, 0.00],
  "m_next":  [0.28, 0.11, 0.43, 0.02],

  "top_docs": [
    {
      "doc_id": "doc-7",
      "source_id": "web:example.com/page",
      "c": 0.62,
      "v": [0.18, 0.00, 0.55, 0.00],
      "e": [0.02, 0.00, 0.50, 0.00],
      "evidence": { ... }
    }
  ],

  "actions": [
    {"type":"SOFT_BLOCK","target":"DOC","doc_ids":["doc-7","doc-8"]},
    {"type":"TOOL_FREEZE","target":"TOOLS","tool_mode":"TOOLS_DISABLED","horizon_steps":20}
  ]
}
```

---

### 2.13 EnforcementStepEvent (telemetry/audit)
Per-step enforcement state event emitted regardless of `off`.

**JSON schema (v1)**
```json
{
  "event": "enforcement_step_v1",
  "timestamp": "2026-02-17T12:00:00Z",
  "session_id": "sess-123",
  "step": 12,
  "freeze": {
    "active": true,
    "mode": "TOOLS_DISABLED",
    "allowlist": [],
    "freeze_until_step": 25,
    "remaining_horizon": 13
  },
  "quarantine": {
    "active": true,
    "quarantined_sources": [
      {"source_id": "web:evil.example", "until_step": 18, "remaining_horizon": 6}
    ],
    "total_quarantined": 1
  },
  "active_actions": [
    {"type":"TOOL_FREEZE","target":"TOOLS"},
    {"type":"SOURCE_QUARANTINE","target":"SOURCE"}
  ]
}
```

---

## 3) Module interfaces (canonical signatures)

### 3.1 Projector

**Responsibility**
- map `ContentItem.text` to wall pressure vector `v` (K dims),
- provide evidence and polarity gating,
- be deterministic given config.

**Python reference**
```python
from dataclasses import dataclass
from typing import Protocol, Dict, Any, List, Optional
import numpy as np

K = 4

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
    polarity: List[int]              # len K, values in {-1,0,+1}
    debug_scores_raw: List[float]    # len K
    matches: Dict[str, Any]

@dataclass
class ProjectionResult:
    doc_id: str
    v: np.ndarray                    # shape (K,), nonnegative
    evidence: ProjectionEvidence

class Projector(Protocol):
    def project(self, item: ContentItem) -> ProjectionResult:
        ...
```

**Hard requirements**
- `len(v) == K`
- `v[k] >= 0`
- `evidence.polarity` uses {-1,0,+1}
- v1 `π₀` uses **hard polarity gate** (see `math.md`)

---

### 3.2 OmegaCore

**Responsibility**
- implement `math.md` step update:
  `ε-floor`, sum aggregation, `φ`, synergy `S`, update `m`, evaluate `Off`,
  attribution `c_{t,j}`.

**Python reference**
```python
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np

@dataclass
class OmegaParams:
    walls: List[str]
    epsilon: float
    alpha: float
    beta: float
    lam: float                 # lambda
    S: np.ndarray              # KxK
    off_tau: float             # τ
    off_Theta: float           # Θ
    off_Sigma: float           # Σ
    off_theta: float           # θ
    off_N: int                 # N
    attrib_gamma: float        # γ

@dataclass
class OmegaState:
    session_id: str
    m: np.ndarray              # shape (K,)
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
    v: np.ndarray              # (K,)
    e: np.ndarray              # (K,)
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

class OmegaCore:
    def __init__(self, params: OmegaParams):
        self.params = params

    def step(self, state: OmegaState, items: List[ContentItem],
             projections: List[ProjectionResult]) -> OmegaStepResult:
        """
        - `items` and `projections` must align 1:1 by doc_id.
        - Updates state.m and state.step (or returns a new state, implementation choice).
        - Computes attribution with packet-level p.
        """
        ...
```

**Hard requirements**
- determinism given inputs
- `Off` reasons correspond exactly to `math.md`
- attribution uses `e_{t,j}=v_{t,j} ⊙ p_t`, `c = sum(e_{t,j})`
- `top_docs` computed by γ-rule

---

### 3.3 OffPolicy

**Responsibility**
- convert Ω diagnostics into product actions (SOFT_BLOCK, TOOL_FREEZE, etc.)
- be deterministic and auditable.

**Python reference**
```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class OffAction:
    type: str                 # SOFT_BLOCK|SOURCE_QUARANTINE|TOOL_FREEZE|HUMAN_ESCALATE
    target: str               # DOC|SOURCE|TOOLS|AGENT
    doc_ids: Optional[List[str]] = None
    source_ids: Optional[List[str]] = None
    tool_mode: Optional[str] = None   # TOOLS_DISABLED|TOOLS_ALLOWLIST
    allowlist: Optional[List[str]] = None
    horizon_steps: Optional[int] = None
    incident_packet: Optional[Dict[str, Any]] = None

@dataclass
class OffDecision:
    off: bool
    severity: str             # L1|L2|L3
    actions: List[OffAction]

class OffPolicy(Protocol):
    def select_actions(self, step_result: OmegaStepResult,
                       items: List[ContentItem]) -> OffDecision:
        ...
```

**Hard requirements**
- must always return at least one action when `step_result.off == True`
- must include `SOFT_BLOCK` for top docs (unless system chooses full AGENT stop)
- must include `TOOL_FREEZE` when tool wall participates (v1 default)
- must include `HUMAN_ESCALATE` when exfil wall participates (v1 default)

---

### 3.4 ToolGateway

**Responsibility**
- the single chokepoint for tool execution
- enforce current tool mode (freeze/allowlist)
- log all decisions

**Python reference**
```python
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, List

@dataclass
class ToolRequest:
    tool_name: str
    args: Dict[str, Any]
    session_id: str
    step: int

@dataclass
class ToolDecision:
    allowed: bool
    mode: str                 # TOOLS_DISABLED|TOOLS_ALLOWLIST
    reason: str               # TOOL_FREEZE_ACTIVE|NOT_IN_ALLOWLIST|POLICY_BLOCK|OK
    logged: bool = True

class ToolGateway(Protocol):
    def enforce(self, request: ToolRequest,
                current_actions: List[OffAction]) -> ToolDecision:
        """
        - `current_actions` comes from OffPolicy decisions (active freezes/quarantines).
        - If TOOLS_DISABLED is active -> deny.
        - If allowlist mode -> allow only if tool_name in allowlist.
        """
        ...
```

**Hard requirements**
- if a freeze is active, tool execution must not happen (fail closed)
- decisions are logged (for audit)

---

## 4) Cross-module invariants (end-to-end)

1) **Everything untrusted passes through π + Ω** before entering model context.
2) **All tool calls pass through ToolGateway**.
3) `doc_id` alignment is consistent across items, projections, contribs, logs.
4) `walls` ordering is consistent everywhere.
5) On Off, you can reconstruct the decision from:
   - projections (v + evidence),
   - Ω params,
   - Ω step outputs (p, m),
   - thresholds (Off reasons).

---

End of document.
