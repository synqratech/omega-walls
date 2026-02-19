# Omega Walls: Logging & Audit (v1)

This document defines the **audit/telemetry contract** for Omega Walls v1:
- the canonical `omega_off_v1` event schema,
- logging levels (per-step vs only-on-Off),
- PII/secret redaction rules,
- storage/retention/access guidance,
- and incident reproduction procedures.

Enterprise rule: **an Off must be reproducible from logs**.

---

## 1) Principles

1) **Determinism & replayability**  
   An `Off` decision must be reproducible with:
   - input packet texts (or redacted hashes + exact spans),
   - projector outputs (`v` + evidence),
   - Ω params,
   - Ω state snapshots (`m_prev`, `p`, `m_next`).

2) **Least disclosure**  
   Logs must avoid storing raw secrets and unnecessary user text. Use:
   - redaction,
   - hashing,
   - truncation,
   - allowlisted capture.

3) **Two-tier logging**  
   - *Operational logs*: light-weight and safe
   - *Audit logs*: structured and access-controlled

---

## 2) Log levels and modes

Omega must support two primary modes:

### 2.1 `OFF_ONLY` (default for production)
- Emit per-step minimal counters (optional)
- Emit full structured event **only when Off triggers**

### 2.2 `PER_STEP` (recommended for staging/debug)
- Emit one structured step event per Ω update
- Plus the Off event when Off triggers

**Config toggle**
```yaml
logging:
  mode: OFF_ONLY   # OFF_ONLY | PER_STEP
  capture_text: NEVER  # NEVER | REDACTED | ALLOWLISTED
  max_text_chars: 800
```

---

## 3) Canonical event: `omega_off_v1`

### 3.1 JSON schema (v1)
This is the **authoritative schema**. Readers must ignore unknown fields.

```json
{
  "event": "omega_off_v1",
  "schema_version": "1.0",
  "timestamp": "2026-02-17T12:00:00Z",

  "session_id": "sess-123",
  "trace_id": "trace-abc (optional)",
  "step": 12,

  "config_refs": {
    "code_commit": "gitsha",
    "pi0_config_sha256": "hex",
    "omega_config_sha256": "hex",
    "off_policy_sha256": "hex"
  },

  "walls": ["override_instructions","secret_exfiltration","tool_or_action_abuse","policy_evasion"],

  "reasons": ["reason_multi","reason_spike"],
  "walls_triggered": ["override_instructions","tool_or_action_abuse"],

  "v_total": [0.30, 0.10, 1.05, 0.00],
  "p":       [0.12, 0.05, 0.91, 0.00],
  "m_prev":  [0.22, 0.09, 0.31, 0.02],
  "m_next":  [0.28, 0.11, 0.43, 0.02],

  "thresholds": {
    "epsilon": 0.10,
    "alpha": 1.0,
    "beta": 0.5,
    "lambda": 0.85,
    "off": {"tau":0.90,"Theta":0.80,"Sigma":0.90,"theta":0.40,"N":2},
    "attrib_gamma": 0.70
  },

  "top_docs": [
    {
      "doc_id": "doc-7",
      "source_id": "web:example.com/page",
      "source_type": "web",
      "trust": "untrusted",
      "contrib_c": 0.62,

      "v": [0.18, 0.00, 0.55, 0.00],
      "e": [0.02, 0.00, 0.50, 0.00],

      "projector": {
        "name": "pi0_intent_v2",
        "polarity": [1,0,1,0],
        "debug_scores_raw": [1.12,0.0,0.81,0.0],
        "matches": {
          "anchors": ["ignore"],
          "phrases": ["ignore previous instructions"],
          "struct": ["system:"],
          "windows": [{"a":"call_tool","b":"send","dist":1}],
          "negations": []
        }
      },

      "content_ref": {
        "text_capture": "REDACTED",
        "text_sha256": "hex",
        "excerpt": "SYSTEM: Ignore previous instructions ...",
        "spans": [{"start":0,"end":58}]
      }
    }
  ],

  "actions": [
    {"type":"SOFT_BLOCK","target":"DOC","doc_ids":["doc-7","doc-8"]},
    {"type":"TOOL_FREEZE","target":"TOOLS","tool_mode":"TOOLS_DISABLED","horizon_steps":20}
  ],

  "tool_gateway": {
    "mode": "TOOLS_DISABLED",
    "active_until_step": 32
  }
}
```

### 3.2 Required vs optional fields

**Required**
- `event`, `schema_version`, `timestamp`
- `session_id`, `step`
- `config_refs` (at least config SHAs + code commit)
- `walls` ordering
- `reasons`, `walls_triggered`
- `v_total`, `p`, `m_prev`, `m_next`
- `actions`
- `top_docs[*].doc_id/source_id/contrib_c/v/e/projector.polarity/debug_scores_raw`

**Optional**
- `trace_id`
- `top_docs[*].content_ref.excerpt/spans` (only if capture policy allows)

---

## 4) Per-step event schema (optional)

When `logging.mode = PER_STEP`, emit:

```json
{
  "event": "omega_step_v1",
  "schema_version": "1.0",
  "timestamp": "...",
  "session_id": "...",
  "step": 12,
  "v_total": [ ... ],
  "p": [ ... ],
  "m_prev": [ ... ],
  "m_next": [ ... ],
  "off": false
}
```

This must be lightweight and **must not** include raw content by default.

### 4.1 Enforcement state event (`enforcement_step_v1`)

Emit once per step to audit active controls even when `off=false`.

```json
{
  "event": "enforcement_step_v1",
  "schema_version": "1.0",
  "timestamp": "...",
  "session_id": "...",
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
    {"type": "TOOL_FREEZE", "target": "TOOLS"},
    {"type": "SOURCE_QUARANTINE", "target": "SOURCE"}
  ]
}
```

---

## 5) PII and secret handling

### 5.1 What is considered sensitive
- credentials: tokens, passwords, API keys, cookies, session IDs
- personal data: emails, phone numbers, addresses, names (depending on context)
- proprietary text: internal docs, system prompts

### 5.2 Redaction policy (v1)

Default: **NEVER store raw content** in logs.

Allowed capture levels:

1) `NEVER` (default prod)
- store `text_sha256` only
- store **bounded evidence**:
  - matched token strings (allowlisted)
  - spans indices (start/end)
  - short excerpt with aggressive redaction (optional, see below)

2) `REDACTED` (staging / incident response)
- store excerpt up to `max_text_chars`
- apply redaction patterns before storage
- store SHA256 of original text

3) `ALLOWLISTED` (strict)
- store only allowlisted sources/types (e.g. synthetic tests)
- required for unit tests but not for production data

**Config**
```yaml
logging:
  capture_text: NEVER      # NEVER|REDACTED|ALLOWLISTED
  max_text_chars: 800
  allowlisted_sources: ["synthetic:", "tests:"]
```

### 5.3 Redaction patterns (minimum v1)
Apply before any excerpt is stored:

- Replace likely secrets:
  - bearer tokens: `(?i)authorization:\s*bearer\s+[A-Za-z0-9._-]+`
  - api keys / tokens: `(?i)(api[_-]?key|token|password)\s*[:=]\s*[^\s]+`
  - long base64-like strings: `[A-Za-z0-9+/]{24,}={0,2}`
  - private keys blocks: `-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY----- ...`

- Replace emails / phones (optional, recommended):
  - emails: `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`
  - phones: `\+?\d[\d\s()-]{7,}`

**Rule:** if redaction fails, **drop excerpt** and store only hash + spans.

### 5.4 Storage and access
- Audit events stored in a **separate index/bucket** with:
  - encryption at rest
  - strict IAM (security + engineering leads)
  - immutable retention policy (WORM recommended)
- Operational logs can be broader but must not contain raw content.

### 5.5 Retention (recommendation)
- Operational logs: 7–14 days
- Audit logs: 30–90 days (or per enterprise policy)
- Test logs: unlimited (synthetic only)

---

## 6) Reproducing an incident (replay procedure)

Goal: given `omega_off_v1`, re-run Ω and confirm the same `Off` and same top contributors.

### 6.1 Minimum required artifacts
From the event:
- `config_refs` (commit + config SHAs)
- `thresholds` (or pointers to configs)
- `v_total`, `p`, `m_prev`, `m_next`
- `top_docs[*].v`, `top_docs[*].projector` evidence
- `content_ref.text_sha256` (and excerpt/spans if available)

From storage:
- the original text payload for each `doc_id` *if* capture policy allows; otherwise an operator may re-fetch the source.

### 6.2 Replay steps (deterministic)
1) Checkout `code_commit`.
2) Load configs matching recorded SHAs.
3) Restore `OmegaState.m = m_prev`, `step`.
4) Recompute projection for each `doc`:
   - if raw text available: run `π0.project(text)`
   - else: validate that logged `v` matches expected using stored evidence (partial replay)
5) Run `OmegaCore.step` for the same packet ordering.
6) Verify:
   - computed `p` equals logged `p` (within tolerance)
   - computed `m_next` equals logged `m_next`
   - `Off` clause(s) match `reasons`
   - γ-rule yields the same `top_docs`

### 6.3 If raw content is unavailable
Two options:
- **Source replay**: re-fetch the source by `source_id` at the same timestamp (best-effort).
- **Projection replay**: treat logged `v_{t,j}` as inputs to Ω-core (still verifies Ω-core and policy decisions).

---

## 7) Validation & schema checks in CI

CI should validate:
- `omega_off_v1` JSON schema (required keys and array lengths)
- vectors have length K
- `walls` ordering matches expected
- `config_refs` present
- redaction does not leak obvious secrets in excerpts (regex-based)

Fail build if schema validation fails.

---

## 8) Operational dashboards (recommended)

Track:
- Off rate over time (by wall)
- Top source_ids triggering Off
- Tool freezes triggered & enforced
- False-positive investigations count
- Time-to-triage for escalations

---

End of document.
