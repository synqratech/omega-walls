# Security Overview

Document version: v0.2  
Status: Draft  
Last updated: 2026-06-14  
Owner: Founder / Product Security  
Legal status: Operational overview, not a compliance certification.

## 1. Security Model

Omega Walls is a stateful runtime defense layer for AI workflows.

It focuses on:

- trust-boundary inspection before context assembly
- stateful risk accumulation across steps
- fail-closed or controlled actions before risky tool execution

## 2. Core Trust Boundaries

Primary boundaries include:

- untrusted input ingestion
- context assembly and memory carry-over
- model interaction boundary
- tool execution boundary
- output and audit boundary

## 3. Operational Posture

Default posture is self-hosted/customer-controlled.

Vendor access to customer runtime content is not required by default.

## 4. Defensive Principles

- explicit policy outcomes (`allow`, `block`, `freeze`, `quarantine`, related controls)
- observable audit events and deterministic reasoning traces where available
- conservative fallback signaling when semantic runtime is degraded

## 5. Limits

Omega Walls reduces risk but does not guarantee prevention of all attacks.
Deployment and integration quality remain critical to effective outcomes.

Cloud semantic mode boundary:
- If external semantic providers are enabled, semantic analysis can send text to that provider endpoint.
- This improves semantic coverage but introduces a data-boundary tradeoff.
- For stricter environments, use local-only modes and explicit fallback controls.
## 6. Production Enforcement Boundaries

### Exact tool approvals

Human approval is a server-owned authorization object, not a request flag. A tool approval
is bound to the tenant, session, actor, tool name, canonical argument hash and deterministic
intent ID. It expires and is consumed atomically once. A changed argument payload or replay
requires a new approval.

### Tool and filesystem containment

`TOOLS_DISABLED` denies every tool. Allowlisted execution uses a separate mode. Any future
filesystem-capable adapter must resolve its target beneath an operator-controlled root and
reject absolute paths, traversal and symlink escapes.

### Network egress

Outbound HTTP is deny-by-default. Eligible destinations require HTTPS, an explicit hostname
and port allowlist, and resolution exclusively to public IP addresses. Production adapters
must disable automatic redirects or revalidate every redirect hop.

### API ingress and proxy trust

The ASGI receive stream enforces a body limit before route buffering. Multipart files, fields
and parts are separately bounded, and every request has a server-side deadline. Forwarded
transport headers are accepted only from configured proxy CIDRs; headers from arbitrary peers
are ignored.

### Attachment sandbox

Structured attachments are selected by verified bytes rather than caller metadata. PDF, DOCX,
image and HTML inputs are preflighted for page, archive expansion, pixel and node limits, then
parsed in a separate process with wall-clock, CPU, heap and file-descriptor limits.

### Stateful attribution

Projection packets require duplicate-free, one-to-one ordered document alignment. If historical
scar mass alone causes `Off` and the current packet contributes zero pressure, attribution is
`state_only`; current documents are not blocked or quarantined as causes. Wall participation uses
explicit current-pressure and decayed-scar thresholds rather than `m > 0`.

