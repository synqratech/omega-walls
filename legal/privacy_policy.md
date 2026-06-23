# Privacy Policy (Early-Stage Template)

Document version: v0.1  
Status: Draft  
Last updated: 2026-05-13  
Owner: Founder / Product Security  
Legal status: Informational template only, not legal advice. Final contracting entity and commercial terms are TBD until signing.

## 1. Contracting Entity and Product Context

Omega Walls is currently operated as an early-stage project by Synqra Tech.
The contracting legal entity for paid commercial engagements is **TBD before commercial signing**.

This policy describes baseline privacy behavior for:

- `Omega Walls OSS`
- `Omega Walls Enterprise` (where separately agreed)

## 2. Deployment Model and Data Intake

Default deployment model is self-hosted / customer-controlled.

By default, vendor systems do **not** receive customer prompts, documents, memory content, tool payloads, or full runtime logs.

Vendor data intake occurs only when explicitly enabled or submitted by the customer, for example:

- support bundles
- incident investigation artifacts
- opt-in telemetry
- explicit managed-service scope in an order form

## 3. Data Categories

Potential categories in scope (depending on customer configuration and support workflow):

- Customer Data: prompts, documents, runtime events, incident artifacts
- Service Data: product version, deployment mode, anonymized instance id, event counters
- Business Data: contact details for account/support/security communication

## 4. Telemetry Baseline

Telemetry is expected to be opt-in and configurable.

Prohibited telemetry fields include raw prompts, raw documents, secrets, tokens, passwords, full stack traces with sensitive paths, IP addresses, hostnames, and direct customer identifiers unless explicitly agreed.

## 5. Retention and Deletion Baseline

Baseline policy for vendor-held artifacts (where collected):

- raw support/incident artifacts: short retention target (for example 30-90 days)
- aggregated service metrics: limited operational retention

Customer self-hosted logs remain customer-controlled.

Deletion requests can be sent to legal or security contacts. Deletion targets are best-effort operational timelines unless stricter terms are defined in a signed agreement.

## 6. Security Incident Notifications

If a confirmed security incident affects vendor-held customer data, notification is made without undue delay under the applicable contract terms.

## 7. Contact

- Legal/privacy: `legal@synqra.tech`
- Security: `security@synqra.tech`
