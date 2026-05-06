# Enterprise Pilot Guide [Enterprise]

This page is a bridge-only overview for OSS users.

## Scope

Enterprise includes:

- enterprise control plane CLI (`omega-walls-enterprise`),
- customer-ready incident/replay/rollback/operations documentation,
- commercial support and SLA-backed operation.

## Boundary with OSS

- OSS core runtime remains in `omega/*`.
- Enterprise implementation and docs remain in `enterprise/*`.
- OSS documentation does not include deep Enterprise runbooks/specs.

## Enterprise Docs Access

Enterprise customer-facing docs are organized as:

- `enterprise/docs/ENG/` (canonical release language, MVP-5 pages)
- `enterprise/docs/RU/` (frozen mirror in this iteration)

## Feature Flag Transparency

Some API capabilities can be technically wired in OSS builds for evaluation.
Commercial usage, operational support, and SLA commitments are Enterprise-only.
