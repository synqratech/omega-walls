# Security Policy

## Reporting a Vulnerability

Do not open public GitHub issues for potential security vulnerabilities.

Send reports to:
- `legal@synqra.tech`

Include:
- affected version/commit
- reproduction steps
- impact summary
- proof-of-concept or logs (redacted)

## Response Targets

- Initial acknowledgment: within 2 business days
- Triage update: within 5 business days
- Remediation timeline: shared after triage based on severity

## Coordinated Disclosure

Please allow time for validation and a coordinated fix before public disclosure.
We will credit responsible reporters unless anonymity is requested.

## Additional Security Baseline Docs

- Public security overview: `security/security_overview.md`
- Vulnerability management baseline: `security/vulnerability_management.md`

## Release and Runtime Security Controls

- Never package a working directory directly. Use
  `scripts/build_clean_source_archive.py` for deterministic source releases.
- `scripts/secret_scan.py` is a blocking release gate; provider-side secret rotation
  remains mandatory after any suspected exposure.
- Production API startup rejects missing, weak, development, or role-reused secrets.
- Numeric wall-space inputs and persisted state reject non-finite values and invalid
  shapes before any state update.
- Audit excerpts are redacted before truncation; production defaults to no raw text capture.
- `TOOLS_DISABLED` is fail-closed. The default registry contains no file-write or
  network-post adapter.
## Runtime Boundary Hardening

- Tool approvals are stored server-side and bound to the exact canonical tool intent:
  tenant, session, actor, tool name, arguments hash and intent ID. Approvals expire
  and are consumed atomically once; request fields such as `human_approved` are never trusted.
- Filesystem targets must pass canonical containment checks. Absolute paths, traversal
  segments and symlink escapes are rejected before any side effect.
- Outbound HTTP is deny-by-default and requires HTTPS, an explicit host/port allowlist,
  public DNS/IP resolution and redirect revalidation by the real adapter.
- Request bodies are limited while streaming, multipart limits are explicit, and request
  execution has a server-side deadline.
- `X-Forwarded-Proto` is trusted only when the immediate peer belongs to a configured
  trusted proxy CIDR.
- Structured attachments are magic-verified, preflighted for parser/decompression limits,
  and parsed in a separate resource-bounded process.
- Projection packets require exact ordered `doc_id` alignment. Historical-only Off events
  never blame zero-contribution current documents.
## Multimodal media boundary

Image attachment bytes are held in a request-scoped in-memory BlobRef store with TTL and SHA-256 integrity checks. Raw or base64 image content is forbidden in semantic source metadata, traces, caches, and logs. Only the selected image-capable provider adapter can resolve a BlobRef. Provider capabilities are re-evaluated for every fallback candidate, and unsupported vision follows the configured semantic failure policy without silent text-only downgrade.

The committed `vision_phase1_frozen_v1` artifact is a deterministic architecture/contract regression gate. It does not constitute a live third-party model quality claim; representative live image evaluation remains mandatory before making external production-quality claims for a provider/model revision.


## Vision Wave C controls

- PDF pages, DOCX embedded media and HTML data-URI images are converted into bounded visual assets. Remote HTML image references are never fetched by the parser.
- Visual assets enter the semantic layer only through ordered, request-scoped BlobRefs. Multi-image packets preserve source order and provenance without serializing raw media into logs, caches or traces.
- Visual egress is evaluated per tenant, provider and data region before an adapter can resolve bytes. Production denies external visual egress by default and permits only `local_vision` in the `local` processing region.
- OpenAI and Anthropic are distinct multimodal adapters. OpenAI-compatible image support is explicit opt-in; the local OpenAI-compatible backend is restricted to loopback endpoints.
- Attachment parsing uses a secret-free persistent broker which launches one-shot parser sandboxes. Parser and OCR processes have no network access, receive sanitized environments and are terminated on deadline/resource violations.
- Semantic caches include tenant and data-region identity. Cached cloud-derived results cannot cross a tenant or residency boundary.
- The committed Wave C frozen and local artifacts prove fixed-fixture architecture and local quality only. A pinned provider/model live report is mandatory before claiming production quality for a cloud visual provider.
