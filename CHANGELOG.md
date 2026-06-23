# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Enterprise release foundation

- Added canonical release, compatibility, deployment, upgrade, rollback, security-update and support policies.
- Added independent runtime/API/config/state/license/policy/vision version metadata and machine-readable version surfaces.
- Added strict release-manifest, compatibility and offline-license schemas plus manifest generation and verification tooling.
- Added local Ed25519 enterprise license verification, feature and major-version entitlements, test-key rejection, key rotation semantics and fail-closed enterprise production profiles.
- Extended the release gate with policy, version, schema, licensing cryptographic self-test and private-key hygiene checks.

### Fixed

- Closed the forwarded-header HTTPS bypass: production CLI disables Uvicorn proxy rewriting, wildcard proxy trust is rejected, and `proxy_tls` accepts forwarded TLS only from configured proxy CIDRs.
- The API CLI now resolves profile selection as explicit `--profile` > `OMEGA_PROFILE` > `prod`; the official API image starts in enforce-mode `prod` rather than `quickstart`.
- Split production profiles: `prod` is stateful text-only enforcement; `prod_vision` is explicit opt-in, local-only, dependency-checked and fail-closed for visual/OCR failures.
- Production startup validates required attachment/vision modules instead of silently degrading when an image is missing dependencies.
- Removed duplicate OCR in the local visual semantic adapter when attachment ingestion already supplied OCR text.
- Bounded RapidOCR/ONNX native thread counts so OCR cannot starve the parent watchdog; fixed-fixture local OCR p95 is now below one second on the release environment.
- Corrected attachment sandbox CPU accounting so the configured CPU budget excludes trusted cold-start/import CPU and remains deterministic.
- Restored the complete release surface: OpenClaw plugin package, public benchmark artifacts, session benchmark fallback fixtures and current characterization/frozen hashes.

### Added

- Separate `Dockerfile.vision` for the opt-in `[vision]` production image.
- Production-profile drift tests, spoofed-forwarded-header regression tests and duplicate-OCR regression coverage.

## [0.2.0] - 2026-06-14

### Added

- Completed Vision Wave C / full multimodal trust boundary:
  - bounded visual PDF page rendering and embedded image extraction for DOCX and data-URI HTML; remote HTML image fetches remain forbidden;
  - ordered multi-image semantic packets through request-scoped BlobRefs;
  - multimodal attachment support in the public SDK and RAG harness;
  - image-capable OpenAI, Anthropic, opt-in OpenAI-compatible and local vision adapters with candidate-level capabilities;
  - local `ocr_pi0` vision plus loopback-only OpenAI-compatible VLM backend;
  - tenant-scoped visual egress policy, provider processing regions and data-residency enforcement before bytes are resolved;
  - frozen architecture gate, real local multiformat quality gate and an explicit credentialed cloud-live benchmark runner.
- Added a persistent, secret-free attachment parser broker that starts before native OCR and launches one-shot resource-bounded parser sandboxes.

### Changed

- Pilot and production profiles now use `projector.mode: hybrid_api`, so configured multimodal semantics participate in actual SDK/API decisions.
- Local visual OCR applies the same security token-boundary repair and confidence filtering as the primary OCR boundary.
- Production parser wall timeout is 20 seconds while CPU, memory, output-size and file-descriptor limits remain bounded.

### Security

- PDF pages and embedded document images cannot bypass visual inspection merely because no extractable text exists.
- Identical media cannot reuse semantic cache entries across tenants or data regions.
- Cloud visual egress is deny-by-default in production; the default production route is local-only.
- Parser and OCR workers receive sanitized environments, no application credentials, no network access and explicit lifecycle/restart controls.

## [0.1.9] - 2026-06-14

### Added

- Completed Vision Wave B / OCR and spatial analysis release contour:
  - strict OCR quality policy for confidence, finite values, geometry, span count, span length and total text;
  - additive `ocr_quality` provenance and API policy-trace fields;
  - uncertainty-aware bounded regional vision policy (`zero|uncertain|always`) with deterministic tile limits;
  - enabled pilot/prod release profiles using local RapidOCR `auto` mode and `uncertain` spatial fallback;
  - deterministic comparative frozen corpus across vision-only, OCR-only, vision+OCR adjudication and region+OCR;
  - fail-closed implementation/data hashes, CI workflow and release-gate configuration.
  - prewarmed persistent RapidOCR worker pool with bounded admission, overload status, deadline, RSS watchdog and deterministic recycling;
  - fixed-fixture live RapidOCR release gate through the production attachment path;
  - request-scoped BlobRef registration for OCR adjudication crops, preserving the Wave A raw-media boundary.

### Changed

- The default `ocr` extra now installs the production RapidOCR/ONNX Runtime path; PaddleOCR is available separately through `ocr-paddle`.
- OCR-only pressure remains provisional until exact span attribution and targeted visual adjudication confirm the same wall.

### Security

- Low-confidence, malformed, out-of-image and unbounded OCR spans cannot become security evidence.
- Spatial re-analysis is bounded and non-recursive; benign UI and quoted/defensive image text are explicitly adjudicated instead of blindly accumulating pressure.
- OCR worker concurrency and pending requests are bounded; overload is explicit and cannot grow an unbounded in-process queue.
- Targeted OCR crops enter the semantic path only as opaque BlobRefs; serialized core metadata is checked for raw-media leakage.

## [0.1.8] - 2026-06-14

### Added

- Completed Vision Wave A / provider-agnostic image semantic boundary:
  - request-scoped opaque BlobRef media storage;
  - candidate-level provider capabilities and explicit unsupported-image semantics;
  - immutable per-request semantic execution traces;
  - strict multimodal contracts, typed attachment OpenAPI and frozen Phase 1 gate.


## [0.1.7] - 2026-06-14

### Fixed

- P1 production security hardening:
  - tool approvals are server-side, exact-intent bound (`tenant/session/actor/tool/args hash/intent`), expiring and atomically single-use; request-provided approval flags are ignored.
  - all future filesystem adapters use canonical path containment and reject traversal, absolute paths and symlink escapes.
  - state-only `Off` events no longer attribute zero-contribution current documents; stale wall traces participate only above explicit decay thresholds.
  - webhook notifier dispatch and runtime tenant binding are corrected; approval retrieval and resolution enforce tenant ownership.
  - outbound HTTP validation is fail-closed against SSRF using HTTPS, host/port allowlists and public DNS/IP resolution.
  - API request bodies are limited while streaming, multipart parsing has bounded files/fields/part size, and requests have a server-side deadline.
  - attachment parsing verifies magic bytes, preflights PDF/DOCX/image/HTML limits and executes structured parsers in a resource-bounded child process.
  - forwarded HTTPS headers are accepted only from explicitly configured trusted proxy CIDRs.
  - projector inputs require one-to-one, duplicate-free, identical-order `doc_id` alignment.

- P0 production security hardening:
  - audit redaction patterns now match real Bearer tokens, secret assignments, private keys, common provider tokens, email addresses and phone numbers; redaction occurs before truncation.
  - projector vectors, Omega parameters/state, semantic-provider payloads and SQLite state reject NaN/Inf, negative values and invalid wall shapes fail-closed.
  - legacy built-ins `write_file` and `network_post` are removed from the default adapter registry and cannot be re-enabled by config or request flags.
  - `TOOLS_DISABLED` is now an absolute deny state; read-only bypasses are forbidden in configuration.
  - production API startup requires strong, separate environment-provided API and HMAC secrets and refuses development credentials.
  - deterministic source archives are built from a clean staging tree and blocked by a secret scanner before publication.

- Enterprise reliability and operator-truthfulness hardening:
  - runtime config sync no longer reports false-green after failed apply; applied hash/version now advance only on successful apply.
  - runtime registry now stores drift diagnostics (`desired_hash`, `desired_version`, `sync_error_at`, `last_applied_sync`).
  - control-plane `sync_state` now separates `last_applied_version` from attempted/desired versions, so failed apply no longer misreports applied version in SQLite state.
  - incident export/replay enterprise docs are now executable end-to-end with explicit required flags and `OMEGA_REPLAY_ENCRYPTION_KEY`.
  - enterprise docs now use source-run CLI as canonical install path (`python scripts/omega_walls_enterprise.py ...`), with `omega-walls-enterprise` marked as future/private alias.
  - rollback path is hardened to managed backups only (`--to-version` filename token under managed `backups_dir`), with arbitrary path rejection.
  - implicit latest-backup rollback now goes through the same managed-backup validator as explicit `--to-version`, including filename regex, link-like rejection, and managed-root boundary checks.
  - Python <3.12 rollback extraction fallback now validates archive members against traversal and link escapes before extraction.
- Hybrid API safety hardening:
  - provider semantic failure no longer suppresses soft-only PI0 signals in degrade mode; hybrid keeps rule pressure when API confirmation is unavailable.
  - `hybrid_redacted` now scrubs additional common secret classes before outbound semantic calls (`AKIA`/`ASIA`, GitHub tokens, Slack tokens, JWT-like blobs, phone numbers).
  - default hybrid config now disables the PI0-clean short fast-path skip (`short_fast_path_skip_on_pi0_clean: false`) for a safer enterprise posture.
- Config safety hardening:
  - unknown `projector.api_perception.semantic_mode` values now fail validation instead of falling back to legacy hybrid cloud behavior.
  - unknown profile names now raise a stable `profile not found: ...` error for both bundled and explicit `config_dir` loading.
  - public root `config/` and packaged `omega/config/resources/` config surfaces are now aligned for shipped layers and public profiles.
- Session/support eval privacy hardening:
  - blind runtime runner boundary now receives a dedicated `RuntimeTurnPayload` without evaluator labels/family/source metadata.
  - `eval_agentdojo_stateful_mini.py` no longer defaults to the legacy single-file pack; explicit `--pack` is required, and the legacy built-in pack now requires explicit `--allow-legacy-runtime-leakage`.

### Added

- Enterprise regression and safety tests for:
  - sync false-green prevention after failed runtime apply.
  - truthful sync-state persistence for failed apply (`applied` vs `attempted`/`desired` version split).
  - rollback path safety (managed backup resolution + archive safety checks).
  - executable enterprise docs contract for incident export/replay and source-run installation path.
- P0 No Label Leakage hardening for session/support eval pipeline:
  - dual-file pack contract (`runtime/session_pack.jsonl` + `labels/session_pack_labels.jsonl`) for session/support builders.
  - strict runtime leakage audit (forbidden keys, non-opaque IDs, label/family ID hints) with fail-closed default.
  - opaque deterministic IDs introduced for runtime payloads (`s_000001`, `a_000001`, `src_000001`).
  - runtime payload hardening across stateful target, stateless baseline A, prefix baseline C, and bare detector baseline D.
- Hybrid API regression coverage for:
  - provider outage parity (`semantic_failed` must not zero-out soft PI0 in degrade mode).
  - expanded outbound redaction patterns in `hybrid_redacted`.
  - hybrid hard-negative corpus parity with benign API-zero composition.
  - explicit short-fast-path safety contract for PI0-clean requests.
- Config loader regression coverage for:
  - invalid `semantic_mode` fail-closed validation.
  - unknown profile fail-fast behavior.
  - root-vs-packaged public config parity and explicit internal-only profile exclusions.
- Eval privacy regression coverage for:
  - runtime-only payload handoff between evaluator and runner.
  - explicit-pack requirement and legacy-pack opt-in contract for AgentDojo session eval.

## [0.1.6] - 2026-05-19

### Fixed

- Release blocker cleanup and launch UX hardening:
  - `omega-walls` no longer crashes on no-arg invocation; root help is shown safely.
  - root `config/profiles/quickstart.yml` now matches packaged quickstart (`guard_mode: monitor`).
  - packaged `sensitive_rules_only` now explicitly sets `projector.api_perception.semantic_mode: rules_only`.
  - OSS GitHub export now excludes internal-only profiles:
    - `config/profiles/sensitive_hybrid_redacted.yml`
    - `config/profiles/sensitive_local_semantic.yml`
- OSS docs contract validator hardened:
  - writable temp-dir probe with fallback, instead of existence-only `C:/tmp` selection.
  - fast default manifest/link validation path; full export check is now optional.

### Added

- New release prepublish artifact guard:
  - `scripts/prepublish_artifact_check.py`
  - validates wheel/sdist contain required launch-fix markers before upload.

## [0.1.5] - 2026-05-06

### Changed

- OSS README onboarding tightened for cross-platform startup:
  - 4-step quickstart now includes optional extras (`[api]`, `[integrations]`, `[attachments]`)
  - dual shell environment examples (Bash + PowerShell) for notifications and provider keys
  - explicit demo + smoke path (`make demo`, `smoke_monitor_mode`, `run_framework_smokes --strict`)
  - added CLI/API one-liner surface in quickstart
- PyPI package page content hardened for installer-first flow:
  - explicit OSS vs Enterprise boundary statement
  - telemetry opt-out documented for Bash and PowerShell (`OMEGA_TELEMETRY=false`)
  - framework integrations link moved to stable docs endpoint
  - project URLs/classifiers aligned with published compatibility claims
- Added release metadata gate workflow:
  - `.github/workflows/pypi-metadata-check.yml` (`python -m build` + `twine check dist/*`)

## [0.1.4] - 2026-04-21

### Added

- Public results snapshot contract for frozen reproducible README metrics:
  - `docs/public_results_snapshot.json`
  - `scripts/sync_readme_results_from_snapshot.py`
- OSS docs/export contract validation:
  - `scripts/validate_oss_docs_contract.py`
  - `tests/test_oss_docs_link_contract.py`
  - `tests/test_public_results_snapshot_contract.py`
- Startup preflight UX hints in terminal output:
  - explicit setup hints when `notifications.enabled=false`
  - explicit Slack/Telegram env hints when channel config is missing
  - semantic fallback hint (`transformers`/`torch`) when semantic projector is inactive

### Changed

- Canonical OSS entrypoint is now `README.md` with:
  - explicit two-phase onboarding (`monitor` first, then required alerts/approvals hardening)
  - framework route map (`install -> adapter wiring -> strict smoke -> alerts setup -> API run`)
  - frozen run-ID results policy and baseline-D model scope disclaimer (`gpt-5.4-mini`)
- GitHub OSS export manifest is tightened to curated English docs and excludes:
  - RU/historical/research docs from public export
  - legacy ambiguous docs and private/internal layers
- `docs/README.md` is reshaped into a strict canonical onboarding order.
- Startup outreach defaults updated to current public contacts:
  - GitHub: `https://github.com/synqratech/omega-walls`
  - LinkedIn: `https://www.linkedin.com/in/anvifedotov/`
  - contact email: `anton.f@synqra.tech`

## [0.1.3] - 2026-04-20

### Added

- Official adapter coverage for six agent frameworks with a unified fail-closed contract:
  - `OmegaLangChainGuard`
  - `OmegaLangGraphGuard`
  - `OmegaLlamaIndexGuard`
  - `OmegaHaystackGuard`
  - `OmegaAutoGenGuard`
  - `OmegaCrewAIGuard`
- Shared adapter runtime contract for custom integrations:
  - `OmegaAdapterRuntime`, `AdapterSessionContext`, `AdapterDecision`, `ToolGateDecision`
  - typed exceptions: `OmegaBlockedError`, `OmegaToolBlockedError`
- OpenClaw integration (in-repo npm plugin):
  - guarded hooks + tool gate + WebFetch guard
  - strict local bridge to `omega-walls-api` with API key + HMAC/nonce/timestamp
- Monitor and explainability surface:
  - explicit `runtime.guard_mode` (`monitor|enforce`)
  - `omega-walls report` and `omega-walls explain --session <id>`
  - API collector status endpoint: `GET /v1/monitor/health`
- Structured logging contract:
  - canonical `OmegaLogEvent` model and normalization
  - sanitized JSON structured logging path
- Secure agent scaffold:
  - Copier template: `templates/secure_agent_template`
  - generator: `scripts/init_secure_agent_template.py`
- Canonical benchmark entrypoint:
  - `scripts/run_benchmark.py`
  - standard outputs: `report.json`, `scorecard.csv`, `dataset_manifest.json`
- Unified validation stands:
  - framework contract/workflow/stress matrix stand
  - real workflow stand for LangChain + OpenClaw e2e
- Custom integration runbook:
  - `docs/custom_integration_from_scratch.md`

### CI / Quality Gates

- FW-001 release gate is now codified as a dedicated CI workflow:
  - `.github/workflows/fw001-release-gate.yml`
  - coverage gate: `>= 85%`
  - perf gate: `<= 15%` overhead vs frozen baseline
- Docs executable validation is enforced:
  - `.github/workflows/docs-examples-smoke.yml`
  - `tests/test_docs_examples.py`
  - `tests/test_docs_reliability_contract.py`
- OpenClaw plugin CI contract added:
  - `.github/workflows/openclaw-plugin-ci.yml`

### Changed

- Documentation is now adoption-first and executable-first:
  - framework quickstart with all six adapters and smoke commands
  - clear OpenClaw and notifications connector docs
  - reliability runbooks for debugging, policy tuning, and continuity
- Packaging and release docs were tightened for reproducibility and hygiene.
- Docker API release path now includes multi-arch GHCR pipeline and runbook.

## [0.1.2] - 2026-04-14

### Added

- 5-minute OSS demo path for repo-clone onboarding:
  - unified orchestrator CLI: `scripts/quick_demo.py`
  - one-command wrappers:
    - `scripts/run_quick_demo.ps1`
    - `scripts/run_quick_demo.sh`
  - compact summary output with:
    - `session_attack_off_rate`
    - `session_benign_off_rate`
    - `mssr_core`
    - `mssr_cross_primary`
    - explicit `blocked behavior observed` signal
  - explicit semantic fallback warning when `semantic_active=false` (run does not fail).
- Advanced quick-demo dataset mode:
  - `--dataset-source agentdojo_runs` builds mini-pack from local AgentDojo cached runs before evaluation.
- Notifications & Human Escalation v1:
  - new notification subsystem (`omega.notifications`) with async dispatcher, approval store, Slack/Telegram providers
  - API callback endpoints:
    - `POST /v1/notifications/callback/slack`
    - `POST /v1/notifications/callback/telegram`
  - approval lifecycle endpoints:
    - `GET /v1/approvals/{approval_id}`
    - `POST /v1/approvals/{approval_id}/resolve`
  - API/runtime response additions (when applicable): `approval_required`, `approval_id`, `approval_status`
  - new config layer `notifications.yml` (bundled + filesystem)
  - monitoring runbook: `docs/monitoring_alerts.md`
- FW-008 Docker multi-arch API packaging:
  - root multi-stage `Dockerfile` + `.dockerignore` for API-only runtime image
  - image hygiene gate script: `scripts/check_docker_image_hygiene.py`
  - CI workflow for multi-arch build/smoke and GHCR publish:
    - `.github/workflows/docker-multiarch-ghcr-api.yml`
  - Docker quickstart docs (`docker run` + `/healthz` + `/v1/scan/attachment`) in README and evaluation docs

### Changed

- README now documents a dedicated **5-Minute Demo (Repo Clone)** flow with:
  - API-key-based default (`hybrid_api`)
  - cross-platform one-command launch examples
  - offline fallback (`--mode pi0`) troubleshooting path.
- `omega-walls` CLI default profile switched from `dev` to `quickstart` for pip-first onboarding.
- Release notes now include latency optimization references in:
  - `docs/reports/omega_latency_optimization_plan_20260403.md`
  - `docs/reports/omega_latency_benchmark_20260403.md`

### Fixed

- Packaging metadata hygiene:
  - added root `LICENSE` file to match declared Apache-2.0 license and eliminate sdist warning.

## [0.1.1] - 2026-04-02

### Changed

- README was simplified for PyPI/GitHub first-contact readability:
  - shorter top-level narrative
  - clearer install + quickstart flow
  - compact optional runtime modes
  - explicit security model and limitations
  - cleaned documentation section
- SDK default profile switched to `quickstart` for low-friction onboarding in clean environments.
- `omega-walls-api` now returns a friendly missing-dependency message (with install hint for `omega-walls[api]`) instead of raw traceback when API extras are absent.

## [0.1.0] - 2026-04-01

First public package release for `omega-walls`.

### Added

- Public SDK facade and import contract:
  - `from omega import OmegaWalls`
  - typed result model: `DetectionResult`, `GuardDecision`, `GuardAction`
  - typed SDK error model: `OmegaConfigError`, `OmegaAPIError`, `OmegaRuntimeError`, etc.
- Package-safe config loading from bundled resources (`omega.config.resources`), so install/import works without copying `config/` folders.
- CLI entrypoints in package metadata:
  - `omega-walls`
  - `omega-walls-api`
- Low-friction `quickstart` profile for local onboarding.
- Package install contract checks:
  - wheel install smoke script (`scripts/smoke_package_install.py`)
  - CI workflow for Linux/Windows installability (`.github/workflows/package-install-smoke.yml`)

### Changed

- README quickstart was rewritten as "3-step" onboarding with clear split:
  - `rule-only`
  - `hybrid_api`
- Optional dependency groups are explicit and decomposed:
  - `omega-walls[api]`
  - `omega-walls[integrations]`
  - `omega-walls[attachments]`
- `pyproject.toml` metadata was completed for publication:
  - `authors`, `license`, `classifiers`, `keywords`, `project.urls`.

### Packaging hygiene

- Distribution scope constrained to `omega*` packages.
- Exclusions for local artifacts/models/secrets are enforced (`MANIFEST.in` + tests).

## [2026-03-09] - Rule-based hardening milestone (stateful PI firewall, pre-v1 OSS snapshot)

### Added

- Rule-based hardening pipeline with reproducible seed-based cycles (`seed=41`):
  - `scripts/run_rule_cycle.py`
  - `scripts/analyze_deepset_fn.py`
  - `scripts/extract_rule_pareto.py`
- Runtime normalization and anti-obfuscation preprocessing in `pi0`:
  - NFKC normalization
  - zero-width cleanup
  - bounded wrapper decoding (`base64-lite`, `url-lite`)
  - markdown/html context extraction for analysis text
- Token/gapped rules engine (DNF-lite style) integrated into `pi0` scoring (override-focused first, then targeted expansions).
- Deterministic hardening regression suite:
  - `tests/test_rb_hardening_suite.py`
  - deterministic fuzz helpers and case builders
  - fixed-size regression packs with blocking vs observe contracts
- Attachment ingestion pilot for local retriever:
  - PDF (`pypdf`), DOCX (`python-docx`), HTML (`bs4/lxml`)
  - extraction flags (`text_empty`, `scan_like`, hidden-html contexts)
  - per-format eval script and artifacts (`scripts/eval_attachment_ingestion.py`)
- HTTP API layer for attachment scan:
  - `POST /v1/scan/attachment`
  - JSON/multipart inputs (binary, base64, or extracted text)
  - structured verdict output (`risk_score`, `verdict`, `reasons`, `evidence_id`, `policy_trace`)
- API security hardening:
  - proxy-TLS enforcement checks
  - API key + HMAC request signing and anti-replay nonce cache
  - safe structured logs (no raw payload logging)
  - optional RS256 JWS attestation support
- Session-based benchmark framework (stateful, no reset inside session):
  - pack builder: `scripts/build_session_benchmark_pack.py`
  - evaluator: `scripts/eval_session_pi_gate.py`
  - metrics: session off-rates, time-to-off, late-detect, never-detected by family, cross-session slice
- Post-patch comparative orchestration:
  - `scripts/run_post_patch_contour.py`
  - external anchors: `scripts/eval_pint_omega.py`, `scripts/eval_wainjectbench_text.py`
  - unified report builder: `scripts/build_comparative_report.py`
- Reproducibility and release docs:
  - `docs/implementation/30_reproducibility_snapshot_2026-03-09.md`
  - `docs/implementation/31_oss_repo_curation_internal.md`
  - `docs/implementation/32_external_assets_bootstrap.md`
  - `docs/implementation/33_wainjectbench_text_eval_2026-03-09.md`
- WAInjectBench reporting utility:
  - `scripts/build_wainjectbench_text_report.py`
  - SVG chart assets under `docs/assets/wainjectbench_text_eval/<run_id>/`

### Changed

- Attachment evaluation gate semantics:
  - `summary_core` now excludes deferred-policy cases (`zip_deferred_runtime`, `scan_like`, `text_empty`)
  - deferred reasons are reported separately to avoid distorting core benign gate.
- Session benchmark reporting schema:
  - split into `summary_core_text_intrinsic`, `summary_context_required`, `summary_all`
  - `cross_session` remains separate and is not mixed into core gate.
- Baseline comparison behavior in session eval:
  - baseline deltas are computed only when `--baseline-report` is explicitly provided.

### Fixed

- Targeted FP cleanup for weak markers and soft-directive ambiguity:
  - tighter context gates for weak tokens (for example, `skip/previous/above`) before counting as anchor/leak intent
  - stricter soft-directive handling requiring actionable cues or role cues for risky scoring paths.
- Targeted FN recovery for narrow contact-exfil phrasing (`handphone/phone number/contact`) via bounded secret-exfil intent logic.
- Multiple evaluator and runbook stability fixes for reproducible Windows runs (`.venv` preferred, avoid `.vendor` ABI mixups).

### Metrics snapshot (frozen in this milestone)

- Rule-cycle progression on deepset:
  - `attack_off_rate`: `0.5833 -> 0.7500`
  - `benign_off_rate`: stayed `0.0000`
  - `fn_total`: `25 -> 15`
- Strict PI holdout:
  - passing runs with `benign_off_rate=0.0000`, `attack_off_rate` up to `1.0000`
- Attachment eval:
  - `summary_core.benign_off_rate=0.0000` after deferred-policy split
- Session benchmark:
  - strong gains on text-intrinsic cocktail/distributed slices
  - remaining residual tail in context-required and some cross-session misses (documented in reproducibility snapshot).

### Compatibility

- No breaking public runtime API changes for existing core projector usage.
- New APIs and scripts are additive.

### References

- Repro snapshot: `docs/implementation/30_reproducibility_snapshot_2026-03-09.md`
- Rule-cycle runbook: `docs/implementation/25_rule_cycle_baseline_and_repro_runbook.md`
- Attachment/API security docs: `docs/implementation/26_api_attachment_security_hardening.md`
- Strict gate and comparative docs:
  - `docs/implementation/28_strict_pi_gate.md`
  - `docs/implementation/29_post_patch_contour_and_comparative.md`

