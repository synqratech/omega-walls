# Changelog

All notable changes to this project are documented in this file.

This project follows a date-based changelog style for now (pre-tag OSS phase).

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
