# OSS Limitations & Roadmap (Honest Boundaries)

This page describes current OSS boundaries and the practical roadmap direction.
It is intentionally conservative: no compliance-style claims, no guaranteed dates.

## Current Limitations (OSS, today)

1. Direct single-step attacks remain a growth area.
- Current behavior is generally stronger on multi-step/stateful attack patterns than on some one-shot direct prompts.
- This is why monitor-first rollout and policy tuning remain mandatory before enforce in production.

2. No full multimodal/image inspection yet.
- Text-centric trust boundaries are implemented.
- Image-only attack payloads are not fully covered in OSS at this stage.

3. Semantic enrichment depends on external LLM providers.
- Hybrid runtime uses provider APIs for semantic signals (OpenAI baseline validated with `gpt-5.4-mini`).
- Provider/model drift can change exact benchmark numbers across time.
- In cloud semantic mode, text can leave the runtime boundary and be processed by the configured provider endpoint.
- This is a conscious tradeoff for semantic quality, not a "zero external transfer" posture.

4. Rule-only fallback is a safety continuity mode, not semantic parity mode.
- During provider quota/outage fallback, protection continuity stays on.
- Semantic depth is reduced compared with healthy hybrid operation.

5. Single-instance orchestrator scope in MVP.
- Current key/quota orchestration is designed for one runtime instance per deployment.
- Cross-node distributed key/state sync is not part of current OSS MVP.

## What This Means Operationally

- Treat OSS as a stateful runtime defense layer with explicit boundaries, not as a universal security guarantee.
- Keep alerts/approvals enabled before enforce rollout.
- Re-run smoke/eval when changing provider/model family or major config thresholds.

## Roadmap Direction (Public, non-binding)

1. Improve direct-attack recall while preserving low false-positive pressure.
- Focus: close the gap between one-shot direct prompts and distributed multi-step scenarios.

2. Add multimodal trust-boundary support.
- Focus: image/document-embedded attack surfaces in agent pipelines.

3. Reduce external semantic dependency over time.
- Focus: progressively introduce internal/tuned detection components for semantic enrichment.

4. Expand reproducible benchmark coverage.
- Focus: stronger public benchmark contract and wider attack-family coverage with pinned protocols/results.

5. Strengthen production operations ergonomics.
- Focus: clearer fallback observability, recovery workflows, and safer default runbooks at scale.

## Status Note

This document is maintained as an OSS public boundary statement.
For enterprise support scope/SLA commitments, use the Enterprise Pilot surface.
