# Rule-Based Roadmap (Rule-Only Track, No Hybrid Yet)

Date: 2026-03-09
Owner: Omega Walls team
Scope: Continue improving `pi0` rule-based firewall before enabling hybrid mode.

## 1) Intent and Constraints

- Keep development on rule-only (`projector.mode=pi0`) in this roadmap.
- Focus on distributed/cocktail and context-heavy prompt injection scenarios.
- Do not change public API contracts while hardening rules.
- Keep safety-first: no relaxed thresholds that increase harmful leakage risk.

## 2) Success Criteria (North Star)

Primary quality targets:

1. Session benchmark (stateful):
- `summary_core_text_intrinsic.session_attack_off_rate >= 0.98`
- `summary_core_text_intrinsic.session_benign_off_rate <= 0.02` (target `<= 0.01`)
- `cross_session.session_attack_off_rate >= 0.92`

2. WAInjectBench text:
- increase `attack_off_rate` without increasing `benign_off_rate`
- short target: `attack_off_rate >= 0.52` with `benign_off_rate <= 0.015`

3. Strict PI gate:
- keep benign gate stable (`<= 0.02`, target `0.0`)
- no regression in roleplay/tool/leak families

4. Deepset + whitebox:
- no material regression from frozen baseline
- keep whitebox guardrails green (`base_detect_rate >= 0.95`, `bypass_rate <= 0.20`)

5. Attachment eval:
- keep deferred-policy separation (`zip + scan_like + text_empty`)
- `summary_core` benign gate must not regress

## 3) Fixed Benchmark Pack (Run Every Week)

Run the same fixed set with seed `41`:

1. Rule cycle (`deepset + whitebox + FN + pareto`)
2. Strict PI gate
3. Session benchmark (`session_pack_seed41_v1.jsonl`)
4. Attachment eval (per-format)
5. WAInjectBench text

Reporting requirement:

- Publish one run manifest and one compact weekly delta table.
- Compare only against explicit baseline report paths (no implicit baseline fallback).

## 4) Work Plan by Iteration (4 timeboxed sprints)

## Sprint R1: FP Cleanup on Weak Markers (1 week)

Goal: reduce false positives in benign text sources (`email/comment/web_text`) without recall loss.

Changes:

1. Tighten weak marker context gates:
- `skip|previous|above` require nearby `instruction|prompt|policy|system|developer`.
- apply to override and weak prompt-leak branches.

2. Soft-directive guard:
- `i want you ...` requires action verb and/or role cue.

3. Regression additions:
- lock known benign FP cases in unit tests.

Acceptance:

- session core benign off rate drops toward `<= 0.02`
- no drop > `0.02` in session core attack off rate
- strict/deepset no-regression

## Sprint R2: Context-Required Chain Rules (1 week)

Goal: close context-heavy misses (`EIA_wo_EI`, `VWA_adv_wo_EI`) using narrow multi-turn chain logic.

Changes:

1. Add chain templates:
- `open tab -> read doc -> fill form -> do not ask user -> retry`
- require stage count and coercive cue; keep capped bonuses.

2. Keep slices separated:
- continue reporting `text_intrinsic` vs `context_required` independently.

Acceptance:

- measurable gain on context-required recall (target +0.10 to +0.20 absolute)
- core benign gate unchanged or better

## Sprint R3: Obfuscation and Wrapper Robustness (1 week)

Goal: improve recall on disguised text without FP drift.

Changes:

1. Normalize wrappers and obfuscation variants:
- dotted/spaced tokens
- bounded wrapper decode signals
- markdown/html context cues

2. Add observe-to-block transition only after evidence:
- keep risky transforms in observe mode until stable.

Acceptance:

- improved obfuscation slice detect-rate
- no benign off regression on strict and session benchmarks

## Sprint R4: Targeted Multilingual Rule Pack (EN/DE/ES, narrow) (1 week)

Goal: close high-impact multilingual misses only.

Changes:

1. Add narrow dictionaries and gapped clauses for:
- imperative override
- SQL/DB tool abuse
- prompt-leak wrappers

2. Add paired hard-negatives in same languages.

Acceptance:

- reduced non-ASCII misses in tracked families
- zero new hard-negative FP in multilingual suite

## 5) Rule-Only Safety Guardrails (Always On)

1. Do not change global thresholds during the same sprint as pattern expansion.
2. Add tests before adding new lexical markers.
3. Keep evidence fields for every new gate/bonus.
4. Track per-family gains and losses; reject patches with hidden regressions.

## 6) Weekly Execution Template

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

# 1) Fast regression
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_pi0.py `
  tests/test_pi0_gapped_rules.py `
  tests/test_pi0_deepset_patterns.py `
  tests/test_rb_hardening_suite.py `
  tests/test_eval_session_pi_gate.py `
  tests/test_attachment_eval_report.py

# 2) Rule cycle
.\.venv\Scripts\python.exe scripts/run_rule_cycle.py --label rb_weekly --seed 41

# 3) Strict PI gate
.\.venv\Scripts\python.exe scripts/eval_strict_pi_gate.py --profile dev --seed 41 --weekly-regression

# 4) Session benchmark
.\.venv\Scripts\python.exe scripts/eval_session_pi_gate.py `
  --profile dev `
  --pack tests/data/session_benchmark/session_pack_seed41_v1.jsonl `
  --seed 41 `
  --mode pi0 `
  --weekly-regression

# 5) Attachment eval
.\.venv\Scripts\python.exe scripts/eval_attachment_ingestion.py --profile dev --seed 41 --weekly-regression

# 6) WAInjectBench text
.\.venv\Scripts\python.exe scripts/eval_wainjectbench_text.py `
  --profile dev `
  --root data/WAInjectBench/text `
  --seed 41 `
  --weekly-regression
```

## 7) Stop/Continue Rule-Only Decision

Continue rule-only only if both are true for 2 consecutive weekly runs:

1. net gain in attack metrics on at least two independent benchmark families
2. no benign gate regression in strict/session/attachment core

If gains plateau for 2 consecutive weeks, freeze rule expansion and prepare hybrid track as next primary path.
