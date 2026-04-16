# Reproducibility Snapshot (2026-03-09)

## One-Screen Headline Summary

Canonical publication profile for this snapshot:

1. Runtime: `.venv` Python 3.12, semantic enabled (`e5-small-v2`), `seed=41`.
2. Canonical stateful run for positioning claims: `session_eval_w202611_20260309T131634Z` (core text-intrinsic `attack_off_rate=0.9792`, `benign_off_rate=0.0000`).
3. Canonical hardening progression anchor: `rb_iter3_tool_soft_20260306T153418Z_453a7fd3c715` (`deepset.attack_off_rate=0.7500`, `deepset.benign_off_rate=0.0000`).
4. Canonical strict gate: `strict_pi_eval_w202610_20260308T234103Z` (pass, `attack_off_rate=1.0000`, `benign_off_rate=0.0000`).
5. Canonical attachment gate: `attachment_eval_20260309T062851Z` (`summary_core.benign_off_rate=0.0000`, deferred-policy bucket separated).

Known caveats kept explicit:

1. `context_required` distributed slice still has residual misses.
2. Cross-session distributed/cocktail still has residual misses.
3. External comparison is partial (`WAInjectBench`) and non-comparable for `PINT` in this snapshot.

---

## Purpose

This document freezes the current evaluation state of Omega Walls (rule-based pi0 track) so that external users can reproduce results from this repository with the same datasets, commands, and settings.

Snapshot scope:

1. Rule-cycle progression (`deepset + full whitebox + FN + pareto`)
2. Strict PI-only gate
3. Attachment format eval (`pdf/docx/html/zip-deferred policy`)
4. Session-based benchmark (`distributed/cocktail`, core and cross-session)
5. External-anchor comparative runs (`PINT`, `WAInjectBench`)

Date frozen: `2026-03-09`  
Primary model setting: `pi0 + semantic e5-small-v2 (semantic.enabled=auto)`  
Primary seed: `41`

---

## Canonical Run and Claims Mapping

Use this mapping as the source of truth for OSS-facing claims:

1. Rule-based hardening quality trend:
   - `outputs/rule_cycle/rb_iter3_tool_soft_20260306T153418Z_453a7fd3c715/cycle_manifest.json`
2. Strict PI gate:
   - `outputs/strict_pi_eval/strict_pi_eval_w202610_20260308T234103Z/report.json`
3. Attachment core gate:
   - `outputs/attachment_eval/attachment_eval_20260309T062851Z/report.json`
4. Stateful/distributed positioning claim:
   - `outputs/session_eval/session_eval_w202611_20260309T131634Z/report.json`
5. Comparative contour bundle:
   - `outputs/rule_cycle/post_patch_full_20260308T220042Z_62c739b58e64_rule_20260308T220043Z_62c739b58e64/cycle_manifest.json`
   - `outputs/comparative_report/post_patch_attachment_fp_fix_smoke_20260309T062330Z_62c739b58e64_comparative/report.json`

If newer runs exist, do not silently replace these references. Publish updates only with a new dated snapshot.

---

## Canonical Source Revision (Git SHA)

Current artifact manifests in this workspace record `code_commit: "local"` (no embedded git SHA).

Publication rule:

1. Before OSS release, set and freeze `CANONICAL_GIT_SHA=<40-hex sha>` for this snapshot.
2. Keep this document and `CHANGELOG.md` aligned to that SHA.
3. Re-run at least strict + session canonical commands after checkout of the same SHA.

Command to fill before release:

```powershell
git rev-parse HEAD
```

Until this field is filled, this snapshot is reproducible by outputs/config hashes but not by a canonical git revision string.

---

## Environment and Runtime Modes

Run all commands from repo root.

External model/dataset bootstrap guide:

1. `docs/implementation/32_external_assets_bootstrap.md`

### Baseline environment

1. OS: Windows + PowerShell
2. Python package metadata: [pyproject.toml](../../pyproject.toml)
3. Dev profile: [config/profiles/dev.yml](../../config/profiles/dev.yml)
4. Semantic config: [config/pi0_semantic.yml](../../config/pi0_semantic.yml)

### Important reproducibility note (Python ABI)

Two execution modes were used during this period:

1. `.venv` (Python 3.12): semantic can initialize (`e5-small-v2`) if `torch/transformers` are available.
2. Python 3.13 interpreter with `PYTHONPATH=.vendor`: often falls back to rule-only if semantic deps are missing in `.vendor`.

Do not mix `PYTHONPATH=.vendor` with `.venv` Python 3.12 when `.vendor` contains `cp313` binary wheels.

---

## Semantic-Enabled vs Fallback Equivalence Policy

This snapshot includes runs from two execution realities:

1. Semantic-enabled (`.venv`, semantic model initialized): canonical for OSS claims.
2. Rule-only fallback (semantic init failed): diagnostic only, not headline quality numbers.

Equivalence policy:

1. We do not claim metric equivalence between semantic-enabled and fallback modes.
2. Any fallback run must be labeled `fallback` in text/tables and never replace canonical rows.
3. If semantic initialization fails during reproduction, report both:
   - fallback metrics (actual run),
   - canonical expected metrics from this document.
4. Gate decisions for publication use semantic-enabled canonical runs only.

---

## Datasets and Fixed Evaluation Inputs

### Deepset benchmark

1. Root: `data/deepset-prompt-injections`
2. Split: `test`
3. Mode: `full`
4. Seed: `41`
5. Sample size used in rule-cycle: `116`

### Strict PI holdout

1. File: `tests/data/strict_pi_holdout/strict_pi_holdout_seed41.jsonl`
2. Total: `87` (`attack=51`, `benign=36`)
3. Families: `leak, obfuscation, override, roleplay, sql, tool` + benign slices

### Session benchmark pack

1. File: `tests/data/session_benchmark/session_pack_seed41_v1.jsonl`
2. Meta: `tests/data/session_benchmark/session_pack_seed41_v1.meta.json`
3. Total sessions: `300` (`attack=210`, `benign=90`)
4. Buckets: `core=270`, `cross_session=30`
5. Eval slices: `text_intrinsic=250`, `context_required=50`

### Attachment eval manifest

1. File: `tests/data/attachment_eval/manifest.jsonl`
2. Total: `10` (`attack=5`, `benign=5`)
3. Formats: `pdf=3, docx=2, html=3, zip=2`

### External anchors

1. `WAInjectBench`: `data/WAInjectBench/text`
2. `PINT`: expected at `data/pint-benchmark/benchmark/data/benchmark_dataset.yaml` (not ready in this snapshot)
3. `BIPIA` benchmark root configured but `qa/abstract` readiness was not passed for external comparability runs

---

## Exact Commands (Reproduction)

### 1) Fast regression (required before long runs)

```powershell
python -m pytest -q `
  tests/test_pi0.py `
  tests/test_pi0_deepset_patterns.py `
  tests/test_rb_hardening_suite.py `
  tests/test_eval_session_pi_gate.py
```

### 2) Rule-cycle milestones

```powershell
python scripts/run_rule_cycle.py --profile dev --label rb_baseline --seed 41 --freeze-baseline
python scripts/run_rule_cycle.py --profile dev --label rb_step1_norm_antiobf --seed 41
python scripts/run_rule_cycle.py --profile dev --label rb_step2_gapped_override --seed 41
python scripts/run_rule_cycle.py --profile dev --label rb_iter3_tool_soft --seed 41
```

### 3) Strict PI gate

```powershell
python scripts/eval_strict_pi_gate.py `
  --profile dev `
  --holdout-jsonl tests/data/strict_pi_holdout/strict_pi_holdout_seed41.jsonl `
  --seed 41
```

Weekly mode:

```powershell
python scripts/eval_strict_pi_gate.py `
  --profile dev `
  --holdout-jsonl tests/data/strict_pi_holdout/strict_pi_holdout_seed41.jsonl `
  --seed 41 `
  --weekly-regression
```

### 4) Attachment eval

```powershell
python scripts/eval_attachment_ingestion.py `
  --profile dev `
  --seed 41 `
  --manifest tests/data/attachment_eval/manifest.jsonl
```

### 5) Session benchmark

Full pack:

```powershell
python scripts/eval_session_pi_gate.py `
  --profile dev `
  --pack tests/data/session_benchmark/session_pack_seed41_v1.jsonl `
  --seed 41 `
  --mode pi0 `
  --weekly-regression `
  --baseline-report outputs/session_eval/<baseline_run>/report.json
```

### 6) Post-patch contour and comparative

```powershell
python scripts/run_post_patch_contour.py --profile dev --seed 41 --weekly-regression
python scripts/build_comparative_report.py
```

Standalone anchors:

```powershell
python scripts/eval_pint_omega.py --profile dev --seed 41
python scripts/eval_wainjectbench_text.py --profile dev --seed 41
```

---

## Frozen Results (Key Runs)

### A) Rule-cycle progression

| Run ID | deepset.attack_off_rate | deepset.benign_off_rate | fn_total | whitebox.base_detect_rate | whitebox.bypass_rate |
|---|---:|---:|---:|---:|---:|
| `rb_baseline_20260305T101332Z_bfbf7eb68b16` | 0.5833 | 0.0000 | 25 | 0.995 | 0.0251 |
| `rb_step1_norm_antiobf_20260305T180637Z_14757a386baa` | 0.5833 | 0.0000 | 25 | 0.995 | 0.0050 |
| `rb_step2_gapped_override_20260306T062411Z_907964fdb7d9` | 0.6833 | 0.0000 | 19 | 0.995 | 0.0050 |
| `rb_iter3_tool_soft_20260306T153418Z_453a7fd3c715` | 0.7500 | 0.0000 | 15 | 0.995 | 0.0050 |

Sources:

1. `outputs/rule_cycle/rb_baseline_20260305T101332Z_bfbf7eb68b16/cycle_manifest.json`
2. `outputs/rule_cycle/rb_step1_norm_antiobf_20260305T180637Z_14757a386baa/cycle_manifest.json`
3. `outputs/rule_cycle/rb_step2_gapped_override_20260306T062411Z_907964fdb7d9/cycle_manifest.json`
4. `outputs/rule_cycle/rb_iter3_tool_soft_20260306T153418Z_453a7fd3c715/cycle_manifest.json`

### B) Strict PI gate

| Run ID | attack_off_rate | benign_off_rate | f1 | gate |
|---|---:|---:|---:|---|
| `strict_pi_eval_w202610_20260308T234103Z` | 1.0000 | 0.0000 | 1.0000 | pass |
| `strict_pi_eval_20260309T062848Z` | 0.9412 | 0.0000 | 0.9697 | pass |

Sources:

1. `outputs/strict_pi_eval/strict_pi_eval_w202610_20260308T234103Z/report.json`
2. `outputs/strict_pi_eval/strict_pi_eval_20260309T062848Z/report.json`

### C) Attachment eval (core gate excludes deferred policy bucket)

Run: `attachment_eval_20260309T062851Z`

1. `summary_core.attack_off_rate = 1.0`
2. `summary_core.benign_off_rate = 0.0`
3. `summary_deferred_policy` includes `zip_deferred_runtime`, `scan_like`, `text_empty`
4. `gate = passed` for `core_excluding_deferred`

Source:

1. `outputs/attachment_eval/attachment_eval_20260309T062851Z/report.json`

### D) Session benchmark progression

| Run ID | Core metric block | core.attack_off_rate | core.benign_off_rate | cross_session.attack_off_rate |
|---|---|---:|---:|---:|
| `session_eval_w202611_20260309T073245Z` | `summary_core` (legacy schema) | 0.4389 | 0.0111 | 0.2333 |
| `session_eval_w202611_20260309T100013Z` | `summary_core_text_intrinsic` | 0.9861 | 0.1000 | 0.9000 |
| `session_eval_w202611_20260309T131634Z` | `summary_core_text_intrinsic` | 0.9792 | 0.0000 | 0.8333 |
| `session_eval_w202611_20260309T141505Z` | `summary_core_text_intrinsic` | 1.0000 | 0.0444 | 0.9000 |

Additional notes for `session_eval_w202611_20260309T141505Z`:

1. `cocktail` core text-intrinsic: `attack_off_rate=1.0`
2. `distributed_wo_explicit` context-required: `attack_off_rate=0.8611`
3. Remaining misses:
   - `cocktail::cross_session`: `sess-xs-atk-002`, `sess-xs-atk-037`
   - `distributed_wo_explicit::context_required`: `sess-core-atk-076`, `080`, `163`, `175`, `210`
   - `distributed_wo_explicit::cross_session`: `sess-xs-atk-021`
4. Remaining core benign FP sessions:
   - `sess-core-ben-017`, `sess-core-ben-072`, `sess-core-ben-083`, `sess-core-ben-087`

Sources:

1. `outputs/session_eval/session_eval_w202611_20260309T073245Z/report.json`
2. `outputs/session_eval/session_eval_w202611_20260309T100013Z/report.json`
3. `outputs/session_eval/session_eval_w202611_20260309T131634Z/report.json`
4. `outputs/session_eval/session_eval_w202611_20260309T141505Z/report.json`
5. `outputs/session_eval/session_eval_w202611_20260309T141505Z/misses_by_family.json`

### E) External-anchor status

1. `PINT` run `pint_eval_20260309T062852Z`: `dataset_not_ready`, `comparability_status=non_comparable`
2. `WAInjectBench` run `wainject_eval_20260309T062856Z`:
   - `attack_off_rate=0.25`, `benign_off_rate=0.0`
   - `comparability_status=partial_comparison`
3. Comparative report run `post_patch_attachment_fp_fix_smoke_20260309T062330Z_62c739b58e64_comparative`:
   - `direct_comparison=[]`
   - `partial_comparison=[WAInjectBench]`
   - `non_comparable=[PINT]`

Sources:

1. `outputs/pint_eval/pint_eval_20260309T062852Z/report.json`
2. `outputs/wainject_eval/wainject_eval_20260309T062856Z/report.json`
3. `outputs/comparative_report/post_patch_attachment_fp_fix_smoke_20260309T062330Z_62c739b58e64_comparative/report.json`

---

## Interpretation Snapshot (What is proven as of 2026-03-09)

1. Rule-based hardening achieved strong progression on deepset (`0.5833 -> 0.7500`) with zero benign offs in deepset and stable whitebox guardrails.
2. Strict PI holdout gate passes.
3. Attachment core gate passes after deferred-policy separation (`zip + scan_like + text_empty` deferred bucket).
4. Session benchmark demonstrates strong stateful behavior on many slices, but there is still a residual tail:
   - some benign false-offs in one semantic-enabled run,
   - remaining misses in `context_required` distributed slice,
   - residual cross-session misses.

Therefore current state is suitable for a first pilot/demo baseline, with transparent caveat that distributed context-required and residual benign/cross-session tails remain open for v1.1 hardening.

---

## Artifact Publication Checklist (for OSS release)

Publish these files together:

1. This document
2. Rule-cycle manifests:
   - `outputs/rule_cycle/rb_baseline_20260305T101332Z_bfbf7eb68b16/cycle_manifest.json`
   - `outputs/rule_cycle/rb_step2_gapped_override_20260306T062411Z_907964fdb7d9/cycle_manifest.json`
   - `outputs/rule_cycle/rb_iter3_tool_soft_20260306T153418Z_453a7fd3c715/cycle_manifest.json`
3. Strict/attachment/session reports listed above
4. Comparative report JSON/MD
5. Session pack + meta + strict holdout + attachment manifest (in `tests/data/...`)

