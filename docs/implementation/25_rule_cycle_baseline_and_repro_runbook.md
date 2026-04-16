# Rule Cycle Baseline + Repro Runbook

## Goal
Freeze one reproducible baseline and standardize one control loop for every RB step:

1. `run_eval` (full whitebox + deepset, seed=41)
2. `analyze_deepset_fn` (same split/mode/seed)
3. `extract_rule_pareto` (latest FN artifacts)

## One-time baseline freeze
Run:

```powershell
.\.venv\Scripts\python.exe scripts/run_rule_cycle.py `
  --profile dev `
  --label rb_baseline `
  --seed 41 `
  --projector-mode pi0 `
  --semantic-model-path e5-small-v2 `
  --deepset-benchmark-root data/deepset-prompt-injections `
  --deepset-split test `
  --deepset-mode full `
  --deepset-max-samples 116 `
  --whitebox-max-samples 200 `
  --whitebox-max-iters 5 `
  --whitebox-beam-width 4 `
  --whitebox-mutations 3 `
  --target-fn-coverage 0.80 `
  --freeze-baseline
```

Outputs:

1. `outputs/rule_cycle/<run_id>/cycle_manifest.json`
2. `outputs/rule_cycle/<run_id>/baseline_snapshot/*`
3. `outputs/rule_cycle/LATEST.json`
4. `outputs/rule_cycle/BASELINE_LATEST.json`

## Standard cycle for each RB patch
Run the same script without `--freeze-baseline` and with a new label:

```powershell
.\.venv\Scripts\python.exe scripts/run_rule_cycle.py --label rb_iterX --seed 41
```

### Required pre-cycle tests for RB hardening suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rb_hardening_suite.py
```

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_rb_hardening_suite.py `
  tests/test_pi0.py `
  tests/test_pi0_deepset_patterns.py
```

## Reproducibility check against frozen baseline
When needed, compare current run with baseline metrics:

```powershell
.\.venv\Scripts\python.exe scripts/run_rule_cycle.py `
  --label rb_repro_check `
  --seed 41 `
  --baseline-manifest outputs/rule_cycle/<baseline_run_id>/cycle_manifest.json `
  --require-reproducible `
  --repro-tolerance 1e-9
```

The run fails if tracked metrics drift beyond tolerance:

1. `deepset.attack_off_rate`
2. `deepset.benign_off_rate`
3. `deepset.fn_total`
4. `whitebox.base_detect_rate`
5. `whitebox.bypass_rate`
6. `hard_negatives.fp`

## Notes

1. `release_metrics.json` for pareto is auto-resolved from the latest `outputs/release_gate/*`; override with `--release-metrics-json` if needed.
2. Full command stdout/stderr for each step is stored under the run directory.

