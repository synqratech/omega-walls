# Post-Patch Contour + Comparative Report

## Current Snapshot Pointer

For frozen contour/comparative outputs and exact run IDs as of `2026-03-09`, see:

1. [30_reproducibility_snapshot_2026-03-09.md](30_reproducibility_snapshot_2026-03-09.md)

This runbook adds one orchestration entrypoint for post-patch validation and one unified comparative report.

## 1) Full contour run

```powershell
.\.venv\Scripts\python.exe scripts/run_post_patch_contour.py `
  --profile dev `
  --seed 41 `
  --weekly-regression `
  --semantic-model-path e5-small-v2 `
  --deepset-benchmark-root data/deepset-prompt-injections `
  --deepset-split test `
  --deepset-mode full `
  --deepset-max-samples 116 `
  --whitebox-max-samples 200 `
  --whitebox-max-iters 5 `
  --whitebox-beam-width 4 `
  --whitebox-mutations 3 `
  --bipia-benchmark-root data/BIPIA-main/benchmark `
  --bipia-mode full `
  --bipia-split test `
  --bipia-repeats 2 `
  --pint-dataset data/pint-benchmark/benchmark/data/benchmark_dataset.yaml `
  --wainject-root data/WAInjectBench/text
```

Output:

- `artifacts/post_patch_contour/<run_id>/manifest.json`
- step stdout/stderr captures for all subprocesses
- references to internal and external reports

## 2) Standalone anchor runs

```powershell
.\.venv\Scripts\python.exe scripts/eval_pint_omega.py --profile dev --seed 41
.\.venv\Scripts\python.exe scripts/eval_wainjectbench_text.py --profile dev --seed 41
```

Outputs:

- `artifacts/pint_eval/<run_id>/report.json`
- `artifacts/wainject_eval/<run_id>/report.json`
- `artifacts/external_refs/wainjectbench_refs.json`

## 3) Unified comparative report

```powershell
.\.venv\Scripts\python.exe scripts/build_comparative_report.py
```

Output:

- `artifacts/comparative_report/<run_id>/report.json`
- `artifacts/comparative_report/<run_id>/report.md`

## Notes

- Evidence policy is `benchmark-maintainer only`.
- `BIPIA` is included in external-compare only when `prepare_bipia_contexts` marks `qa/abstract` readiness as `true`.
- `WAInjectBench` is currently emitted as `partial_comparison` unless a benchmark-maintainer detector leaderboard table is available.
