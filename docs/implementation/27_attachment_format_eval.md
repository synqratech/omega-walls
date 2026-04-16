# Attachment Format Eval (PDF/DOCX/HTML/ZIP)

## Current Snapshot Pointer

For frozen results and exact run IDs as of `2026-03-09`, see:

1. [30_reproducibility_snapshot_2026-03-09.md](30_reproducibility_snapshot_2026-03-09.md)

## Scope

1. Dataset manifest includes benign + attack samples for `pdf`, `docx`, `html`, `zip`.
2. Metrics are reported per format and globally:
   - `tp/fp/tn/fn`
   - `precision/recall`
   - `attack_off_rate/benign_off_rate`
   - `parse_success_rate`
   - `text_empty_rate/scan_like_rate`

## Command

```powershell
.\.venv\Scripts\python.exe scripts/eval_attachment_ingestion.py `
  --profile dev `
  --manifest tests/data/attachment_eval/manifest.jsonl `
  --seed 41 `
  --weekly-regression `
  --require-gate
```

## Gate

Defaults are controlled by CLI flags:

1. `--gate-attack-off-rate-ge`
2. `--gate-benign-off-rate-le`
3. `--gate-precision-ge`
4. `--gate-recall-ge`
5. `--gate-per-format-parse-success-ge`
6. `--gate-required-formats` (default: `pdf,docx,html,zip`)
7. `--gate-include-deferred` (optional, off by default)

By default gate uses `summary_core` (excludes deferred-policy bucket):
1. `zip_deferred_runtime`
2. `scan_like`
3. `text_empty`

Deferred bucket is reported separately as `summary_deferred_policy` and includes
`deferred_reasons_breakdown` for transparent diagnostics.

`status=gate_failed` is written to report when checks fail.  
With `--require-gate`, process exits with non-zero code.

## Weekly Regression

1. Use fixed `--seed 41` for reproducibility.
2. Optional baseline comparison:

```powershell
.\.venv\Scripts\python.exe scripts/eval_attachment_ingestion.py `
  --seed 41 `
  --baseline-report outputs/attachment_eval/<previous_run>/report.json
```

3. Report includes `baseline_compare.summary_delta` and `per_format_delta`.

## Pilot API Debug

Use profile `pilot` to enable document scan report endpoint:

```powershell
.\.venv\Scripts\python.exe scripts/run_api_server.py --profile pilot
```

Enabled:

1. `POST /v1/scan/attachment/document_scan_report`
2. `POST /v1/scan/attachment?debug=true`

