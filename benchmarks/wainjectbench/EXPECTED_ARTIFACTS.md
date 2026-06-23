# WAInjectBench Expected Artifacts

A successful run produces:

- `artifacts/wainject_eval/<run_id>/report.json`
- `artifacts/wainject_eval/<run_id>/rows.jsonl`
- `artifacts/wainject_eval/<run_id>/wainjectbench_refs.json`

Published frozen snapshot:

- `../results/wainject_frozen_20260324.json`

Expected checks:

- `status == "ok"`
- `comparability_status == "partial_comparison"`
- summary metrics include `attack_off_rate`, `benign_off_rate`, `precision`, `recall`
