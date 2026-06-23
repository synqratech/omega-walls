# AgentDojo Expected Artifacts

A successful run produces:

- `artifacts/agentdojo_stateful_mini_eval/<run_id>/report.json`
- `artifacts/agentdojo_stateful_mini_eval/<run_id>/rows.jsonl`
- `artifacts/agentdojo_stateful_mini_eval/<run_id>/misses_by_family.json`

Published frozen snapshot:

- `../results/agentdojo_frozen_20260330.json`

Expected high-level checks for the frozen track:

- `status == "ok"`
- `all_reports_blind_eval == true` (for aggregate run format)
- scoped stateful slices (`summary_core`, `summary_cross_primary`) are reported separately from `summary_all`
