# PromptShield Expected Artifacts

A successful run produces:

- `artifacts/promptshield_eval/<run_id>/report.json`
- `artifacts/promptshield_eval/<run_id>/rows.jsonl`

Published frozen snapshot:

- `../results/promptshield_frozen_20260324.json`

Expected checks:

- `status == "ok"`
- `comparability_status == "non_comparable"`
- summary metrics include `attack_off_rate`, `benign_off_rate`, `precision`, `recall`, `balanced_accuracy`
