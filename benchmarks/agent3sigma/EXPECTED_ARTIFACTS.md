# Agent3Sigma Expected Artifacts

A successful full stateful-only run produces:

- `artifacts/agent3sigma_stateful_only_prod_api/<run_id>/report.json`
- `artifacts/agent3sigma_stateful_only_prod_api/<run_id>/rows.jsonl`
- `artifacts/agent3sigma_stateful_only_prod_api/<run_id>/misses_by_family.json`

A successful full `malicious_skill` SkillBox shadow run produces:

- `artifacts/agent3sigma_malicious_skill_full_run_skillbox_only/<run_id>/report.json`
- `artifacts/agent3sigma_malicious_skill_full_run_skillbox_only/<run_id>/rows.jsonl`
- `artifacts/agent3sigma_malicious_skill_full_run_skillbox_only/<run_id>/misses_by_family.json`

Published frozen snapshot:

- `../results/agent3sigma_frozen_20260622.json`

Expected checks:

- `status == "ok"`
- `comparability_status == "scoped_stateful_family_mix"`
- headline metrics include `session_attack_off_rate`, `session_benign_off_rate`, `precision`, `recall`
- targeted `malicious_skill` diagnostics include `skillbox_source_mismatch_count` and `simulated_skillbox_attack_recall`
