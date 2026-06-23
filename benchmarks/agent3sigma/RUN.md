# Agent3Sigma Run

Run the full stateful-only benchmark baseline:

```bash
python scripts/eval_session_pi_gate.py \
  --profile prod_api \
  --mode hybrid_api \
  --pack tests/data/session_benchmark/agent3sigma_stage_advance_v1/runtime/session_pack.jsonl \
  --labels-pack tests/data/session_benchmark/agent3sigma_stage_advance_v1/labels/session_pack_labels.jsonl \
  --strict-projector \
  --enable-effects-shadow \
  --artifacts-root artifacts/agent3sigma_stateful_only_prod_api
```

Run the full `malicious_skill` family slice in SkillBox-only shadow mode:

```bash
python scripts/eval_session_pi_gate.py \
  --profile prod_api \
  --mode hybrid_api \
  --pack artifacts/agent3sigma_malicious_skill_full_input/runtime/session_pack.jsonl \
  --labels-pack artifacts/agent3sigma_malicious_skill_full_input/labels/session_pack_labels.jsonl \
  --strict-projector \
  --provenance-mode segmented \
  --artifacts-root artifacts/agent3sigma_malicious_skill_full_run_skillbox_only
```

Notes:

- The frozen public headline is the first command above.
- The second command is a targeted diagnostic slice used to track `malicious_skill` recovery work.
- For live provider quality claims, use an unsandboxed run path; sandboxed local runs can fail with environment-specific egress errors that are not representative of the runtime logic.
