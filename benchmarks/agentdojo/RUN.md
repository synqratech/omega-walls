# AgentDojo Run

Recommended reproduction command (stateful API-hybrid path):

```bash
python scripts/eval_agentdojo_stateful_mini.py \
  --profile dev \
  --mode hybrid_api \
  --pack tests/data/session_benchmark/agentdojo_cocktail_mini_v1.jsonl \
  --seed 41 \
  --strict-projector \
  --blind-eval \
  --api-provider openai \
  --api-model gpt-5.4-mini \
  --api-key-env OPENAI_API_KEY \
  --artifacts-root artifacts/agentdojo_stateful_mini_eval
```

Notes:

- `--blind-eval` is required for the published protocol.
- For no-key local smoke only, you can use `--mode pi0` (not comparable to frozen hybrid run).
