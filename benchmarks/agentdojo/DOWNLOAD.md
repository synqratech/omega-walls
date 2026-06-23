# AgentDojo Download

Omega does not ship the full AgentDojo dataset in this repository.

Prepare input assets locally:

1. Obtain AgentDojo source assets according to its license and terms.
2. Place run assets under:
   - `data/AgentDojo/runs`
3. Build the Omega mini stateful benchmark pack:

```bash
python scripts/build_agentdojo_cocktail_mini_pack.py \
  --runs-root data/AgentDojo/runs \
  --out tests/data/session_benchmark/agentdojo_cocktail_mini_v1.jsonl \
  --meta-out tests/data/session_benchmark/agentdojo_cocktail_mini_v1.meta.json \
  --seed 41
```

Licensing and distribution of external benchmark assets remain the responsibility of the operator.
