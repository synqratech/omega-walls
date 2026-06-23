# PromptShield Run

Run the published split protocol:

```bash
python scripts/eval_promptshield_text.py \
  --profile dev \
  --root data/PromptShield \
  --split validation \
  --seed 41 \
  --artifacts-root artifacts/promptshield_eval
```

Optional bounded mode for local diagnostics:

```bash
python scripts/eval_promptshield_text.py \
  --profile dev \
  --root data/PromptShield \
  --split validation \
  --max-samples 200 \
  --max-seconds 900 \
  --artifacts-root artifacts/promptshield_eval
```
