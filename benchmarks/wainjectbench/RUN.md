# WAInjectBench Run

Run the full text-track evaluation:

```bash
python scripts/eval_wainjectbench_text.py \
  --profile dev \
  --root data/WAInjectBench/text \
  --seed 41 \
  --artifacts-root artifacts/wainject_eval
```

Optional diagnostics (non-comparable):

```bash
python scripts/eval_wainjectbench_text.py \
  --profile dev \
  --root data/WAInjectBench/text \
  --sessionized-diagnostic \
  --artifacts-root artifacts/wainject_eval
```
