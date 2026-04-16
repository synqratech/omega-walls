# External Assets Bootstrap (Models + Datasets)

## Purpose

This document defines how to fetch heavyweight external assets that are not intended to be stored in this Git repository, and how to verify readiness before running canonical evals.

Use this together with:

1. `docs/implementation/30_reproducibility_snapshot_2026-03-09.md`
2. `docs/implementation/25_rule_cycle_baseline_and_repro_runbook.md`

## Canonical Runtime Assumptions

1. Use `.venv` (Python 3.12) as canonical runtime.
2. Do not mix `.venv` with `PYTHONPATH=.vendor` (ABI mismatch risk on Windows).
3. Use `seed=41` for reproducibility runs.

## Asset Matrix (What to Download)

### A) Required for canonical rule-based evals

1. Semantic model:
   - Local path: `e5-small-v2`
   - Upstream: `https://huggingface.co/intfloat/e5-small-v2`
   - Used by: `run_rule_cycle`, `run_eval`, strict/session/contour runs with semantic enabled
2. Deepset benchmark dataset:
   - Local path: `data/deepset-prompt-injections`
   - Upstream: `https://huggingface.co/datasets/deepset/prompt-injections`
   - Required file contract:
     - `data/deepset-prompt-injections/data/train-*.parquet`
     - `data/deepset-prompt-injections/data/test-*.parquet`
   - Used by: `run_rule_cycle`, `run_eval`, `analyze_deepset_fn`
3. WAInjectBench (text subset at minimum):
   - Local path: `data/WAInjectBench`
   - Upstream repo: `https://github.com/Norrrrrrr-lyn/WAInjectBench`
   - Upstream HF dataset card: `https://huggingface.co/datasets/Norrrrrrr/WAInjectBench`
   - Used by: session benchmark pack builder + comparative text-anchor eval

### B) Required only for selected comparative/validation stages

1. BIPIA benchmark source:
   - Local path: `data/BIPIA-main`
   - Upstream: `https://github.com/microsoft/BIPIA`
   - Note: `qa`/`abstract` context files are license-gated and must be generated locally.
2. PINT benchmark framework:
   - Local path: `data/pint-benchmark`
   - Upstream: `https://github.com/lakeraai/pint-benchmark`
   - Note: project can run with your own YAML; public benchmark dataset access can be limited.
3. Open-Prompt-Injection (used for rule mining/test-case enrichment):
   - Local path: `data/Open-Prompt-Injection-main`
   - Upstream: `https://github.com/liu00222/Open-Prompt-Injection`

### C) Optional for hybrid/pi_theta training track

1. Base encoder model:
   - `microsoft/deberta-v3-base`
   - Local path expected by current config: `deberta-v3-base` (or override path in config/scripts)

## Download Commands (PowerShell)

Run from repository root.

### 1) Tooling

```powershell
python -m pip install -U huggingface_hub datasets pyarrow
```

### 2) Models

```powershell
huggingface-cli download intfloat/e5-small-v2 `
  --local-dir e5-small-v2 `
  --local-dir-use-symlinks False
```

Optional (hybrid/pi_theta track):

```powershell
huggingface-cli download microsoft/deberta-v3-base `
  --local-dir deberta-v3-base `
  --local-dir-use-symlinks False
```

### 3) Datasets

Deepset benchmark:

```powershell
huggingface-cli download --repo-type dataset deepset/prompt-injections `
  --local-dir data/deepset-prompt-injections `
  --local-dir-use-symlinks False
```

WAInjectBench:

```powershell
huggingface-cli download --repo-type dataset Norrrrrrr/WAInjectBench `
  --local-dir data/WAInjectBench `
  --local-dir-use-symlinks False
```

BIPIA source:

```powershell
git clone https://github.com/microsoft/BIPIA.git data/BIPIA-main
```

PINT framework:

```powershell
git clone https://github.com/lakeraai/pint-benchmark.git data/pint-benchmark
```

Open-Prompt-Injection:

```powershell
git clone https://github.com/liu00222/Open-Prompt-Injection.git data/Open-Prompt-Injection-main
```

## BIPIA License-Gated Context Build

BIPIA `qa` and `abstract` contexts are not fully redistributable as raw files. Build locally:

1. NewsQA guidance: `https://github.com/Maluuba/newsqa`
2. XSum guidance: `https://github.com/EdinburghNLP/XSum`
3. Then run:

```powershell
python scripts/prepare_bipia_contexts.py `
  --benchmark-root data/BIPIA-main/benchmark `
  --newsqa-data-dir <PATH_TO_NEWSQA> `
  --strict
```

For strict BIPIA validation also set upstream commit:

```powershell
$env:BIPIA_COMMIT="<UPSTREAM_COMMIT_SHA>"
```

## Readiness Checks (Before Long Runs)

### 1) Fast filesystem contract checks

```powershell
@'
from pathlib import Path

checks = {
    "e5-small-v2": Path("e5-small-v2").exists(),
    "deepset_train_parquet": any(Path("data/deepset-prompt-injections/data").glob("train-*.parquet")),
    "deepset_test_parquet": any(Path("data/deepset-prompt-injections/data").glob("test-*.parquet")),
    "wainject_text_root": Path("data/WAInjectBench/text").exists(),
    "bipia_root": Path("data/BIPIA-main/benchmark").exists(),
    "pint_yaml_default": Path("data/pint-benchmark/benchmark/data/benchmark_dataset.yaml").exists(),
}
print(checks)
'@ | python -
```

### 2) Semantic availability sanity check

```powershell
python scripts/run_eval.py --help
```

Then enforce semantic on first real eval run:

```powershell
python scripts/run_rule_cycle.py `
  --profile dev `
  --label asset_bootstrap_smoke `
  --seed 41 `
  --semantic-model-path e5-small-v2
```

## What Is Usually NOT Published to OSS

1. Local model directories (`e5-small-v2`, `deberta-v3-base`, etc.)
2. Full heavyweight benchmark mirrors under `data/` (unless license permits and size policy allows)
3. Generated artifacts under `outputs/`

Publish instructions + manifests + small deterministic test packs instead.

## License and Compliance Note

This project references third-party models/datasets. Users are responsible for:

1. Accepting upstream licenses and usage terms.
2. Respecting redistribution limits (especially BIPIA `qa`/`abstract` source material).
3. Pinning exact revisions for regulated environments.

