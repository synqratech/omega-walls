# Benchmark Data Sources

This page defines the canonical dataset profile used by `scripts/run_benchmark.py`.

## Default profile: `core_oss_v1`

All datasets below are local and repo-reproducible.

1. `tests/data/hard_negatives_50.jsonl`  
   Origin: internal canonical hard negatives.
2. `tests/data/redteam_pos_20.jsonl`  
   Origin: internal canonical positives.
3. `tests/data/redteam_obf_20.jsonl`  
   Origin: internal canonical obfuscation positives.
4. `tests/data/attack_layers/v1/manifest_all.jsonl`  
   Origin: internal attack-layers benchmark manifest.
5. `tests/data/attack_layers/v1/manifest.meta.json`  
   Origin: internal attack-layers benchmark metadata.
6. `tests/data/session_benchmark/session_pack_seed41_v1.jsonl`  
   Origin: internal canonical session benchmark.
7. `tests/data/session_benchmark/session_pack_stateful_focus_v1.jsonl`  
   Origin: internal stateful-focus session benchmark.

## Reproducibility rules

- Every benchmark run writes `dataset_manifest.json`.
- Each dataset record includes:
  - `dataset_id`
  - `path` and `path_rel`
  - `source_type`
  - `source_url`
  - `sha256`
  - deterministic `stats`
- Benchmark run is invalid when manifest is incomplete (`failed_reproducibility`).

## Baseline D model provenance

`baseline_d_bare_llm_detector` is model metadata, not dataset provenance.
It is fixed in run metadata (`model`, `base_url`, `timeout_sec`, `retries`) in `report.json`.
