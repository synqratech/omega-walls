# Benchmarks

This directory is the public benchmark surface for Omega Walls.

It provides a reproducibility contract for four benchmark tracks:

1. AgentDojo (stateful multi-step focus)
2. Agent3Sigma (repository-native stateful family benchmark)
3. WAInjectBench (text track)
4. PromptShield (text evaluation track)

Each track follows the same contract:

1. Download
2. Run
3. Expected artifacts
4. Caveats

## Frozen Public Runs

| Benchmark | Frozen run ID | Comparability status | Snapshot |
|---|---|---|---|
| AgentDojo | `agentdojo_prepilot_blind_20260330_193616` | scoped (stateful/multi-step positioning) | [`results/agentdojo_frozen_20260330.json`](results/agentdojo_frozen_20260330.json) |
| Agent3Sigma | `session_eval_20260620T075139Z` | scoped (stateful family mix; targeted `malicious_skill` shadow tracked separately) | [`results/agent3sigma_frozen_20260622.json`](results/agent3sigma_frozen_20260622.json) |
| WAInjectBench | `wainject_eval_w202613_20260324T125013Z` | `partial_comparison` | [`results/wainject_frozen_20260324.json`](results/wainject_frozen_20260324.json) |
| PromptShield | `promptshield_eval_w202613_20260324T164314Z` | `non_comparable` | [`results/promptshield_frozen_20260324.json`](results/promptshield_frozen_20260324.json) |

Catalog: [`results/benchmark_index.json`](results/benchmark_index.json)

## Benchmark Pages

- [AgentDojo](agentdojo/README.md)
- [Agent3Sigma](agent3sigma/README.md)
- [WAInjectBench](wainjectbench/README.md)
- [PromptShield](promptshield/README.md)

## Reproducibility Notes

- Frozen numbers are point-in-time snapshots.
- Exact values can drift across provider/model/runtime versions.
- The expected outcome is reproducible protocol and metric trend, not strict bit-identical numbers.
- Full raw benchmark dumps are intentionally not committed to this public surface.
