# Runtime Config (Lean OSS)

This folder intentionally contains only runtime-facing config files needed for
the OSS launch paths documented in the root `README.md`.

Kept here:
- Core runtime layers (`pi0`, `projector`, `omega`, `off_policy`, `tools`, `api`, `retriever`, `source_policy`)
- Runtime profiles (`profiles/dev.yml`, `profiles/quickstart.yml`)

Removed from Lean OSS:
- Benchmark/eval/training/release-gate configs (`bipia`, `deepset`, `pitheta_*`, `release_gate`, pilot/deepset profiles)

Important:
- The runtime loads bundled defaults from `omega/config/resources` by default.
- This top-level folder is a lean reference surface for OSS readers.
