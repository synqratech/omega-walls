#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-cloud_run_001}"
DATA_DIR="${DATA_DIR:-artifacts/pitheta_data/${RUN_ID}}"
TRAIN_DIR="${TRAIN_DIR:-artifacts/pitheta_train/${RUN_ID}/best}"
BASELINE_REPORT="${BASELINE_REPORT:-artifacts/deepset_eval/latest_deepset_report.json}"
REQUIRE_SEMANTIC="${REQUIRE_SEMANTIC:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN}" CHECK_TOKENIZER_PREFLIGHT="${CHECK_TOKENIZER_PREFLIGHT:-1}" bash "${SCRIPT_DIR}/check_models.sh"

ARGS=(
  scripts/eval_pitheta.py
  --checkpoint "${TRAIN_DIR}"
  --data-dir "${DATA_DIR}"
  --baseline-report "${BASELINE_REPORT}"
  --strict-gates
)

if [[ "${REQUIRE_SEMANTIC}" == "1" ]]; then
  ARGS+=(--require-semantic)
fi

"${PYTHON_BIN}" "${ARGS[@]}"
