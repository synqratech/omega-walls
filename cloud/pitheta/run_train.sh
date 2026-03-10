#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-cloud_run_001}"
DATA_DIR="${DATA_DIR:-artifacts/pitheta_data/${RUN_ID}}"
TRAIN_DIR="${TRAIN_DIR:-artifacts/pitheta_train/${RUN_ID}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN}" CHECK_TOKENIZER_PREFLIGHT="${CHECK_TOKENIZER_PREFLIGHT:-1}" bash "${SCRIPT_DIR}/check_models.sh"

TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-cloud/pitheta/train_config.yml}"
TRAIN_CONFIG_EFFECTIVE="${TRAIN_CONFIG_EFFECTIVE:-/tmp/pitheta_train_config_${RUN_ID}.yml}"
PITHETA_MODEL_PATH="${PITHETA_MODEL_PATH:-/workspace/models/deberta-v3-base}"

"${PYTHON_BIN}" - "$TRAIN_CONFIG_PATH" "$TRAIN_CONFIG_EFFECTIVE" "$PITHETA_MODEL_PATH" <<'PY'
import sys
from pathlib import Path
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
model_path = str(sys.argv[3])
raw = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
node = raw.get("pitheta_train")
if not isinstance(node, dict):
    raise ValueError("train config must contain 'pitheta_train' mapping")
node["base_model"] = model_path
dst.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
PY

"${PYTHON_BIN}" scripts/build_pitheta_dataset.py \
  --registry cloud/pitheta/dataset_registry.yml \
  --output-dir "${DATA_DIR}" \
  --seed 41 \
  --strict

"${PYTHON_BIN}" scripts/train_pitheta_lora.py \
  --train-config "${TRAIN_CONFIG_EFFECTIVE}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${TRAIN_DIR}"
