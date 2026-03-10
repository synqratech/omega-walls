#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PITHETA_MODEL_PATH="${PITHETA_MODEL_PATH:-}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECK_TOKENIZER_PREFLIGHT="${CHECK_TOKENIZER_PREFLIGHT:-1}"

if [[ -z "${PITHETA_MODEL_PATH}" ]]; then
  if [[ -d "${REPO_ROOT}/deberta-v3-base" ]]; then
    PITHETA_MODEL_PATH="${REPO_ROOT}/deberta-v3-base"
  else
    PITHETA_MODEL_PATH="/workspace/models/deberta-v3-base"
  fi
fi

if [[ -z "${SEMANTIC_MODEL_PATH}" ]]; then
  if [[ -d "${REPO_ROOT}/e5-small-v2" ]]; then
    SEMANTIC_MODEL_PATH="${REPO_ROOT}/e5-small-v2"
  else
    SEMANTIC_MODEL_PATH="/workspace/models/e5-small-v2"
  fi
fi

die() {
  echo "[check_models] ERROR: $*" >&2
  exit 1
}

warn() {
  echo "[check_models] WARN: $*" >&2
}

preflight_tokenizer() {
  local model_path="$1"
  "${PYTHON_BIN}" - "$model_path" <<'PY'
import os
import sys
from pathlib import Path

raw_model_path = str(sys.argv[1])
if os.name == "nt" and raw_model_path.startswith("/mnt/") and len(raw_model_path) > 6:
    # WSL-style path -> Windows drive path, e.g. /mnt/d/foo -> D:/foo.
    drive = raw_model_path[5].upper()
    tail = raw_model_path[6:].replace("/", "\\")
    raw_model_path = f"{drive}:{tail}"

model_path = Path(raw_model_path).resolve()
if not model_path.exists():
    raise RuntimeError(f"tokenizer preflight path does not exist: {model_path}")
try:
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"failed to import transformers: {exc}") from exc

errors = []
tok = None
for use_fast in (True, False):
    try:
        tok = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            use_fast=use_fast,
        )
        break
    except Exception as exc:
        errors.append(f"use_fast={use_fast}: {exc}")

if tok is None:
    try:
        from transformers import DebertaV2Tokenizer

        tok = DebertaV2Tokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
    except Exception as exc:
        errors.append(f"DebertaV2Tokenizer fallback: {exc}")

try:
    if tok is None:
        raise RuntimeError("all tokenizer loading strategies failed")
except Exception as exc:
    msg = str(exc)
    joined = " | ".join(errors)
    hint = " Install sentencepiece in train environment."
    raise RuntimeError(f"tokenizer preflight failed for {model_path}: {msg}. {joined}.{hint}") from exc

if tok is None:
    raise RuntimeError(f"tokenizer preflight returned None for {model_path}")

print(f"[check_models] tokenizer preflight OK: {model_path}")
PY
}

check_dir() {
  local path="$1"
  local name="$2"
  if [[ ! -d "$path" ]]; then
    die "${name} directory not found: ${path}"
  fi
}

check_model_files() {
  local model_dir="$1"
  local model_name="$2"
  local config_file="${model_dir}/config.json"
  if [[ ! -f "$config_file" ]]; then
    die "${model_name} is missing config.json in ${model_dir}"
  fi
  local weights_bin="${model_dir}/pytorch_model.bin"
  local weights_safe="${model_dir}/model.safetensors"
  if [[ ! -f "$weights_bin" && ! -f "$weights_safe" ]]; then
    die "${model_name} is missing weights file (pytorch_model.bin or model.safetensors) in ${model_dir}"
  fi
}

ensure_repo_link() {
  local target="$1"
  local link_path="$2"
  local link_name="$3"

  if [[ -L "$link_path" ]]; then
    ln -sfn "$target" "$link_path"
    return
  fi

  if [[ -d "$link_path" ]]; then
    if [[ -f "${link_path}/config.json" ]]; then
      echo "[check_models] ${link_name} already exists in repo: ${link_path}"
      return
    fi
    die "${link_name} path exists but does not look like a model directory: ${link_path}"
  fi

  if [[ -e "$link_path" ]]; then
    die "${link_name} path exists and is not a directory/symlink: ${link_path}"
  fi

  ln -s "$target" "$link_path"
}

check_dir "$PITHETA_MODEL_PATH" "PiTheta base model"
check_dir "$SEMANTIC_MODEL_PATH" "Semantic model"

check_model_files "$PITHETA_MODEL_PATH" "PiTheta base model"
check_model_files "$SEMANTIC_MODEL_PATH" "Semantic model"

if [[ "${CHECK_TOKENIZER_PREFLIGHT}" == "1" ]]; then
  preflight_tokenizer "$PITHETA_MODEL_PATH"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi not found; GPU check skipped"
else
  nvidia-smi >/dev/null 2>&1 || warn "nvidia-smi failed; continuing"
fi

export PITHETA_MODEL_PATH
export SEMANTIC_MODEL_PATH

ensure_repo_link "$PITHETA_MODEL_PATH" "${REPO_ROOT}/deberta-v3-base" "PiTheta base model link"
ensure_repo_link "$SEMANTIC_MODEL_PATH" "${REPO_ROOT}/e5-small-v2" "Semantic model link"

echo "[check_models] OK"
echo "[check_models] PITHETA_MODEL_PATH=${PITHETA_MODEL_PATH}"
echo "[check_models] SEMANTIC_MODEL_PATH=${SEMANTIC_MODEL_PATH}"
echo "[check_models] CHECK_TOKENIZER_PREFLIGHT=${CHECK_TOKENIZER_PREFLIGHT}"
