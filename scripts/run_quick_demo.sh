#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="./.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  if [[ -f "./.venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="./.venv/Scripts/python.exe"
  else
    echo "Expected existing project venv python at ./.venv/bin/python or ./.venv/Scripts/python.exe" >&2
    exit 1
  fi
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" scripts/quick_demo.py "$@"
