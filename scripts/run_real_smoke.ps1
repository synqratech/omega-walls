$ErrorActionPreference = 'Stop'

$python = 'python'
if (Test-Path '.venv\Scripts\python.exe') {
  $python = '.venv\Scripts\python.exe'
}

if (-not $env:OMEGA_MODEL_PATH) {
  Write-Error "Set OMEGA_MODEL_PATH to a local model directory before running real smoke."
  exit 1
}

Write-Host "Using Python: $python"
& $python scripts/smoke_real_rag.py --model-path $env:OMEGA_MODEL_PATH --top-k 3 --max-new-tokens 80 --strict
